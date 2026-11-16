# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Veriflow: a plan-verify engineering agent harness for PLE.

Built on the kit's `taskbridge` protocol (`docs/protocol.md`). The agent
works in four phases:

  recon   -> explore the container, read the task files, write a plan
  solve   -> work the plan with bash / read / write (+ MCP when present)
  verify  -> after `declare_deliverables`, independently re-derive every
             key number with a fresh script, check units and clauses,
             fix discrepancies
  deliver -> final summary (goes to the grader)

Two deterministic guards sit outside the model: deliverable paths declared
by the model are checked to exist and be non-empty before the run can end,
and the verify phase is forced by the clock even if the model never declares
deliverables. Token usage is reported per call so the run's cost is visible
for the score-vs-cost plot the Stage 1 writeup needs.
"""

import asyncio
import json
import os
import re
from pathlib import Path

import openai
import taskbridge

HERE = Path(__file__).resolve().parent
MODEL = os.environ["MODEL_NAME"]
MAX_EFFORT_MODELS = ("claude", "glm", "gpt-6", "deepseek-v4")
EFFORT = os.environ.get("REASONING_EFFORT") or (
    "max" if any(m in MODEL.lower() for m in MAX_EFFORT_MODELS) else "high"
)
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "32000"))
MAX_TURNS = int(os.environ.get("MAX_TURNS", "120"))
VERIFY_SEC = int(os.environ.get("VERIFY_SEC", "1800"))   # reserve for verify
WRAP_UP_SEC = int(os.environ.get("WRAP_UP_SEC", "600"))  # reserve for summary
VERIFY_ROUNDS = int(os.environ.get("VERIFY_ROUNDS", "2"))
RESULT_CHARS = 20_000  # a longer tool result loses its middle
RETRY_WAITS = (5, 15, 30, 60, 120)
CONTEXT_FULL = re.compile(
    r"context.?(length|window)|prompt is too long|maximum prompt length"
    r"|exceed[^.]{0,40}(token|context)",
    re.I,
)
CUT_KEEP = 1000
MAX_NUDGES = 3
CACHE_MARK = "claude" in MODEL.lower()

client = openai.AsyncOpenAI(max_retries=5, timeout=1800)


def _read_prompt(name):
    return (HERE / "prompts" / name).read_text(encoding="utf-8")


def tool(name, description, **params):
    """Builds an OpenAI function-tool definition dictionary."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": v} for k, v in params.items()},
                "required": list(params),
            },
        },
    }


BASE_TOOLS = [
    tool(
        "bash",
        "Run a bash command (120 s limit) and return its exit code and output.",
        command="string",
    ),
    tool("read_file", "Read a text file.", path="string"),
    tool(
        "write_file",
        "Create or overwrite a file; parent folders are created.",
        path="string",
        content="string",
    ),
    tool(
        "declare_deliverables",
        "Declare the deliverable file paths you have written, exactly as the "
        "task asked for them. The harness checks each one exists and is "
        "non-empty, then starts the verification phase.",
        paths="array",
    ),
]

MCP_TOOLS = [
    tool(
        "mcp_list_tools",
        "List the tools of one of the task's MCP servers.",
        server="string",
    ),
    tool(
        "mcp_call_tool",
        "Call one tool on one of the task's MCP servers.",
        server="string",
        tool_name="string",
        arguments="object",
    ),
]


def clip(text):
    """Truncates `text` in the middle if it exceeds `RESULT_CHARS`."""
    if len(text) <= RESULT_CHARS:
        return text
    half = RESULT_CHARS // 2
    return (
        f"{text[:half]}\n[... {len(text) - RESULT_CHARS} characters cut"
        f" ...]\n{text[-half:]}"
    )


def marked(messages):
    """Marks the system prompt and newest message for Claude prompt caching."""
    if not CACHE_MARK:
        return messages
    out = list(messages)
    for i in {0, len(out) - 1}:
        m, content = out[i], out[i].get("content")
        if isinstance(content, str) and content:
            out[i] = {
                **m,
                "content": [{
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }],
            }
        else:
            out[i] = {**m, "cache_control": {"type": "ephemeral"}}
    return out


class Harness:
    """One task run: the model loop plus the phase machine."""

    def __init__(self, session):
        self.session = session
        self.messages = []
        self.turns = 0
        self.phase = "recon"          # recon -> solve -> verify -> deliver
        self.deliverables = []        # paths declared by the model
        self.verified = False         # a verify pass completed
        self.verify_rounds = 0
        self.nudges = 0
        self.head = ""                # head of an answer cut off at the limit
        self.waits = iter(RETRY_WAITS)
        self.total_in = 0
        self.total_out = 0
        self.mcp_clients = {}
        self.tools = list(BASE_TOOLS)
        if session.mcp_servers:
            self.tools += MCP_TOOLS

    async def mcp_client(self, name):
        """Returns (and caches) an MCPClient for the named task server."""
        if name not in self.mcp_clients:
            server = next(
                (s for s in self.session.mcp_servers if s.get("name") == name),
                None,
            )
            if server is None:
                raise taskbridge.TaskError(f"no MCP server named {name}")
            client = taskbridge.MCPClient(self.session, str(server["url"]))
            await client.initialize()
            self.mcp_clients[name] = client
        return self.mcp_clients[name]

    async def run_tool(self, fn):
        """Executes one model tool call in the task container."""
        name = fn["name"]
        try:
            args = json.loads(fn["arguments"] or "{}")
            if not isinstance(args, dict):
                raise TypeError("not a JSON object")
            fn["arguments"] = json.dumps(args, allow_nan=False)
        except (TypeError, ValueError) as exc:
            fn["arguments"] = "{}"
            return (
                f"error: the arguments are not a JSON object ({exc}); send "
                "the call again"
            )
        try:
            if name == "bash":
                result = await self.session.exec(args["command"])
                return clip(
                    f"exit code {result['exit_code']}\n{result['output']}"
                )
            if name == "read_file":
                return clip(
                    (await self.session.read_file(args["path"])).decode(
                        "utf-8", "replace"
                    )
                )
            if name == "write_file":
                return (
                    "wrote "
                    f"{await self.session.write_file(args['path'], args['content'])}"
                    f" bytes to {args['path']}"
                )
            if name == "declare_deliverables":
                return await self.declare(args.get("paths"))
            if name == "mcp_list_tools":
                client = await self.mcp_client(args["server"])
                tools = await client.list_tools()
                return clip(json.dumps(
                    [{"name": t["name"], "description": t["description"],
                      "inputSchema": t["inputSchema"]} for t in tools],
                    indent=1,
                ))
            if name == "mcp_call_tool":
                client = await self.mcp_client(args["server"])
                result = await client.call_tool(
                    args["tool_name"], args.get("arguments") or {}
                )
                return clip(json.dumps(result, indent=1)[:RESULT_CHARS])
            return f"error: there is no tool named {name}"
        except (taskbridge.TaskError, taskbridge.MCPError, KeyError,
                ValueError) as exc:
            return f"error: {type(exc).__name__}: {exc}"

    async def declare(self, paths):
        """Deterministic deliverable check; moves the run to verify."""
        if not isinstance(paths, list) or not paths:
            return ("error: declare_deliverables needs a non-empty list of "
                    "file paths")
        if not all(isinstance(p, str) and p for p in paths):
            return "error: every deliverable path must be non-empty text"
        self.deliverables = list(paths)
        missing, empty, ok = [], [], []
        for path in paths:
            result = await self.session.read_result(path)
            if not result.get("ok"):
                missing.append(f"{path} ({result.get('error')})")
            elif not result.get("size"):
                empty.append(path)
            else:
                ok.append(f"{path} ({result['size']} bytes)")
        report = []
        if ok:
            report.append("found:\n- " + "\n- ".join(ok))
        if missing:
            report.append("MISSING:\n- " + "\n- ".join(missing))
        if empty:
            report.append("EMPTY (0 bytes):\n- " + "\n- ".join(empty))
        if missing or empty:
            return (
                "Deliverable check FAILED — fix these before verification:\n"
                + "\n".join(report)
            )
        self.phase = "verify"
        return (
            "Deliverable check passed:\n" + "\n".join(report)
            + "\n\nStarting the verification phase: re-read each "
            "deliverable, recompute its key numbers with a fresh script, "
            "check units and clauses, and fix every discrepancy."
        )

    async def shorten(self, messages):
        """Truncates the older half of long tool results to CUT_KEEP chars."""
        long = [
            n
            for n, m in enumerate(messages)
            if m["role"] == "tool" and len(m["content"]) > 2 * CUT_KEEP
        ]
        if not long:
            return False
        cut = long[: max(1, len(long) // 2)]
        for n in cut:
            messages[n]["content"] = (
                messages[n]["content"][:CUT_KEEP]
                + "\n[... cut to save context; run the tool again if needed]"
            )
        await self.session.log(f"context full: cut {len(cut)} old tool results")
        return True

    async def recover(self, exc, until):
        """Handles a failed model call; returns whether to retry."""
        status = getattr(exc, "status_code", None)  # None: no HTTP answer
        if status == 400 and CONTEXT_FULL.search(str(exc)):
            return await self.shorten(self.messages)
        wait = next(self.waits, None)
        if wait is None or not (
            status is None or status in (408, 409, 429) or status >= 500
        ):
            return False
        left = self.session.time_left()
        if left is not None and left > until:
            wait = min(wait, left - until)
        await self.session.log(
            f"model call failed ({str(exc)[:200]}); again in {wait:.0f} s"
        )
        await asyncio.sleep(wait)
        return True

    def time_state(self):
        """Returns (left, wrap_up, verify_due) for the clock checks."""
        left = self.session.time_left()  # None when the task has no limit
        if left is None:
            return None, False, False
        return (
            left,
            left < WRAP_UP_SEC,
            left < WRAP_UP_SEC + VERIFY_SEC and self.phase in ("recon", "solve"),
        )

    async def solve(self):
        """Runs the phase machine until the final summary is ready."""
        system = _read_prompt("system.md").format(workdir=self.session.workdir)
        verify_prompt = _read_prompt("verify.md")
        self.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": self.session.instruction},
        ]
        while True:
            left, wrap_up, verify_due = self.time_state()
            if self.turns >= MAX_TURNS:
                return await self.finish(
                    "Turn budget reached. " + self.summary_line()
                )
            if verify_due and not self.verified:
                # Clock forces verification even if the model never declared
                # deliverables: ask it to declare them first, then verify.
                self.messages.append({
                    "role": "user",
                    "content": (
                        "Verification time is approaching. Stop solving, "
                        "write any remaining deliverables now, call "
                        "declare_deliverables with their paths, then verify "
                        "them as instructed."
                    ),
                })
                self.phase = "verify"
            elif wrap_up and (
                not self.messages or self.messages[-1].get("role") != "user"
            ):
                self.messages.append({
                    "role": "user",
                    "content": (
                        "Time is almost up. Stop and reply with your summary: "
                        "each deliverable, where it is, the key results, and "
                        "what you verified."
                    ),
                })
            if self.phase == "verify" and not self.verified:
                self.messages.append(
                    {"role": "user", "content": verify_prompt}
                )
                self.verified = True  # one verify pass, model-driven
                self.verify_rounds += 1
            self.turns += 1
            try:
                reply = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=MODEL,
                        messages=marked(self.messages),
                        tools=self.tools,
                        tool_choice="none" if wrap_up else "auto",
                        reasoning_effort=EFFORT,
                        max_completion_tokens=MAX_OUTPUT_TOKENS,
                    ),
                    None if left is None or wrap_up else left - WRAP_UP_SEC,
                )
            except asyncio.TimeoutError:
                continue
            except openai.APIError as exc:
                if await self.recover(exc, WRAP_UP_SEC):
                    continue
                raise
            self.waits = iter(RETRY_WAITS)
            if reply.usage:
                details = getattr(reply.usage, "prompt_tokens_details", None)
                self.total_in += reply.usage.prompt_tokens
                self.total_out += reply.usage.completion_tokens
                await self.session.usage(
                    MODEL,
                    reply.usage.prompt_tokens,
                    reply.usage.completion_tokens,
                    getattr(details, "cached_tokens", 0) or 0,
                )
            message = reply.choices[0].message
            cut = reply.choices[0].finish_reason == "length"
            text = self.head + (message.content or "")
            if text:
                self.session.final_message = text
            turn = message.model_dump(exclude_none=True)
            if wrap_up or not message.tool_calls:
                if wrap_up or (message.content and not cut):
                    return await self.finish(text)
                if self.nudges == MAX_NUDGES:
                    return await self.finish(text or self.summary_line())
                self.nudges, self.head = self.nudges + 1, text
                if message.content:
                    self.messages.append(turn)
                if self.phase in ("recon", "solve") and not self.deliverables:
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "You replied without calling a tool and without "
                            "declaring deliverables. If the deliverables are "
                            "written, call declare_deliverables with their "
                            "paths. Otherwise continue the task with tools."
                        ),
                    })
                else:
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "Your reply was cut off; continue where it "
                            "stopped." if cut else
                            "Continue the task, and finish with your summary."
                        ),
                    })
                continue
            self.nudges, self.head = 0, ""
            if self.phase == "recon":
                self.phase = "solve"
            self.messages.append(turn)
            for call in turn["tool_calls"]:
                await self.session.log(
                    f"{call['function']['name']} "
                    f"{call['function']['arguments'][:200]}"
                )
                content = await self.run_tool(call["function"])
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": content,
                })
            if left is not None:
                self.messages[-1][
                    "content"
                ] += f"\n[{int(self.session.time_left() // 60)} minutes left]"

    def summary_line(self):
        """Fallback final message when the model never produced one."""
        if self.deliverables:
            return (
                "Deliverables written: " + ", ".join(self.deliverables)
                + f". ({self.turns} turns, {self.total_in} input / "
                f"{self.total_out} output tokens.)"
            )
        return (
            f"No deliverables were declared. ({self.turns} turns, "
            f"{self.total_in} input / {self.total_out} output tokens.)"
        )

    async def finish(self, text):
        """Appends run stats and returns the final message."""
        stats = (
            f"\n\n[veriflow: {self.turns} turns, {self.total_in} input / "
            f"{self.total_out} output tokens, phase={self.phase}, "
            f"deliverables={len(self.deliverables)}]"
        )
        return (text or self.summary_line()) + stats


async def solve(session):
    """Entry point for taskbridge.run."""
    return await Harness(session).solve()


if __name__ == "__main__":
    taskbridge.run(solve)
