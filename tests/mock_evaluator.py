"""Mock PLE evaluator: speaks the kit WebSocket protocol (docs/protocol.md).

The evaluator is the WebSocket *client*: it connects to the harness server,
sends the task message, then answers the harness's exec / read_file /
write_file / http requests. The task container is a real temp directory:
`exec` runs bash in it, reads/writes use real files, `http` answers canned
MCP JSON-RPC. Every frame is recorded so tests can assert on the run.
No Docker, no model.
"""

import asyncio
import base64
import json
import subprocess
import tempfile
from pathlib import Path

import websockets


class MockEvaluator:
    def __init__(self, instruction, workdir_files=None, mcp_servers=None,
                 agent_timeout_sec=21600, cancel_after=None):
        self.instruction = instruction
        self.workdir_files = workdir_files or {}
        self.mcp_servers = mcp_servers or []
        self.agent_timeout_sec = agent_timeout_sec
        self.cancel_after = cancel_after
        self.frames = []          # every frame received from the harness
        self.results = []         # every result frame sent
        self.done = None          # the done/error frame, once received
        self.tmp = tempfile.TemporaryDirectory()
        self.workdir = Path(self.tmp.name)
        for name, content in self.workdir_files.items():
            path = self.workdir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                content if isinstance(content, bytes) else content.encode()
            )

    async def run(self, harness_url):
        """Connects to the harness, sends the task, answers until done."""
        async with websockets.connect(
                harness_url, max_size=16 * 2**20, ping_interval=None) as ws:
            await ws.send(json.dumps({
                "type": "task", "protocol": 1, "session": "mock-session",
                "instruction": self.instruction,
                "workdir": str(self.workdir),
                "limits": {
                    "agent_timeout_sec": self.agent_timeout_sec,
                    "exec_timeout_sec": 600,
                    "max_output_bytes": 100000,
                    "max_write_bytes": 10000000,
                },
                "mcp_servers": self.mcp_servers,
            }))
            cancel_task = None
            if self.cancel_after is not None:
                async def _cancel():
                    await asyncio.sleep(self.cancel_after)
                    try:
                        await ws.send(json.dumps(
                            {"type": "cancel", "reason": "timeout"}))
                    except Exception:
                        pass
                cancel_task = asyncio.create_task(_cancel())
            try:
                async for frame in ws:
                    msg = json.loads(frame)
                    self.frames.append(msg)
                    if msg.get("type") in ("done", "error"):
                        self.done = msg
                        break
                    if msg.get("type") in ("log", "usage"):
                        continue
                    result = self._answer(msg)
                    self.results.append(result)
                    await ws.send(json.dumps(result))
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                if cancel_task is not None:
                    cancel_task.cancel()

    def cleanup(self):
        """Removes the temp workdir. Call after assertions."""
        self.tmp.cleanup()

    def _answer(self, msg):
        kind, rid = msg.get("type"), msg.get("id")
        if kind == "exec":
            proc = subprocess.run(
                ["bash", "-c", msg.get("command", "")],
                cwd=self.workdir, capture_output=True, text=True,
                timeout=min(msg.get("timeout_sec") or 120, 600),
            )
            out = proc.stdout + proc.stderr
            return {"type": "result", "id": rid, "ok": True,
                    "exit_code": proc.returncode, "output": out[:100000],
                    "truncated": len(out) > 100000, "timed_out": False}
        if kind == "read_file":
            path = self._resolve(msg.get("path", ""))
            if not path.is_file():
                return {"type": "result", "id": rid, "ok": False,
                        "error": f"no such file: {msg.get('path')}"}
            data = path.read_bytes()
            return {"type": "result", "id": rid, "ok": True,
                    "content_b64": base64.b64encode(data[:10_000_000]).decode(),
                    "size": len(data), "truncated": len(data) > 10_000_000}
        if kind == "write_file":
            path = self._resolve(msg.get("path", ""))
            path.parent.mkdir(parents=True, exist_ok=True)
            data = base64.b64decode(msg.get("content_b64") or "")
            mode = "ab" if msg.get("append") else "wb"
            with open(path, mode) as fh:
                fh.write(data)
            return {"type": "result", "id": rid, "ok": True,
                    "size": path.stat().st_size}
        if kind == "http":
            return {"type": "result", "id": rid, "ok": True, "status": 200,
                    "headers": {"Content-Type": "application/json"},
                    "body_b64": base64.b64encode(
                        self._mcp_reply(msg)).decode(),
                    "truncated": False}
        return {"type": "result", "id": rid, "ok": False,
                "error": f"unknown request type: {kind}"}

    def _resolve(self, path):
        p = Path(path)
        return p if p.is_absolute() else self.workdir / p

    def _mcp_reply(self, msg):
        """Canned Streamable-HTTP MCP answers for the docs server."""
        try:
            body = json.loads(base64.b64decode(msg.get("body_b64") or b""))
        except ValueError:
            body = {}
        method = body.get("method")
        mid = body.get("id")
        if method == "initialize":
            result = {"protocolVersion": "2025-06-18",
                      "capabilities": {},
                      "serverInfo": {"name": "mock-docs", "version": "1.0"}}
        elif method == "tools/list":
            result = {"tools": [{
                "name": "get_document",
                "description": "Return the text of a project document.",
                "inputSchema": {"type": "object",
                                "properties": {"doc": {"type": "string"}}},
            }]}
        elif method == "tools/call":
            result = {"content": [{"type": "text",
                                   "text": "SPEC: feeder load 250 kVA, "
                                   "voltage 480 V, 3-phase."}]}
        else:
            result = {}
        return json.dumps({"jsonrpc": "2.0", "id": mid,
                           "result": result}).encode()
