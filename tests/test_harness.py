"""End-to-end tests for the veriflow harness.

Runs `serve.py` as a subprocess against a mock evaluator (kit protocol)
and a mock OpenAI-compatible model endpoint. No API key, no Docker.
"""

import asyncio
import os
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS_DIR = HERE.parent

sys.path.insert(0, str(HERE))
from mock_evaluator import MockEvaluator  # noqa: E402
from mock_model import MockModel, text_turn, tool_turn  # noqa: E402


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class HarnessTest(unittest.TestCase):
    def run_case(self, instruction, script, timeout=60, model_delay=0.0,
                 env_extra=None, **evaluator_kw):
        """Starts the harness + mocks, runs one task, returns the evaluator."""
        port = free_port()
        evaluator = MockEvaluator(instruction, **evaluator_kw)
        self.addCleanup(evaluator.cleanup)
        with MockModel(script, delay=model_delay) as model:
            env = dict(os.environ)
            env.update({
                "PORT": str(port),
                "MODEL_NAME": "mock-model",
                "OPENAI_BASE_URL": model.base_url,
                "OPENAI_API_KEY": "test-key",
                "MAX_TURNS": "30",
                "VERIFY_SEC": "1800",
                "WRAP_UP_SEC": "600",
                "PYTHONPATH": str(HARNESS_DIR),
            })
            if env_extra:
                env.update(env_extra)
            proc = subprocess.Popen(
                [sys.executable, str(HARNESS_DIR / "serve.py")],
                cwd=HARNESS_DIR, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            try:
                deadline = time.time() + 15
                while time.time() < deadline:
                    if proc.poll() is not None:
                        raise AssertionError(
                            f"harness exited early: {proc.stdout.read()}")
                    try:
                        with socket.create_connection(
                                ("127.0.0.1", port), timeout=0.5):
                            break
                    except OSError:
                        time.sleep(0.2)
                else:
                    raise AssertionError("harness did not start listening")
                asyncio.run(asyncio.wait_for(
                    evaluator.run(f"ws://127.0.0.1:{port}"),
                    timeout=timeout,
                ))
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                if proc.stdout:
                    proc.stdout.close()
        return evaluator

    def test_smoke_write_and_done(self):
        """The kit smoke shape: write a file, declare it, finish."""
        ev = self.run_case(
            "Write a file hello.txt in the workdir containing exactly "
            "'hello' and finish.",
            [
                tool_turn([("write_file", {
                    "path": "hello.txt", "content": "hello"})]),
                tool_turn([("declare_deliverables", {
                    "paths": ["hello.txt"]})]),
                text_turn("Verified: hello.txt exists with 'hello'. "
                          "Deliverable: hello.txt."),
            ],
        )
        self.assertIsNotNone(ev.done, "no done frame received")
        self.assertEqual(ev.done["type"], "done")
        self.assertEqual(
            (ev.workdir / "hello.txt").read_text(), "hello")
        self.assertIn("hello.txt", ev.done["final_message"])
        self.assertIn("[veriflow:", ev.done["final_message"])
        kinds = [f["type"] for f in ev.frames]
        self.assertIn("usage", kinds, "no usage frames reported")

    def test_calc_with_verification_round(self):
        """A calculation task: compute with code, write the report, verify."""
        ev = self.run_case(
            "Compute the 3-phase current for a 250 kVA load at 480 V and "
            "write the result with its unit into report.md, then finish.",
            [
                tool_turn([("bash", {"command":
                    "python3 -c \"import math; "
                    "print(f'{250000/(math.sqrt(3)*480):.1f} A')\" > calc.txt"})]),
                tool_turn([("read_file", {"path": "calc.txt"})]),
                tool_turn([("write_file", {
                    "path": "report.md",
                    "content": "# Feeder current\n\nLoad 250 kVA at 480 V, "
                    "3-phase: I = 300.7 A.\n"})]),
                tool_turn([("declare_deliverables", {
                    "paths": ["report.md"]})]),
                text_turn("Verified: recomputed 300.7 A with a fresh "
                          "script; matches report.md. Deliverable: "
                          "report.md."),
            ],
        )
        self.assertIsNotNone(ev.done)
        self.assertEqual(ev.done["type"], "done")
        report = (ev.workdir / "report.md").read_text()
        self.assertIn("300.7 A", report)
        execs = [f for f in ev.frames
                 if f.get("type") == "exec"]
        self.assertTrue(execs, "harness never ran a command")

    def test_mcp_tools(self):
        """A task whose answer comes from an MCP server tool."""
        ev = self.run_case(
            "Use the docs MCP server to get the feeder spec, write it into "
            "spec.md, and finish.",
            [
                tool_turn([("mcp_list_tools", {"server": "docs"})]),
                tool_turn([("mcp_call_tool", {
                    "server": "docs", "tool_name": "get_document",
                    "arguments": {"doc": "feeder-spec"}})]),
                tool_turn([("write_file", {
                    "path": "spec.md",
                    "content": "# Feeder spec\n\nLoad 250 kVA, 480 V, "
                    "3-phase (from docs MCP).\n"})]),
                tool_turn([("declare_deliverables", {
                    "paths": ["spec.md"]})]),
                text_turn("Verified spec.md against the MCP document. "
                          "Deliverable: spec.md."),
            ],
            mcp_servers=[{"name": "docs",
                          "url": "http://127.0.0.1:8000/mcp/",
                          "transport": "streamable-http"}],
        )
        self.assertIsNotNone(ev.done)
        self.assertEqual(ev.done["type"], "done")
        spec = (ev.workdir / "spec.md").read_text()
        self.assertIn("250 kVA", spec)
        https = [f for f in ev.frames if f.get("type") == "http"]
        self.assertTrue(https, "harness never called the MCP server")

    def test_cancel_ends_with_done(self):
        """A cancelled task still ends with a done frame, not a crash."""
        ev = self.run_case(
            "Do a long task.",
            [text_turn("This should never be reached.")],
            timeout=30, model_delay=8.0,
            env_extra={"WRAP_UP_SEC": "2", "VERIFY_SEC": "2"},
            agent_timeout_sec=6, cancel_after=2,
        )
        self.assertIsNotNone(ev.done, "no done frame after cancel")
        self.assertEqual(ev.done["type"], "done")
        self.assertIn("time is up", ev.done["final_message"])


if __name__ == "__main__":
    unittest.main()
