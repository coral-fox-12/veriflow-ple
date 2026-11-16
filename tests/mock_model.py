"""Mock OpenAI-compatible model endpoint for harness tests.

Serves POST /v1/chat/completions from a scripted queue of replies. Each
reply is either a tool-calls turn or a final text turn. Replies are served
in order; when the queue runs out, the last reply repeats as a final text
turn so the run always terminates.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def tool_turn(calls):
    """One scripted turn: a list of (name, arguments-dict) tool calls."""
    return {
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant", "content": None,
                "tool_calls": [
                    {"id": f"call_{i}", "type": "function",
                     "function": {"name": name,
                                  "arguments": json.dumps(args)}}
                    for i, (name, args) in enumerate(calls)
                ],
            },
        }],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50,
                  "total_tokens": 150},
    }


def text_turn(text):
    """One scripted turn: a final text reply with no tool calls."""
    return {
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": text},
        }],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50,
                  "total_tokens": 150},
    }


class MockModel:
    """Threaded HTTP server; `script` is a list of reply dicts."""

    def __init__(self, script, delay=0.0):
        self.script = list(script)
        self.delay = delay
        self.calls = []  # request bodies received, in order
        self._httpd = None
        self._thread = None

    def __enter__(self):
        model = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                model.calls.append(body)
                if model.delay:
                    time.sleep(model.delay)
                idx = min(len(model.calls) - 1, len(model.script) - 1)
                reply = model.script[idx] if model.script else text_turn("")
                payload = json.dumps({
                    "id": "chatcmpl-mock", "object": "chat.completion",
                    "created": 0, "model": body.get("model", "mock"),
                    **reply,
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except BrokenPipeError:
                    pass  # client cancelled (e.g. the cancel test)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}/v1"
