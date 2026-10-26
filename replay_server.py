"""A stand-in for the LLM: an OpenAI-compatible server that replays recorded model answers.

The lab has no live model, so this little server takes the model's place. Mellea talks to
it exactly as it would to a real one (``POST /v1/chat/completions``), and gets back the
answer that the real Gemma model gave to the same prompt when we recorded it. Everything
else stays real on your machine: Mellea's Instruct-Validate-Repair loop, running the unit
tests, deciding pass/fail, and building the repair prompt from the test failure.

How it finds the right answer:
  * The first line of the task text identifies the prompt.
  * The request number for that prompt (1st, 2nd, ...) is the turn in the IVR loop. Mellea
    re-sends the original task on every repair turn, so the text alone can't tell turn 2
    from turn 3; counting is reliable because each prompt gets its own session.
  * A prompt with no recording gets an honest "no recording" answer, which the unit tests
    then fail. Nothing is made up.

Run standalone (``python replay_server.py``) or let the notebook start it in the background.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RECORDED = Path(__file__).resolve().parent / "data" / "recorded"
NO_RECORDING = "```python\n# (no recorded answer for this prompt: try one of the 20 sample prompts)\n```"


def _key(text: str) -> str:
    """First non-empty line of the task: what identifies a prompt."""
    return next((line.strip() for line in text.splitlines() if line.strip()), "")[:300]


class Replay:
    def __init__(self, recorded: Path = RECORDED):
        # prompt key -> list of the model's answers in turn order
        self.answers: dict[str, list[str]] = {}
        for f in sorted(recorded.glob("trace_*.json")):
            t = json.loads(f.read_text())
            self.answers[_key(t["instruction"])] = [a.get("raw") or f"```python\n{a['code']}\n```" for a in t["attempts"]]
        self.turn: dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()

    def reset(self) -> None:
        with self.lock:
            self.turn.clear()

    def answer(self, user_text: str) -> str:
        key = _key(user_text)
        with self.lock:
            if "The following requirements failed before:" not in user_text:
                self.turn[key] = 0  # a first turn starts the count over (re-running a cell)
            i = self.turn[key]
            self.turn[key] += 1
        recorded = self.answers.get(key)
        if not recorded:
            return NO_RECORDING
        return recorded[min(i, len(recorded) - 1)]


def make_handler(replay: Replay, delay: float):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the notebook quiet
            pass

        def _send(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                self._send({"object": "list", "data": [{"id": "replay", "object": "model", "owned_by": "lab"}]})
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            user = next((m["content"] for m in reversed(req.get("messages", [])) if m.get("role") == "user"), "")
            if isinstance(user, list):  # content parts
                user = " ".join(p.get("text", "") for p in user if isinstance(p, dict))
            text = replay.answer(str(user))
            time.sleep(delay)  # a little thinking time, so the loop feels like a real model
            self._send({
                "id": "replay-1", "object": "chat.completion", "created": int(time.time()), "model": req.get("model", "replay"),
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })

    return Handler


_RUNNING: dict[Path, tuple[str, Replay]] = {}


def start(port: int = 0, delay: float = 0.4, recorded: Path = RECORDED) -> tuple[str, Replay]:
    """Start the replay server in a background thread; returns (base_url, replay).

    ``port=0`` lets the OS pick a free port, so several people on one shared machine never clash.
    Calling it again (re-running the notebook's setup cell) reuses the server already running.
    """
    if recorded in _RUNNING:
        url, replay = _RUNNING[recorded]
        replay.reset()
        return url, replay
    replay = Replay(recorded)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(replay, delay))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _RUNNING[recorded] = (f"http://127.0.0.1:{server.server_address[1]}/v1", replay)
    return _RUNNING[recorded]


if __name__ == "__main__":
    url, r = start()
    print(f"replaying {len(r.answers)} recorded prompts at {url}  (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
