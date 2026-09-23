"""A local HTTP service that speaks several decision dialects.

Why a server at all: every agent stack already has a way to call *something* - a
TypeSafe-compatible `/v1/systemone`, an MCP server, a plain JSON endpoint. Instead of
asking each stack to adopt an SDK, run one process here and point the stack at it.
Same local model, no cloud round trip, and the page content never leaves the machine.

Dialects served:

* `POST /v1/systemone`  - the TypeSafe Jev / Laya wire format. Anything written against
  Jev's HTTP API works by changing one base URL (`TYPESAFE_BASE_URL`).
* `POST /v1/decide`     - this project's native shape: `{state, questions}` with a
  `table` shortcut that turns a raw observation into questions for you.
* `POST /v1/table`      - give it an observation and a goal, get the chosen index back.
* `GET  /healthz`       - liveness plus what the backend actually is.

Deliberately stdlib-only and single-file-ish: no web framework to conflict with yours.
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from .decider import Decider
from .page import build_element_table, table_to_questions

MAX_BODY_BYTES = 4_000_000


class _State:
    decider: Optional[Decider] = None
    backend_name: str = ""
    started_at: float = 0.0
    calls: int = 0
    errors: int = 0
    lock = threading.Lock()


def _decider() -> Decider:
    with _State.lock:
        if _State.decider is None:
            _State.decider = Decider(backend=os.environ.get("LOCALDECIDE_BACKEND") or None,
                                     max_options_per_question=int(os.environ.get("LOCALDECIDE_MAX_OPTIONS", "20")))
            _State.backend_name = getattr(_State.decider.backend, "name", "?")
            _State.started_at = time.time()
        return _State.decider


def _handle_systemone(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The Jev/Laya wire format: everything is passed through after validation."""
    state = payload.get("state")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object")
    decision = _decider().decide(state if state is not None else "", questions)
    if not decision.ok:
        raise RuntimeError(decision.error or "decision failed")
    assert decision.answers is not None
    return {"answers": decision.answers.raw, "model": payload.get("model", "localdecide"),
            "usage": decision.answers.usage, "latency_ms": decision.latency_ms,
            "backend": decision.answers.backend}


def _handle_table(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Observation + goal in, chosen index out. The convenience dialect."""
    observation = payload.get("observation") or payload
    goal = payload.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("goal is required")
    table = build_element_table(observation)
    questions = table_to_questions(table, goal)
    decision = _decider().decide(table.state(text_chars=int(payload.get("text_chars", 1200))), questions)
    if not decision.ok:
        raise RuntimeError(decision.error or "decision failed")
    assert decision.answers is not None
    answers = decision.answers
    operation = answers.choice("operation")
    result: Dict[str, Any] = {"operation": operation, "confidence": answers.confidence("operation"),
                              "latency_ms": decision.latency_ms, "backend": answers.backend}
    if operation in ("CLICK", "TYPE_TEXT", "SELECT") and f"{operation.lower()}_target" in answers.raw:
        target = answers.choice(f"{operation.lower()}_target")
        element = table.by_index().get(target)
        result["target"] = target
        result["label"] = element.label if element else ""
        result["handle"] = element.handle if element else None
        result["confidence"] = min(result["confidence"], answers.confidence(f"{operation.lower()}_target"))
    result["answers"] = answers.raw
    return result


class Handler(BaseHTTPRequestHandler):
    server_version = "localdecide"
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # keep the console clean
        pass

    def _send(self, code: int, body: Dict[str, Any]) -> None:
        data = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        # CORS: browser extensions and local web consoles call this service from
        # origins like chrome-extension://... — the same convention Ollama uses for
        # a localhost-only tool service. The server binds to 127.0.0.1 by default.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        if code == 204:  # no content: no body, no Content-Type/Length
            self.end_headers()
            return
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:  # noqa: N802
        # CORS preflight — headers go out via _send.
        self._send(204, {})

    def do_HEAD(self) -> None:  # noqa: N802
        # Health checkers often probe with HEAD; answer without a body.
        path = self.path.rstrip("/")
        code = 200 if path in ("/healthz", "/", "/v1/models") else 404
        self._send(code, {})

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")
        if path == "/v1/models":
            # TypeSafe-compatible model list (their /v1/models returns hosted Jev
            # variants; here the served model is whichever local checkpoint loaded).
            if not _State.backend_name:
                _decider()  # force the backend load so the name is real, not ""
            self._send(200, {"models": [{"name": _State.backend_name or "localdecide",
                                         "description": "The local decision model behind this service",
                                         "release_date": None}]})
            return
        if path in ("/healthz", "/"):
            try:
                decider = _decider()
                self._send(200, {"ok": True, "backend": _State.backend_name, "calls": _State.calls,
                                 "errors": _State.errors, "uptime_s": round(time.time() - _State.started_at, 1),
                                 "max_options_per_question": decider.max_options_per_question})
            except Exception as error:
                self._send(500, {"ok": False, "error": str(error)[:300]})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send(413, {"error": "bad body length"})
                return
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("body must be an object")
        except Exception as error:
            self._send(400, {"error": f"bad request: {error}"})
            return

        try:
            if path == "/v1/systemone":
                body = _handle_systemone(payload)
            elif path in ("/v1/decide",):
                state = payload.get("state", "")
                questions = payload.get("questions") or {}
                if not questions:
                    raise ValueError("questions is required")
                decision = _decider().decide(state, questions)
                if not decision.ok:
                    raise RuntimeError(decision.error or "decision failed")
                assert decision.answers is not None
                body = {"answers": decision.answers.raw, "usage": decision.answers.usage,
                        "latency_ms": decision.latency_ms, "backend": decision.answers.backend}
            elif path == "/v1/table":
                body = _handle_table(payload)
            else:
                self._send(404, {"error": f"no such endpoint: {path}"})
                return
        except Exception as error:
            with _State.lock:
                _State.errors += 1
            self._send(422, {"error": f"{type(error).__name__}: {error}"[:400]})
            return
        with _State.lock:
            _State.calls += 1
        self._send(200, body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()


def serve(host: str = "127.0.0.1", port: int = 8791) -> None:
    """Run the service until interrupted. Loads the model on the first decision."""
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"localdecide serving on http://{host}:{port}")
    print("  POST /v1/systemone   TypeSafe/Jev-compatible wire format")
    print("  POST /v1/decide      native {state, questions}")
    print("  POST /v1/table       {goal, observation} -> chosen index")
    print("  GET  /healthz        liveness")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    serve(host=os.environ.get("LOCALDECIDE_HOST", "127.0.0.1"),
          port=int(os.environ.get("LOCALDECIDE_PORT", "8791")))
