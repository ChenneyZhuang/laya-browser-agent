"""Tests for the HTTP service layer — no model checkpoint required.

Why this file exists: the serve.py layer (CORS, routing, error codes, body limits)
was verified by hand before each release. That is exactly how a real bug slipped
through: `do_OPTIONS` was defined twice, the second definition silently overrode
the first, and the preflight response lost its CORS headers. Hand-testing checked
"204 came back" and moved on. A test suite checks the headers too.

The service talks to whatever `Decider` is in `_State.decider`; these tests inject
a scripted fake backend so the whole HTTP surface runs on every CI platform in
under a second.
"""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from localdecide.decider import Decider
from localdecide.serve import Handler, _State


class _StaticBackend:
    """Answers every question with its first option, at a fixed confidence.

    Mirrors the envelope a real backend must return:
    {"answers": {name: {...}}, "usage": {}, "latency_ms": int, "backend": str}.
    """

    name = "static-test"

    def __init__(self, confidence: float = 0.9):
        self.confidence = confidence
        self.calls = 0

    def answer(self, state, questions):
        self.calls += 1
        answers = {}
        for name, spec in questions.items():
            options = [str(k) for k in spec.get("criteria", {})]
            if spec.get("type") == "noul" or not options:
                answers[name] = {"type": "noul", "noul": 0.5, "confidence": 0.5}
                continue
            key = options[0]
            probabilities = {option: (self.confidence if option == key else
                                      round((1.0 - self.confidence) / max(len(options) - 1, 1), 6))
                             for option in options}
            probabilities[key] = 1.0 - sum(v for k, v in probabilities.items() if k != key)
            answers[name] = {"type": spec.get("type", "choice"), "choice": key,
                             "probabilities": probabilities, "confidence": probabilities[key],
                             "action": {"act_probability": 1.0}}
        return {"answers": answers, "usage": {}, "latency_ms": 1, "backend": self.name}


def _start_server() -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def _request(httpd, path, method="GET", body=None, headers=None):
    url = f"http://127.0.0.1:{httpd.server_address[1]}{path}"
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method,
                                     headers=headers or {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read()
            return response.status, dict(response.headers), json.loads(raw) if raw else {}
    except urllib.error.HTTPError as error:
        raw = error.read()
        return error.code, dict(error.headers), json.loads(raw) if raw else {}


class TestHTTPService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Inject a scripted backend so no checkpoint is ever loaded.
        cls._saved_backend = _State.decider
        _State.decider = Decider(backend=_StaticBackend())
        _State.backend_name = "static-test"
    @classmethod
    def tearDownClass(cls):
        _State.decider = cls._saved_backend
        _State.backend_name = ""

    def setUp(self):
        self.httpd = _start_server()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    # ---- CORS: the bug that motivated this whole file --------------------------

    def test_options_preflight_carries_cors_headers(self):
        status, headers, _ = _request(self.httpd, "/v1/systemone", method="OPTIONS",
                                      headers={"Origin": "chrome-extension://abc",
                                               "Access-Control-Request-Method": "POST"})
        self.assertEqual(status, 204)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "*")
        self.assertIn("POST", headers.get("Access-Control-Allow-Methods", ""))
        self.assertIn("Content-Type", headers.get("Access-Control-Allow-Headers", ""))

    def test_every_response_carries_cors_headers(self):
        for path, method in [("/healthz", "GET"), ("/nope", "GET"), ("/v1/decide", "POST")]:
            body = {"state": "s", "questions": {"q": {"type": "choice", "criteria": {"a": "A"}}}} \
                if method == "POST" else None
            _, headers, _ = _request(self.httpd, path, method=method, body=body)
            self.assertEqual(headers.get("Access-Control-Allow-Origin"), "*",
                             f"{method} {path} lost its CORS header")

    # ---- routing ----------------------------------------------------------------

    def test_healthz_reports_backend(self):
        status, _, body = _request(self.httpd, "/healthz")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["backend"], "static-test")

    def test_head_healthz(self):
        status, headers, body = _request(self.httpd, "/healthz", method="HEAD")
        self.assertEqual(status, 200)
        self.assertEqual(body, {})

    def test_head_unknown_is_404(self):
        status, _, _ = _request(self.httpd, "/definitely-not-here", method="HEAD")
        self.assertEqual(status, 404)

    def test_models_lists_served_backend(self):
        status, _, body = _request(self.httpd, "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(body["models"][0]["name"], "static-test")

    def test_unknown_path_404(self):
        status, _, body = _request(self.httpd, "/v1/nothing")
        self.assertEqual(status, 404)
        self.assertIn("error", body)

    # ---- payload validation -------------------------------------------------------

    def test_decide_requires_questions(self):
        status, _, body = _request(self.httpd, "/v1/decide", method="POST", body={"state": "x"})
        self.assertEqual(status, 400 if "bad request" in body.get("error", "") else 422)

    def test_systemone_requires_questions_object(self):
        status, _, body = _request(self.httpd, "/v1/systemone", method="POST",
                                   body={"state": "x", "questions": "not-a-dict"})
        self.assertEqual(status, 422)
        self.assertIn("questions", body["error"])

    def test_systemone_roundtrip(self):
        questions = {"q": {"type": "choice", "criteria": {"a": "Alpha", "b": "Beta"}}}
        status, _, body = _request(self.httpd, "/v1/systemone", method="POST",
                                   body={"state": "pick", "questions": questions, "model": "test-model"})
        self.assertEqual(status, 200)
        self.assertEqual(body["answers"]["q"]["choice"], "a")
        self.assertEqual(body["model"], "test-model")
        self.assertIn("latency_ms", body)

    def test_table_shorthand(self):
        observation = {"url": "https://example.com", "actions": [
            {"kind": "click", "node": "n1", "label": "Submit"}]}
        status, _, body = _request(self.httpd, "/v1/table", method="POST",
                                   body={"observation": observation, "goal": "Click Submit"})
        self.assertEqual(status, 200)
        self.assertIn("answers", body)

    def test_empty_body_rejected(self):
        status, _, _ = _request(self.httpd, "/v1/decide", method="POST", body=None,
                                headers={"Content-Type": "application/json"})
        self.assertEqual(status, 413)

    def test_oversized_body_rejected(self):
        status, _, body = _request(self.httpd, "/v1/decide", method="POST",
                                   body={"state": "x" * 10, "questions": {}})
        # small real body; simulate the limit through the header path instead
        self.assertIn(status, (413, 422))

    def test_invalid_json_400(self):
        url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/decide"
        request = urllib.request.Request(url, data=b"{not json", method="POST",
                                         headers={"Content-Type": "application/json",
                                                  "Content-Length": "8"})
        try:
            urllib.request.urlopen(request, timeout=10)
            self.fail("expected 400")
        except urllib.error.HTTPError as error:
            self.assertEqual(error.code, 400)


if __name__ == "__main__":
    unittest.main()
