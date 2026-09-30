"""Model-free backend compatibility and HTTP wire tests."""

from __future__ import annotations

import json
import sys
import threading
import types
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from localdecide.backends.base import HTTPBackend, LayaMLXBackend, LayaTorchBackend
from localdecide.decider import Decider, choice, noul, score
from localdecide.serve import Handler, _State
from tests.test_serve import _StaticBackend


class _WireHandler(BaseHTTPRequestHandler):
    request_body = None
    request_headers = None

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        type(self).request_body = json.loads(self.rfile.read(length))
        type(self).request_headers = dict(self.headers)
        payload = {"answers": {"pick": {"type": "choice", "choice": "a",
                                          "probabilities": {"a": 1.0, "b": 0.0},
                                          "confidence": 1.0}},
                   "usage": {}, "backend": "wire-fake"}
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


@pytest.fixture
def wire_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _WireHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_backend_real_local_wire_carries_model_and_key(wire_server):
    url = f"http://127.0.0.1:{wire_server.server_address[1]}/v1/systemone"
    backend = HTTPBackend(url, api_key="wire-secret", model="wire-model", timeout=4.0)
    result = Decider(backend=backend, retries=0).decide(
        "state", {"pick": choice("pick", {"a": "A", "b": "B"})})
    assert result.ok, result.error
    assert _WireHandler.request_body["model"] == "wire-model"
    assert _WireHandler.request_body["questions"]["pick"]["type"] == "choice"
    assert _WireHandler.request_headers["Authorization"] == "Bearer wire-secret"
    assert _WireHandler.request_headers["Accept"] == "application/json"


@pytest.fixture
def local_handler_server():
    saved = (_State.decider, _State.backend_name, _State.started_at, _State.calls, _State.errors)
    _State.decider = Decider(backend=_StaticBackend())
    _State.backend_name = "static-test"
    server = None
    thread = None
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield server
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)
        _State.decider, _State.backend_name, _State.started_at, _State.calls, _State.errors = saved


def test_http_backend_roundtrips_choice_score_and_noul_through_handler(local_handler_server):
    url = f"http://127.0.0.1:{local_handler_server.server_address[1]}/v1/systemone"
    result = Decider(backend=HTTPBackend(url, timeout=4.0), retries=0).decide(
        "state", {
            "pick": choice("pick", {"a": "A", "b": "B"}),
            "rating": score("rate", ["low", "mid", "high"]),
            "valid": noul("is it valid?"),
        })
    assert result.ok, result.error
    assert result.answers is not None
    assert result.answers.choice("pick") == "a"
    assert result.answers.score("rating") == 0.0
    assert result.answers.noul("valid") == 0.5
    assert result.answers.raw["rating"]["type"] == "score"
    assert set(result.answers.raw["rating"]["probabilities"]) == {"0", "1", "2"}


def test_http_backend_rejects_score_without_probabilities(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: _FakeResponse({
        "answers": {"rating": {"type": "score", "score": 1.0}},
    }))
    result = Decider(backend=HTTPBackend("http://wire.invalid"), retries=0).decide(
        "state", {"rating": score("rate", ["low", "mid", "high"])})
    assert not result.ok
    assert "score probabilities" in (result.error or "")


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_http_timeout_is_bounded_by_decider_remaining_deadline(monkeypatch):
    observed = {}

    def fake_urlopen(request, timeout):
        observed["timeout"] = timeout
        return _FakeResponse({"answers": {"pick": {"type": "choice", "choice": "a",
                                                      "probabilities": {"a": 1.0, "b": 0.0},
                                                      "confidence": 1.0}}})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = Decider(backend=HTTPBackend("http://wire.invalid", timeout=60.0),
                     timeout=0.05, retries=0).decide(
                         "state", {"pick": choice("pick", {"a": "A", "b": "B"})})
    assert result.ok, result.error
    assert 0 < observed["timeout"] <= 0.05


def test_http_backend_rejects_malformed_envelope(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: _FakeResponse({"answers": []}))
    result = Decider(backend=HTTPBackend("http://wire.invalid"), retries=0).decide(
        "state", {"pick": choice("pick", {"a": "A", "b": "B"})})
    assert not result.ok
    assert "malformed" in (result.error or "")


def test_laya_backend_alias_subfolder_and_revision_are_forwarded_without_model_load(monkeypatch):
    calls = []

    class FakeAgent:
        def predict(self, state, questions):
            return {"answers": {}, "usage": {}}

    mlx = types.ModuleType("laya_mlx")
    mlx.load = lambda *args, **kwargs: (calls.append(("mlx", args, kwargs)) or FakeAgent())
    torch = types.ModuleType("laya")
    torch.load = lambda *args, **kwargs: (calls.append(("torch", args, kwargs)) or FakeAgent())
    monkeypatch.setitem(sys.modules, "laya_mlx", mlx)
    monkeypatch.setitem(sys.modules, "laya", torch)

    LayaMLXBackend(revision="mlx-revision")._load()
    LayaTorchBackend(revision="torch-revision")._load()

    assert calls == [
        ("mlx", ("ichenney/laya-browser-v32b",), {"subfolder": "v32b", "revision": "mlx-revision"}),
        ("torch", ("ichenney/laya-browser-v32b",), {"subfolder": "v32b", "revision": "torch-revision"}),
    ]


def test_legacy_browser_alias_pins_the_historical_v10s_revision(monkeypatch):
    calls = []

    class FakeAgent:
        def predict(self, state, questions):
            return {"answers": {}, "usage": {}}

    mlx = types.ModuleType("laya_mlx")
    mlx.load = lambda *args, **kwargs: (calls.append((args, kwargs)) or FakeAgent())
    monkeypatch.setitem(sys.modules, "laya_mlx", mlx)

    LayaMLXBackend(model="browser-legacy")._load()

    assert calls == [(("cklxx/laya-browser",), {
        "subfolder": "v10s",
        "revision": "adf912be85ff9221ee171551778456b133c1af75",
    })]
