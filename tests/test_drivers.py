"""Model-free protocol tests for browser driver failure paths."""

from __future__ import annotations

import json
import sys
import types
import urllib.request
import warnings

import pytest

from localdecide.drivers import CDPDriver, PlaywrightDriver


class _HTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_cdp_target_url_filter_does_not_fall_back_to_first_page(monkeypatch):
    pages = [
        {"type": "page", "url": "https://wrong.example", "webSocketDebuggerUrl": "ws://wrong"},
        {"type": "page", "url": "https://also-wrong.example", "webSocketDebuggerUrl": "ws://wrong2"},
    ]
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: _HTTPResponse(pages))
    monkeypatch.setitem(sys.modules, "websocket", types.SimpleNamespace(create_connection=lambda *args, **kwargs: None))
    with pytest.raises(RuntimeError, match="no attachable page"):
        CDPDriver(target_url_contains="https://required.example")


def test_cdp_runtime_evaluate_exception_details_are_not_dropped():
    class FakeSocket:
        def send(self, message):
            self.message = json.loads(message)

        def recv(self):
            return json.dumps({"id": self.message["id"], "result": {},
                               "exceptionDetails": {"text": "Uncaught", "exception": {
                                   "description": "Error: synthetic failure"}}})

    driver = CDPDriver.__new__(CDPDriver)
    driver._ws = FakeSocket()
    driver._id = 0
    with pytest.raises(RuntimeError, match="synthetic failure"):
        driver._eval("(() => { throw new Error('synthetic failure') })()")


def test_playwright_constructor_failure_stops_runtime_and_restores_event_loop(monkeypatch):
    import asyncio

    previous = asyncio.new_event_loop()
    asyncio.set_event_loop(previous)
    state = {"stopped": 0}

    class Chromium:
        def launch(self, **kwargs):
            raise RuntimeError("synthetic launch failure")

    class Runtime:
        chromium = Chromium()

        def stop(self):
            state["stopped"] += 1

    class Starter:
        def start(self):
            return Runtime()

    playwright_module = types.ModuleType("playwright")
    sync_module = types.ModuleType("playwright.sync_api")
    sync_module.sync_playwright = lambda: Starter()
    playwright_module.sync_api = sync_module
    monkeypatch.setitem(sys.modules, "playwright", playwright_module)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_module)

    try:
        with pytest.raises(RuntimeError, match="synthetic launch failure"):
            PlaywrightDriver()
        assert state["stopped"] == 1
        assert asyncio.get_event_loop() is previous
    finally:
        previous.close()
        asyncio.set_event_loop(None)


def test_playwright_constructor_without_current_loop_has_no_deprecation_warning(monkeypatch):
    import asyncio

    class Page:
        pass

    state = {"closed": 0, "stopped": 0}

    class Browser:
        def new_page(self):
            return Page()

        def close(self):
            state["closed"] += 1

    class Chromium:
        def launch(self, **kwargs):
            return Browser()

    class Runtime:
        chromium = Chromium()

        def stop(self):
            state["stopped"] += 1

    class Starter:
        def start(self):
            return Runtime()

    playwright_module = types.ModuleType("playwright")
    sync_module = types.ModuleType("playwright.sync_api")
    sync_module.sync_playwright = lambda: Starter()
    playwright_module.sync_api = sync_module
    monkeypatch.setitem(sys.modules, "playwright", playwright_module)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_module)

    asyncio.set_event_loop(None)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            driver = PlaywrightDriver()
            driver.close()
            driver.close()
        assert not [warning for warning in caught if warning.category is DeprecationWarning]
        assert state == {"closed": 1, "stopped": 1}
    finally:
        asyncio.set_event_loop(None)
