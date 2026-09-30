"""Model-free browser safety regressions on a synthetic page.

The tests are optional locally because the base contract suite deliberately does not
install Playwright or a browser. CI installs the browser extra and runs these against
the real DOM executor, never against a user profile or a live website.
"""

from __future__ import annotations

import socket
import time

import pytest

pytest.importorskip("playwright.sync_api", reason="synthetic browser tests need Playwright")
pytest.importorskip("websocket", reason="synthetic CDP tests need websocket-client")

from localdecide.drivers import CDPDriver, PlaywrightDriver
from localdecide.loop import ElementRef
from localdecide.page import build_element_table


HTML = """
<style>#cover { position: fixed; inset: 0; background: rgba(0,0,0,.1); }</style>
<button id="delete" onclick="document.body.dataset.clicked='delete'">Delete</button>
<button id="other">Other</button>
<input id="query" aria-label="Query">
<select id="country" aria-label="Country"><option value="au">Australia</option><option value="jp">Japan</option></select>
<div id="status"></div>
"""


def _ref(table, label):
    element = next(item for item in table.elements if item.label == label)
    return ElementRef(element.index, element.label, element.role, element.handle,
                      {**element.meta, "options": element.options},
                      options=list(element.options), sensitive=element.sensitive)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _connect_cdp(port):
    last_error = None
    for _ in range(40):
        try:
            return CDPDriver(endpoint=f"http://127.0.0.1:{port}")
        except Exception as error:  # Chromium may need a short beat before /json is ready.
            last_error = error
            time.sleep(0.05)
    raise AssertionError("isolated Chromium never exposed its local CDP endpoint") from last_error


@pytest.fixture(params=("playwright", "cdp"), ids=("playwright", "cdp"))
def driver(request):
    if request.param == "playwright":
        instance = PlaywrightDriver(start_url="about:blank", settle_ms=0)
        instance._page.set_content(HTML)
        try:
            yield instance
        finally:
            instance.close()
        return

    # Playwright owns a temporary, isolated Chromium profile here. No user_data_dir or
    # persistent browser profile is read; the only external interface under test is CDP.
    from playwright.sync_api import sync_playwright

    port = _free_port()
    runtime = sync_playwright().start()
    browser = None
    instance = None
    try:
        browser = runtime.chromium.launch(
            headless=True,
            args=[
                "--remote-debugging-address=127.0.0.1",
                f"--remote-debugging-port={port}",
                "--remote-allow-origins=*",
            ],
        )
        page = browser.new_page()
        page.set_content(HTML)
        instance = _connect_cdp(port)
        # The page is only the fixture's mutation/inspection handle. All observations
        # and actions in the test go through CDPDriver.
        instance._page = page
        yield instance
    finally:
        if instance is not None:
            instance.close()
        if browser is not None:
            browser.close()
        runtime.stop()


def _table(driver):
    return build_element_table(driver.observe())


def test_renamed_or_replaced_target_fails_closed(driver):
    table = _table(driver)
    target = _ref(table, "Delete")
    driver._page.evaluate("document.querySelector('#delete').textContent = 'Delete renamed'")
    result = driver.execute("CLICK", target)
    assert not result["ok"]
    assert "identity changed" in result["detail"]

    table = _table(driver)
    target = _ref(table, "Delete renamed")
    driver._page.evaluate("document.querySelector('#delete').replaceWith(document.createElement('button'))")
    result = driver.execute("CLICK", target)
    assert not result["ok"]
    assert "missing" in result["detail"] or "identity" in result["detail"]


def test_observation_handles_do_not_cross_document_boundaries(driver):
    table = _table(driver)
    target = _ref(table, "Delete")
    # The new document deliberately has the same first target and label. A document-local
    # counter alone would reuse localdecide-1 and click the new, unrelated node.
    driver._page.set_content(HTML)
    result = driver.execute("CLICK", target)
    assert not result["ok"]
    assert "missing" in result["detail"] or "identity" in result["detail"]
    assert driver._page.locator("body").get_attribute("data-clicked") is None


def test_reorder_does_not_use_stale_coordinates(driver):
    table = _table(driver)
    target = _ref(table, "Delete")
    driver._page.evaluate("document.body.insertBefore(document.querySelector('#other'), document.querySelector('#delete'))")
    result = driver.execute("CLICK", target)
    assert result["ok"], result
    assert driver._page.locator("body").get_attribute("data-clicked") == "delete"


def test_overlay_disabled_and_readonly_targets_fail_closed(driver):
    table = _table(driver)
    target = _ref(table, "Delete")
    driver._page.evaluate("document.body.insertAdjacentHTML('beforeend', '<div id=cover></div>')")
    result = driver.execute("CLICK", target)
    assert not result["ok"]
    assert "covered" in result["detail"]

    driver._page.evaluate("document.querySelector('#cover').remove(); document.querySelector('#delete').disabled=true")
    result = driver.execute("CLICK", target)
    assert not result["ok"]
    assert "disabled" in result["detail"]

    table = _table(driver)
    query = _ref(table, "Query")
    driver._page.evaluate("document.querySelector('#query').readOnly=true")
    result = driver.execute("TYPE_TEXT", query, "safe")
    assert not result["ok"]
    assert "read-only" in result["detail"]


def test_select_and_type_resolve_current_stable_nodes(driver):
    table = _table(driver)
    query = _ref(table, "Query")
    country = _ref(table, "Country")
    driver._page.evaluate("document.body.insertBefore(document.querySelector('#country'), document.querySelector('#delete'))")
    typed = driver.execute("TYPE_TEXT", query, "hello")
    selected = driver.execute("SELECT", country, "jp")
    assert typed["ok"], typed
    assert selected["ok"], selected
    assert driver._page.locator("#query").input_value() == "hello"
    assert driver._page.locator("#country").input_value() == "jp"


def test_select_uses_actual_value_and_rejects_disabled_options(driver):
    table = _table(driver)
    country = _ref(table, "Country")
    driver._page.evaluate("""
      const select = document.querySelector('#country');
      select.insertAdjacentHTML('beforeend',
        '<option value="disabled-value" disabled>Disabled label</option>' +
        '<optgroup disabled><option value="group-disabled">Group disabled</option></optgroup>' +
        '<option value="actual-value">Visible label</option>');
    """)
    refreshed = next(element for element in _table(driver).elements if element.label == "Country")
    assert "disabled-value" not in {option["value"] for option in refreshed.options}
    assert "group-disabled" not in {option["value"] for option in refreshed.options}

    disabled = driver.execute("SELECT", country, "disabled-value")
    assert not disabled["ok"]
    assert "disabled" in disabled["detail"]

    selected = driver.execute("SELECT", country, "actual-value")
    assert selected["ok"], selected
    assert driver._page.locator("#country").input_value() == "actual-value"


def test_select_rejects_a_change_handler_that_reverts_the_value(driver):
    country = _ref(_table(driver), "Country")
    driver._page.evaluate("""
      const select = document.querySelector('#country');
      select.addEventListener('change', () => { select.value = 'au'; });
    """)
    result = driver.execute("SELECT", country, "jp")
    assert not result["ok"]
    assert "did not retain" in result["detail"]
    assert driver._page.locator("#country").input_value() == "au"


def test_cdp_rechecks_target_between_mouse_events(driver):
    if not isinstance(driver, CDPDriver):
        pytest.skip("the mouse-event gap is specific to CDP execution")
    driver._page.set_content("""
      <button id="race" onmousedown="this.replaceWith(document.createElement('button'))"
              onclick="document.body.dataset.clicked='race'">Race target</button>
    """)
    target = _ref(_table(driver), "Race target")
    events = []
    real_call = driver._call

    def record(method, **params):
        events.append((method, params))
        return real_call(method, **params)

    driver._call = record
    result = driver.execute("CLICK", target)
    assert not result["ok"]
    assert "during click" in result["detail"]
    assert driver._page.locator("body").get_attribute("data-clicked") is None
    assert any(method == "Input.dispatchMouseEvent" and params.get("type") == "mouseReleased"
               and params.get("x", 0) < 0 and params.get("y", 0) < 0
               for method, params in events)


def test_cdp_fails_closed_when_target_moves_between_mouse_events(driver):
    if not isinstance(driver, CDPDriver):
        pytest.skip("the mouse-event gap is specific to CDP execution")
    driver._page.set_content("""
      <button id="moving" onmousedown="this.style.transform='translateX(120px)'"
              onclick="document.body.dataset.clicked='moving'">Moving target</button>
    """)
    target = _ref(_table(driver), "Moving target")
    events = []
    real_call = driver._call

    def record(method, **params):
        events.append((method, params))
        return real_call(method, **params)

    driver._call = record
    result = driver.execute("CLICK", target)
    assert not result["ok"]
    assert "moved during click" in result["detail"]
    assert driver._page.locator("body").get_attribute("data-clicked") is None
    assert any(method == "Input.dispatchMouseEvent" and params.get("type") == "mouseReleased"
               and params.get("x", 0) < 0 and params.get("y", 0) < 0
               for method, params in events)
