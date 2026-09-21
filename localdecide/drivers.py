"""Browser drivers: the part that touches the real world.

Both drivers do the same two things:

* `observe()` reads the page and returns the *raw* observation (they do not decide
  what is actionable - `page.build_element_table` does).
* `execute()` performs one operation against one element, resolving the element only
  from what was observed, and reports whether the page changed.

Two flavours because they suit different jobs:

* `PlaywrightDriver` - clean install, drives its own browser, good default.
* `CDPDriver`        - attach to a Chrome you are already logged into. No second
  profile, no re-login; it is the pragmatic choice for anything behind a login.

Neither driver takes screenshots for the decision path: the model reads text. A
screenshot is only ever for *your* logs, not for the loop.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .loop import ElementRef

OBSERVE_JS = r"""
() => {
  const out = [];
  const seen = new Set();
  const clean = (s) => (s || '').replace(/\s+/g, ' ').trim().slice(0, 120);
  const roleOf = (el) => {
    const explicit = el.getAttribute && el.getAttribute('role');
    if (explicit) return explicit;
    const tag = el.tagName.toLowerCase();
    const map = {a: 'link', button: 'button', input: el.type || 'input', select: 'select',
                 textarea: 'textbox', summary: 'button', option: 'option'};
    return map[tag] || tag;
  };
  const labelOf = (el) => {
    if (el.labels && el.labels.length) return clean(el.labels[0].innerText);
    const aria = el.getAttribute && (el.getAttribute('aria-label') || el.getAttribute('title'));
    if (aria) return clean(aria);
    if (el.tagName === 'INPUT' || el.tagName === 'SELECT' || el.tagName === 'TEXTAREA') {
      if (el.placeholder) return clean(el.placeholder);
      if (el.value) return clean(el.value);
    }
    const text = clean(el.innerText || el.textContent);
    if (text) return text;
    const img = el.querySelector && el.querySelector('img[alt]');
    return img ? clean(img.getAttribute('alt')) : '';
  };
  const visible = (el) => {
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
    // A zero-size or clipped ancestor is the usual way pages hide controls without
    // `hidden` or `display:none` - and the naive bounding-box test passes for them,
    // because the button still has its own width and height. Walk up to check.
    let node = el;
    while (node && node !== document.documentElement) {
      const parentStyle = window.getComputedStyle(node);
      if (parentStyle.display === 'none' || parentStyle.visibility === 'hidden') return false;
      if (parentStyle.overflow === 'hidden' || parentStyle.overflow === 'clip') {
        const rect = node.getBoundingClientRect();
        // 1px tolerance: sub-pixel layout makes exact zero unreliable.
        if (rect.width < 1 || rect.height < 1) return false;
      }
      node = node.parentElement;
    }
    const rect = el.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) return false;
    // Skip anything scrolled entirely out of a clipped ancestor.
    const clip = el.getBoundingClientRect();
    if (clip.bottom < 0 && el.offsetParent === null) return false;
    return true;
  };
  const nodes = document.querySelectorAll(
    'a[href], button, input, select, textarea, [role=button], [role=link], [role=menuitem], ' +
    '[role=tab], [role=option], [role=checkbox], [role=radio], summary, [onclick]');
  nodes.forEach((el, i) => {
    if (out.length >= 120) return;
    if (!visible(el)) return;
    const tag = el.tagName.toLowerCase();
    const role = roleOf(el);
    const clickable = tag === 'a' || tag === 'button' || tag === 'summary' ||
                      ['button','link','menuitem','tab','option','checkbox','radio'].includes(role) ||
                      el.hasAttribute('onclick');
    const editable = (tag === 'input' && !['checkbox','radio','submit','button','hidden'].includes(el.type)) ||
                     tag === 'textarea' ||
                     (role === 'textbox' || role === 'searchbox');
    const selectable = tag === 'select';
    if (!clickable && !editable && !selectable) return;
    const label = labelOf(el);
    if (!label) return;
    const key = tag + '|' + role + '|' + label + '|' + i;
    if (seen.has(key)) return;
    seen.add(key);
    const item = {kind: selectable ? 'select' : (editable ? 'fill' : 'click'),
                  label, role, node: key, index: out.length + 1};
    if (editable) item.current_value = el.value || '';
    if (selectable) {
      item.current_value = el.value || '';
      item.options = Array.from(el.options).slice(0, 200).map(o => ({label: clean(o.textContent), value: o.value || ''}));
    }
    if (el.type === 'checkbox' || el.type === 'radio' || role === 'checkbox' || role === 'radio') item.checked = !!el.checked;
    if (el.disabled) item.disabled = true;
    const rect = el.getBoundingClientRect();
    item.bbox = {x: Math.round(rect.x), y: Math.round(rect.y), w: Math.round(rect.width), h: Math.round(rect.height)};
    out.push(item);
  });
  return out;
}
"""


class _BaseDriver:
    """Shared bookkeeping: turning the raw DOM read into an observation dict."""

    def __init__(self, *, text_chars: int = 1500) -> None:
        self.text_chars = text_chars
        self._last_signature: Optional[str] = None

    def _observe_common(self, url: str, title: str, text: str, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        signature = f"{url}|{len(actions)}|{text[:200]}"
        changed = signature != self._last_signature
        self._last_signature = signature
        return {"url": url, "title": title, "text": (text or "")[: self.text_chars],
                "actions": actions, "page_changed": changed}


class PlaywrightDriver(_BaseDriver):
    """Drive a browser with Playwright. `pip install localdecide[playwright]`.

    headless=False is usually the right choice while you are getting a flow working:
    you want to watch what the decisions actually do.
    """

    def __init__(self, *, headless: bool = True, start_url: str = "about:blank",
                 user_data_dir: Optional[str] = None, text_chars: int = 1500,
                 settle_ms: int = 300) -> None:
        super().__init__(text_chars=text_chars)
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("playwright is not installed: pip install playwright && playwright install chromium") from exc
        self.settle_ms = settle_ms
        # Playwright's sync API binds to the event loop that is current *at start()* time,
        # and close() closes that loop. A second driver created later in the same process
        # then finds "Event loop is closed!" - which is what happened when the real-website
        # battery created several drivers in a row. The fix is a fresh loop per driver,
        # started explicitly and restored after, so each driver owns its loop lifecycle.
        import asyncio

        try:
            self._previous_loop = asyncio.get_event_loop()
        except RuntimeError:
            self._previous_loop = None
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._pw = sync_playwright().start()
        launch: Dict[str, Any] = {"headless": headless}
        if user_data_dir:
            self._context = self._pw.chromium.launch_persistent_context(user_data_dir, **launch)
            self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
        else:
            self._browser = self._pw.chromium.launch(**launch)
            self._context = None
            self._page = self._browser.new_page()
        if start_url and start_url != "about:blank":
            self._page.goto(start_url, wait_until="domcontentloaded")
            self._settle()

    def _settle(self) -> None:
        """Give a just-navigated page a moment to finish firing its own loads.

        Reading the DOM while a page is still navigating raises "Execution context was
        destroyed" - correct behaviour from the browser, but noise for a driver that is
        simply observing a little too early. Waiting for network idle and then a short
        beat is enough in practice; anything longer belongs to the caller's policy.
        """
        try:
            self._page.wait_for_load_state("networkidle", timeout=2500)
        except Exception:
            pass  # a page with long-poll connections never goes idle; carry on
        self._page.wait_for_timeout(self.settle_ms)

    def observe(self) -> Dict[str, Any]:
        from playwright.sync_api import Error as PlaywrightError
        for attempt in range(3):
            try:
                actions = self._page.evaluate(OBSERVE_JS)
                title = self._page.title()
                body = self._page.query_selector("body")
                text = self._page.inner_text("body") if body else ""
                return self._observe_common(self._page.url, title, text, actions)
            except PlaywrightError as error:
                # A navigation destroyed the execution context between calls. Settle and retry.
                if "Execution context was destroyed" in str(error) or "navigation" in str(error).lower():
                    if attempt < 2:
                        self._settle()
                        continue
                raise
        raise RuntimeError("observe failed after retries")

    def execute(self, operation: str, element: Optional[ElementRef], text: Optional[str] = None) -> Dict[str, Any]:
        from playwright.sync_api import Error as PlaywrightError
        try:
            if operation == "SCROLL_DOWN":
                self._page.mouse.wheel(0, 800)
            elif operation == "SCROLL_UP":
                self._page.mouse.wheel(0, -800)
            elif operation == "WAIT":
                self._page.wait_for_timeout(400)
            elif operation in ("CLICK", "TYPE_TEXT", "SELECT"):
                if element is None:
                    return {"ok": False, "detail": "no element"}
                handle = self._resolve(element)
                box = handle.bounding_box()
                if box is None:
                    return {"ok": False, "detail": "element not visible"}
                if operation == "CLICK":
                    handle.click(timeout=5000)
                elif operation == "TYPE_TEXT" and text is not None:
                    handle.fill(text, timeout=5000)
                elif operation == "SELECT":
                    # The loop already resolved the observed option to a value; drive the
                    # real <select> by its bounding box so no selector is ever invented.
                    if not text:
                        return {"ok": False, "detail": "SELECT with no option value"}
                    if not self._select_by_bbox(box, text):
                        return {"ok": False, "detail": f"option {text!r} not present in the dropdown"}
                else:
                    return {"ok": False, "detail": f"{operation} needs a payload the loop did not provide"}
        except PlaywrightError as error:
            return {"ok": False, "detail": f"{type(error).__name__}: {str(error)[:120]}"}
        # A click can trigger navigation. Let it land before anyone reads the DOM again.
        self._settle()
        signature = f"{self._page.url}|{self._page.title()}"
        changed = signature != getattr(self, "_last_url_signature", None)
        self._last_url_signature = signature
        return {"ok": True, "detail": "ok", "page_changed": changed}

    def _resolve(self, element: ElementRef):
        meta = element.meta or {}
        bbox = meta.get("bbox") or {}
        x = bbox.get("x", 0) + max(1, bbox.get("w", 2) // 2)
        y = bbox.get("y", 0) + max(1, bbox.get("h", 2) // 2)
        return _PointHandle(self._page, x, y)

    def _select_by_bbox(self, box: Dict[str, int], value: str) -> bool:
        """Select an option in the <select> whose box we were given.

        Driving a native dropdown by keyboard is the only approach that works without
        inventing a CSS selector: focus the control, walk its options with the keyboard,
        and read back `value` to confirm the choice landed. If the option is not there,
        say so - a silent failure here would look like the model picked wrong.
        """
        x = box["x"] + max(1, box["width"] // 2)
        y = box["y"] + max(1, box["height"] // 2)
        self._page.mouse.click(x, y)
        element_handle = self._page.evaluate_handle(
            "([x, y]) => document.elementFromPoint(x, y)", [x, y]
        )
        element = element_handle.as_element()
        if element is None:
            return False
        options = element.evaluate("(el) => el.tagName === 'SELECT' ? "
                                   "Array.from(el.options).map(o => ({value: o.value, label: o.textContent})) : null")
        if not options:
            # Not a native select - a custom dropdown. Try typing the option label.
            self._page.keyboard.type(value)
            self._page.keyboard.press("Enter")
            return True
        index = next((i for i, option in enumerate(options) if option["value"] == value), None)
        if index is None:
            return False
        # Home, then down N times: works regardless of how the page renders the list.
        self._page.keyboard.press("Home")
        for _ in range(index):
            self._page.keyboard.press("ArrowDown")
        self._page.keyboard.press("Enter")
        self._page.wait_for_timeout(120)
        current = element.evaluate("(el) => el.value")
        return str(current) == str(value)

    def close(self) -> None:
        # Idempotent: the loop is closed on the first call, and a second close must be a
        # no-op rather than an exception. The real-website battery hit this because both
        # BrowserDecider.run()'s finally block and the caller's finally block closed the
        # same driver, and the second close crashed on the already-closed loop - turning
        # a completed run into a spurious DRIVER ERROR.
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            if getattr(self, "_context", None) is not None:
                self._context.close()
            elif getattr(self, "_browser", None) is not None:
                self._browser.close()
        finally:
            self._pw.stop()
            # Restore the caller's event loop so a subsequent driver (or any asyncio
            # work in the same process) starts clean instead of finding a closed loop.
            import asyncio
            asyncio.set_event_loop(self._previous_loop)


class _PointHandle:
    """Minimal click/type helper at a coordinate. Keeps the driver honest: the model
    picked an element we observed, and this is where that element was."""

    def __init__(self, page, x: int, y: int) -> None:
        self.page, self.x, self.y = page, x, y

    def bounding_box(self):
        return {"x": self.x, "y": self.y, "width": 2, "height": 2}

    def click(self, timeout: int = 5000) -> None:
        self.page.mouse.click(self.x, self.y)

    def fill(self, text: str, timeout: int = 5000) -> None:
        self.page.mouse.click(self.x, self.y)
        self.page.keyboard.press("Meta+A" if self.page.evaluate("navigator.platform.includes('Mac')") else "Control+A")
        self.page.keyboard.type(text)


class CDPDriver(_BaseDriver):
    """Attach to an already-running Chrome over CDP. No `pip install`, no new profile.

    Start Chrome with remote debugging, log in once by hand, then let this drive it:

        /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222

    Requires `websocket-client` (`pip install websocket-client`) for the CDP socket.
    """

    def __init__(self, endpoint: str = "http://127.0.0.1:9222", *, text_chars: int = 1500, target_url_contains: str = "") -> None:
        super().__init__(text_chars=text_chars)
        try:
            from websocket import create_connection  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("websocket-client is not installed: pip install websocket-client") from exc
        import urllib.request
        with urllib.request.urlopen(f"{endpoint}/json", timeout=10) as response:
            tabs = json.loads(response.read().decode("utf-8"))
        pages = [tab for tab in tabs if tab.get("type") == "page"]
        if target_url_contains:
            pages = [tab for tab in pages if target_url_contains in tab.get("url", "")] or pages
        if not pages:
            raise RuntimeError("no attachable page found over CDP")
        self.target = pages[0]
        self._ws = create_connection(self.target["webSocketDebuggerUrl"], timeout=30)
        self._id = 0

    def _call(self, method: str, **params: Any) -> Dict[str, Any]:
        self._id += 1
        message = json.dumps({"id": self._id, "method": method, "params": params})
        self._ws.send(message)
        while True:
            raw = self._ws.recv()
            data = json.loads(raw)
            if data.get("id") == self._id:
                if "error" in data:
                    raise RuntimeError(data["error"].get("message", "CDP error"))
                return data.get("result", {})

    def _eval(self, expression: str) -> Any:
        result = self._call("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        return result.get("result", {}).get("value")

    def observe(self) -> Dict[str, Any]:
        return self._observe_common(
            self._eval("location.href") or "",
            self._eval("document.title") or "",
            self._eval("document.body ? document.body.innerText : ''") or "",
            self._eval(f"({OBSERVE_JS})()") or [],
        )

    def execute(self, operation: str, element: Optional[ElementRef], text: Optional[str] = None) -> Dict[str, Any]:
        try:
            if operation == "SCROLL_DOWN":
                self._eval("window.scrollBy(0, 800)")
            elif operation == "SCROLL_UP":
                self._eval("window.scrollBy(0, -800)")
            elif operation == "WAIT":
                import time
                time.sleep(0.3)
            elif operation in ("CLICK", "TYPE_TEXT"):
                if element is None:
                    return {"ok": False, "detail": "no element"}
                x, y = self._center(element)
                for event_type in ("mousePressed", "mouseReleased"):
                    self._call("Input.dispatchMouseEvent", type=event_type, x=x, y=y,
                               button="left", clickCount=1)
                if operation == "TYPE_TEXT" and text is not None:
                    self._call("Input.insertText", text=text)
            else:
                return {"ok": False, "detail": f"unsupported {operation}"}
        except Exception as error:  # noqa: BLE001 - driver failures are data, not crashes
            return {"ok": False, "detail": f"{type(error).__name__}: {str(error)[:120]}"}
        import time
        time.sleep(0.25)
        signature = f"{self._eval('location.href')}|{self._eval('document.title')}"
        changed = signature != getattr(self, "_last_url_signature", None)
        self._last_url_signature = signature
        return {"ok": True, "detail": "ok", "page_changed": changed}

    def _center(self, element: ElementRef) -> tuple:
        bbox = (element.meta or {}).get("bbox") or {}
        x = int(bbox.get("x", 0) + max(1, bbox.get("w", 2) // 2))
        y = int(bbox.get("y", 0) + max(1, bbox.get("h", 2) // 2))
        return x, y

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self._ws.close()
        except Exception:
            pass


def open_driver(name: str = "playwright", **kwargs: Any) -> Any:
    """`open_driver("cdp")` / `open_driver("playwright", headless=False)`."""
    return CDPDriver(**kwargs) if name == "cdp" else PlaywrightDriver(**kwargs)
