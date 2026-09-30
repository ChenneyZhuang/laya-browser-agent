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

import asyncio
import json
import warnings
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
    const sensitive = tag === 'input' && String(el.type || '').toLowerCase() === 'password';
    const label = labelOf(el) || (sensitive ? 'Password field' : '');
    if (!label) return;
    if (!el.__localdecide_handle) {
      if (!window.__localdecide_document_id) {
        const random = window.crypto && typeof window.crypto.randomUUID === 'function'
          ? window.crypto.randomUUID()
          : Date.now().toString(36) + '-' + Math.random().toString(36).slice(2);
        window.__localdecide_document_id = random;
      }
      window.__localdecide_next_handle = (window.__localdecide_next_handle || 0) + 1;
      el.__localdecide_handle = 'localdecide-' + window.__localdecide_document_id + '-' +
                                window.__localdecide_next_handle;
    }
    const key = el.__localdecide_handle;
    if (seen.has(key)) return;
    seen.add(key);
    const item = {kind: selectable ? 'select' : (editable ? 'fill' : 'click'),
                  label, role, node: key, index: out.length + 1, sensitive};
    if (editable && !sensitive) item.current_value = el.value || '';
    if (selectable) {
      item.current_value = el.value || '';
      item.options = Array.from(el.options)
        .filter(o => !o.disabled && !(o.parentElement && o.parentElement.disabled))
        .slice(0, 200)
        .map(o => ({label: clean(o.textContent), value: o.value || ''}));
    }
    if (el.type === 'checkbox' || el.type === 'radio' || role === 'checkbox' || role === 'radio') item.checked = !!el.checked;
    if (el.disabled) item.disabled = true;
    item.identity = {tag, role, type: String(el.type || ''), label,
                     disabled: !!el.disabled, read_only: !!el.readOnly, sensitive};
    const rect = el.getBoundingClientRect();
    item.bbox = {x: Math.round(rect.x), y: Math.round(rect.y), w: Math.round(rect.width), h: Math.round(rect.height)};
    out.push(item);
  });
  return out;
}
"""


def _current_event_loop() -> Optional[asyncio.AbstractEventLoop]:
    """Return the caller's loop without the deprecated idle-loop lookup.

    Python 3.14 deprecates ``asyncio.get_event_loop_policy()`` itself. There is
    still no public getter for a loop that is set but not running, so inspect the
    policy's thread-local slot through that legacy compatibility API after checking
    for a running loop. Suppress only that known policy deprecation. Keeping the
    idle loop is important: ``PlaywrightDriver.close()`` must restore it exactly as
    the constructor found it.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        pass
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^'asyncio\.get_event_loop_policy' is deprecated",
            category=DeprecationWarning,
        )
        policy = asyncio.get_event_loop_policy()
    local = getattr(policy, "_local", None)
    loop = getattr(local, "_loop", None)
    if loop is None or loop.is_closed():
        return None
    return loop


class _BaseDriver:
    """Shared bookkeeping: turning the raw DOM read into an observation dict."""

    def __init__(self, *, text_chars: int = 1500) -> None:
        self.text_chars = text_chars
        self._last_signature: Optional[str] = None

    def _observe_common(self, url: str, title: str, text: str, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        safe_actions: List[Dict[str, Any]] = []
        for action in actions:
            safe = dict(action)
            role = str(safe.get("role", "") or "").lower()
            input_type = str(safe.get("type", "") or "").lower()
            sensitive = bool(safe.get("sensitive")) or role == "password" or input_type == "password"
            if sensitive:
                supplied = {str(safe.get(key, "") or "") for key in ("value", "current_value")}
                label = str(safe.get("label", "") or "").strip()
                if not label or label in supplied:
                    safe["label"] = "Password field"
                safe.pop("value", None)
                safe.pop("current_value", None)
                safe["sensitive"] = True
            safe_actions.append(safe)
        actions = safe_actions
        signature = f"{url}|{len(actions)}|{text[:200]}"
        changed = signature != self._last_signature
        self._last_signature = signature
        return {"url": url, "title": title, "text": (text or "")[: self.text_chars],
                "actions": actions, "page_changed": changed}


class PlaywrightDriver(_BaseDriver):
    """Drive a browser with Playwright. From the repository root: `pip install -e '.[playwright]'`.

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
        self._previous_loop = _current_event_loop()
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._closed = False
        try:
            self._pw = sync_playwright().start()
            launch: Dict[str, Any] = {"headless": headless}
            if user_data_dir:
                self._context = self._pw.chromium.launch_persistent_context(user_data_dir, **launch)
                self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
            else:
                self._browser = self._pw.chromium.launch(**launch)
                self._page = self._browser.new_page()
            if start_url and start_url != "about:blank":
                self._page.goto(start_url, wait_until="domcontentloaded")
                self._settle()
        except BaseException:
            self.close()
            raise

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

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
                handle = self._resolve(element, operation, text)
                if operation == "CLICK":
                    handle.click(timeout=5000)
                elif operation == "TYPE_TEXT" and text is not None:
                    handle.fill(text, timeout=5000)
                elif operation == "SELECT":
                    if not text:
                        return {"ok": False, "detail": "SELECT with no option value"}
                    handle.select_option(text, timeout=5000)
                    actual = handle.evaluate("el => String(el.value)")
                    if actual != str(text):
                        return {"ok": False, "detail": "select change handler did not retain the requested value"}
                else:
                    return {"ok": False, "detail": f"{operation} needs a payload the loop did not provide"}
        except PlaywrightError as error:
            return {"ok": False, "detail": f"{type(error).__name__}: {str(error)[:120]}"}
        # A click can trigger navigation. Let it land before anyone reads the DOM again.
        self._settle()
        # The page "changed" if URL/title moved OR the interactive surface moved:
        # SPAs and dynamic panels reveal content without touching location. A
        # click that visibly changed the DOM is progress even when the URL is not.
        dom_sig = self._page.evaluate(
            "() => JSON.stringify([document.visibilityState,"
            " Array.from(document.querySelectorAll('button, a, input, select, textarea'))"
            ".filter(e => e.offsetParent !== null).length,"
            " (document.body ? document.body.innerText.length : 0)])"
        )
        signature = f"{self._page.url}|{self._page.title()}|{dom_sig}"
        changed = signature != getattr(self, "_last_url_signature", None)
        self._last_url_signature = signature
        return {"ok": True, "detail": "ok", "page_changed": changed}

    def _resolve(self, element: ElementRef, operation: str, value: Optional[str] = None):
        """Resolve and validate the same DOM node that observation assigned a handle to."""
        if not isinstance(element.handle, str) or not element.handle:
            raise RuntimeError("target has no stable observation handle")
        payload = {"handle": element.handle, "expected": element.meta.get("identity", {}),
                   "operation": operation, "value": value}
        handle = self._page.evaluate_handle(
            """(payload) => {
                const labelOf = (el) => {
                  if (el.labels && el.labels.length) return (el.labels[0].innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 120);
                  const aria = el.getAttribute && (el.getAttribute('aria-label') || el.getAttribute('title'));
                  if (aria) return aria.replace(/\\s+/g, ' ').trim().slice(0, 120);
                  if (el.placeholder) return el.placeholder.replace(/\\s+/g, ' ').trim().slice(0, 120);
                  const text = (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim();
                  return text.slice(0, 120);
                };
                const roleOf = (el) => el.getAttribute('role') ||
                  ({a:'link',button:'button',input:el.type || 'input',select:'select',textarea:'textbox',summary:'button'}[el.tagName.toLowerCase()] || el.tagName.toLowerCase());
                const el = Array.from(document.querySelectorAll('*')).find(node => node.__localdecide_handle === payload.handle);
                if (!el || !el.isConnected) throw new Error('target is missing or was replaced');
                const identity = payload.expected || {};
                const actualLabel = labelOf(el) || (String(el.type || '').toLowerCase() === 'password' ? 'Password field' : '');
                if (identity.label && actualLabel !== identity.label) throw new Error('target identity changed: label');
                if (identity.tag && el.tagName.toLowerCase() !== identity.tag) throw new Error('target identity changed: tag');
                if (identity.role && roleOf(el) !== identity.role) throw new Error('target identity changed: role');
                if (el.disabled) throw new Error('target is disabled');
                if (payload.operation === 'TYPE_TEXT' && (el.readOnly || String(el.type || '').toLowerCase() === 'password' || identity.sensitive))
                  throw new Error(String(el.type || '').toLowerCase() === 'password' || identity.sensitive
                    ? 'password input rejected: vault access is not provided by this project'
                    : 'target is read-only');
                if (payload.operation === 'SELECT' && el.tagName.toLowerCase() !== 'select') throw new Error('target is not a select');
                if (payload.operation === 'SELECT') {
                  const option = Array.from(el.options).find(item => String(item.value) === String(payload.value));
                  if (!option) throw new Error('option is not present in the dropdown');
                  if (option.disabled || (option.parentElement && option.parentElement.disabled))
                    throw new Error('option is disabled');
                }
                const style = getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0' || rect.width < 1 || rect.height < 1)
                  throw new Error('target is not actionable');
                const top = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
                if (top && top !== el && !el.contains(top)) throw new Error('target is covered');
                return el;
            }""", payload).as_element()
        if handle is None:
            raise RuntimeError("target is not a DOM element")
        return handle

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
                try:
                    self._context.close()
                except Exception:
                    pass
            elif getattr(self, "_browser", None) is not None:
                try:
                    self._browser.close()
                except Exception:
                    pass
        finally:
            try:
                if getattr(self, "_pw", None) is not None:
                    self._pw.stop()
            except Exception:
                pass
            # Restore the caller's event loop so a subsequent driver (or any asyncio
            # work in the same process) starts clean instead of finding a closed loop.
            import asyncio
            try:
                asyncio.set_event_loop(self._previous_loop)
            finally:
                owned_loop = getattr(self, "_loop", None)
                if owned_loop is not None and not owned_loop.is_closed():
                    owned_loop.close()


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
            pages = [tab for tab in pages if target_url_contains in tab.get("url", "")]
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
                result = dict(data.get("result", {}) or {})
                if "exceptionDetails" in data:
                    result["exceptionDetails"] = data["exceptionDetails"]
                return result

    def _eval(self, expression: str) -> Any:
        result = self._call("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        if result.get("exceptionDetails"):
            details = result["exceptionDetails"]
            exception = details.get("exception", {}) if isinstance(details, dict) else {}
            message = exception.get("description") or details.get("text") or "Runtime.evaluate failed"
            raise RuntimeError(message)
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
            elif operation in ("CLICK", "TYPE_TEXT", "SELECT"):
                if element is None:
                    return {"ok": False, "detail": "no element"}
                target = self._target_state(element, operation)
                if operation == "CLICK":
                    x, y = target["x"], target["y"]
                    pressed = False
                    try:
                        # Assume the event may have reached Chromium even if the transport
                        # reports an error; cleanup is harmless when it did not.
                        pressed = True
                        self._call("Input.dispatchMouseEvent", type="mousePressed", x=x, y=y,
                                   button="left", clickCount=1)
                        # CDP has no atomic press/release plus identity operation. Re-resolve
                        # the handle and geometry in the gap; this is fail-closed cleanup,
                        # not a transactional race solution.
                        current = self._target_state(element, operation)
                        if (current["x"], current["y"]) != (x, y):
                            raise RuntimeError("target moved during click")
                        self._call("Input.dispatchMouseEvent", type="mouseReleased", x=x, y=y,
                                   button="left", clickCount=1)
                        pressed = False
                    except Exception as error:
                        cleanup = ""
                        if pressed:
                            try:
                                self._release_mouse_outside_target()
                            except Exception as cleanup_error:
                                cleanup = f"; mouse cleanup failed: {cleanup_error}"
                        return {"ok": False, "detail": f"target changed during click: {error}{cleanup}"}
                elif operation == "TYPE_TEXT" and text is not None:
                    self._eval(self._target_expression(element, operation, action="focus"))
                    self._call("Input.insertText", text=text)
                elif operation == "SELECT" and text:
                    changed = self._eval(self._target_expression(element, operation, value=text, action="select"))
                    if not changed:
                        return {"ok": False, "detail": f"option {text!r} not present in the dropdown"}
                else:
                    return {"ok": False, "detail": f"{operation} needs a payload the loop did not provide"}
            else:
                return {"ok": False, "detail": f"unsupported {operation}"}
        except Exception as error:
            return {"ok": False, "detail": f"{type(error).__name__}: {str(error)[:120]}"}
        import time
        time.sleep(0.25)
        # Same SPA-aware signature as the Playwright driver: a dynamic panel that
        # reveals content without navigating is still progress.
        dom_sig = self._eval(
            "JSON.stringify([Array.from(document.querySelectorAll('button, a, input, select, textarea'))"
            ".filter(e => e.offsetParent !== null).length,"
            " (document.body ? document.body.innerText.length : 0)])"
        )
        signature = f"{self._eval('location.href')}|{self._eval('document.title')}|{dom_sig}"
        changed = signature != getattr(self, "_last_url_signature", None)
        self._last_url_signature = signature
        return {"ok": True, "detail": "ok", "page_changed": changed}

    def _target_expression(self, element: ElementRef, operation: str, *, action: str = "validate",
                           value: str = "") -> str:
        payload = json.dumps({"handle": element.handle, "expected": element.meta.get("identity", {}),
                              "operation": operation, "action": action, "value": value})
        return f"""(() => {{
          const payload = {payload};
          const labelOf = (el) => {{
            if (el.labels && el.labels.length) return (el.labels[0].innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 120);
            const aria = el.getAttribute && (el.getAttribute('aria-label') || el.getAttribute('title'));
            if (aria) return aria.replace(/\\s+/g, ' ').trim().slice(0, 120);
            if (el.placeholder) return el.placeholder.replace(/\\s+/g, ' ').trim().slice(0, 120);
            return (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 120);
          }};
          const roleOf = (el) => el.getAttribute('role') ||
            ({{a:'link',button:'button',input:el.type || 'input',select:'select',textarea:'textbox',summary:'button'}}[el.tagName.toLowerCase()] || el.tagName.toLowerCase());
          const el = Array.from(document.querySelectorAll('*')).find(node => node.__localdecide_handle === payload.handle);
          if (!el || !el.isConnected) throw new Error('target is missing or was replaced');
          const identity = payload.expected || {{}};
          const actualLabel = labelOf(el) || (String(el.type || '').toLowerCase() === 'password' ? 'Password field' : '');
          if (identity.label && actualLabel !== identity.label) throw new Error('target identity changed: label');
          if (identity.tag && el.tagName.toLowerCase() !== identity.tag) throw new Error('target identity changed: tag');
          if (identity.role && roleOf(el) !== identity.role) throw new Error('target identity changed: role');
          if (el.disabled) throw new Error('target is disabled');
          if (payload.operation === 'TYPE_TEXT' && (el.readOnly || String(el.type || '').toLowerCase() === 'password' || identity.sensitive))
            throw new Error(String(el.type || '').toLowerCase() === 'password' || identity.sensitive
              ? 'password input rejected: vault access is not provided by this project' : 'target is read-only');
          if (payload.operation === 'SELECT' && el.tagName.toLowerCase() !== 'select') throw new Error('target is not a select');
          const style = getComputedStyle(el), rect = el.getBoundingClientRect();
          if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0' || rect.width < 1 || rect.height < 1)
            throw new Error('target is not actionable');
          const top = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
          if (top && top !== el && !el.contains(top)) throw new Error('target is covered');
          if (payload.action === 'focus') {{ el.focus(); if (typeof el.select === 'function') el.select(); return true; }}
          if (payload.action === 'select') {{
            const option = Array.from(el.options || []).find(item => String(item.value) === String(payload.value));
            if (!option) return false;
            if (option.disabled || (option.parentElement && option.parentElement.disabled))
              throw new Error('option is disabled');
            el.focus(); el.value = payload.value;
            el.dispatchEvent(new Event('input', {{bubbles: true}}));
            el.dispatchEvent(new Event('change', {{bubbles: true}}));
            if (String(el.value) !== String(payload.value))
              throw new Error('select change handler did not retain the requested value');
            return true;
          }}
          return {{x: Math.round(rect.left + rect.width / 2), y: Math.round(rect.top + rect.height / 2)}};
        }})()"""

    def _target_state(self, element: ElementRef, operation: str) -> Dict[str, Any]:
        if not isinstance(element.handle, str) or not element.handle:
            raise RuntimeError("target has no stable observation handle")
        state = self._eval(self._target_expression(element, operation))
        if not isinstance(state, dict) or "x" not in state or "y" not in state:
            raise RuntimeError("target validation returned no current geometry")
        return state

    def _release_mouse_outside_target(self) -> None:
        """Release CDP's pressed button away from any actionable page target."""
        self._call("Input.dispatchMouseEvent", type="mouseReleased", x=-1, y=-1,
                   button="left", clickCount=1)

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
