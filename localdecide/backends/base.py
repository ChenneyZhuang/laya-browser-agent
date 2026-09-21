"""Backend protocol and resolution.

Backends are duck-typed: anything with `answer(state, questions)` works. The two
bundled ones wrap the open-weight Laya runtime in its MLX (Apple Silicon) and
PyTorch (everywhere) flavours.

Why a backend is a runtime choice, not a code choice: the *decision quality* comes
from the checkpoint, and both runtimes load the same checkpoint. The runtime only
decides speed and where you can run it.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import time
from typing import Any, Dict, Mapping, Optional, Protocol, runtime_checkable

State = Any
Questions = Mapping[str, Mapping[str, Any]]


class BackendError(RuntimeError):
    """Anything that means "do not trust this decision"."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


@runtime_checkable
class Backend(Protocol):
    name: str

    def answer(self, state: State, questions: Questions) -> Dict[str, Any]:
        """Return {"answers": {...}, "usage": {...}} for one state."""


# ── bundled backends ─────────────────────────────────────────────────────────

class LayaMLXBackend:
    """Apple Silicon via MLX. `pip install laya-mlx`.

    Fastest path on a Mac: measured 10-30 ms per decision on an M4 for short states.
    """

    name = "laya-mlx"
    # The hosted checkpoint that is actually fine-tuned for browser decisions. The base
    # `convaiinnovations/laya` checkpoints are near-chance at picking a control
    # (documented top-1 ~0.10 among ~45 candidates), so browser work needs this one.
    BROWSER_MODEL = "cklxx/laya-browser"
    BROWSER_SUBFOLDER = "v10s"

    def __init__(self, model: str = "browser", subfolder: str | None = None, **kwargs: Any) -> None:
        try:
            import laya_mlx  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise BackendError(
                "backend_missing",
                "laya-mlx is not installed. On Apple Silicon: pip install laya-mlx",
            ) from exc
        self._laya = importlib.import_module("laya_mlx")
        self._kwargs = kwargs
        if model == "browser":
            model, subfolder = self.BROWSER_MODEL, subfolder or self.BROWSER_SUBFOLDER
        self._model_arg: Any = model
        self._subfolder: Optional[str] = subfolder
        self._agent = None

    def _load(self):
        if self._agent is None:
            self._agent = self._laya.load(self._model_arg, subfolder=self._subfolder, **self._kwargs)
        return self._agent

    def answer(self, state: State, questions: Questions) -> Dict[str, Any]:
        agent = self._load()
        started = time.monotonic()
        try:
            result = agent.predict(state, dict(questions))
        except Exception as exc:  # surface as a typed failure so callers fail open
            raise BackendError("backend_failed", f"{type(exc).__name__}: {exc}") from exc
        return {
            "answers": result.get("answers", {}),
            "usage": result.get("usage", {}),
            "latency_ms": int((time.monotonic() - started) * 1000),
            "backend": self.name,
        }


class LayaTorchBackend:
    """Any platform with PyTorch. `pip install laya`.

    Slower than the MLX path on a Mac and needs the CUDA/CPU PyTorch wheel, but it
    is the portable option for Linux servers, Windows, and Intel Macs.
    """

    name = "laya-torch"

    def __init__(self, model: str = "browser", subfolder: str | None = None, **kwargs: Any) -> None:
        try:
            import laya  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise BackendError(
                "backend_missing",
                "laya is not installed. pip install laya (pulls torch and transformers)",
            ) from exc
        self._laya = importlib.import_module("laya")
        self._kwargs = kwargs
        if model == "browser":
            model, subfolder = LayaMLXBackend.BROWSER_MODEL, subfolder or LayaMLXBackend.BROWSER_SUBFOLDER
        self._model_arg: Any = model
        self._subfolder: Optional[str] = subfolder
        self._agent = None

    def _load(self):
        if self._agent is None:
            self._agent = self._laya.load(self._model_arg, subfolder=self._subfolder, **self._kwargs)
        return self._agent

    def answer(self, state: State, questions: Questions) -> Dict[str, Any]:
        agent = self._load()
        started = time.monotonic()
        try:
            result = agent.predict(state, dict(questions))
        except Exception as exc:
            raise BackendError("backend_failed", f"{type(exc).__name__}: {exc}") from exc
        return {
            "answers": result.get("answers", {}),
            "usage": result.get("usage", {}),
            "latency_ms": int((time.monotonic() - started) * 1000),
            "backend": self.name,
        }


class HTTPBackend:
    """Talk to a `systemone`-shaped HTTP service instead of loading a model here.

    Point this at `localdecide.serve` on another machine, at a TypeSafe-compatible
    gateway, or at any service that speaks the same request/response contract.
    """

    name = "http"

    def __init__(self, url: str, api_key: str | None = None, model: str = "local", timeout: float = 60.0) -> None:
        self.url = url
        self.api_key = api_key or os.environ.get("LOCALDECIDE_API_KEY", "")
        self.model = model
        self.timeout = timeout

    def answer(self, state: State, questions: Questions) -> Dict[str, Any]:
        import json
        import urllib.error
        import urllib.request

        body = json.dumps({"model": self.model, "state": state, "questions": questions}).encode()
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise BackendError(f"http_{error.code}", error.read().decode("utf-8", "ignore")[:200]) from None
        except Exception as error:  # network, timeout, malformed
            raise BackendError("network", str(error)[:200]) from None
        if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
            raise BackendError("malformed", "reply has no answers")
        payload.setdefault("backend", self.name)
        return payload


_BUILTINS = {"laya-mlx": LayaMLXBackend, "laya-torch": LayaTorchBackend}


def _mlx_available() -> bool:
    import platform

    if platform.machine() != "arm64" or platform.system() != "Darwin":
        return False
    return importlib.util.find_spec("laya_mlx") is not None


def resolve_backend(spec: Any = None, **kwargs: Any) -> Backend:
    """Turn a user-supplied spec into a backend instance.

    Accepts: None (auto-detect), a backend name, an `http(s)://` URL, or any object
    that already satisfies the `Backend` protocol.
    """
    if spec is None or spec == "auto":
        if _mlx_available():
            return LayaMLXBackend(**kwargs)
        if importlib.util.find_spec("laya") is not None:
            return LayaTorchBackend(**kwargs)
        raise BackendError(
            "no_backend",
            "No local decision runtime found. Apple Silicon: pip install 'localdecide[mlx]'. "
            "Other platforms: pip install 'localdecide[torch]'.",
        )
    if isinstance(spec, str) and spec.startswith(("http://", "https://")):
        return HTTPBackend(spec, **kwargs)
    if isinstance(spec, str):
        if spec in _BUILTINS:
            return _BUILTINS[spec](**kwargs)
        raise BackendError("unknown_backend", f"{spec!r} is not a known backend: {sorted(_BUILTINS)}")
    if hasattr(spec, "answer"):  # already a backend
        return spec
    raise BackendError("unknown_backend", f"cannot use {type(spec).__name__} as a backend")
