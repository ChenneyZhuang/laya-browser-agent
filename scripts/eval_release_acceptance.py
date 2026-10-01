#!/usr/bin/env python3
"""Prepare and (on the parent Linux/CUDA host) run the v37 release gate.

The preparation path is intentionally boring: it only uses the standard library,
checks the checkpoint and the original wire JSONL, and writes an auditable
manifest.  PyTorch and Laya are imported only from the explicit ``--execute``
path.  In particular, importing this module on a Mac must not make the optional
CUDA runtime a prerequisite.

The execute path is deliberately not a general benchmark.  It loads exactly one
checkpoint with the Laya 0.3.4-compatible ``load(path, device="cuda")`` call and
uses the project's real ``LayaTorchBackend`` around that already-loaded agent.
The original wire ``state`` and ``questions`` are retained as the input audit;
``Decider`` may deploy coarse-to-fine transformed questions to the backend when a
choice is wider than its configured option budget.  Every raw backend envelope
and the questions sent with it are recorded.  Scoring uses only successful
``decision.answers.raw`` (the merged, validated result); ``prediction`` remains
the last raw backend envelope for audit and is not the scoring source.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _datetime
import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
import platform
import re
import subprocess
import sys
import tempfile
import time
import types
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_SCORER = REPO_ROOT / "reports" / "v37" / "evaluate_wide.py"
DEFAULT_FRAMEWORK_FILES = (
    REPO_ROOT / "localdecide" / "backends" / "base.py",
    REPO_ROOT / "localdecide" / "decider.py",
)
EXPECTED_ITEMS = 95
DEFAULT_MAX_OPTIONS_PER_QUESTION = 20
DEFAULT_BASELINE_MEMORY_MIB = 1218
DEFAULT_MAX_LEN = 4096
DEFAULT_HEAD_MAX_LEN = 1024
DEFAULT_WARMUP = 3
FIXED_DTYPE_NAME = "bf16"
GPU_LOCK_PATH = pathlib.Path("/mnt/d/Jev-Training/logs/release_acceptance/gpu.lock")
RUNTIME_LAYA_VERSION = "0.3.4"
RICH_STATE_KEYS = frozenset({"state_rich", "rich_state", "rendered_state"})
SUPPORTED_OPERATION_SEMANTICS = frozenset({
    "CLICK", "TYPE_TEXT", "SELECT", "SCROLL_DOWN", "SCROLL_UP", "WAIT", "DONE", "BLOCKED",
})
SCORING_SOURCE = "decision.answers.raw (merged validated answers)"


class AcceptanceError(RuntimeError):
    """A release-gate input or runtime safety check failed."""


class PrepareError(AcceptanceError):
    """The input set cannot be accepted as the fixed release wire."""


class SafetyError(AcceptanceError):
    """The GPU host is not safe to use for the release run."""


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _require_absolute(path: pathlib.Path, option: str) -> pathlib.Path:
    path = pathlib.Path(path)
    if not path.is_absolute():
        raise PrepareError(f"{option} must be an absolute path")
    return path


def _json_loads(line: str, source: pathlib.Path, line_number: int) -> Any:
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise PrepareError(f"invalid JSON in {source} line {line_number}: {exc.msg}") from exc


def load_items(path: pathlib.Path) -> list[dict[str, Any]]:
    """Load JSONL without importing any project or model runtime."""
    path = _require_absolute(path, "--items")
    if not path.is_file():
        raise PrepareError(f"items file does not exist: {path}")
    items: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = _json_loads(line, path, line_number)
                if not isinstance(value, dict):
                    raise PrepareError(f"items line {line_number} is not an object")
                items.append(value)
    except OSError as exc:
        raise PrepareError(f"cannot read items: {exc}") from exc
    return items


def _validate_question(name: str, question: Any) -> None:
    if not isinstance(question, dict):
        raise PrepareError(f"question {name!r} is not an object")
    kind = question.get("type")
    if kind not in {"choice", "score", "noul"}:
        raise PrepareError(f"question {name!r} has unsupported type {kind!r}")
    if "instructions" not in question:
        raise PrepareError(f"question {name!r} has no instructions")
    if kind == "choice":
        criteria = question.get("criteria")
        # The fixed browser wire can contain a one-element target question (for
        # example the only visible TYPE_TEXT field).  Decider's answer contract
        # still validates that single offered key, so preparation must preserve
        # rather than reject it.
        if not isinstance(criteria, dict) or len(criteria) < 1:
            raise PrepareError(f"choice question {name!r} has invalid criteria")
    elif kind == "score":
        criteria = question.get("criteria")
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise PrepareError(f"score question {name!r} has invalid criteria")


def validate_items(items: Sequence[Mapping[str, Any]], *, expected_count: int = EXPECTED_ITEMS) -> list[dict[str, Any]]:
    """Validate the fixed original wire contract and return detached item copies."""
    if len(items) != expected_count:
        raise PrepareError(f"expected exactly {expected_count} non-empty wire items, got {len(items)}")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for position, original in enumerate(items, 1):
        item = dict(original)
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise PrepareError(f"wire item {position} has no string id")
        if item_id in seen:
            raise PrepareError(f"duplicate wire item id: {item_id}")
        seen.add(item_id)
        if RICH_STATE_KEYS.intersection(item):
            raise PrepareError(f"{item_id}: rich/altered state rendering is not allowed")
        if not isinstance(item.get("state"), (dict, list, str)):
            raise PrepareError(f"{item_id}: state is missing or has an invalid type")
        questions = item.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise PrepareError(f"{item_id}: questions are missing")
        for name, question in questions.items():
            _validate_question(str(name), question)
        suite = item.get("suite")
        if suite not in {"pick", "restraint"}:
            raise PrepareError(f"{item_id}: invalid suite {suite!r}")
        if not isinstance(item.get("state_name"), str) or not isinstance(item.get("slug"), str):
            raise PrepareError(f"{item_id}: state_name/slug are required")
        if suite == "pick":
            if not isinstance(item.get("accepted"), list):
                raise PrepareError(f"{item_id}: pick item has no accepted labels")
            if not isinstance(item.get("expected_operation"), str):
                raise PrepareError(f"{item_id}: pick item has no original expected operation")
            if item.get("expected_op_semantic") not in {"CLICK", "TYPE_TEXT", "ANY"}:
                raise PrepareError(f"{item_id}: invalid expected_op_semantic")
        else:
            if not isinstance(item.get("expected_operation"), str):
                raise PrepareError(f"{item_id}: restraint item has no expected operation")
        labels = item.get("index_labels")
        if not isinstance(labels, dict):
            raise PrepareError(f"{item_id}: index_labels are required")
        # The evaluator resolves the model's sparse string choice through this
        # mapping.  It must not infer a dense position from the mapping order.
        if not all(isinstance(key, str) for key in labels):
            raise PrepareError(f"{item_id}: index_labels keys must be strings")
        validated.append(copy.deepcopy(item))
    return validated


def render_wire_item(item: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Return the original wire fields; never rebuild them from a rich state."""
    if RICH_STATE_KEYS.intersection(item):
        raise PrepareError("rich/altered rendering is not allowed")
    state = copy.deepcopy(item["state"])
    questions = copy.deepcopy(item["questions"])
    return state, questions


def _hash_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PrepareError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _file_record(path: pathlib.Path, *, relative_to: pathlib.Path | None = None) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise PrepareError(f"cannot stat {path}: {exc}") from exc
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(relative_to)) if relative_to else None,
        "size": size,
        "sha256": _hash_file(path),
    }


def hash_checkpoint(model_dir: pathlib.Path) -> list[dict[str, Any]]:
    """Hash every regular file below the checkpoint, including hidden files."""
    model_dir = _require_absolute(model_dir, "--model-dir")
    if not model_dir.is_dir():
        raise PrepareError(f"model directory does not exist: {model_dir}")
    records: list[dict[str, Any]] = []
    try:
        paths = sorted((path for path in model_dir.rglob("*") if path.is_file()), key=lambda path: path.as_posix())
    except OSError as exc:
        raise PrepareError(f"cannot enumerate checkpoint: {exc}") from exc
    if not paths:
        raise PrepareError(f"checkpoint directory is empty: {model_dir}")
    known_weight = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
    if not any(path.suffix.lower() in known_weight or path.name.startswith("model") for path in paths):
        raise PrepareError(f"checkpoint has no recognized model weight file: {model_dir}")
    for path in paths:
        records.append(_file_record(path, relative_to=model_dir))
    return records


def _validate_checkpoint_configs(model_dir: pathlib.Path) -> list[str]:
    errors: list[str] = []
    for name in ("config.json", "rl_agent_config.json"):
        path = model_dir / name
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, dict):
                errors.append(f"{name} is not an object")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{name} is not valid JSON: {exc}")
    return errors


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception as exc:  # metadata must never make prepare import a runtime
        return f"metadata-error:{type(exc).__name__}"


def environment_versions(*, torch_module: Any = None, laya_module: Any = None) -> dict[str, Any]:
    """Collect version evidence through metadata, without importing torch/Laya."""
    versions: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "laya_distribution": _package_version("laya"),
        "torch_distribution": _package_version("torch"),
    }
    if torch_module is not None:
        versions["torch_runtime"] = getattr(torch_module, "__version__", None)
    if laya_module is not None:
        versions["laya_runtime"] = getattr(laya_module, "__version__", None)
    return versions


def _hash_inputs_and_framework(items_path: pathlib.Path) -> dict[str, Any]:
    if not DEFAULT_SCORER.is_file():
        raise PrepareError(f"scorer file is missing: {DEFAULT_SCORER}")
    framework = [
        _file_record(DEFAULT_SCORER),
        *(_file_record(path) for path in DEFAULT_FRAMEWORK_FILES),
        _file_record(SCRIPT_PATH),
    ]
    return {
        "items": _file_record(items_path),
        "scorer_and_framework": framework,
    }


def build_manifest(model_dir: pathlib.Path, items_path: pathlib.Path, items: Sequence[Mapping[str, Any]],
                   output_dir: pathlib.Path, baseline_memory_mib: int) -> dict[str, Any]:
    model_dir = _require_absolute(model_dir, "--model-dir")
    items_path = _require_absolute(items_path, "--items")
    output_dir = _require_absolute(output_dir, "--output-dir")
    model_resolved = model_dir.resolve()
    output_resolved = output_dir.resolve()
    if output_resolved == model_resolved or model_resolved in output_resolved.parents:
        raise PrepareError("--output-dir must not be inside the checkpoint directory")
    if output_resolved in model_resolved.parents:
        raise PrepareError("checkpoint directory must not be inside --output-dir")
    if baseline_memory_mib <= 0:
        raise PrepareError("--baseline-memory-mib must be positive")
    config_errors = _validate_checkpoint_configs(model_dir)
    if config_errors:
        raise PrepareError("checkpoint config validation failed: " + "; ".join(config_errors))
    validated = validate_items(items)
    hashes = _hash_inputs_and_framework(items_path)
    return {
        "schema": "localdecide.release_acceptance.v1",
        "created_utc": _utc_now(),
        "model_dir": str(model_dir),
        "items": str(items_path),
        "output_dir": str(output_dir),
        "wire": {
            "source": "reports/v37/wide_items.jsonl",
            "count": len(validated),
            "unique_ids": len({item["id"] for item in validated}),
            "rendering": "original_wire_state_and_questions",
            "backend_questions": "Decider_deployed_questions_may_be_coarse_to_fine_transformed",
            "coarse_to_fine_threshold": DEFAULT_MAX_OPTIONS_PER_QUESTION,
            "rich_rendering_rejected": True,
        },
        "settings": {
            "max_len": DEFAULT_MAX_LEN,
            "head_max_len": DEFAULT_HEAD_MAX_LEN,
            "max_options_per_question": DEFAULT_MAX_OPTIONS_PER_QUESTION,
            "dtype": FIXED_DTYPE_NAME,
            "warmup": DEFAULT_WARMUP,
            "baseline_memory_mib": baseline_memory_mib,
        },
        "scoring": {
            "source": SCORING_SOURCE,
            "prediction": "last_backend_call_raw_envelope",
            "validated_answers": "successful_decision.answers.raw",
        },
        "checkpoint_files": hash_checkpoint(model_dir),
        "input_scorer_framework_hashes": hashes,
        "environment_versions": environment_versions(),
    }


def _private_dir(path: pathlib.Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path, 0o700)
    except OSError as exc:
        raise AcceptanceError(f"cannot create private output directory: {exc}") from exc


def write_private_json(path: pathlib.Path, value: Any) -> None:
    _private_dir(path.parent)
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                        prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = pathlib.Path(handle.name)
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True))
            handle.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except OSError as exc:
        if temporary:
            try:
                temporary.unlink()
            except OSError:
                pass
        raise AcceptanceError(f"cannot write private JSON {path}: {exc}") from exc


def write_private_jsonl(path: pathlib.Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _private_dir(path.parent)
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                        prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = pathlib.Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True, separators=(",", ":")))
                handle.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except OSError as exc:
        if temporary:
            try:
                temporary.unlink()
            except OSError:
                pass
        raise AcceptanceError(f"cannot write private JSONL {path}: {exc}") from exc


def _destination_type(value: Any) -> str | None:
    if isinstance(value, str):
        return value.split(":", 1)[0].lower()
    device = getattr(value, "device", None)
    if device is not None and device is not value:
        return _destination_type(device)
    value_type = getattr(value, "type", None)
    if isinstance(value_type, str):
        return value_type.lower()
    return None


def _reject_cpu_mps_destination(args: Sequence[Any], kwargs: Mapping[str, Any]) -> None:
    candidates: list[Any] = []
    if "device" in kwargs:
        candidates.append(kwargs["device"])
    # Module.to accepts a device/tensor as its first positional argument, while
    # dtype-only positional calls must remain allowed.
    if args:
        first = args[0]
        destination = _destination_type(first)
        if destination in {"cpu", "mps"}:
            candidates.append(first)
    for candidate in candidates:
        destination = _destination_type(candidate)
        if destination in {"cpu", "mps"}:
            raise SafetyError(f"refusing runtime device fallback to {destination}")


@contextmanager
def guard_torch_module_to(torch_module: Any) -> Iterator[None]:
    """Guard the class method only for load, restoring it even on failure."""
    module_class = torch_module.nn.Module
    original = module_class.to

    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        _reject_cpu_mps_destination(args, kwargs)
        return original(self, *args, **kwargs)

    module_class.to = guarded
    try:
        yield
    finally:
        module_class.to = original


def guard_model_instance_to(model: Any) -> Callable[[], None]:
    """Install a per-model guard; dtype-only ``model.to(dtype)`` remains legal."""
    original = getattr(model, "to", None)
    if not callable(original):
        return lambda: None

    def guarded(*args: Any, **kwargs: Any) -> Any:
        _reject_cpu_mps_destination(args, kwargs)
        return original(*args, **kwargs)

    setattr(model, "to", guarded)

    def restore() -> None:
        try:
            setattr(model, "to", original)
        except Exception:
            pass

    return restore


def _model_objects(agent: Any) -> list[Any]:
    result: list[Any] = []
    seen: set[int] = set()
    if callable(getattr(agent, "to", None)):
        result.append(agent)
        seen.add(id(agent))
    for name in ("model", "_model", "network", "_network", "encoder", "_encoder"):
        value = getattr(agent, name, None)
        if value is not None and id(value) not in seen:
            seen.add(id(value))
            result.append(value)
    return result


def install_agent_device_guards(agent: Any) -> list[Callable[[], None]]:
    return [guard_model_instance_to(model) for model in _model_objects(agent)]


def _ensure_repo_on_path() -> None:
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def configure_agent(agent: Any, torch_module: Any, *, max_len: int = DEFAULT_MAX_LEN,
                    head_max_len: int = DEFAULT_HEAD_MAX_LEN) -> None:
    cfg = getattr(agent, "cfg", None)
    if not isinstance(cfg, dict):
        raise AcceptanceError("loaded Laya agent has no dict cfg")
    cfg["max_len"] = max_len
    cfg["head_max_len"] = head_max_len
    agent.dtype = torch_module.bfloat16


def validate_cuda_device_dtype(agent: Any, torch_module: Any) -> None:
    """Fail closed if a model moved off CUDA or lost the fixed bf16 setting."""
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        raise SafetyError("CUDA is unavailable during prediction")
    expected_dtype = torch_module.bfloat16
    if getattr(agent, "dtype", None) != expected_dtype:
        raise SafetyError("agent dtype is not torch.bfloat16 during prediction")
    # current_device is also a useful device-context sanity check, but do not
    # assume that the runtime exposes a particular GPU index in the agent.
    current_device = cuda.current_device()
    if not isinstance(current_device, int) or current_device < 0:
        raise SafetyError("CUDA current device is invalid")
    checked_any = False
    for model in _model_objects(agent):
        parameters = getattr(model, "parameters", None)
        if callable(parameters):
            try:
                values = list(parameters())
            except Exception as exc:
                raise SafetyError(f"cannot inspect model parameters: {exc}") from exc
            for parameter in values:
                checked_any = True
                device_type = _destination_type(getattr(parameter, "device", None))
                if device_type != "cuda":
                    raise SafetyError(f"model parameter is on {device_type or 'unknown'}, not CUDA")
                # Laya uses bf16 autocast with fp32 master parameters. Checking
                # parameter dtype against bf16 would reject its actual runtime;
                # agent.dtype above is the effective forward/autocast contract.
        device = getattr(model, "device", None)
        if device is not None and not checked_any and _destination_type(device) != "cuda":
            raise SafetyError("model device is not CUDA")


def _exception_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _reset_backend_call_records(backend: Any) -> None:
    reset = getattr(backend, "reset_call_records", None)
    if callable(reset):
        reset()
        return
    records = getattr(backend, "call_records", None)
    if isinstance(records, list):
        records.clear()
    if hasattr(backend, "last_payload"):
        backend.last_payload = None


def _backend_call_records(backend: Any) -> list[dict[str, Any]]:
    records = getattr(backend, "call_records", None)
    if not isinstance(records, list):
        return []
    return copy.deepcopy(records)


class _RecordingBackend:
    """Model-free call recorder for a caller-supplied duck-typed backend."""

    def __init__(self, backend: Any):
        self._backend = backend
        self.name = getattr(backend, "name", type(backend).__name__)
        self.last_payload: Any = None
        self.call_records: list[dict[str, Any]] = []

    def reset_call_records(self) -> None:
        self.call_records.clear()
        self.last_payload = None

    def answer(self, state: Any, questions: Mapping[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "questions": copy.deepcopy(dict(questions)),
            "raw_envelope": None,
            "error": None,
        }
        try:
            payload = self._backend.answer(state, questions)
            record["raw_envelope"] = copy.deepcopy(payload)
            self.last_payload = copy.deepcopy(payload)
        except Exception as exc:
            record["error"] = _exception_text(exc)
            self.last_payload = None
            self.call_records.append(record)
            raise
        self.call_records.append(record)
        return payload


def make_preloaded_backend(agent: Any, *, laya_module: Any = None,
                           before_predict: Callable[[], None] | None = None) -> Any:
    """Make an actual LayaTorchBackend instance around an already-loaded agent.

    The instance is allocated without calling its importing constructor.  Its
    inherited ``answer`` implementation is still used; a bound instance wrapper
    records the complete raw backend envelope and the exact questions sent for
    every call, including failed calls.  It optionally checks CUDA immediately
    before every call.
    """
    _ensure_repo_on_path()
    from localdecide.backends.base import LayaTorchBackend  # lazy, model-free

    backend = object.__new__(LayaTorchBackend)
    backend._laya = laya_module
    backend._kwargs = {}
    backend._model_arg = None
    backend._subfolder = None
    backend._agent = agent
    backend.last_payload = None
    backend.call_records = []

    def reset_call_records(self: Any) -> None:
        self.call_records.clear()
        self.last_payload = None

    backend.reset_call_records = types.MethodType(reset_call_records, backend)
    inherited_answer = LayaTorchBackend.answer

    def answer(self: Any, state: Any, questions: Mapping[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "questions": copy.deepcopy(dict(questions)),
            "raw_envelope": None,
            "error": None,
        }
        try:
            if before_predict is not None:
                before_predict()
            payload = inherited_answer(self, state, questions)
            record["raw_envelope"] = copy.deepcopy(payload)
            self.last_payload = copy.deepcopy(payload)
            if before_predict is not None:
                before_predict()
        except Exception as exc:
            record["error"] = _exception_text(exc)
            if record["raw_envelope"] is None:
                self.last_payload = None
            self.call_records.append(record)
            raise
        self.call_records.append(record)
        return payload

    backend.answer = types.MethodType(answer, backend)
    return backend


def smoke_typed_contract(backend: Any) -> dict[str, Any]:
    """Exercise Decider's real choice/score/noul validation through the backend."""
    _ensure_repo_on_path()
    _reset_backend_call_records(backend)
    from localdecide import choice, noul, score
    from localdecide.decider import Decider

    questions = {
        "choice_smoke": choice("Choose one smoke option", {"alpha": "A", "beta": "B"}),
        "score_smoke": score("Place the smoke state", ["low", "mid", "high"]),
        "noul_smoke": noul("Is the smoke contract usable?"),
    }
    result = Decider(backend=backend, retries=0, fail_open=False).decide(
        {"release_acceptance": "typed-contract-smoke"}, questions)
    if not result.ok or result.answers is None:
        raise AcceptanceError(result.error or "typed contract smoke failed")
    raw = result.answers.raw
    return {
        "ok": True,
        "types": {name: value.get("type") for name, value in raw.items()},
        "validated": True,
        "backend_calls": _backend_call_records(backend),
    }


def _finite_count(value: Any) -> int:
    if isinstance(value, float):
        return int(not math.isfinite(value))
    if isinstance(value, dict):
        return sum(_finite_count(key) + _finite_count(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return sum(_finite_count(item) for item in value)
    return 0


def _raw_answer(payload: Any, name: str) -> Mapping[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    answers = payload.get("answers")
    if isinstance(answers, dict):
        value = answers.get(name)
    else:
        value = payload.get(name)
    return value if isinstance(value, dict) else None


def _probability(answer: Mapping[str, Any] | None, selected: Any) -> float | None:
    if not answer or not isinstance(answer.get("probabilities"), dict):
        return None
    probabilities = answer["probabilities"]
    value = probabilities.get(str(selected))
    if value is None and selected in probabilities:
        value = probabilities[selected]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    value = float(value)
    return value if 0.0 <= value <= 1.0 else None


def _normalise_operation_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalised = value.strip().upper().replace("-", "_").replace(" ", "_")
    return normalised if normalised in SUPPORTED_OPERATION_SEMANTICS else None


def _operation_semantic(item: Mapping[str, Any], operation_key: Any) -> str:
    """Decode an operation only from an explicitly offered original-wire key.

    Numeric operation keys are not positions.  They are accepted only when the
    original criteria value itself (or a single explicit semantic field in a
    structured value) names exactly one supported operation.  Descriptions,
    substring matches, ordering, and ``index_ops`` are deliberately excluded.
    """
    questions = item.get("questions")
    operation_question = questions.get("operation") if isinstance(questions, Mapping) else None
    criteria = operation_question.get("criteria") if isinstance(operation_question, Mapping) else None
    if not isinstance(criteria, Mapping):
        raise ValueError("operation criteria are missing from the original wire question")
    if operation_key not in criteria:
        raise ValueError(f"operation key {operation_key!r} was not offered by the original wire question")

    direct = _normalise_operation_value(operation_key)
    if direct is not None:
        return direct

    value = criteria[operation_key]
    candidates: set[str] = set()
    if isinstance(value, str):
        semantic = _normalise_operation_value(value)
        if semantic is not None:
            candidates.add(semantic)
    elif isinstance(value, Mapping):
        for field in ("operation", "operation_semantic", "semantic_operation", "semantic", "op"):
            if field in value:
                semantic = _normalise_operation_value(value[field])
                if semantic is not None:
                    candidates.add(semantic)
                else:
                    raise ValueError(f"operation criteria key {operation_key!r} has an unsupported semantic value")
    if len(candidates) != 1:
        raise ValueError(f"operation criteria key {operation_key!r} does not map uniquely to a supported operation")
    return next(iter(candidates))


def _validated_answers(decision: Any) -> Mapping[str, Any] | None:
    if not getattr(decision, "ok", False):
        return None
    answers = getattr(decision, "answers", None)
    raw = getattr(answers, "raw", None)
    return raw if isinstance(raw, Mapping) else None


def _row_result(item: Mapping[str, Any], payload: Any, decision: Any, latency_ms: float,
                error: str | None, warmup: bool, *, backend_calls: Sequence[Mapping[str, Any]] = (),
                wire_state: Any = None, wire_questions: Mapping[str, Any] | None = None,
                max_options_per_question: int | None = None) -> dict[str, Any]:
    validated_answers = _validated_answers(decision)
    row_error = error
    if row_error is None and validated_answers is None:
        row_error = "decision did not provide successful validated answers"

    raw_operation = _raw_answer(validated_answers, "operation")
    operation_key = raw_operation.get("choice") if raw_operation else None
    operation: str | None = None
    if row_error is None:
        if operation_key is None:
            row_error = "validated operation answer has no selected key"
        else:
            try:
                operation = _operation_semantic(item, operation_key)
            except ValueError as exc:
                row_error = str(exc)

    target_name = f"{operation.lower()}_target" if operation in {"CLICK", "TYPE_TEXT", "SELECT"} else ""
    raw_target = _raw_answer(validated_answers, target_name) if target_name else None
    target = raw_target.get("choice") if raw_target else None
    if row_error is None and item.get("suite") == "pick" and target_name and raw_target is None:
        row_error = f"validated answer has no {target_name} key"
    labels = item.get("index_labels") if isinstance(item.get("index_labels"), dict) else {}
    label = labels.get(str(target)) if target is not None else None
    accepted = set(item.get("accepted", [])) if isinstance(item.get("accepted"), list) else set()
    suite = item.get("suite")
    if suite == "restraint":
        semantic_operation_ok = operation == item.get("expected_operation")
        handoff_operation_ok = semantic_operation_ok
        target_ok = True
    else:
        expected_semantic = item.get("expected_op_semantic")
        semantic_operation_ok = expected_semantic == "ANY" or operation == expected_semantic
        handoff_operation_ok = operation == item.get("expected_operation")
        target_ok = label in accepted
    scoreable = row_error is None and validated_answers is not None
    semantic_ok = bool(scoreable and semantic_operation_ok and target_ok)
    handoff_ok = bool(scoreable and handoff_operation_ok and target_ok)
    validated = bool(getattr(decision, "ok", False)) and validated_answers is not None
    raw_gold = {key: copy.deepcopy(value) for key, value in item.items() if key not in {"state", "questions"}}
    calls = copy.deepcopy(list(backend_calls))
    raw_nonfinite = sum(_finite_count(call.get("raw_envelope")) for call in calls
                        if isinstance(call, Mapping) and call.get("raw_envelope") is not None)
    if not calls:
        raw_nonfinite = _finite_count(payload)
    transformed = bool(wire_questions is not None and any(
        call.get("questions") != wire_questions for call in calls if isinstance(call, Mapping)
    ))
    return {
        "id": item.get("id"),
        "suite": suite,
        "state": item.get("state_name"),
        "scenario": item.get("scenario") or item.get("slug"),
        "slug": item.get("slug"),
        "gold": raw_gold,
        "prediction": copy.deepcopy(payload),
        "prediction_source": "last_backend_call_raw_envelope",
        "validated_answers": copy.deepcopy(validated_answers),
        "scoring_source": SCORING_SOURCE,
        "backend_calls": calls,
        "original_wire_state": copy.deepcopy(wire_state),
        "original_wire_questions": copy.deepcopy(wire_questions),
        "question_transformed": transformed,
        "max_options_per_question": max_options_per_question,
        "validated": validated,
        "error": row_error,
        "error_count": int(row_error is not None),
        "nonfinite_count": raw_nonfinite,
        "warmup": warmup,
        "latency_ms": round(float(latency_ms), 3),
        "operation_key": operation_key,
        "operation": operation,
        "op": operation,
        "target": target,
        "label": label,
        "operation_selected_probability": _probability(raw_operation, operation_key),
        "target_selected_probability": _probability(raw_target, target),
        "operation_semantic_ok": bool(semantic_operation_ok and scoreable),
        "operation_handoff_ok": bool(handoff_operation_ok and scoreable),
        "target_ok": bool(target_ok and scoreable),
        "semantic_ok": semantic_ok,
        "handoff_ok": handoff_ok,
        "joint": {
            "semantic_ok": semantic_ok,
            "handoff_ok": handoff_ok,
        },
    }


def _synchronize(torch_module: Any) -> None:
    if torch_module is None:
        return
    cuda = getattr(torch_module, "cuda", None)
    synchronize = getattr(cuda, "synchronize", None)
    if callable(synchronize):
        synchronize()


def evaluate_items(items: Sequence[Mapping[str, Any]], backend: Any, *, torch_module: Any = None,
                   decider: Any = None, warmup: int = DEFAULT_WARMUP,
                   before_predict: Callable[[], None] | None = None) -> list[dict[str, Any]]:
    """Evaluate wire items and retain raw calls plus merged validated answers.

    A normal item has one backend call.  A wide choice has the Decider's
    coarse-to-fine first pass and final pass; both are retained, while scoring
    uses only the successful merged ``decision.answers.raw``.
    """
    recording_backend = backend
    if decider is not None:
        decider_backend = getattr(decider, "backend", backend)
        if not hasattr(decider_backend, "call_records"):
            decider_backend = _RecordingBackend(decider_backend)
            decider.backend = decider_backend
        recording_backend = decider_backend
    elif not hasattr(recording_backend, "call_records"):
        recording_backend = _RecordingBackend(recording_backend)
    if decider is None:
        _ensure_repo_on_path()
        from localdecide.decider import Decider
        decider = Decider(backend=recording_backend, retries=0, fail_open=False,
                          max_options_per_question=DEFAULT_MAX_OPTIONS_PER_QUESTION)
    elif getattr(decider, "backend", None) is not recording_backend:
        decider.backend = recording_backend
    max_options_per_question = getattr(decider, "max_options_per_question", None)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        state, questions = render_wire_item(item)
        wire_state = copy.deepcopy(state)
        wire_questions = copy.deepcopy(questions)
        if before_predict is not None:
            # This hook is useful for a backend supplied by a test; the real
            # backend normally carries it in make_preloaded_backend so Decider's
            # every-call path is guarded too.
            before_predict()
        _reset_backend_call_records(recording_backend)
        _synchronize(torch_module)
        started = time.perf_counter()
        error: str | None = None
        decision: Any = None
        try:
            decision = decider.decide(state, questions)
            if not getattr(decision, "ok", False):
                error = str(getattr(decision, "error", None) or "decision failed")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _synchronize(torch_module)
        elapsed = (time.perf_counter() - started) * 1000.0
        payload = copy.deepcopy(getattr(recording_backend, "last_payload", None))
        calls = _backend_call_records(recording_backend)
        rows.append(_row_result(
            item, payload, decision, elapsed, error, index < warmup,
            backend_calls=calls, wire_state=wire_state, wire_questions=wire_questions,
            max_options_per_question=max_options_per_question,
        ))
    return rows


def percentile(values: Sequence[float], quantile: float) -> float | None:
    """Linear-interpolated percentile, with the conventional [0, 1] quantile."""
    if not values:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _accuracy(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return round(sum(bool(row.get(key)) for row in rows) / len(rows), 6)


def slice_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "errors": sum(int(row.get("error_count", 0)) for row in rows),
        "nonfinite_values": sum(int(row.get("nonfinite_count", 0)) for row in rows),
        "operation_semantic": _accuracy(rows, "operation_semantic_ok"),
        "operation_handoff": _accuracy(rows, "operation_handoff_ok"),
        "target": _accuracy(rows, "target_ok"),
        "joint_semantic": _accuracy(rows, "semantic_ok"),
        "joint_handoff": _accuracy(rows, "handoff_ok"),
    }


def _slices(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = sorted({str(row.get(field)) for row in rows})
    return {value: slice_metrics([row for row in rows if str(row.get(field)) == value]) for value in values}


def operation_ece(rows: Sequence[Mapping[str, Any]], *, correct_key: str = "operation_semantic_ok",
                 bins: int = 10) -> dict[str, Any]:
    """ECE of operation accuracy versus the model's selected operation probability."""
    samples: list[tuple[float, bool]] = []
    for row in rows:
        value = row.get("operation_selected_probability")
        if row.get("error") is not None or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        value = float(value)
        if math.isfinite(value) and 0.0 <= value <= 1.0:
            samples.append((value, bool(row.get(correct_key))))
    if not samples:
        return {"ece": None, "n": 0, "bins": [], "definition": "accuracy versus selected operation probability"}
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for confidence, correct in samples:
        index = min(bins - 1, int(confidence * bins))
        buckets[index].append((confidence, correct))
    details: list[dict[str, Any]] = []
    total = len(samples)
    ece = 0.0
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        mean_confidence = sum(value for value, _ in bucket) / len(bucket)
        accuracy = sum(correct for _, correct in bucket) / len(bucket)
        gap = abs(accuracy - mean_confidence)
        ece += len(bucket) / total * gap
        details.append({
            "lower": round(index / bins, 6),
            "upper": round((index + 1) / bins, 6),
            "n": len(bucket),
            "accuracy": round(accuracy, 6),
            "mean_selected_probability": round(mean_confidence, 6),
            "gap": round(gap, 6),
        })
    return {
        "ece": round(ece, 6),
        "n": total,
        "bins": details,
        "definition": "accuracy versus selected operation probability",
        "accuracy_key": correct_key,
    }


def latency_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    measured = [float(row["latency_ms"]) for row in rows if not row.get("warmup")]
    return {
        "warmup_excluded": sum(int(bool(row.get("warmup"))) for row in rows),
        "n": len(measured),
        "p50_ms": None if not measured else round(percentile(measured, 0.50) or 0.0, 3),
        "p95_ms": None if not measured else round(percentile(measured, 0.95) or 0.0, 3),
        "synchronized": True,
    }


def build_report(rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], *, smoke: Mapping[str, Any]) -> dict[str, Any]:
    restraint = [row for row in rows if row.get("suite") == "restraint"]
    picks = [row for row in rows if row.get("suite") == "pick"]
    joint_explanation = (
        "null: agent.predict exposes separate marginal probabilities for operation and target; "
        "no real joint probability was returned, so multiplying marginals would fabricate an independence assumption"
    )
    configured_options = sorted({row.get("max_options_per_question") for row in rows
                                 if row.get("max_options_per_question") is not None})
    manifest_options = manifest.get("settings", {}).get("max_options_per_question") \
        if isinstance(manifest.get("settings"), Mapping) else None
    max_options_per_question = configured_options[0] if len(configured_options) == 1 else manifest_options
    return {
        "schema": "localdecide.release_acceptance.report.v1",
        "created_utc": _utc_now(),
        "manifest": dict(manifest),
        "smoke": dict(smoke),
        "counts": {
            "rows": len(rows),
            "unique_ids": len({row.get("id") for row in rows}),
            "errors": sum(int(row.get("error_count", 0)) for row in rows),
            "nonfinite_values": sum(int(row.get("nonfinite_count", 0)) for row in rows),
        },
        "overall": slice_metrics(rows),
        "pick": slice_metrics(picks),
        "restraint": slice_metrics(restraint),
        "slices": {
            "suite": {value: slice_metrics([row for row in rows if row.get("suite") == value])
                      for value in sorted({str(row.get("suite")) for row in rows})},
            "state": _slices(rows, "state"),
            "scenario": _slices(rows, "scenario"),
        },
        "latency": latency_summary(rows),
        "decider": {
            "max_options_per_question": max_options_per_question,
            "coarse_to_fine_cases": sum(bool(row.get("question_transformed")) for row in rows),
            "wire_input_cases": len(rows),
            "backend_call_count": sum(len(row.get("backend_calls", [])) for row in rows),
        },
        "scoring_source": SCORING_SOURCE,
        "prediction_source": "last_backend_call_raw_envelope",
        "calibration": {
            "operation_ece": operation_ece(rows, correct_key="operation_semantic_ok"),
            "operation_ece_handoff": operation_ece(rows, correct_key="operation_handoff_ok"),
            "joint_ece": None,
            "joint_ece_explanation": joint_explanation,
        },
        "rendering": {
            "wire_input": "original_wire_state_and_questions",
            "backend_questions": "Decider_deployed_questions_may_be_coarse_to_fine_transformed",
        },
        "raw_predictions_file": "raw_predictions.jsonl",
    }


def _command_output(runner: Callable[..., Any], argv: Sequence[str]) -> str:
    try:
        result = runner(list(argv), capture_output=True, text=True, check=False)
    except OSError as exc:
        raise SafetyError(f"safety command failed: {argv[0]}: {exc}") from exc
    if isinstance(result, str):
        return result
    return_code = getattr(result, "returncode", 0)
    if return_code not in (None, 0):
        detail = str(getattr(result, "stderr", "")).strip()[:200]
        raise SafetyError(f"safety command failed ({return_code}): {argv[0]} {detail}")
    return str(getattr(result, "stdout", ""))


def _parse_csv_numbers(text: str) -> tuple[float, float]:
    line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 2:
        raise SafetyError(f"could not parse nvidia-smi GPU stats: {line!r}")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise SafetyError(f"could not parse nvidia-smi GPU stats: {line!r}") from exc


def _compute_pids(text: str) -> set[int]:
    result: set[int] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("no running"):
            continue
        first = line.split(",", 1)[0].strip()
        try:
            result.add(int(first))
        except ValueError:
            continue
    return result


def _is_python_argv(argv: list[str]) -> bool:
    if not argv:
        return False
    first = pathlib.Path(argv[0]).name.lower()
    if first == "python" or first.startswith("python3"):
        return True
    if first == "env":
        return any(pathlib.Path(token).name.lower().startswith("python") for token in argv[1:3])
    return False


def _other_train_eval_python(pid: int, command: str, own_pid: int) -> bool:
    if pid == own_pid:
        return False
    # Only inspect a process whose executable is Python.  A shell command that
    # happens to contain "python evaluate..." is not a competing process.
    try:
        argv = command.strip().split()
    except Exception:
        return False
    if not _is_python_argv(argv):
        return False
    lowered = command.lower()
    return bool(re.search(r"(?:^|[\s_./-])(train|training|eval|evaluate|evaluation)(?:[\s_.:/-]|$)", lowered))


def evaluate_gpu_safety(memory_used_mib: float, gpu_util_percent: float, baseline_memory_mib: float,
                        compute_pids: Iterable[int], ps_lines: Iterable[tuple[int, str]],
                        *, own_pid: int | None = None) -> dict[str, Any]:
    own_pid = os.getpid() if own_pid is None else own_pid
    if not all(math.isfinite(float(value)) for value in (memory_used_mib, gpu_util_percent, baseline_memory_mib)):
        raise SafetyError("GPU safety telemetry is non-finite")
    other_compute = sorted(int(pid) for pid in compute_pids if int(pid) != own_pid)
    other_python = sorted(pid for pid, command in ps_lines if _other_train_eval_python(int(pid), command, own_pid))
    reasons: list[str] = []
    delta = abs(float(memory_used_mib) - float(baseline_memory_mib))
    if delta > 256.0:
        reasons.append(f"GPU memory delta {delta:.1f} MiB exceeds 256 MiB")
    if float(gpu_util_percent) > 10.0:
        reasons.append(f"GPU utilization {float(gpu_util_percent):.1f}% exceeds 10%")
    if other_compute:
        reasons.append(f"other active CUDA compute processes: {other_compute}")
    if other_python:
        reasons.append(f"other train/eval Python processes: {other_python}")
    result = {
        "memory_used_mib": float(memory_used_mib),
        "baseline_memory_mib": float(baseline_memory_mib),
        "memory_delta_mib": delta,
        "gpu_util_percent": float(gpu_util_percent),
        "other_compute_pids": other_compute,
        "other_train_eval_python_pids": other_python,
        "own_pid": own_pid,
        "ok": not reasons,
        "reasons": reasons,
    }
    if reasons:
        raise SafetyError("; ".join(reasons))
    return result


def check_gpu_safety(baseline_memory_mib: int, *, runner: Callable[..., Any] | None = None,
                     own_pid: int | None = None) -> dict[str, Any]:
    runner = subprocess.run if runner is None else runner
    stats = _command_output(runner, ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                     "--format=csv,noheader,nounits"])
    memory_used, utilization = _parse_csv_numbers(stats)
    compute = _command_output(runner, ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                       "--format=csv,noheader,nounits"])
    ps = _command_output(runner, ["ps", "-axo", "pid=,command="])
    ps_lines: list[tuple[int, str]] = []
    for line in ps.splitlines():
        match = re.match(r"\s*(\d+)\s+(.*)$", line)
        if match:
            ps_lines.append((int(match.group(1)), match.group(2)))
    return evaluate_gpu_safety(memory_used, utilization, baseline_memory_mib,
                               _compute_pids(compute), ps_lines, own_pid=own_pid)


@contextmanager
def gpu_lock(path: pathlib.Path = GPU_LOCK_PATH) -> Iterator[Any]:
    """Hold the shared non-blocking CUDA release lock for the whole execution."""
    try:
        import fcntl
    except ImportError as exc:
        raise SafetyError("fcntl is required for the Linux execute lock") from exc
    path = pathlib.Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        handle = path.open("a+", encoding="utf-8")
        os.chmod(path, 0o600)
    except OSError as exc:
        raise SafetyError(f"cannot create GPU lock {path}: {exc}") from exc
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise SafetyError(f"GPU release lock is already held: {path}") from exc
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
    except BaseException:
        raise


def load_laya_agent(model_dir: pathlib.Path) -> tuple[Any, Any, Any]:
    """Import and load the runtime only from explicit Linux/CUDA execution."""
    if platform.system() != "Linux":
        raise SafetyError("--execute is Linux-only")
    os.environ.setdefault("DISABLE_TORCH_NATIVE_BMM", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if _package_version("laya") != RUNTIME_LAYA_VERSION:
        raise SafetyError("expected existing laya 0.3.4 runtime; refusing environment drift")
    import torch
    try:
        torch.backends.python_native.disable_operations("bmm")
    except (AttributeError, ImportError):
        pass
    if not torch.cuda.is_available():
        raise SafetyError("CUDA unavailable; refusing to load on CPU")
    import laya

    with guard_torch_module_to(torch):
        # Laya 0.3.4's public API has no revision argument.  Do not add one.
        agent = laya.load(str(model_dir), device="cuda")
    configure_agent(agent, torch)
    install_agent_device_guards(agent)
    return torch, laya, agent


def execute(model_dir: pathlib.Path, items_path: pathlib.Path, output_dir: pathlib.Path,
            baseline_memory_mib: int) -> dict[str, Any]:
    if platform.system() != "Linux":
        raise SafetyError("--execute is Linux-only")
    items = validate_items(load_items(items_path))
    manifest = build_manifest(model_dir, items_path, items, output_dir, baseline_memory_mib)
    _private_dir(output_dir)
    with gpu_lock():
        gpu = check_gpu_safety(baseline_memory_mib)
        torch_module, laya_module, agent = load_laya_agent(model_dir)
        validator = lambda: validate_cuda_device_dtype(agent, torch_module)
        backend = make_preloaded_backend(agent, laya_module=laya_module, before_predict=validator)
        smoke = smoke_typed_contract(backend)
        rows = evaluate_items(items, backend, torch_module=torch_module)
        if len(rows) != EXPECTED_ITEMS or len({row["id"] for row in rows}) != EXPECTED_ITEMS:
            raise AcceptanceError("runtime did not produce exactly one complete row per wire id")
        report = build_report(rows, manifest, smoke=smoke)
        report["gpu_safety"] = gpu
        report["environment_versions"] = environment_versions(torch_module=torch_module, laya_module=laya_module)
        write_private_jsonl(output_dir / "raw_predictions.jsonl", rows)
        write_private_json(output_dir / "report.json", report)
        manifest = dict(manifest)
        manifest["executed_utc"] = _utc_now()
        manifest["runtime"] = {"laya_api": "load(path, device='cuda')", "version_target": RUNTIME_LAYA_VERSION}
        write_private_json(output_dir / "manifest.json", manifest)
        return {
            "mode": "execute",
            "rows": len(rows),
            "errors": report["counts"]["errors"],
            "output_dir": str(output_dir),
        }


def prepare(model_dir: pathlib.Path, items_path: pathlib.Path, output_dir: pathlib.Path,
            baseline_memory_mib: int) -> dict[str, Any]:
    items = validate_items(load_items(items_path))
    manifest = build_manifest(model_dir, items_path, items, output_dir, baseline_memory_mib)
    _private_dir(output_dir)
    write_private_json(output_dir / "manifest.json", manifest)
    return {"mode": "prepare", "ok": True, "rows": len(items), "output_dir": str(output_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true", help="validate and hash inputs; never import model runtimes")
    modes.add_argument("--execute", action="store_true", help="run on the parent Linux/CUDA host")
    parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--items", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-memory-mib", type=int, default=DEFAULT_BASELINE_MEMORY_MIB)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.prepare:
            result = prepare(args.model_dir, args.items, args.output_dir, args.baseline_memory_mib)
        else:
            result = execute(args.model_dir, args.items, args.output_dir, args.baseline_memory_mib)
    except AcceptanceError as exc:
        print(f"release acceptance refused: {exc}", file=sys.stderr)
        return 2
    # Deliberately print only aggregate status/path metadata, never state, goal,
    # questions, or raw model text.
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if result.get("mode") == "execute" and result.get("errors", 0):
        return 1  # artifacts remain available, but do not mark a failed gate complete
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
