"""Model-free tests for the release acceptance harness.

These tests intentionally never import or execute PyTorch/Laya.  The runtime
contract is exercised through a tiny fake agent, while the safety, hashing,
rendering, and Decider-validation paths are tested directly.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import pathlib
import sys
import types

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("eval_release_acceptance_test_module",
                                              ROOT / "scripts" / "eval_release_acceptance.py")
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _synthetic_item(*, item_id: str, suite: str, state_name: str, slug: str,
                    target_count: int) -> dict:
    target_keys = [f"target-{index:02d}" for index in range(1, target_count + 1)]
    index_labels = {
        key: f"synthetic-label-{slug}-{index:02d}"
        for index, key in enumerate(target_keys, 1)
    }
    if suite == "pick":
        operation_keys = ("CLICK", "TYPE_TEXT")
        expected_operation = "CLICK"
        expected_op_semantic = "CLICK"
        accepted = [index_labels[target_keys[0]]]
    else:
        operation_keys = ("DONE", "BLOCKED")
        expected_operation = "DONE"
        expected_op_semantic = None
        accepted = []
    return {
        "id": f"synthetic/{item_id}",
        "suite": suite,
        "state_name": state_name,
        "slug": slug,
        "goal": "select the synthetic target",
        "state": {"page": {"text": f"{state_name} state {slug}"}},
        "questions": {
            "operation": {
                "type": "choice",
                "criteria": {key: key for key in operation_keys},
                "instructions": {},
            },
            "click_target": {
                "type": "choice",
                "criteria": {key: f"synthetic target {key}" for key in target_keys},
                "instructions": {},
            },
        },
        "accepted": accepted,
        "expected_operation": expected_operation,
        "expected_op_semantic": expected_op_semantic,
        "index_labels": index_labels,
        "index_ops": {},
        "synthetic": True,
    }


def _synthetic_items() -> list[dict]:
    items = [
        _synthetic_item(
            item_id=f"inbox-{index:02d}", suite="pick", state_name="synthetic-inbox",
            slug=f"inbox-{index:02d}", target_count=39,
        )
        for index in range(1, 37)
    ]
    items.extend(
        _synthetic_item(
            item_id=f"calendar-{index:02d}", suite="pick", state_name="synthetic-calendar",
            slug=f"calendar-{index:02d}", target_count=21,
        )
        for index in range(1, 12)
    )
    items.extend(
        _synthetic_item(
            item_id=f"pick-{index:02d}", suite="pick", state_name="synthetic-small",
            slug=f"pick-{index:02d}", target_count=3,
        )
        for index in range(1, 33)
    )
    items.extend(
        _synthetic_item(
            item_id=f"restraint-{index:02d}", suite="restraint", state_name="synthetic-small",
            slug=f"restraint-{index:02d}", target_count=3,
        )
        for index in range(1, 17)
    )
    return items


def _write_synthetic_items(path: pathlib.Path) -> pathlib.Path:
    path.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in _synthetic_items()) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def items_path(tmp_path: pathlib.Path) -> pathlib.Path:
    return _write_synthetic_items(tmp_path / "synthetic_release_acceptance.jsonl")


def _items() -> list[dict]:
    return _synthetic_items()


def test_synthetic_fixture_has_expected_invariants_and_width_distribution(items_path):
    items = runner.load_items(items_path)
    assert len(items) == 95
    assert len({item["id"] for item in items}) == 95
    assert all(item["synthetic"] is True for item in items)
    assert {item["suite"] for item in items} == {"pick", "restraint"}
    assert sum(item["suite"] == "pick" for item in items) == 79
    assert sum(item["suite"] == "restraint" for item in items) == 16

    widths = [len(item["questions"]["click_target"]["criteria"]) for item in items]
    wide = [item for item, width in zip(items, widths) if width > runner.DEFAULT_MAX_OPTIONS_PER_QUESTION]
    small = [item for item, width in zip(items, widths) if width <= runner.DEFAULT_MAX_OPTIONS_PER_QUESTION]
    assert len(wide) == 47
    assert len(small) == 48
    assert {item["state_name"] for item in wide} == {"synthetic-inbox", "synthetic-calendar"}
    assert sum(item["state_name"] == "synthetic-inbox" for item in wide) == 36
    assert sum(item["state_name"] == "synthetic-calendar" for item in wide) == 11
    assert sorted(widths[:47]) == [21] * 11 + [39] * 36
    assert all(1 <= width <= runner.DEFAULT_MAX_OPTIONS_PER_QUESTION for width in widths[47:])
    assert "reports" not in items_path.parts
    assert "v37" not in items_path.parts


def test_synthetic_fixture_gold_is_consistent_and_runner_legal(items_path):
    items = runner.validate_items(runner.load_items(items_path))
    for item in items:
        operation = item["questions"]["operation"]
        targets = item["questions"]["click_target"]["criteria"]
        assert item["synthetic"] is True
        assert item["expected_operation"] in operation["criteria"]
        assert set(item["index_labels"]) == set(targets)
        assert all(label.startswith("synthetic-label-") for label in item["index_labels"].values())
        if item["suite"] == "pick":
            assert item["expected_op_semantic"] == item["expected_operation"] == "CLICK"
            first_target = next(iter(targets))
            assert item["accepted"] == [item["index_labels"][first_target]]
        else:
            assert item["expected_operation"] == "DONE"
            assert item["expected_op_semantic"] is None
            assert item["accepted"] == []


def test_synthetic_fixture_does_not_need_reports_assets(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "DEFAULT_SCORER", tmp_path / "missing-scorer.py")
    items = runner.validate_items(_items())
    assert len(items) == 95
    assert all(item["synthetic"] for item in items)


class FakeAgent:
    def __init__(self, answers=None):
        self.answers = answers
        self.cfg = {}
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        if self.answers is not None:
            return {"answers": self.answers(questions), "usage": {"fake": True}}
        answers = {}
        for name, question in questions.items():
            kind = question["type"]
            if kind == "choice":
                keys = list(question["criteria"])
                answers[name] = {
                    "type": "choice", "choice": keys[0],
                    "probabilities": {str(key): (1.0 if key == keys[0] else 0.0) for key in keys},
                    "confidence": 1.0, "action": {"act_probability": 1.0},
                }
            elif kind == "score":
                levels = len(question["criteria"])
                answers[name] = {
                    "type": "score", "score": 1.0,
                    "legend": {str(i): value for i, value in enumerate(question["criteria"])},
                    "probabilities": {str(i): (0.8 if i == 1 else 0.1) for i in range(levels)},
                    "confidence": 0.8, "action": {"act_probability": 1.0},
                }
            elif kind == "noul":
                answers[name] = {
                    "type": "noul", "noul": 0.75, "confidence": 0.75,
                    "action": {"act_probability": 1.0},
                }
        return {"answers": answers, "usage": {"fake": True}}


def _backend(agent):
    return runner.make_preloaded_backend(agent)


def _item(*, suite="pick", expected="CLICK", semantic="CLICK", op_keys=("CLICK", "TYPE_TEXT"),
          target_keys=("2", "7")):
    questions = {
        "operation": {"type": "choice", "criteria": {key: key for key in op_keys}, "instructions": {}},
        "click_target": {"type": "choice", "criteria": {key: key for key in target_keys}, "instructions": {}},
    }
    return {
        "id": "synthetic/one", "suite": suite, "state_name": "synthetic", "slug": "one",
        "goal": "choose the useful target", "state": {"page": {"text": "synthetic"}},
        "questions": questions, "accepted": ["synthetic-label-good"],
        "expected_operation": expected, "expected_op_semantic": semantic,
        "index_labels": {"2": "synthetic-label-wrong", "7": "synthetic-label-good"}, "index_ops": {},
        "synthetic": True,
    }


def test_module_import_and_prepare_never_import_torch_or_laya(tmp_path, items_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "laya", None)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"fake weights")
    output = tmp_path / "output"
    result = runner.prepare(checkpoint.resolve(), items_path.resolve(), output.resolve(), 1218)
    assert result["rows"] == 95
    assert "torch" not in runner.__dict__
    assert "laya" not in runner.__dict__
    if os.name == "posix":
        assert (output / "manifest.json").stat().st_mode & 0o777 == 0o600
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["checkpoint_files"]) == 1
    assert manifest["wire"]["rendering"] == "original_wire_state_and_questions"
    assert manifest["wire"]["backend_questions"].startswith("Decider_deployed")
    assert manifest["settings"]["max_options_per_question"] == 20


def test_duplicate_ids_in_95_item_wire_are_rejected():
    items = _items()
    items[-1] = dict(items[-1], id=items[0]["id"])
    with pytest.raises(runner.PrepareError, match="duplicate"):
        runner.validate_items(items)


def test_fixed_wire_renders_exact_state_and_questions_and_rejects_rich():
    item = _items()[0]
    state, questions = runner.render_wire_item(item)
    assert state == item["state"]
    assert questions == item["questions"]
    state["page"]["text"] = "changed only in copy"
    assert item["state"]["page"]["text"] != "changed only in copy"
    with pytest.raises(runner.PrepareError, match="rich"):
        runner.render_wire_item({**item, "state_rich": {}})


def test_sparse_target_indices_are_resolved_by_wire_key_not_position():
    item = _item()

    def answers(questions):
        out = FakeAgent().predict(None, questions)["answers"]
        out["operation"]["choice"] = "CLICK"
        out["operation"]["probabilities"] = {"CLICK": 0.9, "TYPE_TEXT": 0.1}
        out["operation"]["confidence"] = 0.9
        out["click_target"]["choice"] = "7"
        out["click_target"]["probabilities"] = {"2": 0.1, "7": 0.9}
        out["click_target"]["confidence"] = 0.9
        return out

    rows = runner.evaluate_items([item], _backend(FakeAgent(answers)))
    assert rows[0]["target"] == "7"
    assert rows[0]["label"] == "synthetic-label-good"
    assert rows[0]["semantic_ok"] is True
    assert rows[0]["handoff_ok"] is True


def test_wide_second_round_scores_merged_answers_and_audits_every_call():
    item = _item(target_keys=tuple(str(index) for index in range(1, 22)))
    item["accepted"] = ["synthetic-label-merged"]
    item["index_labels"] = {
        str(index): ("synthetic-label-merged" if index == 2 else f"synthetic-label-other-{index}")
        for index in range(1, 22)
    }
    original = copy.deepcopy(item)

    def answers(questions):
        result = {}
        for name, question in questions.items():
            keys = list(question["criteria"])
            if name == "operation":
                chosen, probabilities = "CLICK", {"CLICK": 0.9, "TYPE_TEXT": 0.1}
            elif name == "click_target__chunk0":
                chosen = "3"
                probabilities = {key: (0.51 if key == "3" else 0.49 if key == "5" else 0.0) for key in keys}
            elif name == "click_target__chunk1":
                chosen = "2"
                probabilities = {key: (0.99 if key == "2" else 0.01 if key == "4" else 0.0) for key in keys}
            else:
                # The final round only contains the chunk winners.  Its winner
                # is deliberately not the winner after probability recombination.
                chosen = "3"
                probabilities = {key: (0.5001 if key == "3" else 0.4999 if key == "2" else 0.0) for key in keys}
            result[name] = {
                "type": "choice", "choice": chosen, "probabilities": probabilities,
                "confidence": probabilities[chosen], "action": {"act_probability": 1.0},
            }
        return result

    rows = runner.evaluate_items([item], _backend(FakeAgent(answers)))
    row = rows[0]
    assert row["operation_key"] == "CLICK"
    assert row["operation"] == "CLICK"
    assert row["target"] == "2"
    assert row["semantic_ok"] is True
    assert row["prediction"]["answers"]["click_target"]["choice"] == "3"
    assert row["validated_answers"]["click_target"]["choice"] == "2"
    assert len(row["backend_calls"]) == 2
    assert "operation" in row["backend_calls"][0]["questions"]
    assert "operation" not in row["backend_calls"][1]["questions"]
    assert row["question_transformed"] is True
    assert item == original


def test_numeric_operation_missing_from_original_criteria_cannot_score():
    item = _item()

    def answers(questions):
        out = FakeAgent().predict(None, questions)["answers"]
        out["operation"]["choice"] = "2"
        return out

    row = runner.evaluate_items([item], _backend(FakeAgent(answers)))[0]
    assert row["error_count"] == 1
    assert row["semantic_ok"] is False
    assert row["operation_key"] is None


def test_numeric_operation_offered_key_uses_only_unambiguous_criteria_value():
    item = _item()
    item["questions"]["operation"]["criteria"] = {"1": "CLICK", "2": "TYPE_TEXT"}

    def answers(questions):
        out = FakeAgent().predict(None, questions)["answers"]
        out["operation"] = {
            "type": "choice", "choice": "1", "probabilities": {"1": 0.9, "2": 0.1},
            "confidence": 0.9, "action": {"act_probability": 1.0},
        }
        out["click_target"]["choice"] = "7"
        out["click_target"]["probabilities"] = {"2": 0.1, "7": 0.9}
        return out

    row = runner.evaluate_items([item], _backend(FakeAgent(answers)))[0]
    assert row["operation_key"] == "1"
    assert row["operation"] == "CLICK"
    assert row["operation_selected_probability"] == pytest.approx(0.9)
    assert row["semantic_ok"] is True


def test_numeric_operation_ambiguous_criteria_value_is_an_error():
    item = _item()
    item["questions"]["operation"]["criteria"] = {"1": "CLICK or TYPE_TEXT", "2": "WAIT"}

    def answers(questions):
        out = FakeAgent().predict(None, questions)["answers"]
        out["operation"] = {
            "type": "choice", "choice": "1", "probabilities": {"1": 0.9, "2": 0.1},
            "confidence": 0.9, "action": {"act_probability": 1.0},
        }
        return out

    row = runner.evaluate_items([item], _backend(FakeAgent(answers)))[0]
    assert row["error_count"] == 1
    assert row["operation_key"] == "1"
    assert row["operation"] is None
    assert row["semantic_ok"] is False


@pytest.mark.parametrize("operation", ["WAIT", "BLOCKED", "DONE", "SCROLL_DOWN", "SCROLL_UP"])
def test_valid_non_target_operation_is_a_pick_miss_not_contract_error(operation):
    item = _item(op_keys=(operation, "CLICK"))
    row = runner.evaluate_items([item], _backend(FakeAgent()))[0]
    assert row["validated"] is True
    assert row["error"] is None
    assert row["operation"] == operation
    assert row["operation_selected_probability"] == 1.0
    assert row["target"] is None
    assert row["semantic_ok"] is False


def test_restraint_is_a_joint_operation_result_for_both_rules():
    item = _item(suite="restraint", expected="DONE", semantic=None, op_keys=("DONE", "BLOCKED"), target_keys=("2",))
    item["accepted"] = []

    def answers(questions):
        out = FakeAgent().predict(None, questions)["answers"]
        out["operation"]["choice"] = "DONE"
        out["operation"]["probabilities"] = {"DONE": 0.8, "BLOCKED": 0.2}
        out["operation"]["confidence"] = 0.8
        return out

    rows = runner.evaluate_items([item], _backend(FakeAgent(answers)))
    assert rows[0]["semantic_ok"] is True
    assert rows[0]["handoff_ok"] is True
    assert rows[0]["target_ok"] is True


def test_real_backend_wrapper_and_decider_validate_choice_score_noul_shapes():
    agent = FakeAgent()
    backend = _backend(agent)
    smoke = runner.smoke_typed_contract(backend)
    assert smoke["validated"] is True
    assert smoke["types"] == {"choice_smoke": "choice", "score_smoke": "score", "noul_smoke": "noul"}
    assert isinstance(backend, __import__("localdecide.backends.base", fromlist=["LayaTorchBackend"]).LayaTorchBackend)
    assert agent.calls


@pytest.mark.parametrize("bad_answer", [
    {"type": "noul", "noul": float("nan")},
    {"type": "choice", "choice": "a", "probabilities": {"a": 0.2, "b": 0.2}},
])
def test_invalid_nan_or_probability_payload_is_an_error(bad_answer):
    from localdecide import choice, noul
    from localdecide.decider import Decider

    question = noul("yes?") if bad_answer["type"] == "noul" else choice("pick", {"a": "A", "b": "B"})

    class Backend:
        name = "fake-invalid"

        def answer(self, state, questions):
            return {"answers": {next(iter(questions)): bad_answer}}

    result = Decider(backend=Backend(), retries=0, fail_open=True).decide("s", {"q": question})
    assert not result.ok


def test_invalid_raw_prediction_counts_nonfinite_and_keeps_row_error():
    item = _item()

    def answers(questions):
        out = FakeAgent().predict(None, questions)["answers"]
        out["operation"]["probabilities"]["CLICK"] = float("nan")
        return out

    rows = runner.evaluate_items([item], _backend(FakeAgent(answers)))
    assert rows[0]["error_count"] == 1
    assert rows[0]["nonfinite_count"] >= 1
    assert rows[0]["prediction"]["answers"]["operation"]["probabilities"]["CLICK"] != rows[0]["prediction"]["answers"]["operation"]["probabilities"]["CLICK"]
    assert rows[0]["backend_calls"][0]["raw_envelope"]["answers"]["operation"]["probabilities"]["CLICK"] != 1.0


def test_backend_exception_is_recorded_and_smoke_calls_do_not_leak_into_case():
    class FlakyAgent(FakeAgent):
        def __init__(self):
            super().__init__()
            self.fail = False

        def predict(self, state, questions):
            if self.fail:
                raise RuntimeError("synthetic backend failure")
            return super().predict(state, questions)

    agent = FlakyAgent()
    backend = _backend(agent)
    runner.smoke_typed_contract(backend)
    agent.fail = True
    row = runner.evaluate_items([_item()], backend)[0]
    assert row["error_count"] == 1
    assert row["prediction"] is None
    assert len(row["backend_calls"]) == 1
    assert row["backend_calls"][0]["raw_envelope"] is None
    assert "synthetic backend failure" in row["backend_calls"][0]["error"]


def test_gpu_guard_limits_and_ignores_broad_shell_false_positive():
    own = 100
    with pytest.raises(runner.SafetyError, match="memory delta"):
        runner.evaluate_gpu_safety(1500, 0, 1218, [], [], own_pid=own)
    with pytest.raises(runner.SafetyError, match="utilization"):
        runner.evaluate_gpu_safety(1218, 11, 1218, [], [], own_pid=own)
    with pytest.raises(runner.SafetyError, match="compute"):
        runner.evaluate_gpu_safety(1218, 0, 1218, [101], [], own_pid=own)
    with pytest.raises(runner.SafetyError, match="train/eval"):
        runner.evaluate_gpu_safety(1218, 0, 1218, [],
                                   [(own, "python evaluate_self.py"),
                                    (101, "/usr/bin/python3 train_model.py"),
                                    (102, "/bin/zsh -lc python evaluate_other.py")], own_pid=own)
    assert runner.evaluate_gpu_safety(1218, 0, 1218, [],
                                      [(own, "python evaluate_self.py"),
                                       (102, "/bin/zsh -lc python evaluate_other.py")], own_pid=own)["ok"]


def test_percentile_is_linear_interpolated_and_latency_excludes_warmup():
    assert runner.percentile([1, 2, 3, 4], 0.5) == 2.5
    assert runner.percentile([1, 2, 3, 4], 0.95) == pytest.approx(3.85)
    rows = [{"latency_ms": 10, "warmup": True, "error": None},
            {"latency_ms": 1, "warmup": False, "error": None},
            {"latency_ms": 3, "warmup": False, "error": None},
            {"latency_ms": 5, "warmup": False, "error": None}]
    summary = runner.latency_summary(rows)
    assert summary["warmup_excluded"] == 1
    assert summary["p50_ms"] == 3
    assert summary["p95_ms"] == pytest.approx(4.8)
    assert summary["synchronized"] is True


def test_hashes_include_all_checkpoint_files_and_framework_evidence(tmp_path, items_path):
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "nested").mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"one")
    (checkpoint / "nested" / "tokenizer.json").write_text("{}", encoding="utf-8")
    records = runner.hash_checkpoint(checkpoint.resolve())
    assert [record["relative_path"] for record in records] == [
        "model.safetensors", str(pathlib.Path("nested") / "tokenizer.json")
    ]
    manifest = runner.build_manifest(checkpoint.resolve(), items_path.resolve(), _items(),
                                     (tmp_path / "out").resolve(), 1218)
    framework_names = {pathlib.Path(record["path"]).name for record in manifest["input_scorer_framework_hashes"]["scorer_and_framework"]}
    assert {"evaluate_wide.py", "base.py", "decider.py", "eval_release_acceptance.py"} <= framework_names


def test_file_lock_is_exclusive(tmp_path):
    lock_path = tmp_path / "logs" / "release_acceptance" / "gpu.lock"
    if os.name == "posix":
        with runner.gpu_lock(lock_path):
            with pytest.raises(runner.SafetyError, match="already held"):
                with runner.gpu_lock(lock_path):
                    pass
    else:
        entered = False
        with pytest.raises(runner.SafetyError) as exc_info:
            with runner.gpu_lock(lock_path):
                entered = True
        assert str(exc_info.value) == "fcntl is required for the Linux execute lock"
        assert entered is False
        assert not lock_path.exists()


def test_file_lock_fails_closed_when_fcntl_import_is_unavailable(monkeypatch, tmp_path):
    lock_path = tmp_path / "logs" / "release_acceptance" / "gpu.lock"
    monkeypatch.setitem(sys.modules, "fcntl", None)
    entered = False
    with pytest.raises(runner.SafetyError) as exc_info:
        with runner.gpu_lock(lock_path):
            entered = True
    assert str(exc_info.value) == "fcntl is required for the Linux execute lock"
    assert entered is False
    assert not lock_path.exists()


def test_module_and_instance_to_guards_refuse_cpu_mps_but_allow_dtype():
    class Module:
        def __init__(self):
            self.calls = []

        def to(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return self

    fake_torch = types.SimpleNamespace(nn=types.SimpleNamespace(Module=Module))
    module = Module()
    with runner.guard_torch_module_to(fake_torch):
        with pytest.raises(runner.SafetyError):
            module.to("cpu")
        with pytest.raises(runner.SafetyError):
            module.to(device="mps")
        module.to("bfloat16")
    module.to("cpu")  # global wrapper was restored

    restore = runner.guard_model_instance_to(module)
    with pytest.raises(runner.SafetyError):
        module.to("cpu")
    module.to("bfloat16")
    restore()
    module.to("cpu")


def test_bf16_autocast_allows_fp32_master_parameters_and_post_predict_guard():
    bf16 = object()
    fp32 = types.SimpleNamespace(is_floating_point=True)
    parameter = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda"), dtype=fp32)
    agent = FakeAgent()
    agent.dtype = bf16
    agent.model = types.SimpleNamespace(parameters=lambda: [parameter])
    fake_torch = types.SimpleNamespace(bfloat16=bf16, cuda=types.SimpleNamespace(
        is_available=lambda: True, current_device=lambda: 0))
    runner.validate_cuda_device_dtype(agent, fake_torch)
    calls = []
    backend = runner.make_preloaded_backend(agent, before_predict=lambda: calls.append("guard"))
    runner.smoke_typed_contract(backend)
    assert calls == ["guard", "guard"]
    parameter.device.type = "cpu"
    with pytest.raises(runner.SafetyError, match="not CUDA"):
        runner.validate_cuda_device_dtype(agent, fake_torch)


def test_execute_cli_preserves_error_exit_status(monkeypatch, tmp_path, items_path):
    monkeypatch.setattr(runner, "execute", lambda *args: {"mode": "execute", "errors": 1})
    assert runner.main(["--execute", "--model-dir", str(tmp_path), "--items", str(items_path),
                        "--output-dir", str(tmp_path / "out")]) == 1


def test_full_95_fake_run_has_complete_unique_raw_rows_and_null_joint_ece():
    items = runner.validate_items(_items())
    original = copy.deepcopy(items)
    rows = runner.evaluate_items(items, _backend(FakeAgent()), warmup=3)
    assert len(rows) == 95
    assert len({row["id"] for row in rows}) == 95
    assert sum(row["warmup"] for row in rows) == 3
    assert all("gold" in row and "prediction" in row for row in rows)
    assert items == original
    assert sum(row["question_transformed"] for row in rows) == 47
    assert sum(len(row["backend_calls"]) for row in rows) == 142
    report = runner.build_report(rows, {"schema": "test"}, smoke={"ok": True})
    assert report["calibration"]["joint_ece"] is None
    assert "marginal" in report["calibration"]["joint_ece_explanation"]
    assert report["scoring_source"] == "decision.answers.raw (merged validated answers)"
    assert report["decider"]["max_options_per_question"] == 20
    assert report["decider"]["coarse_to_fine_cases"] == 47
    assert report["rendering"]["wire_input"] == "original_wire_state_and_questions"
    assert report["slices"]["suite"]["pick"]["n"] == 79
    assert report["slices"]["suite"]["restraint"]["n"] == 16
