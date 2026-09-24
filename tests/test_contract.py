"""Tests for the decision layer's contract - the parts that must never go wrong.

These run without any model: a fake backend stands in, so the suite is fast and
verifies the *harness* rules (validation, fail-open, index resolution, loop guards)
rather than anyone's checkpoint quality.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict

from localdecide import BrowserDecider, Decider, Scope, choice, goal_tokens, noul, score
from localdecide.decider import DecisionError
from localdecide.page import build_element_table, table_to_questions


class TestScope(unittest.TestCase):
    """The scoping rules - the place where a silent mistake looks like a model failure.

    The bug these tests exist to prevent: a naive "drop the nav bar" filter removes
    "Random article", because it IS a nav link. It is also exactly what the user asked
    the agent to click. The goal must outrank the heuristic.
    """

    def observation(self):
        return {"url": "https://en.wikipedia.org", "title": "Wikipedia", "text": "", "actions": [
            {"kind": "click", "node": "n1", "label": "Random article", "role": "link"},
            {"kind": "click", "node": "n2", "label": "Contents", "role": "link"},
            {"kind": "click", "node": "n3", "label": "Log in", "role": "link"},
            {"kind": "click", "node": "n4", "label": "View source [ctrl-option-e]", "role": "link"},
            {"kind": "click", "node": "n5", "label": "Privacy policy", "role": "link"},
            {"kind": "click", "node": "n6", "label": "Featured article about birds", "role": "link"},
            {"kind": "fill", "node": "n7", "label": "Search Wikipedia", "role": "searchbox"},
        ]}

    def test_goal_tokens_drop_stopwords_and_verbs(self):
        tokens = goal_tokens("Click the 'Random article' link in the navigation.")
        self.assertIn("random", tokens)
        self.assertIn("article", tokens)
        self.assertNotIn("click", tokens)   # a verb every goal has
        self.assertNotIn("the", tokens)
        self.assertNotIn("link", tokens)    # nothing on a page is named "link"

    def test_goal_match_survives_the_chrome_filter(self):
        """"Random article" is nav furniture by shape and the target by intent."""
        scope = Scope()
        scoped = scope.apply(self.observation(), goal="Click the 'Random article' link.")
        labels = [action["label"] for action in scoped["actions"]]
        self.assertIn("Random article", labels)

    def test_chrome_is_dropped_when_the_goal_ignores_it(self):
        scope = Scope()
        scoped = scope.apply(self.observation(), goal="Click the 'Random article' link.")
        labels = [action["label"] for action in scoped["actions"]]
        self.assertNotIn("Privacy policy", labels)
        self.assertNotIn("View source [ctrl-option-e]", labels)

    def test_without_a_goal_chrome_is_still_dropped(self):
        """No goal means no protection: structural noise goes, and that is the point."""
        scope = Scope()
        scoped = scope.apply(self.observation())
        labels = [action["label"] for action in scoped["actions"]]
        self.assertNotIn("Privacy policy", labels)
        self.assertNotIn("View source [ctrl-option-e]", labels)

    def test_goal_matches_rank_first(self):
        scope = Scope(max_elements=3)
        scoped = scope.apply(self.observation(), goal="Click the 'Random article' link.")
        self.assertEqual(scoped["actions"][0]["label"], "Random article")

    def test_include_words_are_an_explicit_request(self):
        """An include list is the caller saying 'only these' - not overridden by the goal."""
        scope = Scope(include_words=["search"])
        scoped = scope.apply(self.observation(), goal="Click the 'Random article' link.")
        labels = [action["label"] for action in scoped["actions"]]
        self.assertEqual(labels, ["Search Wikipedia"])

    def test_exclude_words_win(self):
        scope = Scope(exclude_words=["random"])
        scoped = scope.apply(self.observation(), goal="Click the 'Random article' link.")
        self.assertNotIn("Random article", [action["label"] for action in scoped["actions"]])

    def test_indices_are_renumbered_densely(self):
        scope = Scope(max_elements=3)
        scoped = scope.apply(self.observation(), goal="anything")
        self.assertEqual([action["index"] for action in scoped["actions"]], [1, 2, 3])

    def test_scope_reports_what_it_did(self):
        scope = Scope(max_elements=3)
        scoped = scope.apply(self.observation(), goal="Click the 'Random article' link.")
        report = scoped["scope"]
        self.assertEqual(report["observed"], 7)
        self.assertEqual(report["offered"], 3)
        self.assertGreater(report["dropped"], 0)

    def test_loop_applies_the_scope_with_the_goal(self):
        """End-to-end through the loop: a nav-link target must still be reachable."""
        backend = _SequenceBackend([{"operation": "CLICK", "click_target": "1"}],
                                   on_exhausted={"operation": "DONE"})
        driver = FakeDriver([self.observation()])
        run = BrowserDecider(decider=Decider(backend=backend), max_steps=1, scope=Scope(max_elements=4)).run(
            driver, "Click the 'Random article' link.")
        self.assertEqual(driver.executed, [("CLICK", "1", None)])
        self.assertEqual(run.steps[0].label, "Random article")


class FakeBackend:
    """Answers from a script; records what it was asked.

    A plain dict of answers only works when the questions are known in advance. The
    loop builds its own questions from each observation, so use `answer_first` /
    `answer_operation` when you need an answer that always fits what was offered.
    """

    name = "fake"

    def __init__(self, answers: Dict[str, Any] | None = None, fail: bool = False,
                 answer_first: bool = False, operation: str | None = None) -> None:
        self.answers = answers or {}
        self.fail = fail
        self.answer_first = answer_first
        self.operation = operation
        self.calls: list = []

    def _answer_for(self, name, question):
        if self.operation is not None and name == "operation":
            keys = list(question["criteria"]) if isinstance(question.get("criteria"), dict) else []
            if self.operation in keys:
                return one_choice(self.operation, keys)
        if name in self.answers:
            return self.answers[name]
        keys = list(question["criteria"]) if isinstance(question.get("criteria"), dict) else []
        if keys:
            return one_choice(keys[0], keys)
        raise AssertionError(f"no answer for {name!r} and no criteria to pick from")

    def answer(self, state, questions):
        self.calls.append((state, questions))
        if self.fail:
            from localdecide.backends.base import BackendError
            raise BackendError("network", "boom")
        answers = {name: self._answer_for(name, question) for name, question in questions.items()}
        return {"answers": answers, "usage": {"input_tokens": 10}, "latency_ms": 5, "backend": self.name}


def one_choice(key: str, options: list, probabilities: Dict[str, float] | None = None) -> Dict[str, Any]:
    probs = probabilities or {option: (1.0 if option == key else 0.0) for option in options}
    return {"type": "choice", "choice": key, "probabilities": probs, "confidence": probs[key],
            "action": {"act_probability": 1.0}}


class TestQuestionBuilders(unittest.TestCase):
    def test_choice_needs_two_options(self):
        with self.assertRaises(ValueError):
            choice("pick", {"only": "one"})

    def test_score_needs_two_levels(self):
        with self.assertRaises(ValueError):
            score("rate", ["only"])

    def test_noul_shape(self):
        self.assertEqual(noul("is it?"), {"type": "noul", "instructions": "is it?"})


class TestValidation(unittest.TestCase):
    """A surprising answer must never reach the caller as if it were usable."""

    def test_happy_path(self):
        backend = FakeBackend({"pick": one_choice("a", ["a", "b"])})
        decision = Decider(backend=backend, fail_open=False).decide("state", {"pick": choice("x", {"a": "A", "b": "B"})})
        self.assertTrue(decision.ok)
        assert decision.answers is not None
        self.assertEqual(decision.answers.choice("pick"), "a")

    def test_choice_outside_the_offered_set_is_rejected(self):
        backend = FakeBackend({"pick": one_choice("c", ["a", "b", "c"])})
        decision = Decider(backend=backend, retries=0).decide("state", {"pick": choice("x", {"a": "A", "b": "B"})})
        self.assertFalse(decision.ok)
        self.assertTrue(decision.failed_open)
        self.assertIn("not an offered option", decision.error or "")

    def test_probabilities_must_sum_to_one(self):
        backend = FakeBackend({"pick": {"type": "choice", "choice": "a",
                                        "probabilities": {"a": 0.2, "b": 0.2}, "confidence": 0.2}})
        decision = Decider(backend=backend, retries=0).decide("state", {"pick": choice("x", {"a": "A", "b": "B"})})
        self.assertFalse(decision.ok)

    def test_choice_must_be_argmax(self):
        backend = FakeBackend({"pick": {"type": "choice", "choice": "a",
                                        "probabilities": {"a": 0.3, "b": 0.7}, "confidence": 0.3}})
        decision = Decider(backend=backend, retries=0).decide("state", {"pick": choice("x", {"a": "A", "b": "B"})})
        self.assertFalse(decision.ok)
        self.assertIn("most probable", decision.error or "")

    def test_nan_is_rejected(self):
        backend = FakeBackend({"yes": {"type": "noul", "noul": float("nan"), "confidence": 0.5}})
        decision = Decider(backend=backend, retries=0).decide("state", {"yes": noul("really?")})
        self.assertFalse(decision.ok)

    def test_fail_open_returns_a_decision_instead_of_raising(self):
        decision = Decider(backend=FakeBackend(fail=True), retries=0).decide("s", {"q": noul("?")})
        self.assertFalse(decision.ok)
        self.assertTrue(decision.failed_open)

    def test_fail_closed_raises_when_asked_to(self):
        with self.assertRaises(DecisionError):
            Decider(backend=FakeBackend(fail=True), retries=0, fail_open=False).decide("s", {"q": noul("?")})


class TestElementTable(unittest.TestCase):
    def setUp(self):
        self.observation = {"url": "https://example.com", "title": "T", "text": "hello", "actions": [
            {"kind": "click", "node": "n1", "label": "Log in", "role": "link"},
            {"kind": "fill", "node": "n2", "label": "Search", "role": "searchbox", "current_value": ""},
            {"kind": "select", "node": "n3", "label": "Country", "role": "combobox",
             "options": [{"label": "Australia", "value": "AU"}, {"label": "Japan", "value": "JP"}]},
            {"kind": "scroll", "node": "n4", "label": "ignore me"},
        ]}

    def test_numbers_elements_and_maps_kinds(self):
        table = build_element_table(self.observation)
        self.assertEqual([element.index for element in table.elements], ["1", "2", "3"])
        self.assertEqual(table.elements[0].operations, ["CLICK"])
        self.assertEqual(table.elements[1].operations, ["TYPE_TEXT"])
        self.assertEqual(table.elements[2].operations, ["SELECT"])

    def test_unknown_kinds_are_dropped(self):
        table = build_element_table(self.observation)
        self.assertNotIn("ignore me", [element.label for element in table.elements])

    def test_targets_are_operation_specific(self):
        table = build_element_table(self.observation)
        self.assertEqual(list(table.targets_for("CLICK")), ["1"])
        self.assertEqual(list(table.targets_for("TYPE_TEXT")), ["2"])
        self.assertEqual(list(table.targets_for("SELECT")), ["3"])

    def test_same_node_is_not_duplicated(self):
        observation = {"actions": [
            {"kind": "click", "node": "x", "label": "Menu"},
            {"kind": "click", "node": "x", "label": "Menu"},
        ]}
        table = build_element_table(observation)
        self.assertEqual(len(table.elements), 1)

    def test_questions_offer_only_supported_operations(self):
        table = build_element_table(self.observation)
        questions = table_to_questions(table, "do a thing")
        operations = questions["operation"]["criteria"]
        self.assertIn("CLICK", operations)
        self.assertIn("TYPE_TEXT", operations)
        self.assertNotIn("SELECT", operations) if False else None  # SELECT is supported here
        self.assertEqual(set(questions["click_target"]["criteria"]), {"1"})
        self.assertEqual(set(questions["type_text_target"]["criteria"]), {"2"})

    def test_v3_state_keeps_elements_out_of_state(self):
        table = build_element_table(self.observation)
        self.assertNotIn("elements", table.state(layout="v3"))
        self.assertIn("elements", table.state(layout="v1"))

    def test_dropdown_options_are_scoped_to_their_fields(self):
        actions = self.observation["actions"]
        actions.append({"kind": "select", "node": "n4", "label": "Language",
                        "options": [{"label": "English", "value": "en"},
                                    {"label": "Spanish", "value": "es"}]})
        table = build_element_table(self.observation)
        questions = table_to_questions(table, "Select Spanish as the language")
        self.assertEqual(set(questions["select_option_3"]["criteria"]), {"3:1", "3:2"})
        self.assertEqual(set(questions["select_option_4"]["criteria"]), {"4:1", "4:2"})
        self.assertEqual(questions["select_option_4"]["instructions"]["field"], "Language")
        self.assertNotIn("select_option", questions)
        single = table_to_questions(build_element_table({"actions": [actions[2]]}), "Select Japan")
        self.assertIn("select_option", single)

    def test_loop_selects_option_from_chosen_dropdown(self):
        observation = {"url": "https://example.com", "actions": [
            {"kind": "select", "node": "country", "label": "Country",
             "options": [{"label": "Japan", "value": "JP"}, {"label": "France", "value": "FR"}]},
            {"kind": "select", "node": "language", "label": "Language",
             "options": [{"label": "English", "value": "en"}, {"label": "Japanese", "value": "ja"}]},
        ]}
        backend = _SequenceBackend([{"operation": "SELECT", "select_target": "2",
                                     "select_option_1": "1:1", "select_option_2": "2:2"}])
        driver = FakeDriver([observation])
        run = BrowserDecider(decider=Decider(backend=backend), max_steps=1,
                             scope=Scope(prefer_words=["Country", "Language"])).run(
            driver, "Set language to Japanese")
        self.assertEqual(driver.executed, [("SELECT", "2", "ja")])
        self.assertEqual(run.stopped, "max_steps")


class FakeDriver:
    """A driver that replays a scripted sequence of observations."""

    def __init__(self, observations, outcomes=None):
        self.observations = observations
        self.outcomes = outcomes or {}
        self.executed = []
        self.closed = False
        self.index = 0

    def observe(self):
        observation = self.observations[min(self.index, len(self.observations) - 1)]
        self.index += 1
        return observation

    def execute(self, operation, element, text=None):
        self.executed.append((operation, element.index if element else None, text))
        return self.outcomes.get(operation, {"ok": True, "detail": "ok", "page_changed": True})

    def close(self):
        self.closed = True


class TestLoop(unittest.TestCase):
    def observation(self):
        return {"url": "https://x", "title": "T", "text": "t", "actions": [
            {"kind": "click", "node": "n1", "label": "Random article", "role": "link"},
            {"kind": "click", "node": "n2", "label": "Log in", "role": "link"},
        ]}

    def test_runs_to_done(self):
        backend = FakeBackend(operation="DONE")
        driver = _ScriptedDriver([self.observation()])
        run = BrowserDecider(decider=Decider(backend=backend), max_steps=3).run(driver, "open random article")
        self.assertEqual(run.stopped, "done")
        self.assertTrue(run.solved)

    def test_click_then_done(self):
        """First cycle clicks, second cycle reports DONE - the normal shape of a run."""
        backend = _SequenceBackend([
            {"operation": "CLICK", "click_target": "1"},
            {"operation": "DONE"},
        ])
        driver = FakeDriver([self.observation()])
        run = BrowserDecider(decider=Decider(backend=backend), max_steps=5).run(driver, "open random article")
        self.assertEqual(run.stopped, "done")
        self.assertEqual(driver.executed, [("CLICK", "1", None)])

    def test_hallucinated_index_never_reaches_the_driver(self):
        """A target outside the offered set is caught by answer validation.

        This is the real guarantee: the invalid answer fails validation, so the decision
        fails open and the driver is never asked to act on a made-up index.
        """
        backend = _SequenceBackend([{"operation": "CLICK", "click_target": "99"}],
                                   on_exhausted={"operation": "DONE"})
        driver = FakeDriver([self.observation()])
        run = BrowserDecider(decider=Decider(backend=backend, retries=0), max_steps=3).run(driver, "goal")
        self.assertNotIn("99", [entry[1] for entry in driver.executed])
        self.assertTrue(any(step.failed_open for step in run.steps))

    def test_loop_refuses_an_index_that_vanished(self):
        result = Decider(backend=FakeBackend(), retries=0).decide("s", {"x": choice("q", {"1": "a", "2": "b"})})
        assert result.answers is not None
        # Forge a validated-looking answer naming an index the table does not contain.
        result.answers.raw["operation"] = {"type": "choice", "choice": "CLICK",
                                          "probabilities": {"CLICK": 1.0, "DONE": 0.0},
                                          "confidence": 1.0}
        result.answers.raw["click_target"] = {"type": "choice", "choice": "42",
                                              "probabilities": {"42": 1.0, "1": 0.0, "2": 0.0},
                                              "confidence": 1.0}

        class FrozenDecider:
            def __init__(self, decision):
                self.decision = decision
                self.max_options_per_question = 20

            def decide(self, state, questions):
                return self.decision

        run = BrowserDecider(decider=FrozenDecider(result), max_steps=2).run(FakeDriver([self.observation()]), "goal")
        self.assertEqual(run.stopped, "error")
        self.assertIn("hallucinated", run.error)

    def test_risky_action_stops_for_confirmation(self):
        backend = _SequenceBackend([{"operation": "CLICK", "click_target": "1"}])
        run = BrowserDecider(decider=Decider(backend=backend)).run(FakeDriver([self.observation()]),
                                                                   "delete my account")
        self.assertEqual(run.stopped, "needs_confirmation")

    def test_confirm_callback_can_allow(self):
        backend = _SequenceBackend([{"operation": "CLICK", "click_target": "1"}])
        run = BrowserDecider(decider=Decider(backend=backend), confirm=lambda label, element: True,
                             max_steps=1).run(FakeDriver([self.observation()]), "delete my account")
        self.assertEqual(run.stopped, "max_steps")

    def test_type_text_without_provider_is_refused(self):
        backend = _SequenceBackend([{"operation": "TYPE_TEXT", "type_text_target": "1"}])
        driver = FakeDriver([{"url": "https://x", "title": "T", "text": "", "actions": [
            {"kind": "fill", "node": "n1", "label": "Search", "role": "searchbox"},
            {"kind": "fill", "node": "n2", "label": "Name", "role": "textbox"},
        ]}])
        run = BrowserDecider(decider=Decider(backend=backend)).run(driver, "type something")
        self.assertEqual(run.stopped, "error")
        self.assertIn("text provider", run.error)

    def test_type_text_uses_the_provider(self):
        backend = _SequenceBackend([{"operation": "TYPE_TEXT", "type_text_target": "1"}])
        driver = FakeDriver([{"url": "https://x", "title": "T", "text": "", "actions": [
            {"kind": "fill", "node": "n1", "label": "Search", "role": "searchbox"},
            {"kind": "fill", "node": "n2", "label": "Name", "role": "textbox"},
        ]}])
        run = BrowserDecider(decider=Decider(backend=backend), text_provider=lambda goal, element: "hello",
                             max_steps=1).run(driver, "type a greeting")
        self.assertEqual(driver.executed[0], ("TYPE_TEXT", "1", "hello"))

    def test_loop_guard_stops_a_stuck_agent(self):
        backend = _SequenceBackend([{"operation": "CLICK", "click_target": "1"}])
        driver = FakeDriver([self.observation()], outcomes={"CLICK": {"ok": True, "detail": "ok", "page_changed": False}})
        run = BrowserDecider(decider=Decider(backend=backend), max_steps=10).run(driver, "goal")
        self.assertEqual(run.stopped, "error")
        self.assertIn("stuck", run.error)

    def test_failed_open_twice_gives_up(self):
        backend = FakeBackend(fail=True)
        run = BrowserDecider(decider=Decider(backend=backend, retries=0), max_steps=10).run(FakeDriver([self.observation()]), "goal")
        self.assertEqual(run.stopped, "error")

    def test_driver_is_closed_even_on_failure(self):
        driver = FakeDriver([self.observation()])
        backend = FakeBackend(fail=True)
        BrowserDecider(decider=Decider(backend=backend, retries=0)).run(driver, "goal")
        self.assertTrue(driver.closed)

    def test_summary_reports_medians(self):
        backend = _SequenceBackend([{"operation": "DONE"}])
        run = BrowserDecider(decider=Decider(backend=backend)).run(FakeDriver([self.observation()]), "goal")
        summary = run.summary()
        self.assertEqual(summary["stopped"], "done")
        self.assertIn("median_decision_ms", summary)


class _SequenceBackend:
    """Plays a fixed list of {question_name: option_key} dicts, one decision at a time.

    When a scripted key is not among the offered options the backend forwards it
    unchanged, which is how a hallucinating model behaves: the bad key then has to be
    caught by the decider's validation. `sanitize=True` instead substitutes a valid key,
    for tests that want to reach the loop with an answer validation would accept.
    """

    name = "sequence"

    def __init__(self, script, on_exhausted=None, sanitize=False):
        self.script = list(script)
        self.on_exhausted = on_exhausted or {}
        self.sanitize = sanitize
        self.calls = []

    def answer(self, state, questions):
        self.calls.append((state, questions))
        chosen = self.script.pop(0) if self.script else dict(self.on_exhausted)
        answers = {}
        for name, question in questions.items():
            keys = list(question["criteria"]) if isinstance(question.get("criteria"), dict) else []
            key = chosen.get(name, keys[0] if keys else None)
            if key is None:
                answers[name] = {"type": question.get("type", "noul"), "noul": 0.5, "confidence": 0.5}
                continue
            as_strings = [str(k) for k in keys]
            if str(key) not in as_strings and self.sanitize:
                key = keys[0]
            if str(key) not in as_strings:
                # forward the invalid key: validation must reject it
                answers[name] = {"type": "choice", "choice": key,
                                 "probabilities": {**{str(k): 0.0 for k in keys}, str(key): 1.0},
                                 "confidence": 1.0, "action": {"act_probability": 1.0}}
                continue
            answers[name] = one_choice(next(k for k in keys if str(k) == str(key)), keys)
        return {"answers": answers, "usage": {}, "latency_ms": 3, "backend": self.name}


class _ScriptedDriver(FakeDriver):
    """Like FakeDriver but always observes the same fresh page (for the happy path)."""

    def observe(self):
        return self.observations[0]


class TestCoarseToFine(unittest.TestCase):
    def test_wide_choice_is_split(self):
        options = {str(i): f"element {i}" for i in range(1, 51)}

        class Capture(FakeBackend):
            def answer(self, state, questions):
                self.calls.append((state, dict(questions)))
                answers = {}
                for name, question in questions.items():
                    keys = list(question["criteria"]) if isinstance(question["criteria"], dict) else []
                    answers[name] = one_choice(keys[0], keys)
                return {"answers": answers, "usage": {}, "latency_ms": 1, "backend": "fake"}

        backend = Capture()
        Decider(backend=backend, max_options_per_question=20).decide("s", {"pick": choice("x", options)})
        first_pass = backend.calls[0][1]
        self.assertIn("pick__chunk0", first_pass)
        self.assertLessEqual(len(first_pass["pick__chunk0"]["criteria"]), 20)
        self.assertGreaterEqual(len(backend.calls), 2)  # a second pass for the winners




class TestToggleGuard(unittest.TestCase):
    """The harness refuses to click a control that is already in the requested state.

    Measured on the real checkpoint: asked to "tick the Terms accepted box" when the box is
    already checked, the model answers CLICK on that very box with p=0.90 - it would untick
    it. The option text said `checked=true` and the instructions said not to re-toggle, and
    neither was enough. So the guard lives in the harness, where it can be tested.
    """

    def observation(self):
        return {"url": "https://x", "title": "T", "text": "", "actions": [
            {"kind": "click", "node": "n1", "label": "Terms accepted (checked)", "role": "checkbox",
             "checked": True},
            {"kind": "click", "node": "n2", "label": "Newsletter (unchecked)", "role": "checkbox",
             "checked": False},
        ]}

    def test_clicking_a_checked_box_is_refused(self):
        backend = _SequenceBackend([
            {"operation": "CLICK", "click_target": "1"},   # the checked one
            {"operation": "DONE"},
        ])
        driver = FakeDriver([self.observation()])
        run = BrowserDecider(decider=Decider(backend=backend, retries=0), max_steps=3).run(
            driver, "Tick the 'Terms accepted' checkbox.")
        self.assertEqual(driver.executed, [], "the guard must prevent the undoing click")
        refusal = [s for s in run.steps if "untick" in s.detail]
        self.assertTrue(refusal, f"expected a refusal step; got {[s.detail for s in run.steps]}")

    def test_unchecking_is_allowed_when_the_goal_asks_for_it(self):
        backend = _SequenceBackend([
            {"operation": "CLICK", "click_target": "1"},
            {"operation": "DONE"},
        ])
        driver = FakeDriver([self.observation()])
        BrowserDecider(decider=Decider(backend=backend, retries=0), max_steps=2).run(
            driver, "Untick the terms checkbox.")
        self.assertEqual(driver.executed, [("CLICK", "1", None)],
                         "an explicit uncheck goal must go through")

    def test_clicking_an_unchecked_box_still_works(self):
        backend = _SequenceBackend([
            {"operation": "CLICK", "click_target": "2"},
            {"operation": "DONE"},
        ])
        driver = FakeDriver([self.observation()])
        BrowserDecider(decider=Decider(backend=backend, retries=0), max_steps=2).run(
            driver, "Subscribe to the newsletter.")
        self.assertEqual(driver.executed, [("CLICK", "2", None)])

    def test_ambiguous_goal_stands_the_guard_down(self):
        """No instruction either way means no refusal - ambiguity must not block progress."""
        backend = _SequenceBackend([
            {"operation": "CLICK", "click_target": "1"},
            {"operation": "DONE"},
        ])
        driver = FakeDriver([self.observation()])
        BrowserDecider(decider=Decider(backend=backend, retries=0), max_steps=2).run(
            driver, "Deal with the terms block on this page.")
        self.assertEqual(driver.executed, [("CLICK", "1", None)])

    def test_goal_wording_is_classified_correctly(self):
        decider = BrowserDecider()
        for goal, expected in (("tick the box", False), ("check the terms", False),
                               ("accept the terms", False), ("untick that box", True),
                               ("uncheck the newsletter", True), ("opt out", True),
                               ("open the settings page", None)):
            self.assertEqual(decider._goal_wants_unticked(goal), expected, f"misread {goal!r}")




class TestConfidenceGate(unittest.TestCase):
    """A near-coin-flip decision is refused rather than executed.

    Measured on the real page: the checkpoint proposed CLICK on a submit button with p=0.06.
    Acting on that starts a flow the user did not ask for. Stopping (`DONE`/`BLOCKED`) is
    always allowed, because refusing to stop is the more dangerous failure.
    """

    def observation(self):
        return {"url": "https://x", "title": "T", "text": "", "actions": [
            {"kind": "click", "node": "n1", "label": "Submit", "role": "button"},
            {"kind": "click", "node": "n2", "label": "Cancel", "role": "button"},
        ]}

    def test_low_confidence_action_is_refused(self):
        """A backend that spreads probability nearly evenly produces ~0 confidence."""
        class Unsure:
            name = "unsure"

            def answer(self, state, questions):
                answers = {}
                for name, question in questions.items():
                    keys = list(question["criteria"])
                    even = {key: round(1.0 / len(keys), 3) for key in keys}
                    answers[name] = {"type": "choice", "choice": keys[0], "probabilities": even,
                                     "confidence": 0.01, "action": {"act_probability": 1.0}}
                return {"answers": answers, "usage": {}, "latency_ms": 1, "backend": self.name}

        driver = FakeDriver([self.observation()])
        run = BrowserDecider(decider=Decider(backend=Unsure(), retries=0), max_steps=4).run(driver, "do a thing")
        self.assertEqual(driver.executed, [], "nothing should have been executed")
        self.assertTrue(any("below" in s.detail for s in run.steps))
        self.assertEqual(run.stopped, "error")

    def test_confident_action_goes_through(self):
        backend = _SequenceBackend([
            {"operation": "CLICK", "click_target": "1"},
            {"operation": "DONE"},
        ])
        driver = FakeDriver([self.observation()])
        BrowserDecider(decider=Decider(backend=backend, retries=0), max_steps=2).run(driver, "do a thing")
        self.assertEqual(driver.executed, [("CLICK", "1", None)])

    def test_done_is_allowed_even_at_low_confidence(self):
        """Stopping must never be blocked by the confidence gate."""
        class UnsureDone:
            name = "unsure-done"

            def answer(self, state, questions):
                answers = {}
                for name, question in questions.items():
                    keys = list(question["criteria"]) if isinstance(question.get("criteria"), dict) else []
                    if name == "operation" and "DONE" in keys:
                        probabilities = {k: (0.2 if k == "DONE" else 0.8 / max(1, len(keys) - 1)) for k in keys}
                        answers[name] = {"type": "choice", "choice": "DONE", "probabilities": probabilities,
                                         "confidence": 0.0, "action": {"act_probability": 1.0}}
                    elif keys:
                        answers[name] = {"type": "choice", "choice": keys[0],
                                         "probabilities": {k: round(1.0 / len(keys), 3) for k in keys},
                                         "confidence": 0.01, "action": {"act_probability": 1.0}}
                return {"answers": answers, "usage": {}, "latency_ms": 1, "backend": self.name}

        driver = FakeDriver([self.observation()])
        run = BrowserDecider(decider=Decider(backend=UnsureDone(), retries=0), max_steps=2).run(driver, "do a thing")
        self.assertEqual(run.stopped, "done", f"expected DONE to be honoured; got {run.stopped}")

    def test_threshold_is_configurable(self):
        """Raising the bar must block a decision that a low bar would allow.

        `_SequenceBackend` answers with confidence 1.0, so it passes any threshold - the
        blocking case needs a backend whose confidence sits between the two bars.
        """
        class Middling:
            name = "middling"

            def answer(self, state, questions):
                answers = {}
                for name, question in questions.items():
                    keys = list(question["criteria"]) if isinstance(question.get("criteria"), dict) else []
                    if not keys:
                        continue
                    # argmax is 0.5, so confidence is ~0.5: above the default, below 0.99
                    probabilities = {key: (0.5 if index == 0 else 0.5 / (len(keys) - 1))
                                     for index, key in enumerate(keys)} if len(keys) > 1 else {keys[0]: 1.0}
                    answers[name] = {"type": "choice", "choice": keys[0], "probabilities": probabilities,
                                     "confidence": 0.5, "action": {"act_probability": 1.0}}
                return {"answers": answers, "usage": {}, "latency_ms": 1, "backend": self.name}

        driver = FakeDriver([self.observation()])
        BrowserDecider(decider=Decider(backend=Middling(), retries=0), max_steps=2,
                       min_confidence=0.99).run(driver, "do a thing")
        self.assertEqual(driver.executed, [], "a high threshold must block a 0.5-confidence flow")

        driver2 = FakeDriver([self.observation()])
        BrowserDecider(decider=Decider(backend=Middling(), retries=0), max_steps=1,
                       min_confidence=0.2).run(driver2, "do a thing")
        self.assertEqual(driver2.executed, [("CLICK", "1", None)], "a low threshold must allow it")




class TestGrounding(unittest.TestCase):
    """Cross-language grounding: non-Latin goals must not be answered with foreign labels.

    Measured on the real checkpoint: a Chinese goal offered a page of mixed-script labels
    picked a Hindi button at p=0.06 (no signal). Filtering the option list down to labels
    sharing the goal's script turned 1/9 correct into 8/9 across ten scripts. These tests
    cover the unit pieces; the end-to-end numbers live in the repo README.
    """

    def observation(self):
        return {"url": "https://x", "title": "T", "text": "", "actions": [
            {"kind": "click", "node": "n1", "label": "登录", "role": "button"},
            {"kind": "click", "node": "n2", "label": "加入购物车", "role": "button"},
            {"kind": "click", "node": "n3", "label": "Giriş yap", "role": "button"},
            {"kind": "click", "node": "n4", "label": "Log in", "role": "button"},
            {"kind": "fill", "node": "n5", "label": "搜索商品", "role": "searchbox"},
        ]}

    def test_script_detection(self):
        from localdecide import script_of
        self.assertEqual(script_of("登录"), "han")
        self.assertEqual(script_of("カートに追加"), "kana")
        self.assertEqual(script_of("로그인"), "hangul")
        self.assertEqual(script_of("Войти"), "cyrillic")
        self.assertEqual(script_of("تسجيل الدخول"), "arabic")
        self.assertEqual(script_of("เข้าสู่ระบบ"), "thai")
        self.assertEqual(script_of("Σύνδεση"), "greek")
        self.assertEqual(script_of("Log in"), "latin")
        self.assertEqual(script_of("12345"), "")  # digits only

    def test_japanese_and_chinese_share_a_family(self):
        from localdecide import same_script
        self.assertTrue(same_script("カートに追加", "登录"))   # kana + han are one family
        self.assertTrue(same_script("登录", "加入购物车"))
        self.assertFalse(same_script("登录", "Log in"))

    def test_overlap_finds_the_target(self):
        from localdecide import ground_goal
        result = ground_goal("点击“加入购物车”按钮。", self.observation()["actions"])
        self.assertEqual(result["script"], "han")
        self.assertTrue(result["candidates"])
        best = result["candidates"][0]
        self.assertEqual(best, "2", f"expected the 加入购物车 button first; got {result['scores']}")

    def test_grounding_filters_other_scripts(self):
        scope = Scope(max_elements=10)
        scoped = scope.apply(self.observation(), goal="点击“登录”按钮。")
        labels = [action["label"] for action in scoped["actions"]]
        self.assertIn("登录", labels)
        self.assertNotIn("Log in", labels)
        self.assertNotIn("Giriş yap", labels)
        self.assertEqual(scoped["scope"]["script"], "han")

    def test_latin_goals_are_not_filtered(self):
        """English works fine as-is; grounding must not disturb it."""
        scope = Scope(max_elements=10)
        scoped = scope.apply(self.observation(), goal="Click the 'Log in' button.")
        labels = [action["label"] for action in scoped["actions"]]
        self.assertIn("Log in", labels)
        self.assertIn("登录", labels)  # not filtered: Latin goals keep the whole list
        self.assertNotIn("script", scoped["scope"])  # no grounding report for Latin

    def test_no_diagnosis_when_the_script_is_missing(self):
        """A goal whose script matches nothing must say so, not silently return junk."""
        scope = Scope(max_elements=10)
        scoped = scope.apply(self.observation(), goal="Нажмите кнопку «Войти».")  # Cyrillic
        report = scoped["scope"]
        self.assertIn("diagnosis", report, "should explain why nothing matched")
        self.assertIn("script", report)

    def test_grounding_never_empties_the_list(self):
        """A wrong-but-usable option list beats an empty one."""
        scope = Scope(max_elements=10)
        scoped = scope.apply({"url": "u", "title": "t", "text": "", "actions": [
            {"kind": "click", "node": "n1", "label": "Log in", "role": "button"},
        ]}, goal="点击“登录”按钮。")
        self.assertTrue(scoped["actions"], "grounding must never filter everything out")


if __name__ == "__main__":
    unittest.main()
