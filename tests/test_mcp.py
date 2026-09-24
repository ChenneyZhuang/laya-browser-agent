"""Tests for the MCP server — protocol conformance, no model checkpoint required.

Why this file exists: mcp_server.py had 0% test coverage; every "version
negotiation works" claim was verified by hand with a one-off script and never
re-run. Protocol code is exactly the kind of thing where a regression is
invisible until a client breaks. These tests pin the spec behaviour from the
MCP lifecycle document (2025-06-18, §Version Negotiation) plus the tools
surface, using an injected fake decider.
"""

from __future__ import annotations

import json
import unittest

from localdecide.decider import Decider
from localdecide.mcp_server import PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS, _Session


class _StaticBackend:
    """First option wins, calibrated envelope."""

    name = "static-test"

    def answer(self, state, questions):
        answers = {}
        for name, spec in questions.items():
            keys = [str(k) for k in spec.get("criteria", {})]
            if not keys:
                answers[name] = {"type": "noul", "noul": 0.5, "confidence": 0.5}
                continue
            key = keys[0]
            share = round(0.9 if len(keys) == 1 else 0.9, 6)
            probabilities = {k: (share if k == key else round((1 - share) / (len(keys) - 1), 6))
                             for k in keys}
            probabilities[key] = 1.0 - sum(v for k, v in probabilities.items() if k != key)
            answers[name] = {"type": "choice", "choice": key, "probabilities": probabilities,
                             "confidence": probabilities[key], "action": {"act_probability": 1.0}}
        return {"answers": answers, "usage": {}, "latency_ms": 1, "backend": self.name}


def _session() -> _Session:
    session = _Session()
    session._decider = Decider(backend=_StaticBackend())
    return session


class TestMCPConformance(unittest.TestCase):
    # ---- lifecycle: version negotiation (MCP spec 2025-06-18 §Version Negotiation)

    def test_initialize_echoes_supported_version(self):
        current = SUPPORTED_PROTOCOL_VERSIONS[0]
        response = _session().handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                      "params": {"protocolVersion": current}})
        self.assertEqual(response["result"]["protocolVersion"], current)

    def test_initialize_legacy_client_keeps_its_version(self):
        response = _session().handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                      "params": {"protocolVersion": "2024-11-05"}})
        self.assertEqual(response["result"]["protocolVersion"], "2024-11-05")

    def test_initialize_unknown_version_falls_back_to_latest(self):
        response = _session().handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                      "params": {"protocolVersion": "1999-01-01"}})
        self.assertEqual(response["result"]["protocolVersion"], PROTOCOL_VERSION)

    def test_initialize_reports_tools_capability_and_server_info(self):
        response = _session().handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                      "params": {"protocolVersion": PROTOCOL_VERSION}})
        result = response["result"]
        self.assertIn("tools", result["capabilities"])
        self.assertEqual(result["serverInfo"]["name"], "localdecide")

    def test_initialized_notification_produces_no_response(self):
        self.assertEqual(_session().handle({"jsonrpc": "2.0",
                                            "method": "notifications/initialized"}), {})

    # ---- tools surface ----------------------------------------------------------

    def test_tools_list_shape(self):
        result = _session().handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]
        names = [tool["name"] for tool in result["tools"]]
        self.assertEqual(sorted(names), ["decide", "page_decide"])
        for tool in result["tools"]:
            self.assertIn("inputSchema", tool)
            self.assertEqual(tool["inputSchema"]["type"], "object")

    def test_ping(self):
        response = _session().handle({"jsonrpc": "2.0", "id": 3, "method": "ping"})
        self.assertEqual(response["result"], {})

    def test_unknown_method_is_jsonrpc_error(self):
        response = _session().handle({"jsonrpc": "2.0", "id": 4, "method": "no/such"})
        self.assertEqual(response["error"]["code"], -32601)

    # ---- tool calls ---------------------------------------------------------------

    def test_decide_tool_returns_content_json(self):
        response = _session().handle({
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": {"name": "decide", "arguments": {
                "state": "pool pump is broken",
                "questions": {"q": {"type": "choice", "criteria": {"a": "Pool", "b": "Roof"}}}}}})
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertEqual(payload["answers"]["q"]["choice"], "a")
        self.assertEqual(payload["backend"], "static-test")

    def test_page_decide_requires_goal(self):
        response = _session().handle({
            "jsonrpc": "2.0", "id": 6, "method": "tools/call",
            "params": {"name": "page_decide", "arguments": {"observation": {}}}})
        self.assertTrue(response["result"]["isError"])
        self.assertIn("goal", response["result"]["content"][0]["text"])

    def test_page_decide_returns_operation(self):
        response = _session().handle({
            "jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {"name": "page_decide", "arguments": {
                "goal": "Click Submit",
                "observation": {"url": "https://example.com",
                                "actions": [{"kind": "click", "node": "n1", "label": "Submit"}]}}}})
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertEqual(payload["operation"], "CLICK")
        self.assertEqual(payload["backend"], "static-test")

    def test_unknown_tool_is_tool_error(self):
        response = _session().handle({
            "jsonrpc": "2.0", "id": 8, "method": "tools/call",
            "params": {"name": "no_such_tool", "arguments": {}}})
        self.assertTrue(response["result"]["isError"])

    def test_internal_error_is_jsonrpc_error(self):
        class _Broken(Decider):
            def decide(self, state, questions):
                raise RuntimeError("boom")

        session = _session()
        session._decider = _Broken(backend=_StaticBackend())
        response = session.handle({
            "jsonrpc": "2.0", "id": 9, "method": "tools/call",
            "params": {"name": "decide", "arguments": {
                "state": "x", "questions": {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}}}})
        self.assertEqual(response["error"]["code"], -32603)
        self.assertIn("boom", response["error"]["message"])


if __name__ == "__main__":
    unittest.main()
