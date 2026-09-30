"""MCP server: expose the decision model to Claude Desktop, Cursor, and any MCP client.

Why this matters: Model Context Protocol is how agent stacks consume external tools now.
Serving the model over MCP means a Claude Desktop config block or a Cursor setting is the
whole integration - no code, no SDK, no fork. The same local model, the same guarantees:
nothing leaves the machine.

Two tools are exposed:

* `decide`     - typed questions about a state (the primitive)
* `page_decide` - an observation and a goal, get the chosen operation and element

Run it:

    localdecide-mcp                # stdio transport, the standard for desktop clients

Register it (Claude Desktop, `claude_desktop_config.json`):

    { "mcpServers": { "localdecide": { "command": "/path/to/localdecide-mcp" } } }

The protocol is implemented with nothing but stdio and JSON - no MCP SDK dependency - so
this works on any Python the package already supports.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict

from .decider import Decider
from .page import build_element_table, table_to_questions

# Protocol versions this server speaks, newest first. Per the MCP lifecycle
# spec (2025-06-18 §Version Negotiation): if the client requests a version we
# support, respond with the same one; otherwise respond with our latest.
SUPPORTED_PROTOCOL_VERSIONS = ["2025-06-18", "2024-11-05"]
PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]


def _package_version() -> str:
    """Same single source as localdecide.__version__ — metadata, not a constant."""
    from . import __version__
    return __version__


SERVER_INFO = {"name": "localdecide", "version": _package_version()}

TOOLS = [
    {
        "name": "decide",
        "description": (
            "Ask a local, open-weight System 1 decision model typed questions about a state. "
            "Returns calibrated probabilities, never generated text. Use for classification, "
            "routing, scoring, and yes/no judgments where the answer is one of a few options."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "state": {
                    "description": "The state to judge: a string, or a JSON object of facts.",
                    "type": ["string", "object"],
                },
                "questions": {
                    "description": (
                        "Questions keyed by name. Each: {type: choice|score|noul, instructions, "
                        "criteria}. choice takes a dict of option->description; score takes an "
                        "ordered list of levels; noul takes none."
                    ),
                    "type": "object",
                },
            },
            "required": ["state", "questions"],
        },
    },
    {
        "name": "page_decide",
        "description": (
            "Decide the next browser action: give a page observation (url, title, text, and an "
            "actions/elements list from a DOM or accessibility snapshot) plus a goal, and get "
            "the chosen operation (CLICK/TYPE_TEXT/SELECT/SCROLL/WAIT/DONE/BLOCKED) and the "
            "index of the element to act on. All local; the page never leaves this machine."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {"description": "What the agent is trying to do.", "type": "string"},
                "observation": {
                    "description": (
                        "Page observation: {url, title, text, actions: [{kind: click|fill|select, "
                        "label, role, node?}]} or {elements: [...]}."
                    ),
                    "type": "object",
                },
            },
            "required": ["goal", "observation"],
        },
    },
]


class _Session:
    """One stdio MCP session. Loads the model lazily, answers one request at a time."""

    def __init__(self) -> None:
        self._decider: Decider | None = None

    def decider(self) -> Decider:
        if self._decider is None:
            import os

            self._decider = Decider(max_options_per_question=int(os.environ.get("LOCALDECIDE_MAX_OPTIONS", "20")))
        return self._decider

    # -- handlers ----------------------------------------------------------

    def handle(self, message: Any) -> Dict[str, Any]:
        if not isinstance(message, dict):
            return self._error(None, -32600, "invalid request: expected an object")
        if message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            return self._error(message.get("id"), -32600, "invalid request")
        method = message["method"]
        request_id = message.get("id")
        notification = "id" not in message

        def reply(response: Dict[str, Any]) -> Dict[str, Any]:
            return {} if notification else response

        try:
            if method == "initialize":
                params = message.get("params", {})
                if not isinstance(params, dict):
                    return reply(self._error(request_id, -32602, "params must be an object"))
                requested = str(params.get("protocolVersion", "") or "")
                # Spec: same version if we support it, else our latest.
                version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else PROTOCOL_VERSION
                return reply(self._ok(request_id, {
                    "protocolVersion": version,
                    "capabilities": {"tools": {}},
                    "serverInfo": SERVER_INFO,
                }))
            if method == "notifications/initialized":
                return {}  # notification: no response
            if method == "tools/list":
                return reply(self._ok(request_id, {"tools": TOOLS}))
            if method == "tools/call":
                params = message.get("params", {})
                if not isinstance(params, dict):
                    return reply(self._error(request_id, -32602, "params must be an object"))
                try:
                    result = self._call_tool(params)
                except ValueError as error:
                    return reply(self._error(request_id, -32602, str(error)))
                return reply(self._ok(request_id, result))
            if method == "ping":
                return reply(self._ok(request_id, {}))
            return reply(self._error(request_id, -32601, f"method not found: {method}"))
        except Exception as error:
            return reply(self._error(request_id, -32603, f"{type(error).__name__}: {error}"))

    def _call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        if not isinstance(name, str) or not name:
            return self._tool_error("tool name is required")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be an object")
        if name == "decide":
            state = arguments.get("state", "")
            questions = arguments.get("questions")
            if not isinstance(state, (str, dict)) or not isinstance(questions, dict) or not questions:
                return self._tool_error("decide requires state string/object and non-empty questions object")
            decision = self.decider().decide(state, questions)
            if not decision.ok:
                return self._tool_error(f"decision failed open: {decision.error}")
            assert decision.answers is not None
            return {"content": [{"type": "text", "text": json.dumps(
                {"answers": decision.answers.raw, "latency_ms": decision.latency_ms,
                 "backend": decision.answers.backend}, ensure_ascii=False, default=str)}]}
        if name == "page_decide":
            goal = str(arguments.get("goal", "") or "")
            observation = arguments.get("observation")
            if not goal.strip() or not isinstance(observation, dict):
                return self._tool_error("page_decide requires a goal and observation object")
            table = build_element_table(observation)
            questions = table_to_questions(table, goal)
            decision = self.decider().decide(table.state(text_chars=1200), questions)
            if not decision.ok:
                return self._tool_error(f"decision failed open: {decision.error}")
            assert decision.answers is not None
            answers = decision.answers
            operation = answers.choice("operation")
            out: Dict[str, Any] = {"operation": operation,
                                   "confidence": answers.confidence("operation"),
                                   "latency_ms": decision.latency_ms, "backend": answers.backend}
            if operation in ("CLICK", "TYPE_TEXT", "SELECT") and f"{operation.lower()}_target" in answers.raw:
                target = answers.choice(f"{operation.lower()}_target")
                element = table.by_index().get(target)
                out.update(target=target, label=(element.label if element else ""),
                           handle=(element.handle if element else None))
                out["confidence"] = min(out["confidence"], answers.confidence(f"{operation.lower()}_target"))
            return {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False, default=str)}]}
        return self._tool_error(f"unknown tool: {name!r}")

    @staticmethod
    def _tool_error(message: str) -> Dict[str, Any]:
        return {"content": [{"type": "text", "text": message}], "isError": True}

    # -- framing -----------------------------------------------------------

    @staticmethod
    def _ok(request_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def serve(session: _Session | None = None) -> None:
    """Run the MCP server over stdio until stdin closes."""
    session = session or _Session()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            response = _Session._error(None, -32700, "parse error")
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
            continue
        response = session.handle(message)
        if response:  # notifications produce no response
            sys.stdout.write(json.dumps(response, ensure_ascii=False, default=str) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    serve()
