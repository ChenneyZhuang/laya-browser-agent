"""CLI: `localdecide serve|decide|table|doctor`.

Three verbs, matching the three ways people use this:

* `doctor`  - is a local runtime even available on this machine?
* `decide`  - ask typed questions about a state (the primitive)
* `table`   - hand it an observation and a goal, get the chosen index
* `serve`   - run the HTTP dialects so other agents can point at this machine
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from .decider import Decider
from .page import build_element_table, table_to_questions


def _load_json(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def cmd_doctor(args: argparse.Namespace) -> int:
    import importlib.util
    import platform

    machine, system = platform.machine(), platform.system()
    print(f"python      {platform.python_version()} ({machine}, {system})")

    # Hardware context: what this device CAN run decides which extras make sense.
    chip = ""
    if system == "Darwin" and machine == "arm64":
        try:
            import subprocess
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=5).stdout.strip()
            memory_gb = round(int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                                 capture_output=True, text=True,
                                                 timeout=5).stdout.strip()) / 2**30)
            print(f"hardware    {chip}, {memory_gb} GB unified memory")
        except Exception:
            memory_gb = None
    else:
        memory_gb = None

    have_mlx = importlib.util.find_spec("laya_mlx") is not None
    have_torch = importlib.util.find_spec("laya") is not None
    print(f"laya-mlx    {'installed' if have_mlx else 'not installed'}"
          f"{'' if have_mlx else '   (Apple Silicon: pip install laya-mlx)'}")
    print(f"laya        {'installed' if have_torch else 'not installed'}"
          f"{'' if have_torch else '   (any platform: pip install laya)'}")
    for name, module in (("playwright", "playwright"), ("websocket-client", "websocket")):
        found = importlib.util.find_spec(module) is not None
        print(f"{name:<11} {'installed' if found else 'not installed'}")

    # Device-specific verdicts: below-M4 machines and 8 GB machines have real constraints,
    # and telling people plainly beats letting them discover it by crash.
    if system == "Darwin" and machine == "arm64" and not (have_mlx or have_torch):
        print("\nNo local decision runtime yet. On this Mac:")
        print("  pip install 'localdecide[mlx]'      # fastest path on Apple Silicon")
    elif system == "Darwin" and machine == "x86_64" and not (have_mlx or have_torch):
        print("\nNo local decision runtime yet. This is an Intel Mac - use the PyTorch runtime:")
        print("  pip install 'localdecide[torch]'    # laya-mlx does not run on Intel Macs")
    elif system == "Windows" and not (have_mlx or have_torch):
        print("\nNo local decision runtime yet. On Windows:")
        print("  pip install 'localdecide[torch]'")
    elif system == "Linux" and not (have_mlx or have_torch):
        print("\nNo local decision runtime yet. On Linux:")
        print("  pip install 'localdecide[torch]'    # add CUDA torch if you have a GPU")

    if memory_gb is not None and memory_gb < 10 and (have_mlx or have_torch):
        print(f"\nnote: {memory_gb} GB is tight for a 650 MB checkpoint plus a browser. "
              "The loop's subprocess design keeps one model OR one browser resident at a "
              "time, but close other heavy apps during live runs.")

    if not (have_mlx or have_torch):
        return 1
    try:
        decider = Decider()
        print(f"\nbackend     {getattr(decider.backend, 'name', '?')}")
        # A one-decision smoke test proves the checkpoint actually loads and answers,
        # which is the part that fails in practice (first download, version mismatches).
        result = decider.decide("The build finished and all tests passed.",
                                {"ok": {"type": "noul", "instructions": "The text reports a successful outcome"}})
        if result.ok and result.answers is not None:
            latency = result.answers.latency_ms
            print(f"smoke test  OK ({latency} ms, first call includes model load)")
            print("ready. try:  localdecide table --observation obs.json --goal 'Open the login page'")
            return 0
        print(f"smoke test  FAILED: {result.error}")
        return 1
    except Exception as error:
        print(f"\nbackend failed to resolve: {error}")
        return 1


def cmd_decide(args: argparse.Namespace) -> int:
    payload = _load_json(args.questions)
    state = _load_json(args.state) if args.state else ""
    decider = Decider(backend=args.backend, max_options_per_question=args.max_options)
    decision = decider.decide(state, payload)
    if not decision.ok:
        print(json.dumps({"ok": False, "error": decision.error}, ensure_ascii=False, indent=2))
        return 1
    assert decision.answers is not None
    print(json.dumps({"ok": True, "answers": decision.answers.raw,
                      "latency_ms": decision.latency_ms, "backend": decision.answers.backend,
                      "usage": decision.answers.usage}, ensure_ascii=False, indent=2))
    return 0


def cmd_table(args: argparse.Namespace) -> int:
    observation = _load_json(args.observation)
    table = build_element_table(observation)
    questions = table_to_questions(table, args.goal)
    decider = Decider(backend=args.backend, max_options_per_question=args.max_options)
    decision = decider.decide(table.state(text_chars=args.text_chars), questions)
    if not decision.ok:
        print(json.dumps({"ok": False, "error": decision.error}, ensure_ascii=False, indent=2))
        return 1
    assert decision.answers is not None
    answer = decision.answers
    operation = answer.choice("operation")
    out: Dict[str, Any] = {"ok": True, "operation": operation,
                           "confidence": answer.confidence("operation"),
                           "latency_ms": decision.latency_ms, "backend": answer.backend}
    if operation in ("CLICK", "TYPE_TEXT", "SELECT") and f"{operation.lower()}_target" in answer.raw:
        target = answer.choice(f"{operation.lower()}_target")
        element = table.by_index().get(target)
        out.update(target=target, label=(element.label if element else ""),
                   handle=(element.handle if element else None))
        out["confidence"] = min(out["confidence"], answer.confidence(f"{operation.lower()}_target"))
    if args.verbose:
        out["answers"] = answer.raw
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .serve import serve

    serve(host=args.host, port=args.port)
    return 0


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(prog="localdecide",
                                     description="Browser decisions from a local, open-weight System 1 model.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="check what is installed on this machine")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("decide", help="ask typed questions about a state")
    p.add_argument("--questions", required=True, help="JSON file (or - for stdin) with the questions")
    p.add_argument("--state", help="JSON file with the state (optional)")
    p.add_argument("--backend", help="backend name or URL; default auto")
    p.add_argument("--max-options", type=int, default=20, help="options per question before coarse-to-fine kicks in")
    p.set_defaults(func=cmd_decide)

    p = sub.add_parser("table", help="observation + goal -> chosen index")
    p.add_argument("--observation", required=True, help="JSON file (or -) with {url,title,text,actions|elements}")
    p.add_argument("--goal", required=True)
    p.add_argument("--backend")
    p.add_argument("--max-options", type=int, default=20)
    p.add_argument("--text-chars", type=int, default=1200)
    p.add_argument("--verbose", action="store_true", help="include every answer, not just the chosen one")
    p.set_defaults(func=cmd_table)

    p = sub.add_parser("serve", help="run the local HTTP service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8791)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
