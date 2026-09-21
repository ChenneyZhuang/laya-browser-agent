#!/usr/bin/env python3
"""Install the laya-browser-agent skills into every agent this machine has.

Detects Claude Code, Codex, Cursor, and Hermes by their skill directories, copies the
SKILL.md files into each, and reports what it did. --check previews; --uninstall reverses.

No dependencies, standard library only, safe to run repeatedly (idempotent).
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent
SKILLS = REPO / "skills"

# agent name -> (skill directory that exists, target subdirectory)
AGENTS = {
    "Claude Code": (Path.home() / ".claude" / "skills", None),
    "Codex": (Path.home() / ".codex" / "skills", None),
    "Cursor": (Path.home() / ".cursor" / "skills", None),
    "Hermes (project profile)": (Path.home() / ".hermes" / "profiles" / "project" / "skills", None),
    "Hermes (default)": (Path.home() / ".hermes" / "skills", None),
}

MARKER = ".installed-by-laya-browser-agent"


def installed_targets() -> list[tuple[Path, str]]:
    """The (skill_dir, agent) pairs whose agent directory exists."""
    found = []
    for agent, (base, _) in AGENTS.items():
        if base.parent.exists() or base.exists():
            found.append((base, agent))
    return found


def install(dry_run: bool = False) -> int:
    targets = installed_targets()
    if not targets:
        print("No known agent directories found on this machine.")
        print("Copy skills/ manually into your agent's skills folder.")
        return 1
    changed = 0
    for base, agent in targets:
        for skill in sorted(SKILLS.iterdir()):
            if not skill.is_dir():
                continue
            destination = base / skill.name
            print(f"  {agent}: {skill.name} -> {destination}")
            if not dry_run:
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(skill / "SKILL.md", destination / "SKILL.md")
                (destination / MARKER).write_text(REPO.name + "\n")
                changed += 1
    print(f"{'would install' if dry_run else 'installed'} {changed} skill(s) into {len(targets)} agent(s)")
    return 0


def uninstall() -> int:
    removed = 0
    for base, agent in installed_targets():
        for skill_dir in base.glob("*"):
            marker = skill_dir / MARKER
            if marker.exists():
                print(f"  {agent}: removing {skill_dir.name}")
                shutil.rmtree(skill_dir)
                removed += 1
    print(f"removed {removed} skill(s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="preview what would be installed")
    parser.add_argument("--uninstall", action="store_true", help="remove installed skills")
    args = parser.parse_args()
    if args.uninstall:
        return uninstall()
    return install(dry_run=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
