"""CLI guidance tests that do not load a model runtime."""

from __future__ import annotations

import importlib.util
import platform
import subprocess
from argparse import Namespace
from types import SimpleNamespace

import pytest

from localdecide.cli import cmd_doctor


@pytest.mark.parametrize(
    ("system", "machine", "extra"),
    [("Linux", "x86_64", "torch"), ("Windows", "AMD64", "torch"), ("Darwin", "arm64", "mlx")],
)
def test_doctor_prints_a_working_install_path_before_pypi_release(monkeypatch, capsys, system, machine, extra):
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(
        stdout="Apple M4" if _args[0][-1] == "machdep.cpu.brand_string" else str(16 * 1024**3)
    ))

    result = cmd_doctor(Namespace())

    output = capsys.readouterr().out
    assert result == 1
    assert "git clone https://github.com/ChenneyZhuang/laya-browser-agent" in output
    assert f"pip install -e '.[{extra}]'" in output
    assert "localdecide[" not in output
