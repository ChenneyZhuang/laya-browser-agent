"""Regression tests for the memory-gated live-test launcher."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "posix" or shutil.which("bash") is None,
                    reason="bash on a POSIX runner is required to test the macOS live-test launcher")
def test_live_batch_script_fails_when_a_pytest_batch_fails(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    tests = repo / "tests"
    venv_bin = repo / ".venv" / "bin"
    fake_bin = tmp_path / "fake-bin"
    scripts.mkdir(parents=True)
    tests.mkdir()
    (tests / "test_live.py").write_text("# synthetic source checkout marker\n", encoding="utf-8")
    (tests / "_browser_child.py").write_text("# synthetic source checkout marker\n", encoding="utf-8")
    venv_bin.mkdir(parents=True)
    fake_bin.mkdir()

    source_script = Path(__file__).resolve().parents[1] / "scripts" / "run_live_batched.sh"
    shutil.copyfile(source_script, scripts / "run_live_batched.sh")

    calls = tmp_path / "pytest-calls.txt"
    fake_python = venv_bin / "python"
    fake_python.write_text("#!/bin/sh\nprintf 'called\\n' >> \"$CALL_LOG\"\nexit 23\n", encoding="utf-8")
    fake_python.chmod(0o755)

    stubs = {
        "memory_pressure": "#!/bin/sh\nprintf 'System-wide memory free percentage: 90%%\\n'\n",
        "sysctl": "#!/bin/sh\nprintf 'vm.swapusage: total = 1024.00M used = 0.00M free = 1024.00M\\n'\n",
        "sleep": "#!/bin/sh\nexit 0\n",
    }
    for name, content in stubs.items():
        path = fake_bin / name
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)

    env = os.environ.copy()
    env["CALL_LOG"] = str(calls)
    env["PATH"] = str(fake_bin) + os.pathsep + env.get("PATH", "")
    result = subprocess.run([shutil.which("bash"), str(scripts / "run_live_batched.sh")],
                            env=env, capture_output=True, text=True, timeout=15)

    assert result.returncode != 0, result.stdout + result.stderr
    assert calls.read_text(encoding="utf-8").splitlines() == ["called"]


@pytest.mark.skipif(os.name != "posix" or shutil.which("bash") is None,
                    reason="bash on a POSIX runner is required to test the macOS live-test launcher")
def test_live_batch_script_honors_explicit_python_without_a_venv(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    tests = repo / "tests"
    fake_bin = tmp_path / "fake-bin"
    scripts.mkdir(parents=True)
    tests.mkdir()
    (tests / "test_live.py").write_text("# synthetic source checkout marker\n", encoding="utf-8")
    (tests / "_browser_child.py").write_text("# synthetic source checkout marker\n", encoding="utf-8")
    fake_bin.mkdir()

    source_script = Path(__file__).resolve().parents[1] / "scripts" / "run_live_batched.sh"
    shutil.copyfile(source_script, scripts / "run_live_batched.sh")

    calls = tmp_path / "pytest-calls.txt"
    fake_python = tmp_path / "explicit-python"
    fake_python.write_text("#!/bin/sh\nprintf 'called\\n' >> \"$CALL_LOG\"\nexit 23\n", encoding="utf-8")
    fake_python.chmod(0o755)

    stubs = {
        "memory_pressure": "#!/bin/sh\nprintf 'System-wide memory free percentage: 90%%\\n'\n",
        "sysctl": "#!/bin/sh\nprintf 'vm.swapusage: total = 1024.00M used = 0.00M free = 1024.00M\\n'\n",
        "sleep": "#!/bin/sh\nexit 0\n",
    }
    for name, content in stubs.items():
        path = fake_bin / name
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)

    env = os.environ.copy()
    env["CALL_LOG"] = str(calls)
    env["PYTHON"] = str(fake_python)
    env["PATH"] = str(fake_bin) + os.pathsep + env.get("PATH", "")
    result = subprocess.run([shutil.which("bash"), str(scripts / "run_live_batched.sh")],
                            env=env, capture_output=True, text=True, timeout=15)

    assert result.returncode != 0, result.stdout + result.stderr
    assert calls.read_text(encoding="utf-8").splitlines() == ["called"]


@pytest.mark.skipif(os.name != "posix" or shutil.which("bash") is None,
                    reason="bash on a POSIX runner is required to test the macOS live-test launcher")
def test_live_batch_script_explains_source_checkout_requirement(tmp_path):
    repo = tmp_path / "installed"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    source_script = Path(__file__).resolve().parents[1] / "scripts" / "run_live_batched.sh"
    shutil.copyfile(source_script, scripts / "run_live_batched.sh")

    result = subprocess.run([shutil.which("bash"), str(scripts / "run_live_batched.sh")],
                            capture_output=True, text=True, timeout=15)

    assert result.returncode == 2
    assert "require a source checkout" in result.stderr
    assert "installed standalone packages do not include tests" in result.stderr
