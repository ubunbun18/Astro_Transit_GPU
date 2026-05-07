import subprocess
import pytest
import sys

def run_cli(args):
    cmd = [sys.executable, "-m", "astrotransit_gpu.cli"] + args
    return subprocess.run(cmd, capture_output=True, text=True)

def test_cli_help():
    res = run_cli(["--help"])
    assert res.returncode == 0
    assert "AstroTransit-GPU" in res.stdout

def test_cli_check():
    res = run_cli(["check"])
    assert res.returncode == 0
    assert "GPU" in res.stdout

@pytest.mark.gpu
def test_cli_compare_smoke():
    # Run compare with standard preset
    res = run_cli(["compare", "--target", "TIC 261136679", "--preset", "standard", "--n-runs", "1", "--out", "test_compare.md"])
    assert res.returncode == 0
    assert "report written to" in res.stdout

@pytest.mark.gpu
def test_cli_inject_smoke():
    # Renamed inject-run to inject in v1.0.0
    res = run_cli(["inject", "--target", "TIC 261136679", "--periods", "5.0", "--depths", "0.01", "--n-trials", "1", "--out", "test_inject.md"])
    assert res.returncode == 0
    assert "report written to" in res.stdout
