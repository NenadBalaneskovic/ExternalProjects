"""
environment_check.py
--------------------
Environment inspection module for the ONNX Model Generator GUI.

This module performs dependency checks for:
- Python version
- ONNX
- ONNX Runtime
- Torch
- scikit-learn
- TensorFlow
- MLServer
- Triton client
- Git
- Podman

It returns a dictionary with boolean flags that the GUI can use
to update status indicators.

This module is intentionally lightweight and safe to call from
PySimpleGUI or PowerShell.
"""

import subprocess
import sys


# ---------------------------------------------------------------------------
# Helper: Run a command and capture success/failure
# ---------------------------------------------------------------------------
def _check(cmd):
    """
    Run a shell command and return True if it succeeds, False otherwise.
    """
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )
        return result.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Check Python version
# ---------------------------------------------------------------------------
def check_python():
    """
    Check if Python >= 3.10 is installed.
    """
    major = sys.version_info.major
    minor = sys.version_info.minor
    return (major == 3 and minor >= 10) or (major > 3)


# ---------------------------------------------------------------------------
# Check pip packages
# ---------------------------------------------------------------------------
def check_pip_package(pkg_name):
    """
    Check if a pip package is installed.
    """
    return _check(f"pip show {pkg_name}")


# ---------------------------------------------------------------------------
# Check Git
# ---------------------------------------------------------------------------
def check_git():
    return _check("git --version")


# ---------------------------------------------------------------------------
# Check Podman
# ---------------------------------------------------------------------------
def check_podman():
    return _check("podman --version")


# ---------------------------------------------------------------------------
# Main environment check
# ---------------------------------------------------------------------------
def check_all():
    """
    Perform all environment checks and return a dictionary with results.
    """

    return {
        "python": check_python(),
        "onnx": check_pip_package("onnx"),
        "onnxruntime": check_pip_package("onnxruntime"),
        "torch": check_pip_package("torch"),
        "sklearn": check_pip_package("scikit-learn"),
        "tensorflow": check_pip_package("tensorflow"),
        "mlserver": check_pip_package("mlserver"),
        "tritonclient": check_pip_package("tritonclient"),
        "git": check_git(),
        "podman": check_podman(),
    }


# ---------------------------------------------------------------------------
# CLI usage (optional)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    print(json.dumps(check_all(), indent=4))
