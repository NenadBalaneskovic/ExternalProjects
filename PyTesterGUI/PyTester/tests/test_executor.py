"""
Tests for executor/ modules.

These tests validate:
- pytest runner command construction and result structure
- coverage runner command construction and parsing
- report collector aggregation
- log capture (subprocess + python logs)

All subprocess calls are mocked for safety.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import executor.pytest_runner as pytest_runner
import executor.coverage_runner as coverage_runner
import executor.report_collector as report_collector
import executor.log_capture as log_capture


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _dummy_settings():
    return {
        "execution": {
            "pytest": {
                "python_executable": "python",
                "max_duration_seconds": 5,
            },
            "coverage": {
                "python_executable": "python",
                "max_duration_seconds": 5,
            },
            "logs": {
                "capture_python_logs": True,
            },
        },
        "visualization": {
            "output_dir": "workspace/plots"
        }
    }


# ------------------------------------------------------------
# pytest_runner
# ------------------------------------------------------------
def test_pytest_runner_basic():
    settings = _dummy_settings()
    runner = pytest_runner.PytestRunner(settings)

    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.stdout = "1 passed"
    fake_proc.stderr = ""

    with patch("subprocess.run", return_value=fake_proc) as mock_run:
        result = runner.run([Path("test_sample.py")])

    assert result["status"] == "ok"
    assert result["exit_code"] == 0
    assert "passed" in result["stdout"]
    assert "test_sample.py" in result["files"]

    mock_run.assert_called_once()


def test_pytest_runner_timeout():
    settings = _dummy_settings()
    runner = pytest_runner.PytestRunner(settings)

    with patch("subprocess.run", side_effect=TimeoutError("timeout")):
        result = runner.run([Path("test_sample.py")])

    assert result["status"] == "error"
    assert result["exit_code"] == -1


# ------------------------------------------------------------
# coverage_runner
# ------------------------------------------------------------
def test_coverage_runner_basic():
    settings = _dummy_settings()
    runner = coverage_runner.CoverageRunner(settings)

    fake_run = MagicMock()
    fake_run.returncode = 0
    fake_run.stdout = ""
    fake_run.stderr = ""

    fake_report = MagicMock()
    fake_report.returncode = 0
    fake_report.stdout = "module.py 10 2 80% 12-13"
    fake_report.stderr = ""

    with patch("subprocess.run", side_effect=[fake_run, fake_report]):
        result = runner.run([Path("test_sample.py")], Path("."))

    assert result["status"] == "ok"
    assert result["total_coverage"] == 80.0
    assert "module.py" in result["files"]
    assert result["files"]["module.py"]["missing"] == [12, 13]


def test_coverage_runner_timeout():
    settings = _dummy_settings()
    runner = coverage_runner.CoverageRunner(settings)

    with patch("subprocess.run", side_effect=TimeoutError("timeout")):
        result = runner.run([Path("test_sample.py")], Path("."))

    assert result["status"] == "error"
    assert result["exit_code"] == -1


# ------------------------------------------------------------
# report_collector
# ------------------------------------------------------------
def test_report_collector_merges_results():
    collector = report_collector.ReportCollector(_dummy_settings())

    pytest_result = {
        "status": "ok",
        "exit_code": 0,
        "files": ["test_sample.py"],
        "stdout": "1 passed",
        "stderr": "",
    }

    coverage_result = {
        "status": "ok",
        "total_coverage": 80.0,
        "files": {"module.py": {"coverage": 80.0, "missing": [12]}},
        "stdout": "",
        "stderr": "",
    }

    report = collector.collect(pytest_result, coverage_result, logs="LOGS")

    assert report["status"] == "ok"
    assert report["summary"]["exit_code"] == 0
    assert report["summary"]["total_coverage"] == 80.0
    assert report["logs"] == "LOGS"


# ------------------------------------------------------------
# log_capture
# ------------------------------------------------------------
def test_log_capture_python_logs():
    lc = log_capture.LogCapture(_dummy_settings())

    lc.start_python_capture()

    import logging
    logging.getLogger().info("hello world")

    logs = lc.stop_python_capture()
    assert "hello world" in logs


def test_log_capture_subprocess_logs():
    lc = log_capture.LogCapture(_dummy_settings())

    logs = lc.capture_subprocess_logs("OUT", "ERR")
    assert "OUT" in logs
    assert "ERR" in logs


def test_log_capture_merge():
    lc = log_capture.LogCapture(_dummy_settings())

    merged = lc.merge("SUB", "PY")
    assert "SUB" in merged
    assert "PY" in merged
