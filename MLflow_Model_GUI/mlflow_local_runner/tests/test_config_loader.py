# mlflow_local_runner/tests/test_config_loader.py
"""
test_config_loader.py – Tests für ConfigLoader
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from core.config_loader import ConfigLoader


# ---------------------------------------------------------
# FIXTURE: isolierter ConfigLoader mit tmp_path
# ---------------------------------------------------------

@pytest.fixture
def isolated_loader(tmp_path, monkeypatch):
    """
    Leitet get_config_dir() auf ein temporäres Verzeichnis um,
    damit keine echten User-Konfigurationsdateien verwendet werden.
    """
    monkeypatch.setenv("MLFLOW_LOCAL_RUNNER_TESTING", "1")

    # get_config_dir mocken
    with patch(
        "mlflow_local_runner.core.config_loader.get_config_dir",
        return_value=str(tmp_path)
    ):
        loader = ConfigLoader()
        yield loader


# ---------------------------------------------------------
# TEST: Laden ohne existierende Datei
# ---------------------------------------------------------

def test_load_empty_config(isolated_loader):
    config = isolated_loader.load()
    assert config == {}  # Standardverhalten


# ---------------------------------------------------------
# TEST: Speichern + Laden
# ---------------------------------------------------------

def test_save_and_load_config(isolated_loader):
    cfg = {
        "tracking_uri": "http://localhost:5000",
        "registry_uri": "http://localhost:5000",
        "artifact_dir": "/tmp/artifacts"
    }

    isolated_loader.save(cfg)

    loaded = isolated_loader.load()
    assert loaded == cfg


# ---------------------------------------------------------
# TEST: Fehler beim Speichern (z. B. Schreibschutz)
# ---------------------------------------------------------

def test_save_config_error(isolated_loader, monkeypatch):
    def raise_io_error(*args, **kwargs):
        raise IOError("Write error")

    # open() mocken → Fehler erzwingen
    monkeypatch.setattr("builtins.open", raise_io_error)

    with pytest.raises(RuntimeError):
        isolated_loader.save({"a": 1})


# ---------------------------------------------------------
# TEST: Config-Datei wird korrekt angelegt
# ---------------------------------------------------------

def test_config_file_created(isolated_loader):
    isolated_loader.save({"x": 1})

    assert isolated_loader.config_path.exists()
    assert json.loads(isolated_loader.config_path.read_text()) == {"x": 1}
