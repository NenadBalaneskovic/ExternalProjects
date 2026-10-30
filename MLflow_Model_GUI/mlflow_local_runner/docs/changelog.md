# Changelog
Alle nennenswerten Änderungen dieses Projekts werden in diesem Dokument festgehalten.

Das Format orientiert sich an [Keep a Changelog](https://keepachangelog.com/de/1.0.0/)
und verwendet **SemVer** (Semantic Versioning).

---

## [0.1.0] – 2026-05-09
### Added
- **GUI-Grundstruktur** mit folgenden Panels:
  - UploadPanel
  - ConfigPanel
  - RunPanel
  - ResultsPanel
  - AppWindow (Tab-Navigation, Signal-Routing)
- **Dark-Theme Stylesheet (`styles.qss`)**  
  Modernes, ruhiges UI-Design basierend auf VS Code / JupyterLab.

### Core-Module
- `runner.py`  
  - Subprozess-Handling  
  - Parsing von `MODEL_READY` und `METRICS_READY`  
  - Fehlerbehandlung (stderr, fehlende Marker)  
  - Integration mit ArtifactManager und MLflowClientWrapper  

- `mlflow_client.py`  
  - Logging von Metriken, Artefakten, Dataset  
  - Modellregistrierung  
  - Generierung von MLflow-Links  
  - Tokenfreie, offline-fähige Architektur  

- `artifact_manager.py`  
  - Run-spezifische Artefaktordner  
  - Speichern von Logs, Debug-Infos, Modellrepräsentation  
  - Kopieren externer Artefakte  
  - Cleanup-Funktion  

- `config_loader.py`  
  - Persistente JSON-Konfiguration  
  - Automatische Ordnererstellung  
  - Fehlerrobuste Lade-/Speicherlogik  

- `validators.py`  
  - Validierung von Dateien, Ordnern, URIs  
  - Spezialisierte Checks für `.py` und `.csv`  

- `logger.py`  
  - Zentrales Logging-System  
  - Rich‑Konsole + rotierende Logfiles  
  - Einmalige Initialisierung  

- `paths.py`  
  - Plattformübergreifende Pfadverwaltung  
  - App‑Config‑, Log‑ und Artefakt‑Verzeichnisse  

### Examples
- `example_script.py`  
  - Vollständiges ML‑Beispielskript  
  - Ausgabe über stdout‑Marker  
- `example_dataset.csv`  
  - Kleines synthetisches Klassifikations‑Dataset  

### Tests
- `test_gui_components.py`  
  - Unit‑Tests für alle Panels  
- `test_gui_end_to_end.py`  
  - Vollständiger GUI‑Workflow (E2E)  
- `test_runner.py`  
  - Mock‑basierte Tests für Subprozess‑Handling  
- `test_mlflow_client.py`  
  - Vollständiges Mocking von MLflow  
- `test_validators.py`  
  - Tests für alle Validatoren  
- `test_config_loader.py`  
  - Isolierte Tests für persistente Konfiguration  

---

## [Unreleased]
### Planned
- Erweiterte MLflow‑Konfiguration (Experiments, Tags)
- Unterstützung für Hyperparameter‑Tuning
- Live‑Log‑Streaming im RunPanel
- Export von Run‑Konfigurationen
- Plugin‑System für benutzerdefinierte Skripte
