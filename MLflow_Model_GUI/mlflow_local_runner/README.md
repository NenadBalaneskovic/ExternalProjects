# 📄 **README.md**

```markdown
# MLflow Local Runner

Der **MLflow Local Runner** ist ein Desktop‑Tool zum lokalen Ausführen,
Analysieren und Protokollieren von Machine‑Learning‑Skripten.  
Er kombiniert eine moderne GUI (PySide6) mit MLflow‑Tracking und einer
robusten Artefaktverwaltung.

Das Tool ist ideal für:
- Data‑Science‑Prototyping
- lokale Experimente
- reproduzierbare ML‑Runs
- interne Modellvalidierung
- ML‑Workflows ohne Cloud‑Abhängigkeit

---

## 🚀 Features

- GUI‑basierte Bedienung (PySide6)
- Ausführen beliebiger Python‑ML‑Skripte
- Laden eigener CSV‑Datasets
- Automatisches Logging von:
  - Modellrepräsentation
  - Metriken
  - Artefakten
  - Dataset
- MLflow‑Integration (Tracking & Registry)
- Offline‑fähig (kein Server notwendig)
- Dunkles, modernes UI‑Design (styles.qss)
- Vollständige Testabdeckung (pytest + pytest‑qt)

---

## 📦 Installation

### Voraussetzungen
- Python 3.10+
- pip oder conda
- Optional: lokaler MLflow‑Server

### Installation

```bash
pip install -r requirements.txt
python main.py
```

Die Anwendung startet im GUI‑Modus.

---

## 🖥️ GUI‑Überblick

Die Anwendung besteht aus vier Panels:

1. **Upload Panel**  
   Auswahl von Skript (`.py`) und Dataset (`.csv`)

2. **Config Panel**  
   MLflow‑Konfiguration (Tracking URI, Registry URI, Artefakt‑Ordner)

3. **Run Panel**  
   Starten des ML‑Runs, Live‑Logs, Fehlerausgabe

4. **Results Panel**  
   Modellrepräsentation, Metriken, MLflow‑Links, Artefakte

---

## 🧪 Beispielskript

Ein Nutzer‑Skript muss folgende Marker ausgeben:

```
MODEL_READY
<Modellrepräsentation>

METRICS_READY
{"accuracy": 0.95, "f1_score": 0.93}
```

Beispiel: `example_script.py`

---

## 📊 Beispiel‑Dataset

Ein kleines synthetisches Dataset ist enthalten:

```
example_dataset.csv
```

Format:
- numerische Features
- letzte Spalte = Target

---

## 🔧 Architektur

```
mlflow_local_runner/
│
├── core/
│   ├── runner.py
│   ├── mlflow_client.py
│   ├── artifact_manager.py
│   ├── config_loader.py
│   └── script_template.py
│
├── gui/
│   ├── app_window.py
│   ├── upload_panel.py
│   ├── config_panel.py
│   ├── run_panel.py
│   └── results_panel.py
│
├── utils/
│   ├── logger.py
│   ├── paths.py
│   └── validators.py
│
├── assets/
│   └── styles.qss
│
└── tests/
    ├── test_gui_components.py
    ├── test_gui_end_to_end.py
    ├── test_runner.py
    ├── test_mlflow_client.py
    ├── test_validators.py
    └── test_config_loader.py
```

---

## 🧪 Tests

Alle Tests können mit pytest ausgeführt werden:

```bash
pytest -v
```

Enthalten sind:

- Unit‑Tests für alle GUI‑Panels  
- End‑to‑End‑GUI‑Test  
- Runner‑Tests (Subprozess‑Mocking)  
- MLflow‑Client‑Tests (vollständig gemockt)  
- Validator‑Tests  
- Config‑Loader‑Tests  

---

## 📘 Dokumentation

- `user_manual.md` – Benutzerhandbuch  
- `CHANGELOG.md` – Versionshistorie  

---

## 📄 Lizenz

Dieses Projekt verwendet eine interne oder Open‑Source‑Lizenz (z. B. MIT).

---

## 🙌 Autor

Entwickelt von **Nenad Balaneskovic**, 2026.

```

---
