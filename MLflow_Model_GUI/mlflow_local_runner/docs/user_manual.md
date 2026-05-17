# 📘 **user_manual.md**

```markdown
# MLflow Local Runner – User Manual

Dieses Handbuch beschreibt Installation, Bedienung und typische Workflows
für den **MLflow Local Runner** – ein Desktop‑Tool zum lokalen Ausführen,
Analysieren und Protokollieren von Machine‑Learning‑Skripten.

---

## 1. Überblick

Der MLflow Local Runner ermöglicht:

- Ausführen beliebiger Python‑ML‑Skripte
- Laden eigener CSV‑Datasets
- Automatisches Logging von:
  - Modellrepräsentation
  - Metriken
  - Artefakten
  - Dataset
- Integration mit MLflow Tracking & Registry
- GUI‑basierte Bedienung (PySide6)
- Offline‑Fähigkeit (kein MLflow‑Server notwendig)

Die Anwendung ist ideal für:

- Data‑Science‑Prototyping  
- lokale Experimente  
- reproduzierbare ML‑Runs  
- interne Modellvalidierung  

---

## 2. Installation

### 2.1 Voraussetzungen

- Python 3.10+
- pip oder conda
- Optional: lokaler MLflow‑Server

### 2.2 Installation

```bash
pip install -r requirements.txt
python main.py
```

Die Anwendung startet im GUI‑Modus.

---

## 3. Benutzeroberfläche

Die GUI besteht aus vier Hauptbereichen:

1. **Upload** – Auswahl von Skript & Dataset  
2. **Config** – MLflow‑Konfiguration  
3. **Run** – Ausführen des Skripts  
4. **Results** – Anzeige der Ergebnisse  

### 3.1 Upload Panel

Hier wählen Sie:

- ein Python‑Skript (`.py`)
- ein Dataset (`.csv`)

Beide Felder müssen ausgefüllt sein, bevor ein Run gestartet werden kann.

### 3.2 Config Panel

Hier konfigurieren Sie:

- Tracking URI  
- Registry URI  
- Artefakt‑Ordner  

Die Konfiguration wird automatisch gespeichert.

### 3.3 Run Panel

Hier starten Sie den ML‑Run.

Das Panel zeigt:

- Live‑Logs des Skripts  
- Statusmeldungen  
- Fehlerausgaben  

### 3.4 Results Panel

Nach einem erfolgreichen Run werden angezeigt:

- Modellrepräsentation  
- Metriken (Accuracy, F1, etc.)  
- MLflow‑Links  
- Artefakt‑Pfad  

---

## 4. Workflow

### Schritt 1 – Skript auswählen

Klicken Sie auf **„Select Script“** und wählen Sie eine `.py`‑Datei.

Das Skript muss:

- über `stdout` die Marker `MODEL_READY` und `METRICS_READY` ausgeben  
- Metriken als JSON ausgeben  
- das Dataset über die ENV‑Variable `DATASET_PATH` laden  

Beispielskript: `example_script.py`

### Schritt 2 – Dataset auswählen

Klicken Sie auf **„Select Dataset“** und wählen Sie eine `.csv`‑Datei.

Das Dataset muss:

- numerische Features enthalten  
- die Zielvariable in der letzten Spalte haben  

Beispiel: `example_dataset.csv`

### Schritt 3 – MLflow konfigurieren

Im **Config Panel**:

- Tracking URI setzen (z. B. `http://localhost:5000`)
- Registry URI setzen
- Artefakt‑Ordner definieren

### Schritt 4 – Run starten

Im **Run Panel**:

- auf **„Start Run“** klicken  
- Logs erscheinen live  
- Modell & Metriken werden automatisch erkannt  

### Schritt 5 – Ergebnisse ansehen

Im **Results Panel**:

- Modellrepräsentation  
- Metriken  
- MLflow‑Links  
- Artefakt‑Ordner  

---

## 5. Skriptanforderungen

Ein Nutzer‑Skript muss:

1. Das Dataset über `DATASET_PATH` laden  
2. Ein Modell trainieren  
3. Metriken berechnen  
4. Folgende Marker ausgeben:

```
MODEL_READY
<Modellrepräsentation>

METRICS_READY
{"accuracy": 0.95, "f1_score": 0.93}
```

Beispiel: `example_script.py`

---

## 6. Artefaktverwaltung

Der Local Runner speichert:

- Modellrepräsentation (`model_repr.txt`)
- Run‑Log (`run_log.txt`)
- Debug‑Info (`debug_info.txt`)
- kopierte Artefakte (`artifacts/`)

Alle Artefakte liegen unter:

```
~/.config/mlflow_local_runner/artifacts/
```

oder unter Windows:

```
C:/Users/<User>/AppData/Roaming/mlflow_local_runner/artifacts/
```

---

## 7. Fehlersuche

### 7.1 Skript liefert keine Metriken

Ursache: `METRICS_READY` fehlt.

Lösung: Marker korrekt ausgeben.

### 7.2 Skript bricht ab

Ursache: Fehler im Python‑Skript.

Lösung: stderr im RunPanel prüfen.

### 7.3 MLflow‑Logging schlägt fehl

Ursache: Tracking‑URI falsch.

Lösung: Config Panel prüfen.

---

## 8. Tastenkombinationen

| Aktion | Shortcut |
|--------|----------|
| Run starten | **Ctrl + R** |
| Script auswählen | **Ctrl + O** |
| Dataset auswählen | **Ctrl + D** |

---

## 9. Support

Bei Fragen oder Problemen:

- Logs prüfen (`logs/mlflow_local_runner.log`)
- Artefakte prüfen
- Skript‑Ausgabe validieren

---

## 10. Lizenz

Dieses Projekt verwendet eine interne Lizenz oder eine Open‑Source‑Lizenz (z. B. MIT).

```

---
