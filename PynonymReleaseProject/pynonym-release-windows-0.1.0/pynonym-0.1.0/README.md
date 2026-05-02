# 📄 **README‑Abschnitt „Windows‑Installation“**

```markdown
# Installation unter Windows 11

Dieses Release enthält eine vollständig Windows‑kompatible Version von **pynonym**.
Alle Funktionen zur Text‑ und Tabellen‑Anonymisierung (spaCy + Faker) sind verfügbar.
Die Privacy‑Metriken (k‑Anonymität, l‑Diversität, t‑Closeness) sind unter Windows
deaktiviert, da `pycanon` für Python 3.13 nicht verfügbar ist.

Unter Linux/JupyterHub können diese Metriken über `pynonym[privacy]` aktiviert werden.

---

## 1. Voraussetzungen

- Windows 11
- Python 3.12 oder 3.13
- pip installiert
- spaCy‑Modelle liegen im Release‑Ordner (`models/`)

Prüfen:

```powershell
python --version
pip --version
```

---

## 2. Installation

Wechsle in den Release‑Ordner:

```powershell
cd .\pynonym-release-windows-0.1.0
```

Führe den Installer aus:

```powershell
.\install.ps1
```

Der Installer führt folgende Schritte aus:

1. Installation der spaCy‑Modelle (DE + EN)
2. Installation des pynonym‑Wheels
3. Fallback‑Installation aus dem Source‑Tarball

---

## 3. Smoke‑Test ausführen

Nach der Installation:

```powershell
python .\smoke-test\smoke_test.py
```

Der Test prüft:

- Text‑Anonymisierung
- Tabellen‑Anonymisierung
- Pseudonymisierung
- spaCy‑Modelle
- deaktivierte Privacy‑Metriken unter Windows

---

## 4. Nutzung im eigenen Code

### Text-Anonymisierung

```python
from pynonym import anonymize_text

text = "Angela Merkel traf Olaf Scholz in Berlin."
print(anonymize_text(text))
```

### Tabellen-Anonymisierung

```python
import pandas as pd
from pynonym.tables import TableAnonymizationConfig, anonymize_dataframe

df = pd.DataFrame({
    "Name": ["Angela Merkel", "Olaf Scholz"],
    "Stadt": ["Berlin", "Hamburg"],
    "Diagnose": ["A", "B"],
})

config = TableAnonymizationConfig(
    quasi_identifiers=["Stadt"],
    sensitive_attributes=["Diagnose"],
    pseudonymize_columns=["Name"],
    k=2,
    l=2,
    t=0.2,
    language="de",
    seed=42,
)

df_anonym = anonymize_dataframe(df, config)
print(df_anonym)
```

---

## 5. Hinweis zu Privacy-Metriken

Unter Windows:

- `k_anonymity`
- `l_diversity`
- `t_closeness`

sind **deaktiviert** und liefern strukturierte Rückgaben:

```python
{
  "metric": "k-anonymity",
  "value": None,
  "status": "pycanon_not_available",
  "message": "k-anonymity ist unter Windows deaktiviert (pycanon nicht installiert)."
}
```

Unter Linux/JupyterHub können die Metriken aktiviert werden:

```bash
pip install "pynonym[privacy]"
```

---

## 6. Linux-Installation (optional)

Für vollständige Privacy‑Metriken:

```bash
pip install pynonym-0.1.0-py3-none-any.whl
pip install "pynonym[privacy]"
```

---

## 7. Support

Bei Fragen oder Problemen:
- Windows‑Installation: lokale Python‑Umgebung prüfen
- Linux‑Installation: pycanon‑Support aktivieren
```

---

