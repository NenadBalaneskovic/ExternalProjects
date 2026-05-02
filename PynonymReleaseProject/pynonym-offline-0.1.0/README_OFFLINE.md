# 📄 **README_OFFLINE.md**

```markdown
# Offline Installation: pynonym 0.1.0  
Version: 0.1.0  
Zielsystem: Linux (vpc23), Python 3.12  
Modus: Vollständig offline

---

## 📦 Inhalt des Offline‑Bundles

```
pynonym-offline-0.1.0/
│
├── install.sh
│
├── wheels/
│   ├── pynonym-0.1.0-py3-none-any.whl
│   ├── spacy-3.8.14-*.whl
│   ├── numpy-*.whl
│   ├── pandas-*.whl
│   ├── pydantic-*.whl
│   ├── pydantic_core-*.whl
│   ├── requests-*.whl
│   ├── charset_normalizer-*.whl
│   ├── six-*.whl
│   ├── tzdata-*.whl
│   ├── pytz-*.whl
│   ├── python_dateutil-*.whl
│   └── weitere Dependencies …
│
└── models/
    ├── de_core_news_md-3.8.0-py3-none-any.whl
    └── en_core_web_md-3.8.0-py3-none-any.whl
```

Alle Wheels sind **manylinux2014_x86_64** und kompatibel mit Python 3.12.

---

## 🧩 Voraussetzungen

Auf dem Zielsystem müssen vorhanden sein:

- Python **3.12**
- pip **3.12**
- Bash (Standard auf air-gapped Umgebungen)
- Schreibrechte im Installationsverzeichnis

Prüfen:

```bash
python3.12 --version
pip3.12 --version
```

---

## 🚀 Installation (offline)

1. Bundle auf das Zielsystem kopieren  
2. Entpacken:

```bash
tar -xzvf pynonym-offline-0.1.0.tar.gz
cd pynonym-offline-0.1.0
```

3. Installer ausführen:

```bash
chmod +x install.sh
./install.sh
```

Der Installer:

- installiert alle Wheels aus `./wheels`
- installiert die spaCy‑Modelle aus `./models`
- registriert die Modelle (`spacy link`)
- führt einen Funktionstest durch

---

## 🔍 Funktionstest (manuell)

Nach der Installation:

```bash
python3.12 - << 'EOF'
import spacy
import pynonym

nlp = spacy.load("de_core_news_md")
print("spaCy Modell geladen:", nlp.meta["name"])

print("pynonym Version:", pynonym.__version__)
EOF
```

Erwartete Ausgabe:

- Modellname: `de_core_news_md`
- Version: `0.1.0`

---

## 🛠 Fehlerbehebung

### pip findet keine Wheels
Ursache: falscher Pfad oder fehlende Rechte.

Lösung:

```bash
pip3.12 install --no-index --find-links=./wheels spacy
```

### spaCy‑Modell kann nicht geladen werden
Ursache: Modell nicht registriert.

Lösung:

```bash
python3.12 -m spacy link de_core_news_md de_core_news_md
```

### Python‑Version ist nicht 3.12
vpc23 kann mehrere Python‑Versionen haben.

Lösung:

```bash
alias python=python3.12
alias pip=pip3.12
```

---

## 📚 Hinweise

- Alle Wheels sind vollständig offline installierbar.
- Die spaCy‑Modelle sind plattformunabhängig (`py3-none-any`).
- Das Bundle enthält **keine** Internet‑Abhängigkeiten.
- Die Installation verändert keine Systempakete.

---

## ✔ Abschluss

Nach erfolgreicher Installation stehen folgende Komponenten offline zur Verfügung:

- pynonym 0.1.0
- spaCy 3.8.14
- de_core_news_md 3.8.0
- en_core_web_md 3.8.0
- pandas, numpy, Faker, requests, charset-normalizer, six, tzdata, pytz, python-dateutil
- alle benötigten C‑Extensions als manylinux‑Wheels

Das System ist damit vollständig offline‑fähig.
```

---

