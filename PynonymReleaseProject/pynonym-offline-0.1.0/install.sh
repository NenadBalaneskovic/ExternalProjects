#!/bin/bash
set -e

echo "=== Offline Installation von pynonym ==="

# Auto-detect Python
if command -v python3.12 >/dev/null 2>&1; then
    PY=python3.12
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    PY=python
fi

# Auto-detect pip
if command -v pip3.12 >/dev/null 2>&1; then
    PIP=pip3.12
elif command -v pip3 >/dev/null 2>&1; then
    PIP=pip3
else
    PIP=pip
fi

echo "Verwende Python: $($PY --version)"
echo "Verwende Pip: $($PIP --version)"

# 1. Wheels installieren
echo "Installiere Python Wheels..."
$PIP install --no-index --find-links=./wheels ./wheels/*.whl

# 2. spaCy-Modelle installieren
echo "Installiere spaCy Modelle..."
$PIP install --no-index --find-links=./models ./models/*.whl

# 3. spaCy Modelle registrieren
echo "Registriere spaCy Modelle..."
$PY - << EOF
import spacy
import subprocess

models = ["de_core_news_md", "en_core_web_md"]

for m in models:
    print(f"Registriere Modell: {m}")
    subprocess.run(["$PY", "-m", "spacy", "link", m, m], check=False)

print("Modelle erfolgreich registriert.")
EOF

# 4. Test
echo "Prüfe spaCy..."
$PY - << EOF
import spacy
nlp = spacy.load("de_core_news_md")
print("spaCy Modell geladen:", nlp.meta["name"])
EOF

echo "=== Installation abgeschlossen ==="
