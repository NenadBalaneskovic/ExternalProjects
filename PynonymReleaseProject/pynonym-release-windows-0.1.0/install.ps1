Write-Host "=== Installing spaCy models ==="

# Install DE model
pip install ".\models\de_core_news_md-3.8.0-py3-none-any.whl"

# Install EN model
pip install ".\models\en_core_web_md-3.8.0-py3-none-any.whl"


Write-Host "=== Installing pynonym ==="

# Install wheel first (fastest)
pip install ".\pynonym-0.1.0-py3-none-any.whl"

# Fallback: install from source if wheel fails
pip install ".\pynonym-0.1.0.tar.gz"


Write-Host ""
Write-Host "=== Installation complete ==="
Write-Host "Hinweis:"
Write-Host " - pycanon ist unter Windows nicht verfügbar."
Write-Host " - k/l/t-Metriken sind deaktiviert."
Write-Host " - Alle anderen Funktionen (Text, Tabellen, Faker, spaCy) funktionieren vollständig."
