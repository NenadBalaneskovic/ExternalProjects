import sys
from pathlib import Path

# 1. Projekt-Root finden
project_root = Path(__file__).resolve().parent
src_path = project_root / "mlflow_local_runner" / "src"

# 2. src zum Python-Pfad hinzufügen
sys.path.append(str(src_path))

# 3. GUI starten
from PySide6.QtWidgets import QApplication
from gui.app_window import AppWindow

app = QApplication(sys.argv)
window = AppWindow(config={})
window.show()
app.exec()
