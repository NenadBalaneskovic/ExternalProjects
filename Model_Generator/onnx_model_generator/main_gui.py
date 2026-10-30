"""
main_gui.py — Vollständige, funktionsfähige GUI für den ONNX Model Generator
----------------------------------------------------------------------------

Features:
- Dynamische Environment-Status-Anzeige (✔ / ✖)
- Buttons: Install TF, Install Triton, Install Git, Install Podman
- Button: Fix Environment (installiert alle fehlenden Komponenten)
- UTF-8-sichere PowerShell-Ausführung
- Fortschrittsbalken
- Live-Log-Konsole
- Automatische Paket-Nachinstallation
- Git/Podman-Installation über winget
- Robustes Fehler-Handling
"""

import PySimpleGUI as sg
import subprocess
import os

# ---------------------------------------------------------------------------
# Status Symbole
# ---------------------------------------------------------------------------
STATUS_OK = "✔"
STATUS_FAIL = "✖"
STATUS_UNKNOWN = "?"

# ---------------------------------------------------------------------------
# GUI Theme
# ---------------------------------------------------------------------------
sg.theme("DarkBlue3")

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
def log(window, text):
    window["-LOG-"].print(text)

# ---------------------------------------------------------------------------
# Progress bar helper
# ---------------------------------------------------------------------------
def update_progress(window, value):
    window["-PROGRESS-"].update(value)
    window.refresh()

# ---------------------------------------------------------------------------
# Auto-detect Python files
# ---------------------------------------------------------------------------
def detect_python_files(folder):
    if not folder or not os.path.isdir(folder):
        return []
    return [f for f in os.listdir(folder) if f.endswith(".py")]

# ---------------------------------------------------------------------------
# UTF‑8‑sichere PowerShell-Ausführung
# ---------------------------------------------------------------------------
def run_powershell(window, script_path, args):
    cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", script_path] + args

    log(window, f"> Running PowerShell script: {script_path}")
    log(window, f"> FULL PATH: {os.path.abspath(script_path)}")
    update_progress(window, 5)

    # ⭐ FIX: Remove encoding="utf-8" to prevent cp1252 crash on ✔
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        bufsize=1
    )

    # stdout lesen
    while True:
        line = process.stdout.readline()
        if not line:
            break
        line = line.rstrip()
        if line:
            log(window, line)

        # Fortschrittsheuristik
        if "Checking" in line:
            update_progress(window, 20)
        elif "Installing" in line:
            update_progress(window, 40)
        elif "Export" in line or "conversion" in line:
            update_progress(window, 70)
        elif "complete" in line.lower():
            update_progress(window, 100)

    # stderr lesen
    while True:
        err = process.stderr.readline()
        if not err:
            break
        err = err.rstrip()
        if err:
            log(window, "ERROR: " + err)

    process.wait()
    return process.returncode

# ---------------------------------------------------------------------------
# Environment Status Update
# ---------------------------------------------------------------------------
def update_environment_status(window, log_text):

    def set_status(key, ok):
        window[key].update(STATUS_OK if ok else STATUS_FAIL)

    set_status("-PYTHON-", "Python version OK." in log_text)
    set_status("-ONNX-", "Package 'onnx' : INSTALLED" in log_text)
    set_status("-TORCHSTATUS-", "Package 'torch' : INSTALLED" in log_text)
    set_status("-TFSTATUS-", "Package 'tensorflow' : INSTALLED" in log_text)
    set_status("-MLSSTATUS-", "Package 'mlserver' : INSTALLED" in log_text)
    set_status("-TRITONSTATUS-", "Package 'tritonclient' : INSTALLED" in log_text)
    set_status("-GITSTATUS-", "Git found." in log_text)
    set_status("-PODMANSTATUS-", "Podman found." in log_text)

# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------
def validate_environment(window):
    script = os.path.join("scripts", "Validate-Environment.ps1")
    if not os.path.exists(script):
        log(window, "> ERROR: Validate-Environment.ps1 not found.")
        return

    old_log = window["-LOG-"].get()
    update_progress(window, 0)
    rc = run_powershell(window, script, [])
    new_log = window["-LOG-"].get()

    update_environment_status(window, new_log)

    if rc == 0:
        log(window, "> Environment validation PASSED.")
    else:
        log(window, "> Environment validation FAILED.")

# ---------------------------------------------------------------------------
# Install single dependency
# ---------------------------------------------------------------------------
def install_dependency(window, package):
    script = os.path.join("scripts", "Install-Dependencies.ps1")
    if not os.path.exists(script):
        log(window, "> ERROR: Install-Dependencies.ps1 not found.")
        return

    log(window, f"> Installing dependency: {package}")
    update_progress(window, 0)

    rc = run_powershell(window, script, ["-Package", package])

    if rc == 0:
        log(window, f"> Successfully installed {package}.")
    else:
        log(window, f"> ERROR: Failed to install {package}.")

# ---------------------------------------------------------------------------
# Fix Environment (install everything missing)
# ---------------------------------------------------------------------------
def fix_environment(window):
    log(window, "> Fixing environment...")

    missing = [
        "onnx", "onnxruntime", "torch", "scikit-learn",
        "tensorflow", "mlserver", "tritonclient",
        "git", "podman"
    ]

    for item in missing:
        install_dependency(window, item)

    validate_environment(window)

    log_text = window["-LOG-"].get()
    update_environment_status(window, log_text)

# ---------------------------------------------------------------------------
# ONNX model generation
# ---------------------------------------------------------------------------
def generate_model(window, values):
    script = os.path.join("scripts", "Generate-ONNXModel.ps1")
    if not os.path.exists(script):
        log(window, "> ERROR: Generate-ONNXModel.ps1 not found.")
        return

    model_folder = values["-MODEL_FOLDER-"]
    entry_point = values["-ENTRYPOINT-"]
    backend = values["-BACKEND-"]
    output_folder = values["-OUTPUT_FOLDER-"]

    # --- FIX: Prevent empty parameters (critical) ---
    if not model_folder:
        log(window, "> ERROR: Model folder is missing.")
        return
    if not entry_point:
        log(window, "> ERROR: Entry point is missing.")
        return
    if not backend:
        log(window, "> ERROR: Backend is missing.")
        return
    if not output_folder:
        log(window, "> ERROR: Output folder is missing.")
        return

    args = [
        "-ModelFolder", model_folder,
        "-EntryPoint", entry_point,
        "-Backend", backend,
        "-OutputFolder", output_folder
    ]

    update_progress(window, 0)
    rc = run_powershell(window, script, args)

    if rc == 0:
        log(window, "> ONNX model generation completed successfully.")
    else:
        log(window, "> ONNX model generation FAILED.")

# ---------------------------------------------------------------------------
# GUI Layout
# ---------------------------------------------------------------------------

model_section = [
    [
        sg.Text("Model Folder:"),
        sg.Input(key="-MODEL_FOLDER-", size=(50, 1), enable_events=True),
        sg.FolderBrowse()
    ],
    [
        sg.Text("Detected files:"),
        sg.Text("None", key="-DETECTED-")
    ],
    [
        sg.Text("Entry Point:"),
        sg.Combo([], key="-ENTRYPOINT-", size=(30, 1))
    ]
]

backend_section = [
    [sg.Text("Backend Selection")],
    [
        sg.Radio("Torch", "BACKEND", key="-TORCH-"),
        sg.Radio("scikit-learn", "BACKEND", key="-SKLEARN-"),
        sg.Radio("Custom Python", "BACKEND", key="-CUSTOM-"),
        sg.Radio("MLServer", "BACKEND", key="-MLSERVER-"),
        sg.Radio("Triton", "BACKEND", key="-TRITON-")
    ]
]

env_section = [
    [sg.Text("Environment Status")],
    [sg.Text("Python 3.10+"), sg.Text(STATUS_UNKNOWN, key="-PYTHON-")],
    [sg.Text("ONNX"), sg.Text(STATUS_UNKNOWN, key="-ONNX-")],
    [sg.Text("Torch"), sg.Text(STATUS_UNKNOWN, key="-TORCHSTATUS-")],
    [sg.Text("TensorFlow"), sg.Text(STATUS_UNKNOWN, key="-TFSTATUS-"), sg.Button("Install TF")],
    [sg.Text("MLServer"), sg.Text(STATUS_UNKNOWN, key="-MLSSTATUS-")],
    [sg.Text("Triton Client"), sg.Text(STATUS_UNKNOWN, key="-TRITONSTATUS-"), sg.Button("Install Triton")],
    [sg.Text("Git"), sg.Text(STATUS_UNKNOWN, key="-GITSTATUS-"), sg.Button("Install Git")],
    [sg.Text("Podman"), sg.Text(STATUS_UNKNOWN, key="-PODMANSTATUS-"), sg.Button("Install Podman")]
]

output_section = [
    [
        sg.Text("Output Folder:"),
        sg.Input(key="-OUTPUT_FOLDER-", size=(50, 1)),
        sg.FolderBrowse()
    ]
]

log_section = [
    [sg.Text("Log Console")],
    [sg.Multiline(size=(100, 15), key="-LOG-", autoscroll=True, disabled=True)]
]

progress_section = [
    [sg.Text("Progress:"), sg.ProgressBar(100, orientation='h', size=(50, 20), key='-PROGRESS-')]
]

buttons = [
    sg.Button("Validate Environment"),
    sg.Button("Fix Environment"),
    sg.Button("Generate ONNX Model"),
    sg.Button("Exit")
]

layout = [
    [sg.Frame("Model Folder Selection", model_section)],
    [sg.Frame("Backend Selection", backend_section)],
    [sg.Frame("Environment Status", env_section)],
    [sg.Frame("Output Settings", output_section)],
    [sg.Frame("Log Console", log_section)],
    [sg.Frame("Progress", progress_section)],
    [sg.HorizontalSeparator()],
    [sg.Column([buttons], justification="center")]
]

window = sg.Window("ONNX Model Generator", layout, finalize=True)

# ---------------------------------------------------------------------------
# Event Loop
# ---------------------------------------------------------------------------
while True:
    event, values = window.read()

    if event in (sg.WIN_CLOSED, "Exit"):
        break

    if event == "-MODEL_FOLDER-":
        folder = values["-MODEL_FOLDER-"]
        py_files = detect_python_files(folder)
        if py_files:
            window["-DETECTED-"].update(", ".join(py_files))
            window["-ENTRYPOINT-"].update(values=py_files)
        else:
            window["-DETECTED-"].update("None")
            window["-ENTRYPOINT-"].update(values=[])

    if event == "Validate Environment":
        validate_environment(window)

    if event == "Fix Environment":
        fix_environment(window)

    if event == "Install TF":
        install_dependency(window, "tensorflow")

    if event == "Install Triton":
        install_dependency(window, "tritonclient")

    if event == "Install Git":
        install_dependency(window, "git")

    if event == "Install Podman":
        install_dependency(window, "podman")

    if event == "Generate ONNX Model":
        backend = None
        if values["-TORCH-"]: backend = "torch"
        elif values["-SKLEARN-"]: backend = "sklearn"
        elif values["-CUSTOM-"]: backend = "custom"
        elif values["-MLSERVER-"]: backend = "mlserver"
        elif values["-TRITON-"]: backend = "triton"

        values["-BACKEND-"] = backend
        generate_model(window, values)

window.close()
