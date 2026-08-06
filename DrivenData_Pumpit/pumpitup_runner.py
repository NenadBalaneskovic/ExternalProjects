import subprocess
import sys
import csv
import datetime
from pathlib import Path
import os

# ------------------------------------------------------------
# Pfade
# ------------------------------------------------------------
project_root = Path("D:/DrivenData_PumpIt/pumpitup")   # WICHTIG: Projekt-Root
env_prefix = Path("D:/conda_envs/pumpitup")
env_python = env_prefix / "python.exe"
env_scripts = env_prefix / "Scripts"
pumpitup_cli = env_scripts / "pumpitup.exe"

log_file = Path("D:/DrivenData_PumpIt/runner_logs.csv")

# ------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------
def log_event(step, command, output, error, exit_code):
    timestamp = datetime.datetime.now().isoformat()
    with log_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, step, command, output, error, exit_code])

# CSV Header
if not log_file.exists():
    with log_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "step", "command", "output", "error", "exit_code"])


# ------------------------------------------------------------
# 0. Arbeitsverzeichnis setzen
# ------------------------------------------------------------
print("### Setze Arbeitsverzeichnis ###")
os.chdir(project_root)
log_event("set_cwd", f"cd {project_root}", "cwd gesetzt", "", 0)
print("cwd =", os.getcwd())


# ------------------------------------------------------------
# 1. Umgebung prüfen
# ------------------------------------------------------------
print("### 1. Prüfe Umgebung ###")
step = "check_environment"

if not env_prefix.exists():
    msg = "Umgebung existiert NICHT — wird erstellt..."
    print(msg)
    log_event(step, "conda create", msg, "", 0)

    subprocess.check_call([
        "conda", "create",
        "--prefix", str(env_prefix),
        "python=3.12",
        "-y"
    ])
else:
    msg = f"Umgebung existiert: {env_prefix}"
    print(msg)
    log_event(step, "check env exists", msg, "", 0)


# ------------------------------------------------------------
# 2. pumpitup installieren
# ------------------------------------------------------------
print("\n### 2. Installiere pumpitup ###")
step = "install_pumpitup"

cmd = [str(env_python), "-m", "pip", "install", "-e", str(project_root)]

try:
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    print(output.decode())
    log_event(step, " ".join(cmd), output.decode(), "", 0)
except subprocess.CalledProcessError as e:
    print("Fehler:", e)
    log_event(step, " ".join(cmd), "", str(e), e.returncode)

print("pumpitup erfolgreich installiert.")


# ------------------------------------------------------------
# 3. CLI-Befehle ausführen
# ------------------------------------------------------------
print("\n### 3. PumpItUp Shell-Befehle ###\n")

commands = [
    ["baseline"],
    ["tune"],
    ["stack"],
    ["submit"],
]

for cmd in commands:
    step = f"cli_{cmd[0]}"
    print("$ pumpitup", " ".join(cmd))

    full_cmd = [str(pumpitup_cli)] + cmd

    try:
        output = subprocess.check_output(full_cmd, stderr=subprocess.STDOUT, cwd=project_root)
        decoded = output.decode()
        print(decoded)
        log_event(step, " ".join(full_cmd), decoded, "", 0)

    except subprocess.CalledProcessError as e:
        print("Fehler:", e)
        try:
            err_output = subprocess.check_output(full_cmd, stderr=subprocess.STDOUT, cwd=project_root).decode()
        except Exception as inner:
            err_output = str(inner)

        log_event(step, " ".join(full_cmd), "", err_output, e.returncode)


# ------------------------------------------------------------
# 4. ExperimentTracker ausführen
# ------------------------------------------------------------
print("\n### 4. ExperimentTracker ###")
step = "tracker"

from pumpitup.pumpitup.experiments.tracker import ExperimentTracker

try:
    tracker = ExperimentTracker()
    msg = "Tracker initialisiert"
    print(msg)
    log_event(step, "ExperimentTracker()", msg, "", 0)
except Exception as e:
    print("Fehler:", e)
    log_event(step, "ExperimentTracker()", "", str(e), 1)


# ------------------------------------------------------------
# 5. tracker.list()
# ------------------------------------------------------------
print("\n### tracker.list() ###")
step = "tracker_list"

try:
    items = tracker.list()
    for exp in items:
        print(exp)
        log_event(step, "tracker.list()", str(exp), "", 0)
except Exception as e:
    print("Fehler:", e)
    log_event(step, "tracker.list()", "", str(e), 1)


# ------------------------------------------------------------
# 6. tracker.best()
# ------------------------------------------------------------
print("\n### tracker.best() ###")
step = "tracker_best"

try:
    best = tracker.best()
    print(best)
    log_event(step, "tracker.best()", str(best), "", 0)
except Exception as e:
    print("Fehler:", e)
    log_event(step, "tracker.best()", "", str(e), 1)
