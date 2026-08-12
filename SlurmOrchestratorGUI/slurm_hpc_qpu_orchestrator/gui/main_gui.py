"""
main_gui.py
-----------
Main GUI window for the Slurm HPC–QPU Workflow Orchestrator.

This module:
    - builds the GUI layout (build_main_window)
    - runs the GUI event loop (run_gui)
    - delegates workflow parsing, classification, and Slurm generation
      to core modules

It NEVER executes user workflow code.
"""

import PySimpleGUI as sg
from pathlib import Path

from slurm_hpc_qpu_orchestrator.core.ast_parser import ASTParser
from slurm_hpc_qpu_orchestrator.core.workflow_classifier import WorkflowClassifier
from slurm_hpc_qpu_orchestrator.core.slurm_template_engine import SlurmTemplateEngine
from slurm_hpc_qpu_orchestrator.gui.utils.validators import validate_workflow_path


# ----------------------------------------------------------------------
# GUI Layout Construction
# ----------------------------------------------------------------------

def build_main_window() -> sg.Window:
    """
    Build and return the main GUI window.
    This function is pure layout construction and does not start the event loop.
    """

    sg.theme("DarkBlue")

    # -----------------------------
    # Workflow Upload Section
    # -----------------------------
    upload_section = [
        [
            sg.Text("Select Python Workflow:", font=("Arial", 12)),
            sg.Input(key="WORKFLOW_PATH", enable_events=True),
            sg.FileBrowse(button_text="Browse"),
            sg.Button("Upload", key="UPLOAD_BUTTON")
        ]
    ]

    # -----------------------------
    # Workflow Analysis Panel
    # -----------------------------
    analysis_panel = sg.Column(
        [
            [sg.Text("Workflow Analysis", font=("Arial", 14, "bold"))],
            [sg.Multiline(
                key="ANALYSIS_PANEL",
                size=(80, 15),
                disabled=True,
                autoscroll=True
            )]
        ],
        key="ANALYSIS_PANEL"
    )

    # -----------------------------
    # Slurm Preview Panel
    # -----------------------------
    slurm_preview = sg.Multiline(
        key="SLURM_PREVIEW",
        size=(80, 20),
        disabled=True,
        autoscroll=True
    )

    # -----------------------------
    # Manual Credentials Section (QPU + HPC)
    # -----------------------------
    creds_section = sg.Column(
    [
        [sg.Text("QPU Credentials", font=("Arial", 12, "bold"))],
        [sg.Text("API Key:"), sg.Input(key="API_KEY")],
        [sg.Text("Runtime URL:"), sg.Input(key="RUNTIME_URL")],

        [sg.HorizontalSeparator()],

        [sg.Text("HPC Settings", font=("Arial", 12, "bold"))],
        [sg.Text("Partition:"), sg.Input(key="PARTITION", default_text="compute")],
        [sg.Text("Nodes:"), sg.Input(key="NODES", default_text="1")],
        [sg.Text("CPUs per node:"), sg.Input(key="CPUS", default_text="4")],
        [sg.Text("Time limit (HH:MM:SS):"), sg.Input(key="TIME_LIMIT", default_text="01:00:00")],
        [sg.Text("Module load:"), sg.Input(key="MODULE_LOAD", default_text="python/3.10")],
        [sg.Text("Python environment:"), sg.Input(key="PYTHON_ENV", default_text="{{PYTHON_ENV}}")],
    ],
    key="CREDS_SECTION",
    visible=False,
    scrollable=True,
    vertical_scroll_only=True,
    size=(400, 200)  # adjust height as needed
    )


    creds_toggle = sg.Checkbox(
        "Enable manual credentials",
        key="ENABLE_CREDS",
        enable_events=True
    )

    # -----------------------------
    # Generate Slurm Button
    # -----------------------------
    generate_button = sg.Button(
        "Generate Slurm Script",
        key="GENERATE_SLURM",
        button_color=("white", "green")
    )

    # -----------------------------
    # Final Layout
    # -----------------------------
    layout = [
        [sg.Text("Slurm HPC–QPU Workflow Orchestrator", font=("Arial", 18, "bold"))],
        [sg.HorizontalSeparator()],
        [sg.Frame("Workflow Upload", upload_section)],
        [sg.Frame("Workflow Analysis", [[analysis_panel]])],
        [sg.Frame("Slurm Preview", [[slurm_preview]])],
        [creds_toggle],
        [creds_section],
        [generate_button],
    ]

    window = sg.Window(
        "Slurm HPC–QPU Workflow Orchestrator",
        layout,
        finalize=True
    )

    return window


# ----------------------------------------------------------------------
# GUI Event Loop
# ----------------------------------------------------------------------

def run_gui():
    """
    Run the GUI event loop.
    This function orchestrates:
        - workflow upload
        - static AST analysis
        - workflow classification
        - Slurm script generation
    """

    window = build_main_window()

    parser = ASTParser()
    classifier = WorkflowClassifier()

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break

        # --------------------------------------------------------------
        # Toggle credential section
        # --------------------------------------------------------------
        if event == "ENABLE_CREDS":
            window["CREDS_SECTION"].update(visible=values["ENABLE_CREDS"])

        # --------------------------------------------------------------
        # Upload workflow file
        # --------------------------------------------------------------
        if event == "UPLOAD_BUTTON":
            path = values["WORKFLOW_PATH"]
            if not path or not Path(path).exists():
                window["ANALYSIS_PANEL"].update("Error: File not found.")
                continue

            parsed = parser.parse_file(Path(path))
            classification = classifier.classify(parsed)

            analysis_text = (
                f"File: {parsed.file_path}\n"
                f"Imports: {parsed.imports}\n"
                f"Function Calls: {parsed.function_calls}\n"
                f"Contains Loops: {parsed.has_loops}\n\n"
                f"Workflow Type: {classification.workflow_type.name}\n"
                f"Quantum Imports: {classification.quantum_imports}\n"
                f"Quantum Calls: {classification.quantum_calls}\n"
                f"Classical Imports: {classification.classical_imports}\n"
            )

            window["ANALYSIS_PANEL"].update(analysis_text)

        # --------------------------------------------------------------
        # Generate Slurm Script
        # --------------------------------------------------------------
        if event == "GENERATE_SLURM":
            path = values["WORKFLOW_PATH"]
            if not path or not Path(path).exists():
                window["SLURM_PREVIEW"].update("Error: No workflow selected.")
                continue

            parsed = parser.parse_file(Path(path))
            classification = classifier.classify(parsed)

            # Prepare substitutions (HPC + QPU)
            subs = {
                "JOB_NAME": "gui_job",

                # HPC
                "PARTITION": values["PARTITION"] if values["ENABLE_CREDS"] else "{{PARTITION}}",
                "NODES": values["NODES"] if values["ENABLE_CREDS"] else "{{NODES}}",
                "CPUS": values["CPUS"] if values["ENABLE_CREDS"] else "{{CPUS}}",
                "TIME_LIMIT": values["TIME_LIMIT"] if values["ENABLE_CREDS"] else "{{TIME_LIMIT}}",
                "MODULE_LOAD": values["MODULE_LOAD"] if values["ENABLE_CREDS"] else "{{MODULE_LOAD}}",
                "PYTHON_ENV": values["PYTHON_ENV"] if values["ENABLE_CREDS"] else "{{PYTHON_ENV}}",

                # QPU
                "API_KEY": values["API_KEY"] if values["ENABLE_CREDS"] else "{{API_KEY}}",
                "RUNTIME_URL": values["RUNTIME_URL"] if values["ENABLE_CREDS"] else "{{RUNTIME_URL}}",

                "OUTPUT_LOG": "logs/%x_%j.out",
            }

            engine = SlurmTemplateEngine(Path("./generated_slurm_jobs"))
            slurm_script = engine.generate_slurm_script(
                workflow_type=classification.workflow_type,
                substitutions=subs,
                script_name=Path(path).name
            )

            window["SLURM_PREVIEW"].update(slurm_script.script_text)

    window.close()


# ----------------------------------------------------------------------
# Script Entry Point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    run_gui()
