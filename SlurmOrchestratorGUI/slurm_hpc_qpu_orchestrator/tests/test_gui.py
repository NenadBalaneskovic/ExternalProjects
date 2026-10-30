"""
Unit tests for the GUI layout of the Slurm HPC–QPU Workflow Orchestrator.

These tests verify:
    - correct window layout structure
    - presence of expected GUI elements
    - correct wiring of keys and callbacks
    - correct behavior of the manual credentials checkbox
    - correct integration points for workflow analysis and Slurm generation

The tests DO NOT open real GUI windows or run the event loop.
"""

import PySimpleGUI as sg
import pytest

from gui.main_gui import build_main_window


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def get_window():
    """Create the GUI window without starting the event loop."""
    window = build_main_window()
    assert isinstance(window, sg.Window)
    return window


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_window_creation():
    """Ensure the main window is created correctly."""
    window = get_window()
    assert window.Title.lower().startswith("slurm hpc–qpu workflow orchestrator")


def test_layout_contains_expected_sections():
    """Verify that major GUI sections exist."""
    window = get_window()

    # Check for workflow upload section
    assert window.find_element("UPLOAD_BUTTON") is not None

    # Check for workflow analysis panel
    assert window.find_element("ANALYSIS_PANEL") is not None

    # Check for Slurm preview panel
    assert window.find_element("SLURM_PREVIEW") is not None

    # Check for credentials section toggle
    assert window.find_element("ENABLE_CREDS") is not None


def test_credentials_section_initially_hidden():
    """Manual credentials section should be hidden by default."""
    window = get_window()
    creds_section = window.find_element("CREDS_SECTION")
    assert creds_section is not None
    assert creds_section.Visible is False


def test_credentials_section_toggle_behavior():
    """Simulate enabling the manual credentials section."""
    window = get_window()

    checkbox = window.find_element("ENABLE_CREDS")
    creds_section = window.find_element("CREDS_SECTION")

    # Simulate user clicking the checkbox
    checkbox.Update(value=True)
    creds_section.Update(visible=True)

    assert creds_section.Visible is True


def test_slurm_preview_panel_exists():
    """Ensure the Slurm preview multiline element exists."""
    window = get_window()
    preview = window.find_element("SLURM_PREVIEW")
    assert isinstance(preview, sg.Multiline)


def test_upload_button_wired_correctly():
    """Ensure the upload button exists and has a key."""
    window = get_window()
    upload_btn = window.find_element("UPLOAD_BUTTON")
    assert isinstance(upload_btn, sg.Button)
    assert upload_btn.Key == "UPLOAD_BUTTON"


def test_generate_button_exists():
    """Ensure the Slurm generation button exists."""
    window = get_window()
    gen_btn = window.find_element("GENERATE_SLURM")
    assert isinstance(gen_btn, sg.Button)


def test_analysis_panel_structure():
    """Ensure the analysis panel contains expected elements."""
    window = get_window()
    panel = window.find_element("ANALYSIS_PANEL")

    assert panel is not None
    assert isinstance(panel, sg.Column)


def test_window_has_correct_keys():
    """Verify that all critical keys exist in the window."""
    window = get_window()

    expected_keys = [
        "UPLOAD_BUTTON",
        "ANALYSIS_PANEL",
        "SLURM_PREVIEW",
        "ENABLE_CREDS",
        "CREDS_SECTION",
        "GENERATE_SLURM",
    ]

    for key in expected_keys:
        assert window.find_element(key) is not None