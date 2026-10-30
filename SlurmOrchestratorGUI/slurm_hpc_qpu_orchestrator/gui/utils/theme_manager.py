"""
theme_manager.py
----------------
Theme management for the Slurm HPC–QPU Workflow Orchestrator GUI.

This module:
    - loads theme JSON files
    - exposes theme tokens (colors, fonts, layout)
    - applies PySimpleGUI theme overrides
    - maps CSS variables to GUI colors (future-proofing)
    - provides a clean API for main_gui.py

It NEVER executes user workflow code.
"""

import json
from pathlib import Path
import PySimpleGUI as sg


# ----------------------------------------------------------------------
# Theme Dataclass
# ----------------------------------------------------------------------

class GUITheme:
    """
    Simple container for GUI theme tokens.
    """

    def __init__(self, name: str, colors: dict, fonts: dict, layout: dict):
        self.name = name
        self.colors = colors
        self.fonts = fonts
        self.layout = layout

    def __repr__(self):
        return f"GUITheme(name={self.name}, colors={len(self.colors)} tokens)"


# ----------------------------------------------------------------------
# Theme Manager
# ----------------------------------------------------------------------

class ThemeManager:
    """
    Loads and applies GUI themes from JSON files.

    Responsibilities:
        - load theme JSON
        - expose theme tokens
        - apply PySimpleGUI overrides
        - support future theme switching
    """

    def __init__(self, theme_path: Path):
        self.theme_path = theme_path
        self.theme = None

    # ------------------------------------------------------------------
    # Load Theme
    # ------------------------------------------------------------------

    def load(self) -> GUITheme:
        """
        Load theme JSON file and return a GUITheme object.
        """
        with open(self.theme_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.theme = GUITheme(
            name=data.get("theme_name", "Default"),
            colors=data.get("colors", {}),
            fonts=data.get("fonts", {}),
            layout=data.get("layout", {}),
        )

        return self.theme

    # ------------------------------------------------------------------
    # Apply Theme to PySimpleGUI
    # ------------------------------------------------------------------

    def apply(self):
        """
        Apply the loaded theme to PySimpleGUI.
        """
        if self.theme is None:
            raise RuntimeError("Theme not loaded. Call load() first.")

        sg.theme(self.theme.name)

        sg.set_options(
            background_color=self.theme.colors.get("background", "#1E1E1E"),
            text_element_background_color=self.theme.colors.get("background", "#1E1E1E"),
            text_color=self.theme.colors.get("text", "#FFFFFF"),
            input_elements_background_color=self.theme.colors.get("input_background", "#2D2D2D"),
            input_text_color=self.theme.colors.get("input_text", "#FFFFFF"),
            button_color=(
                self.theme.colors.get("button_background", "#3A3A3A"),
                self.theme.colors.get("button_text", "#FFFFFF"),
            ),
            border_width=self.theme.layout.get("frame_border_width", 2),
            font=(
                self.theme.fonts.get("default", "Segoe UI"),
                self.theme.fonts.get("text_size", 10),
            ),
            element_padding=tuple(self.theme.layout.get("element_padding", [5, 5])),
            margins=tuple(self.theme.layout.get("padding", [10, 10])),
        )

    # ------------------------------------------------------------------
    # CSS Variable Mapping (Future-Proofing)
    # ------------------------------------------------------------------

    def css_variables(self) -> dict:
        """
        Return a mapping of CSS-style variables for documentation or future GUI engines.
        """
        if self.theme is None:
            raise RuntimeError("Theme not loaded. Call load() first.")

        return {
            "--background": self.theme.colors.get("background"),
            "--text": self.theme.colors.get("text"),
            "--input-background": self.theme.colors.get("input_background"),
            "--input-text": self.theme.colors.get("input_text"),
            "--button-background": self.theme.colors.get("button_background"),
            "--button-text": self.theme.colors.get("button_text"),
            "--frame-background": self.theme.colors.get("frame_background"),
            "--progress-bar": self.theme.colors.get("progress_bar"),
            "--progress-background": self.theme.colors.get("progress_background"),
        }