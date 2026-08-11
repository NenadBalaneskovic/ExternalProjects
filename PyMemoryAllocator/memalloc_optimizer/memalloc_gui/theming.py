"""
theming.py

Provides:
- Theme dataclass for GUI styling
- ThemeLoader for reading JSON theme files
- apply_theme() for configuring PySimpleGUI with custom colors, fonts, and layout

This module ensures consistent styling across the MemAlloc Optimizer GUI.
"""

import json
from dataclasses import dataclass
from pathlib import Path
import PySimpleGUI as sg


# ============================================================
# Theme Dataclass
# ============================================================

@dataclass(frozen=True)
class Theme:
    name: str
    colors: dict
    fonts: dict
    layout: dict
    progress_bar: dict


# ============================================================
# Theme Loader
# ============================================================

class ThemeLoader:
    """
    Loads a theme JSON file and returns a Theme object.
    """

    def __init__(self, theme_path: Path):
        self.theme_path = theme_path

    def load(self) -> Theme:
        with open(self.theme_path, "r") as f:
            data = json.load(f)

        return Theme(
            name=data.get("theme_name", "Default"),
            colors=data.get("colors", {}),
            fonts=data.get("fonts", {}),
            layout=data.get("layout", {}),
            progress_bar=data.get("progress_bar", {}),
        )


# ============================================================
# Theme Application
# ============================================================

def apply_theme(theme: Theme):
    """
    Apply a Theme object to PySimpleGUI.
    This sets:
    - global theme name
    - background colors
    - text colors
    - input colors
    - button colors
    - frame colors
    - font sizes
    """

    # Set base theme name
    sg.theme(theme.name)

    # Apply global color overrides
    sg.set_options(
        background_color=theme.colors.get("background", "#1E1E1E"),
        text_element_background_color=theme.colors.get("background", "#1E1E1E"),
        text_color=theme.colors.get("text", "#FFFFFF"),
        input_elements_background_color=theme.colors.get("input_background", "#2D2D2D"),
        input_text_color=theme.colors.get("input_text", "#FFFFFF"),
        button_color=(
            theme.colors.get("button_background", "#3A3A3A"),
            theme.colors.get("button_text", "#FFFFFF"),
        ),
        progress_meter_color=(
            theme.colors.get("progress_bar", "#4EA5F7"),
            theme.colors.get("progress_background", "#1C2B45"),
        ),
        border_width=theme.layout.get("frame_border_width", 2),
        font=(theme.fonts.get("default", "Segoe UI"), theme.fonts.get("text_size", 10)),
    )

    # Apply frame styling
    sg.set_options(
        element_padding=tuple(theme.layout.get("element_padding", [5, 5])),
        margins=tuple(theme.layout.get("padding", [10, 10])),
    )

    # Apply progress bar defaults
    sg.set_options(
        progress_meter_size=tuple(theme.progress_bar.get("size", [50, 20])),
    )
