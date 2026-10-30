"""
gui/main_window.py

Main application window for PyDependencyInspector.

Updated to:
- Remove PyInstaller + spec generation
- Use MetadataRunner instead
- Add new Wheel Extractor tab
- Keep Build Report tab (now dependency summary)
"""

from __future__ import annotations

import sys
import logging
from typing import Dict

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QHBoxLayout,
    QVBoxLayout,
)
from PySide6.QtCore import Qt

from .top_bar import TopBar
from .dependency_panel import DependencyPanel
from .documentation_panel import DocumentationPanel
from .log_panel import LogPanel
from .export_dialog import ExportDialog
from .build_report_window import BuildReportWindow
from .wheel_extractor_panel import WheelExtractorPanel

from ..core.dependency_resolver import DependencyResolver, ResolutionResult
from ..core.metadata_runner import MetadataRunner
from ..core.build_report import BuildReportGenerator
from ..config.paths import config
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Project state container
# ---------------------------------------------------------------------------

class ProjectState:
    """
    Holds mutable project-level state shared across GUI components.
    """

    def __init__(self) -> None:
        self.project_name: str = "Untitled Project"
        self.os_profile: str = config.os_profile

        # Dependency resolution
        self.resolution_result: ResolutionResult | None = None

        # Metadata result (replaces PyInstaller build result)
        self.metadata_result = None

        # Build report HTML (for PDF export)
        self.build_report_html: str = ""

        # Documentation cache: package_name → DocumentationResult
        self.documentation_cache: dict[str, object] = {}

        # Last selected package (for documentation panel + PDF export)
        self.last_selected_package: str | None = None

    def cache_documentation(self, package_name: str, doc) -> None:
        self.documentation_cache[package_name] = doc

    def get_cached_documentation(self, package_name: str):
        return self.documentation_cache.get(package_name)


# ---------------------------------------------------------------------------
# MainWindow – public GUI entry point
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    NAV_ITEMS = [
        "Dependencies",
        "Documentation",
        "Logs",
        "Build Report",
        "Wheel Extractor",
    ]

    def __init__(self, project_state: ProjectState | None = None) -> None:
        super().__init__()

        self.project_state = project_state or ProjectState()

        self.setWindowTitle("PyDependencyInspector")
        self.resize(1280, 800)

        self._init_ui()
        self._apply_dark_theme()

    # ------------------------------------------------------------------ #
    # UI initialization
    # ------------------------------------------------------------------ #

    def _init_ui(self) -> None:
        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Top bar
        self.top_bar = TopBar(self.project_state)
        self.top_bar.projectRenamed.connect(self._on_project_renamed)
        self.top_bar.osProfileChanged.connect(self._on_os_profile_changed)
        self.top_bar.runResolutionRequested.connect(self._on_run_resolution)
        self.top_bar.runBuildRequested.connect(self._on_run_metadata)
        self.top_bar.exportRequested.connect(self._on_export_requested)

        root_layout.addWidget(self.top_bar)

        # Main content area
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Navigation list
        self.nav_list = QListWidget()
        self.nav_list.setFixedWidth(180)
        self.nav_list.setStyleSheet("""
            QListWidget {
                background-color: #1a1a1a;
                color: #e0e0e0;
                border-right: 1px solid #333;
            }
            QListWidget::item:selected {
                background-color: #0e639c;
            }
        """)
        for name in self.NAV_ITEMS:
            QListWidgetItem(name, self.nav_list)

        self.nav_list.currentRowChanged.connect(self._on_nav_changed)

        # Stacked views
        self.views: Dict[str, QWidget] = {}
        self.stack = QStackedWidget()

        self._init_views()

        content_layout.addWidget(self.nav_list)
        content_layout.addWidget(self.stack)

        root_layout.addWidget(content)
        self.setCentralWidget(central)

        # Select first view
        self.nav_list.setCurrentRow(0)

    def _init_views(self) -> None:
        """Create and register all functional panels."""
        self.views["Dependencies"] = DependencyPanel(self.project_state)
        self.views["Documentation"] = DocumentationPanel(self.project_state)
        self.views["Logs"] = LogPanel(self.project_state)
        self.views["Build Report"] = BuildReportWindow(self.project_state)
        self.views["Wheel Extractor"] = WheelExtractorPanel(self.project_state)

        # Dependency selection → documentation panel
        self.views["Dependencies"].dependencySelected.connect(
            self._on_dependency_selected
        )

        for name in self.NAV_ITEMS:
            self.stack.addWidget(self.views[name])

    # ------------------------------------------------------------------ #
    # Navigation
    # ------------------------------------------------------------------ #

    def _on_nav_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.NAV_ITEMS):
            return
        name = self.NAV_ITEMS[row]
        self.stack.setCurrentWidget(self.views[name])

    # ------------------------------------------------------------------ #
    # Top bar callbacks
    # ------------------------------------------------------------------ #

    def _on_project_renamed(self, new_name: str) -> None:
        self.project_state.project_name = new_name
        logger.info("Project renamed to '%s'", new_name)

    def _on_os_profile_changed(self, profile: str) -> None:
        self.project_state.os_profile = profile
        logger.info("OS profile changed to '%s'", profile)

    # ------------------------------------------------------------------ #
    # Dependency selection → Documentation
    # ------------------------------------------------------------------ #

    def _on_dependency_selected(self, package_name: str) -> None:
        self.project_state.last_selected_package = package_name
        self.views["Documentation"].load_documentation(package_name)

    # ------------------------------------------------------------------ #
    # Core actions
    # ------------------------------------------------------------------ #

    def _on_run_resolution(self) -> None:
        resolver = DependencyResolver()
        result = resolver.resolve(
            self.project_state.project_name,
            self.project_state.os_profile,
        )
        self.project_state.resolution_result = result

        # Update dependency panel
        self.views["Dependencies"].update_with_resolution(result)

        # Log warnings/errors
        for w in result.warnings:
            logger.warning(w)
        for e in result.errors:
            logger.error(e)

    def _on_run_metadata(self) -> None:
        """
        Replaces old PyInstaller build step.
        Now collects metadata only.
        """
        if not self.project_state.resolution_result:
            logger.error("Cannot collect metadata: no dependency resolution available.")
            return

        runner = MetadataRunner()
        metadata_result = runner.run(
            resolution_warnings=self.project_state.resolution_result.warnings,
            resolution_errors=self.project_state.resolution_result.errors,
            log_callback=self.views["Logs"].append_line,
        )
        self.project_state.metadata_result = metadata_result

        # Generate dependency summary report
        report_gen = BuildReportGenerator()
        report = report_gen.generate(
            project_name=self.project_state.project_name,
            os_profile=self.project_state.os_profile,
            resolution_result=self.project_state.resolution_result,
        )

        self.project_state.build_report_html = report.html or ""

        # Update build report panel
        self.views["Build Report"].update_report(report)

    def _on_export_requested(self) -> None:
        dlg = ExportDialog(self.project_state, parent=self)
        dlg.exec()

    # ------------------------------------------------------------------ #
    # Styling
    # ------------------------------------------------------------------ #

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QWidget {
                background-color: #121212;
                color: #e0e0e0;
            }
        """)


# ---------------------------------------------------------------------------
# Safe QApplication entry point
# ---------------------------------------------------------------------------

def run_app() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()
