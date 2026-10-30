"""
gui/documentation_panel.py

Documentation viewer panel for PyDependencyInspector.

Responsibilities:
- Display documentation for the selected dependency.
- Render Markdown → HTML using MarkdownRenderer.
- Fetch metadata via DocumentationFetcher.
- Update dynamically when DependencyPanel emits dependencySelected.
- Apply dark graphite + cyan styling.

This module is intentionally:
- GUI-only (no resolver logic).
- Modular and clean.
- Driven by ProjectState.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QTextBrowser,
    QSizePolicy,
)
from PySide6.QtCore import Qt, Slot

from ..core.documentation_fetcher import DocumentationFetcher
from ..utils.markdown_renderer import MarkdownRenderer
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class DocumentationPanel(QWidget):
    """
    Displays documentation for a selected dependency.
    """

    def __init__(self, project_state) -> None:
        super().__init__()
        self.project_state = project_state

        self.fetcher = DocumentationFetcher()
        self.renderer = MarkdownRenderer()

        self._init_ui()
        self._apply_styles()

    # ------------------------------------------------------------------ #
    # UI initialization
    # ------------------------------------------------------------------ #

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.title_label = QLabel("Select a package to view documentation")
        self.title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.title_label.setStyleSheet("font-size: 18px; font-weight: 600;")

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addWidget(self.title_label)
        layout.addWidget(self.browser)

    # ------------------------------------------------------------------ #
    # Styling
    # ------------------------------------------------------------------ #

    def _apply_styles(self) -> None:
        self.setStyleSheet("""
            QLabel {
                color: #80cbc4;
            }
            QTextBrowser {
                background-color: #1a1a1a;
                color: #e0e0e0;
                border: 1px solid #333;
                padding: 8px;
                font-size: 13px;
            }
            QTextBrowser a {
                color: #4fc3f7;
            }
            QTextBrowser a:hover {
                color: #81d4fa;
            }
        """)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @Slot(str)
    def load_documentation(self, package_name: str) -> None:
        if not package_name:
            return

        self.project_state.last_selected_package = package_name
        self.title_label.setText(package_name)
        self.browser.setHtml("<p><em>Loading documentation…</em></p>")

        logger.info("Fetching documentation for '%s'", package_name)

        # Check cache
        cached = self.project_state.get_cached_documentation(package_name)
        if cached:
            html = self._build_html(package_name, cached)
            self.browser.setHtml(html)
            return

        # Fetch metadata
        result = self.fetcher.fetch(package_name)
        self.project_state.cache_documentation(package_name, result)

        if not result.success:
            self.browser.setHtml(
                f"<p><strong>Error:</strong> Could not fetch documentation for "
                f"<code>{package_name}</code>.</p>"
            )
            for e in result.errors:
                logger.error(e)
            return

        html = self._build_html(package_name, result)
        self.browser.setHtml(html)

        for w in result.warnings:
            logger.warning(w)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_html(self, package_name: str, result) -> str:
        """
        Build fragment HTML for the documentation viewer.
        No <html>, <head>, <body>, or <style> tags.
        Suitable for QTextBrowser.
        """

        summary = result.summary or "No summary available."
        readme_html = self.renderer.render(result.long_description or "")

        # Project links
        links_html = "<ul>"
        for label, url in result.project_urls.items():
            links_html += f'<li><a href="{url}">{label}</a></li>'
        links_html += f'<li><a href="https://pypi.org/project/{package_name}/">PyPI</a></li>'
        links_html += "</ul>"

        return f"""
            <h2>{package_name}</h2>
            <p><em>{summary}</em></p>

            <h3>Project Links</h3>
            {links_html}

            <hr>
            {readme_html}
        """
