"""
utils/markdown_renderer.py

Responsible for:
- Rendering Markdown into safe HTML for GUI display and PDF export.
- Supporting GitHub-Flavored Markdown (GFM) when available.
- Falling back to a minimal renderer if the 'markdown' package is missing.
- Sanitizing HTML to prevent unsafe tags (script, iframe, etc.).
- Providing a single, stable API for the Documentation Panel.

This module is intentionally:
- GUI-agnostic.
- Pure Python.
- Safe in offline environments.
"""

from __future__ import annotations

import logging
import re
import html
from typing import Optional

logger = logging.getLogger(__name__)


class MarkdownRenderer:
    """
    Converts Markdown text into safe HTML.

    Responsibilities:
    - Use Python-Markdown (if installed) for full GFM support.
    - Fall back to a minimal Markdown-to-HTML converter if unavailable.
    - Sanitize HTML output.
    - Provide a single render() method for GUI and PDF export.
    """

    def __init__(self) -> None:
        try:
            import markdown  # noqa
            self._markdown_available = True
        except Exception:
            self._markdown_available = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def render(self, markdown_text: str) -> str:
        """
        Render Markdown into safe HTML.
        """
        if not markdown_text or not markdown_text.strip():
            return "<p><em>No documentation available.</em></p>"

        if self._markdown_available:
            html_output = self._render_with_markdown(markdown_text)
        else:
            html_output = self._render_fallback(markdown_text)

        # Ensure non-empty HTML for PDF export and GUI
        if not html_output.strip():
            logger.warning("Markdown renderer returned empty output; using fallback wrapper.")
            html_output = f"<pre>{html.escape(markdown_text)}</pre>"

        return self._sanitize_html(html_output)

    # ------------------------------------------------------------------ #
    # Internal helpers – full Markdown renderer
    # ------------------------------------------------------------------ #

    def _render_with_markdown(self, text: str) -> str:
        """
        Render using Python-Markdown with safe extensions.
        """
        try:
            import markdown

            return markdown.markdown(
                text,
                extensions=[
                    "markdown.extensions.fenced_code",
                    "markdown.extensions.tables",
                    "markdown.extensions.nl2br",
                    "markdown.extensions.sane_lists",
                ],
                output_format="html"  # IMPORTANT: PySide6 rejects XHTML silently
            )
        except Exception as exc:
            logger.warning("Markdown rendering failed, using fallback: %s", exc)
            return self._render_fallback(text)

    # ------------------------------------------------------------------ #
    # Internal helpers – fallback renderer
    # ------------------------------------------------------------------ #

    def _render_fallback(self, text: str) -> str:
        """
        Minimal Markdown-to-HTML converter.
        """
        logger.debug("Using fallback Markdown renderer.")

        # Escape HTML entities
        escaped = html.escape(text)

        # Headings
        escaped = re.sub(r"^### (.*)$", r"<h3>\1</h3>", escaped, flags=re.MULTILINE)
        escaped = re.sub(r"^## (.*)$", r"<h2>\1</h2>", escaped, flags=re.MULTILINE)
        escaped = re.sub(r"^# (.*)$", r"<h1>\1</h1>", escaped, flags=re.MULTILINE)

        # Bold
        escaped = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", escaped)

        # Italic
        escaped = re.sub(r"\*(.*?)\*", r"<em>\1</em>", escaped)

        # Inline code
        escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)

        # Paragraphs
        paragraphs = [f"<p>{line}</p>" for line in escaped.split("\n") if line.strip()]
        return "\n".join(paragraphs)

    # ------------------------------------------------------------------ #
    # Internal helpers – sanitization
    # ------------------------------------------------------------------ #

    def _sanitize_html(self, html_text: str) -> str:
        """
        Remove unsafe HTML tags such as <script> and <iframe>.
        """
        # Remove script/iframe tags
        html_text = re.sub(
            r"<\s*(script|iframe).*?>.*?<\s*/\1\s*>",
            "",
            html_text,
            flags=re.I | re.S,
        )

        # Remove javascript: URLs
        html_text = re.sub(
            r'href=["\']javascript:[^"\']*["\']',
            'href="#"',
            html_text,
            flags=re.I,
        )

        return html_text.strip()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def render_markdown(markdown_text: str) -> str:
    renderer = MarkdownRenderer()
    return renderer.render(markdown_text)
