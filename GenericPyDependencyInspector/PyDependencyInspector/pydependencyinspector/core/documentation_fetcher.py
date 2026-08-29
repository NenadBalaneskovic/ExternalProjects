"""
core/documentation_fetcher.py

Responsible for:
- Fetching documentation metadata for a given Python package.
- Retrieving README / long description from the PyPI JSON API.
- Extracting project URLs (homepage, docs, repository).
- Providing a clean, GUI-friendly DocumentationResult object.
- Operating gracefully in offline environments (no hard failures).

This module is intentionally:
- Lightweight (only stdlib + optional requests).
- Resilient (network failures become warnings, not fatal errors).
- Decoupled from GUI logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import logging
import urllib.request
import urllib.error
import html
import re

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DocumentationResult:
    """
    Represents all documentation-related metadata for a package.

    Fields:
    - name: package name
    - summary: short description (from PyPI)
    - long_description: README or long description (HTML or Markdown)
    - project_urls: mapping of label -> URL (homepage, docs, repo, etc.)
    - warnings: non-fatal issues (network errors, missing fields)
    - errors: fatal issues (package not found on PyPI)
    """
    name: str
    summary: str = ""
    long_description: str = ""
    project_urls: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """
        GUI expects a `.success` attribute.
        A documentation fetch is considered successful if:
        - no fatal errors occurred, AND
        - either summary or long_description is available.
        """
        if self.errors:
            return False
        return bool(self.summary or self.long_description)


# ---------------------------------------------------------------------------
# DocumentationFetcher – public API
# ---------------------------------------------------------------------------

class DocumentationFetcher:
    """
    Fetches documentation metadata for Python packages using the PyPI JSON API.

    Responsibilities:
    - Query https://pypi.org/pypi/<package>/json
    - Extract summary, long description, project URLs
    - Provide a DocumentationResult object
    - Handle offline mode gracefully (warnings instead of crashes)

    The GUI will use this to populate the Documentation Panel.
    """

    PYPI_URL_TEMPLATE = "https://pypi.org/pypi/{package}/json"

    def __init__(self, timeout: int = 5) -> None:
        """
        :param timeout: network timeout in seconds
        """
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fetch(self, package_name: str) -> DocumentationResult:
        """
        Fetch documentation metadata for a package.

        :param package_name: e.g. "pandas"
        :return: DocumentationResult
        """
        logger.info("Fetching documentation for package '%s'", package_name)

        result = DocumentationResult(name=package_name)

        # Step 1: Try to fetch PyPI JSON metadata
        pypi_data = self._fetch_pypi_json(package_name, result)

        if pypi_data is None:
            # Offline or package not found — return partial result
            return result

        # Step 2: Extract summary
        info = pypi_data.get("info", {})
        result.summary = info.get("summary", "") or ""

        # Step 3: Extract long description (HTML or Markdown)
        long_desc = info.get("description", "") or ""
        result.long_description = self._sanitize_long_description(long_desc)

        # Step 4: Extract project URLs
        urls = info.get("project_urls", {}) or {}
        result.project_urls = {k: v for k, v in urls.items() if isinstance(v, str)}

        logger.info("Documentation fetch completed for '%s'", package_name)
        return result

    # ------------------------------------------------------------------ #
    # Internal helpers – PyPI JSON API
    # ------------------------------------------------------------------ #

    def _fetch_pypi_json(
        self,
        package_name: str,
        result: DocumentationResult
    ) -> Optional[Dict]:
        """
        Fetch PyPI JSON metadata for a package.

        Returns:
            dict or None (if offline or not found)
        """
        url = self.PYPI_URL_TEMPLATE.format(package=package_name)

        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw)

        except urllib.error.HTTPError as exc:
            msg = f"PyPI returned HTTP {exc.code} for package '{package_name}'."
            logger.warning(msg)
            result.warnings.append(msg)
            if exc.code == 404:
                result.errors.append(f"Package '{package_name}' not found on PyPI.")
            return None

        except urllib.error.URLError as exc:
            msg = f"Network error while fetching PyPI metadata: {exc.reason}"
            logger.warning(msg)
            result.warnings.append(msg)
            return None

        except Exception as exc:
            msg = f"Unexpected error while fetching PyPI metadata: {exc}"
            logger.exception(msg)
            result.warnings.append(msg)
            return None

    # ------------------------------------------------------------------ #
    # Internal helpers – sanitization
    # ------------------------------------------------------------------ #

    def _sanitize_long_description(self, text: str) -> str:
        """
        Clean up the long description for GUI display.

        PyPI returns either:
        - Markdown
        - reStructuredText
        - HTML

        We keep the content as-is but:
        - unescape HTML entities
        - strip excessive whitespace
        - remove dangerous tags (script, iframe)
        """
        if not text:
            return ""

        # Unescape HTML entities (&lt;, &amp;, etc.)
        text = html.unescape(text)

        # Remove script/iframe tags for safety
        text = re.sub(r"<\s*(script|iframe).*?>.*?<\s*/\1\s*>", "", text, flags=re.I | re.S)

        # Normalize whitespace
        text = text.strip()

        return text


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def fetch_documentation(package_name: str) -> DocumentationResult:
    """
    Convenience wrapper for simple usage.

    Example:
        doc = fetch_documentation("pandas")
        print(doc.summary)
    """
    fetcher = DocumentationFetcher()
    return fetcher.fetch(package_name)
