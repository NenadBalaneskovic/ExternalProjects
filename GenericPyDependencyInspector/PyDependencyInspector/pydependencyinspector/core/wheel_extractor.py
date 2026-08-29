"""
core/wheel_extractor.py

Responsible for:
- Scraping PyPI HTML pages to locate wheel files for a given package+version.
- Filtering wheels by Python tag (e.g. cp312) if provided.
- Filtering wheels by platform (e.g. win_amd64, manylinux, macosx) if provided.
- Normalizing wheel URLs so downloads always work.
- Removing invalid and duplicate wheel entries.
- Returning structured wheel metadata.
- Logging success/failure for GUI consumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


@dataclass
class WheelInfo:
    package: str
    version: str
    filename: str
    url: str


class WheelExtractor:
    PYPI_FILES_URL = "https://pypi.org/project/{pkg}/{ver}/#files"
    PYPI_HISTORY_URL = "https://pypi.org/project/{pkg}/#history"

    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def find_wheels(
        self,
        package: str,
        version: str,
        python_tag: Optional[str] = None,
        platform_tag: Optional[str] = None,
    ) -> Tuple[List[WheelInfo], List[str]]:

        logs: List[str] = []
        wheels: List[WheelInfo] = []

        logs.append(f"Searching wheels for {package}=={version}...")
        if python_tag:
            logs.append(f"Filtering for Python tag: {python_tag}")
        if platform_tag:
            logs.append(f"Filtering for platform tag: {platform_tag}")

        # 1. Version page
        url_files = self.PYPI_FILES_URL.format(pkg=package, ver=version)
        logs.append(f"Fetching: {url_files}")

        try:
            wheels_found = self._scrape_files_page(
                package, version, url_files, python_tag, platform_tag
            )
            wheels.extend(wheels_found)
            logs.append(f"Found {len(wheels_found)} wheels on version page.")
        except Exception as exc:
            msg = f"Failed to fetch version page: {exc}"
            logs.append(msg)
            logger.exception(msg)

        # 2. History page
        if not wheels:
            url_hist = self.PYPI_HISTORY_URL.format(pkg=package)
            logs.append(f"No wheels found on version page. Checking history: {url_hist}")

            try:
                wheels_found = self._scrape_history_page(
                    package, version, url_hist, python_tag, platform_tag
                )
                wheels.extend(wheels_found)
                logs.append(f"Found {len(wheels_found)} wheels on history page.")
            except Exception as exc:
                msg = f"Failed to fetch history page: {exc}"
                logs.append(msg)
                logger.exception(msg)

        # Deduplicate wheels
        unique: List[WheelInfo] = []
        seen: Set[str] = set()

        for w in wheels:
            if w.url not in seen:
                seen.add(w.url)
                unique.append(w)

        wheels = unique

        # Final result
        if wheels:
            logs.append(f"Success: {len(wheels)} wheels found.")
        else:
            logs.append("Failure: No wheels found for this package+version.")

        return wheels, logs

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _normalize_url(self, href: str) -> Optional[str]:
        """
        Normalize PyPI wheel URLs.
        Returns None for invalid/anchor links.
        """
        if href.startswith("#"):
            return None  # anchor → invalid

        if href.startswith("/packages"):
            return "https://files.pythonhosted.org" + href

        if href.startswith("packages"):
            return "https://files.pythonhosted.org/" + href

        if href.startswith("http"):
            return href

        return None  # unknown format → skip

    def _scrape_files_page(
        self,
        package: str,
        version: str,
        url: str,
        python_tag: Optional[str],
        platform_tag: Optional[str],
    ) -> List[WheelInfo]:

        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        links = soup.find_all("a", href=True)

        wheels: List[WheelInfo] = []

        for a in links:
            href = a["href"]
            if not href.endswith(".whl"):
                continue

            filename = href.split("/")[-1]

            if python_tag and python_tag not in filename:
                continue

            if platform_tag and platform_tag not in filename:
                continue

            full_url = self._normalize_url(href)
            if not full_url:
                continue

            wheels.append(WheelInfo(package, version, filename, full_url))

        return wheels

    def _scrape_history_page(
        self,
        package: str,
        version: str,
        url: str,
        python_tag: Optional[str],
        platform_tag: Optional[str],
    ) -> List[WheelInfo]:

        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        version_headers = soup.find_all("a", {"class": "release__version"})
        target_section = None

        for header in version_headers:
            if header.text.strip() == version:
                target_section = header.parent
                break

        if not target_section:
            return []

        links = target_section.find_all("a", href=True)
        wheels: List[WheelInfo] = []

        for a in links:
            href = a["href"]
            if not href.endswith(".whl"):
                continue

            filename = href.split("/")[-1]

            if python_tag and python_tag not in filename:
                continue

            if platform_tag and platform_tag not in filename:
                continue

            full_url = self._normalize_url(href)
            if not full_url:
                continue

            wheels.append(WheelInfo(package, version, filename, full_url))

        return wheels
