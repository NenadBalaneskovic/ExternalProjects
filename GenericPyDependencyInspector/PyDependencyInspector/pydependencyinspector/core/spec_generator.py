"""
core/spec_generator.py

Responsible for:
- Generating PyInstaller spec files dynamically.
- Injecting OS-specific settings (binaries, hidden imports, data files).
- Providing a structured SpecGenerationResult object.
- Ensuring reproducible, deterministic spec output.

This module is intentionally:
- GUI-agnostic.
- Pure Python (no PyInstaller imports required).
- Safe to run in offline environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os
import logging
import textwrap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SpecGenerationResult:
    """
    Represents the outcome of generating a spec file.

    Fields:
    - success: True if spec file was written successfully
    - spec_file: path to the generated spec file
    - warnings: non-fatal issues (missing metadata, unknown OS)
    - errors: fatal issues (cannot write file, missing entry point)
    """
    success: bool
    spec_file: str
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def is_successful(self) -> bool:
        return self.success and not self.errors


# ---------------------------------------------------------------------------
# SpecGenerator – public API
# ---------------------------------------------------------------------------

class SpecGenerator:
    """
    Generates PyInstaller spec files for a given package.

    Responsibilities:
    - Determine entry point (via importlib.metadata)
    - Build a deterministic spec file
    - Inject OS-specific settings
    - Return a SpecGenerationResult

    The GUI will call:
        generator = SpecGenerator()
        result = generator.generate("pandas", "Windows 11", "build/pandas.spec")
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate(
        self,
        package_name: str,
        os_profile_str: str,
        output_path: str,
        hidden_imports: Optional[List[str]] = None,
        data_files: Optional[List[tuple]] = None,
        binaries: Optional[List[tuple]] = None,
    ) -> SpecGenerationResult:
        """
        Generate a PyInstaller spec file.

        :param package_name: Name of the Python package (e.g. "pandas")
        :param os_profile_str: Human-readable OS string (e.g. "Windows 11")
        :param output_path: Path where the spec file should be written
        :param hidden_imports: Optional list of hidden imports
        :param data_files: Optional list of (src, dest) tuples
        :param binaries: Optional list of (src, dest) tuples
        :return: SpecGenerationResult
        """
        warnings: List[str] = []
        errors: List[str] = []

        # Step 1: Determine entry point
        entry_point = self._find_entry_point(package_name, warnings, errors)
        if entry_point is None:
            return SpecGenerationResult(
                success=False,
                spec_file=output_path,
                warnings=warnings,
                errors=errors,
            )

        # Step 2: Normalize OS profile
        os_profile = self._normalize_os(os_profile_str, warnings)

        # Step 3: Build spec file content
        spec_content = self._build_spec_content(
            package_name=package_name,
            entry_point=entry_point,
            os_profile=os_profile,
            hidden_imports=hidden_imports or [],
            data_files=data_files or [],
            binaries=binaries or [],
        )

        # Step 4: Write spec file
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(spec_content)
        except Exception as exc:
            msg = f"Failed to write spec file: {exc}"
            logger.exception(msg)
            errors.append(msg)
            return SpecGenerationResult(
                success=False,
                spec_file=output_path,
                warnings=warnings,
                errors=errors,
            )

        logger.info("Spec file generated: %s", output_path)
        return SpecGenerationResult(
            success=True,
            spec_file=output_path,
            warnings=warnings,
            errors=errors,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _find_entry_point(
        self,
        package_name: str,
        warnings: List[str],
        errors: List[str],
    ) -> Optional[str]:
        """
        Determine the entry point of the package via importlib.metadata.

        Returns:
            path to entry script (string) or None
        """
        try:
            import importlib.metadata as metadata
        except ImportError:
            import importlib_metadata as metadata  # Python <3.8

        try:
            dist = metadata.distribution(package_name)
        except metadata.PackageNotFoundError:
            msg = f"Package '{package_name}' not found in current environment."
            logger.error(msg)
            errors.append(msg)
            return None

        # Look for console_scripts entry points
        entry_points = dist.entry_points
        console_scripts = [ep for ep in entry_points if ep.group == "console_scripts"]

        if console_scripts:
            # Use the first console script as entry point
            ep = console_scripts[0]
            return f"{ep.module}:{ep.attr}" if ep.attr else ep.module

        # Fallback: try to import the package and use __main__
        try:
            __import__(package_name)
            return f"{package_name}.__main__"
        except Exception:
            msg = f"Could not determine entry point for package '{package_name}'."
            logger.error(msg)
            errors.append(msg)
            return None

    def _normalize_os(self, os_profile_str: str, warnings: List[str]) -> str:
        """
        Normalize OS profile string to a simple identifier.
        """
        s = os_profile_str.lower()
        if "win" in s:
            return "windows"
        if "linux" in s or "ubuntu" in s:
            return "linux"
        if "mac" in s or "darwin" in s:
            return "macos"

        warnings.append(f"Unknown OS profile '{os_profile_str}', using generic settings.")
        return "unknown"

    def _build_spec_content(
        self,
        package_name: str,
        entry_point: str,
        os_profile: str,
        hidden_imports: List[str],
        data_files: List[tuple],
        binaries: List[tuple],
    ) -> str:
        """
        Build the actual spec file content as a string.
        """
        hidden_imports_str = ",\n        ".join(f"'{h}'" for h in hidden_imports)
        datas_str = ",\n        ".join(f"('{src}', '{dest}')" for src, dest in data_files)
        binaries_str = ",\n        ".join(f"('{src}', '{dest}')" for src, dest in binaries)

        return textwrap.dedent(f"""
        # -*- mode: python ; coding: utf-8 -*-

        block_cipher = None

        a = Analysis(
            ['{entry_point}'],
            pathex=[],
            binaries=[
                {binaries_str}
            ],
            datas=[
                {datas_str}
            ],
            hiddenimports=[
                {hidden_imports_str}
            ],
            hookspath=[],
            runtime_hooks=[],
            excludes=[],
            win_no_prefer_redirects=False,
            win_private_assemblies=False,
            cipher=block_cipher,
        )

        pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

        exe = EXE(
            pyz,
            a.scripts,
            a.binaries,
            a.zipfiles,
            a.datas,
            name='{package_name}',
            debug=False,
            strip=False,
            upx=True,
            console=True,
        )
        """).strip()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def generate_spec(
    package_name: str,
    os_profile_str: str,
    output_path: str,
    hidden_imports: Optional[List[str]] = None,
    data_files: Optional[List[tuple]] = None,
    binaries: Optional[List[tuple]] = None,
) -> SpecGenerationResult:
    """
    Convenience wrapper for simple usage.
    """
    generator = SpecGenerator()
    return generator.generate(
        package_name,
        os_profile_str,
        output_path,
        hidden_imports,
        data_files,
        binaries,
    )
