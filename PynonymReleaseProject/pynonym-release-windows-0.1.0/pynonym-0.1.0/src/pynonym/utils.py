# src/pynonym/utils.py

from __future__ import annotations
from typing import Dict, Any
from faker import Faker

from .config import (
    PynonymConfig,
    get_global_replacement_map,
    get_faker,
)


# ---------------------------------------------------------
# 1. Deterministische Pseudonymisierung (global)
# ---------------------------------------------------------

def pseudonymize_value(value: str, config: PynonymConfig) -> str:
    """
    Erzeugt einen deterministischen Fake-Wert für einen gegebenen String.
    Nutzt die globale Replacement-Map und die globale Faker-Instanz.
    """
    if value is None:
        return value

    value = normalize_string(value)
    if not value:
        return value

    repl_map = get_global_replacement_map()
    faker = get_faker(config)

    if value not in repl_map:
        repl_map[value] = faker.name()

    return repl_map[value]


# ---------------------------------------------------------
# 2. Normalisierung
# ---------------------------------------------------------

def normalize_string(value: Any) -> str:
    """
    Konvertiert beliebige Werte in Strings und trimmt Whitespace.
    """
    if value is None:
        return ""
    return str(value).strip()


def normalize_language(lang: str) -> str:
    """
    Normalisiert Sprachcodes wie 'DE', 'de', 'De' → 'de'.
    """
    if not lang:
        return "de"
    return lang.lower().strip()


# ---------------------------------------------------------
# 3. Utility: sichere Sprachauswahl
# ---------------------------------------------------------

def ensure_valid_language(lang: str) -> str:
    """
    Stellt sicher, dass nur 'de' oder 'en' verwendet wird.
    """
    lang = normalize_language(lang)
    if lang not in ("de", "en"):
        return "de"
    return lang


# ---------------------------------------------------------
# 4. Utility: deterministische Faker-Instanz
# ---------------------------------------------------------

def faker_for_language(lang: str, seed: int | None = None) -> Faker:
    """
    Gibt eine Faker-Instanz für eine bestimmte Sprache zurück.
    Wird selten benötigt, da get_faker(config) global arbeitet.
    """
    lang = ensure_valid_language(lang)

    locale = "de_DE" if lang == "de" else "en_US"
    faker = Faker(locale)

    if seed is not None:
        Faker.seed(seed)

    return faker


# ---------------------------------------------------------
# 5. Utility: Mapping-Reset (für Tests)
# ---------------------------------------------------------

def reset_global_state() -> None:
    """
    Löscht die globale Replacement-Map und setzt Faker zurück.
    Wird in Tests verwendet.
    """
    repl_map = get_global_replacement_map()
    repl_map.clear()

    # Faker-Reset: globale Instanz wirklich zurücksetzen
    import pynonym.config as cfg
    cfg._faker_instance = None
