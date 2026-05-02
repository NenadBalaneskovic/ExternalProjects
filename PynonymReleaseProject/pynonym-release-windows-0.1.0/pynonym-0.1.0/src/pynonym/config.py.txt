# src/pynonym/config.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Dict
from faker import Faker


# ---------------------------------------------------------
# 1. Sprachunterstützung
# ---------------------------------------------------------

Language = Literal["de", "en"]

SPACY_MODELS = {
    "de": "de_core_news_md",
    "en": "en_core_web_md",
}

FAKER_LOCALES = {
    "de": "de_DE",
    "en": "en_US",
}


# ---------------------------------------------------------
# 2. Globale Replacement-Map (Text + Tabellen)
# ---------------------------------------------------------

GLOBAL_REPLACEMENT_MAP: Dict[str, str] = {}


def get_global_replacement_map() -> Dict[str, str]:
    """Gibt die globale Map zurück (für Text + Tabellen)."""
    return GLOBAL_REPLACEMENT_MAP


# ---------------------------------------------------------
# 3. Konfigurationsobjekt
# ---------------------------------------------------------

@dataclass
class PynonymConfig:
    """
    Zentrale Konfiguration für Text- und Tabellenanonymisierung.
    Sprache bestimmt:
    - spaCy-Modell
    - Faker-Locale
    """
    language: Language = "de"
    seed: int | None = None

    def spacy_model(self) -> str:
        return SPACY_MODELS[self.language]

    def faker_locale(self) -> str:
        return FAKER_LOCALES[self.language]


# ---------------------------------------------------------
# 4. Faker-Instanz (deterministisch, global)
# ---------------------------------------------------------

_faker_instance: Faker | None = None


def get_faker(config: PynonymConfig | None = None) -> Faker:
    """
    Gibt eine globale Faker-Instanz zurück.
    Wird für Text + Tabellen verwendet.
    """
    global _faker_instance

    if _faker_instance is None:
        lang = config.language if config else "de"
        locale = FAKER_LOCALES[lang]
        _faker_instance = Faker(locale)

        if config and config.seed is not None:
            Faker.seed(config.seed)

    return _faker_instance