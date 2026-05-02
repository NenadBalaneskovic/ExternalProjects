# src/pynonym/text.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List

import spacy

from .config import PynonymConfig
from .utils import pseudonymize_value


DEFAULT_ENTITIES: Tuple[str, ...] = (
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
)


class TextAnonymizer:
    """
    Zweisprachige Text-Anonymisierung basierend auf spaCy + Faker.
    Nutzt globale Replacement-Map für konsistente Pseudonyme.
    """

    def __init__(
        self,
        config: PynonymConfig | None = None,
        entities_to_anonymize: Tuple[str, ...] = DEFAULT_ENTITIES,
    ):
        self.config = config or PynonymConfig()
        self.entities_to_anonymize = entities_to_anonymize

        model_name = self.config.spacy_model()
        try:
            self.nlp = spacy.load(model_name)
        except OSError as e:
            raise RuntimeError(
                f"spaCy model '{model_name}' is not installed. "
                f"Install it via: python -m spacy download {model_name}"
            ) from e

    def anonymize(self, text: str) -> str:
        if not text:
            return text

        doc = self.nlp(text)
        result = text

        for ent in reversed(doc.ents):
            if ent.label_ not in self.entities_to_anonymize:
                continue

            original = ent.text
            replacement = pseudonymize_value(original, self.config)

            start, end = ent.start_char, ent.end_char
            result = result[:start] + replacement + result[end:]

        return result


def anonymize_text(
    text: str,
    config: PynonymConfig | None = None,
    entities: Tuple[str, ...] = DEFAULT_ENTITIES,
) -> str:
    anonymizer = TextAnonymizer(config=config, entities_to_anonymize=entities)
    return anonymizer.anonymize(text)
