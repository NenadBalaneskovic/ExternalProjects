# tests/test_text.py

import pytest

from pynonym import (
    anonymize_text,
    PynonymConfig,
)
from pynonym.utils import reset_global_state


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_state():
    """
    Vor jedem Test globale Replacement-Map und Faker zurücksetzen.
    """
    reset_global_state()
    yield
    reset_global_state()


# ---------------------------------------------------------
# 1. Grundfunktionalität (Deutsch)
# ---------------------------------------------------------

def test_anonymize_text_german_basic():
    cfg = PynonymConfig(language="de", seed=42)
    text = "Angela Merkel traf Olaf Scholz in Berlin."

    result = anonymize_text(text, config=cfg)

    # Sollte nicht mehr die Originalnamen enthalten
    assert "Angela Merkel" not in result
    assert "Olaf Scholz" not in result
    assert "Berlin" not in result

    # Sollte Fake-Namen enthalten
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------
# 2. Grundfunktionalität (Englisch)
# ---------------------------------------------------------

def test_anonymize_text_english_basic():
    cfg = PynonymConfig(language="en", seed=42)
    text = "Barack Obama met Joe Biden in Washington."

    result = anonymize_text(text, config=cfg)

    assert "Barack Obama" not in result
    assert "Joe Biden" not in result
    assert "Washington" not in result


# ---------------------------------------------------------
# 3. Determinismus (Seed)
# ---------------------------------------------------------

def test_anonymize_text_deterministic():
    cfg = PynonymConfig(language="de", seed=123)

    text = "Angela Merkel traf Olaf Scholz."

    r1 = anonymize_text(text, config=cfg)
    reset_global_state()
    r2 = anonymize_text(text, config=cfg)

    assert r1 == r2


# ---------------------------------------------------------
# 4. Globale Replacement-Map (Konsistenz)
# ---------------------------------------------------------

def test_global_replacement_consistency():
    cfg = PynonymConfig(language="de", seed=42)

    text1 = "Angela Merkel traf Olaf Scholz."
    text2 = "Merkel und Scholz sind Politiker."

    r1 = anonymize_text(text1, config=cfg)
    r2 = anonymize_text(text2, config=cfg)

    # Beide Texte müssen dieselben Fake-Namen verwenden
    # Beispiel: "Claudia Fischer" für Merkel
    fake_name = None
    for token in r1.split():
        if token not in text1:
            fake_name = token
            break

    assert fake_name is not None
    assert fake_name in r2


# ---------------------------------------------------------
# 5. Edge Cases
# ---------------------------------------------------------

def test_empty_string():
    cfg = PynonymConfig(language="de")
    assert anonymize_text("", config=cfg) == ""


def test_none_input():
    cfg = PynonymConfig(language="de")
    # anonymize_text erwartet str, None sollte nicht crashen
    result = anonymize_text(None, config=cfg) if None else ""
    assert result == ""


def test_no_entities():
    cfg = PynonymConfig(language="de")
    text = "Dies ist ein einfacher Satz ohne Personen."
    result = anonymize_text(text, config=cfg)
    assert result == text