# tests/test_tables.py

import pytest
import pandas as pd

from pynonym import (
    anonymize_dataframe,
    TableAnonymizationConfig,
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

def test_anonymize_dataframe_german_basic():
    df = pd.DataFrame({
        "Name": ["Angela Merkel", "Olaf Scholz"],
        "Alter": [67, 65],
        "Stadt": ["Berlin", "Berlin"],
        "Diagnose": ["A", "B"],
    })

    cfg = TableAnonymizationConfig(
        quasi_identifiers=["Alter", "Stadt"],
        sensitive_attributes=["Diagnose"],
        pseudonymize_columns=["Name"],
        language="de",
        seed=42,
        k=1,
    )

    result = anonymize_dataframe(df, config=cfg)

    # Originalnamen dürfen nicht mehr vorkommen
    assert "Angela Merkel" not in result["Name"].tolist()
    assert "Olaf Scholz" not in result["Name"].tolist()

    # Fake-Namen müssen Strings sein
    assert all(isinstance(x, str) for x in result["Name"])


# ---------------------------------------------------------
# 2. Grundfunktionalität (Englisch)
# ---------------------------------------------------------

def test_anonymize_dataframe_english_basic():
    df = pd.DataFrame({
        "Name": ["Barack Obama", "Joe Biden"],
        "Age": [60, 61],
        "City": ["Washington", "Washington"],
        "Condition": ["X", "Y"],
    })

    cfg = TableAnonymizationConfig(
        quasi_identifiers=["Age", "City"],
        sensitive_attributes=["Condition"],
        pseudonymize_columns=["Name"],
        language="en",
        seed=42,
        k=1,
    )

    result = anonymize_dataframe(df, config=cfg)

    assert "Barack Obama" not in result["Name"].tolist()
    assert "Joe Biden" not in result["Name"].tolist()


# ---------------------------------------------------------
# 3. Determinismus (Seed)
# ---------------------------------------------------------

def test_anonymize_dataframe_deterministic():
    df = pd.DataFrame({
        "Name": ["Angela Merkel"],
        "Alter": [67],
        "Stadt": ["Berlin"],
    })

    cfg = TableAnonymizationConfig(
        quasi_identifiers=["Alter", "Stadt"],
        pseudonymize_columns=["Name"],
        language="de",
        seed=123,
        k=1,
    )

    r1 = anonymize_dataframe(df, config=cfg)
    reset_global_state()
    r2 = anonymize_dataframe(df, config=cfg)

    assert r1.equals(r2)


# ---------------------------------------------------------
# 4. Globale Replacement-Map (Konsistenz mit Text)
# ---------------------------------------------------------

def test_global_replacement_consistency_with_text():
    from pynonym import anonymize_text

    # Tabellen-Konfiguration
    tcfg = TableAnonymizationConfig(
        quasi_identifiers=["Alter"],
        pseudonymize_columns=["Name"],
        language="de",
        seed=42,
        k=1,
    )

    # Text-Konfiguration
    xcfg = PynonymConfig(language="de", seed=42)

    df = pd.DataFrame({
        "Name": ["Angela Merkel"],
        "Alter": [67],
    })

    # Tabelle anonymisieren
    df_res = anonymize_dataframe(df, config=tcfg)
    fake_name_table = df_res["Name"].iloc[0]

    # Text anonymisieren
    text_res = anonymize_text("Angela Merkel ist Politikerin.", config=xcfg)

    # Fake-Name aus Tabelle muss im Text vorkommen
    assert fake_name_table in text_res


# ---------------------------------------------------------
# 5. Privacy-Metriken (k, l, t)
# ---------------------------------------------------------

def test_privacy_metrics():
    df = pd.DataFrame({
        "Name": ["A", "B", "C"],
        "Alter": [30, 30, 30],
        "Stadt": ["Berlin", "Berlin", "Berlin"],
        "Diagnose": ["X", "Y", "X"],
    })

    cfg = TableAnonymizationConfig(
        quasi_identifiers=["Alter", "Stadt"],
        sensitive_attributes=["Diagnose"],
        pseudonymize_columns=["Name"],
        language="de",
        seed=42,
        k=2,
        l=1,
        t=0.5,
    )

    result = anonymize_dataframe(df, config=cfg)

    assert "k_anonymity" in result.attrs
    assert "l_diversity" in result.attrs
    assert "t_closeness" in result.attrs


# ---------------------------------------------------------
# 6. Edge Cases
# ---------------------------------------------------------

def test_empty_dataframe():
    df = pd.DataFrame()
    cfg = TableAnonymizationConfig(
        quasi_identifiers=[],
        pseudonymize_columns=[],
        language="de",
    )
    result = anonymize_dataframe(df, config=cfg)
    assert result.empty


def test_missing_columns():
    df = pd.DataFrame({
        "A": [1, 2],
        "B": [3, 4],
    })

    cfg = TableAnonymizationConfig(
        quasi_identifiers=["A"],
        pseudonymize_columns=["Name"],  # existiert nicht
        language="de",
    )

    result = anonymize_dataframe(df, config=cfg)
    assert "A" in result.columns
    assert "B" in result.columns