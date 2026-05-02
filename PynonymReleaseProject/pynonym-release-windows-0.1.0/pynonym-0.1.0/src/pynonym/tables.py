# src/pynonym/tables.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

# ---------------------------------------------------------
# Optionaler Import von pycanon (Linux-only)
# ---------------------------------------------------------
try:
    from pycanon.anonymity import k_anonymity, l_diversity, t_closeness
    HAS_PYCANON = True
except ImportError:
    HAS_PYCANON = False
    k_anonymity = None
    l_diversity = None
    t_closeness = None

from .config import PynonymConfig
from .utils import pseudonymize_value, normalize_string


# ---------------------------------------------------------
# Hilfsfunktion für deaktivierte Privacy-Metriken
# ---------------------------------------------------------

def _metric_unavailable(name: str):
    return {
        "metric": name,
        "value": None,
        "status": "pycanon_not_available",
        "message": f"{name} ist unter Windows deaktiviert (pycanon nicht installiert)."
    }


# ---------------------------------------------------------
# 1. Konfiguration für Tabellenanonymisierung
# ---------------------------------------------------------

@dataclass
class TableAnonymizationConfig:
    quasi_identifiers: List[str]
    sensitive_attributes: Optional[List[str]] = None
    pseudonymize_columns: Optional[List[str]] = None

    k: int = 5
    l: Optional[int] = None
    t: Optional[float] = None

    language: str = "de"
    seed: Optional[int] = None

    def to_pynonym_config(self) -> PynonymConfig:
        return PynonymConfig(language=self.language, seed=self.seed)


# ---------------------------------------------------------
# 2. Tabellen-Anonymizer
# ---------------------------------------------------------

class TableAnonymizer:
    def __init__(self, config: TableAnonymizationConfig):
        self.config = config
        self.pcfg = config.to_pynonym_config()

    def _apply_pseudonymization(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.pseudonymize_columns:
            return df

        df = df.copy()

        for col in self.config.pseudonymize_columns:
            if col not in df.columns:
                continue

            df[col] = df[col].apply(
                lambda v: pseudonymize_value(normalize_string(v), self.pcfg)
            )

        return df

    def _apply_k_anonymity(self, df: pd.DataFrame) -> None:
        if not HAS_PYCANON:
            print("Warnung: pycanon nicht verfügbar. k-Anonymität deaktiviert.")
            df.attrs["k_anonymity"] = _metric_unavailable("k-anonymity")
            return

        result = k_anonymity(
            df,
            qi=self.config.quasi_identifiers,
            k=self.config.k,
        )
        df.attrs["k_anonymity"] = result

    def _apply_l_diversity(self, df: pd.DataFrame) -> None:
        if not self.config.l or not self.config.sensitive_attributes:
            return

        if not HAS_PYCANON:
            print("Warnung: pycanon nicht verfügbar. l-Diversität deaktiviert.")
            df.attrs["l_diversity"] = _metric_unavailable("l-diversity")
            return

        result = l_diversity(
            df,
            qi=self.config.quasi_identifiers,
            sa=self.config.sensitive_attributes,
            l=self.config.l,
        )
        df.attrs["l_diversity"] = result

    def _apply_t_closeness(self, df: pd.DataFrame) -> None:
        if not self.config.t or not self.config.sensitive_attributes:
            return

        if not HAS_PYCANON:
            print("Warnung: pycanon nicht verfügbar. t-Closeness deaktiviert.")
            df.attrs["t_closeness"] = _metric_unavailable("t-closeness")
            return

        result = t_closeness(
            df,
            qi=self.config.quasi_identifiers,
            sa=self.config.sensitive_attributes,
            t=self.config.t,
        )
        df.attrs["t_closeness"] = result

    def anonymize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df = self._apply_pseudonymization(df)

        self._apply_k_anonymity(df)
        self._apply_l_diversity(df)
        self._apply_t_closeness(df)

        return df


def anonymize_dataframe(
    df: pd.DataFrame,
    config: TableAnonymizationConfig,
) -> pd.DataFrame:
    anonymizer = TableAnonymizer(config)
    return anonymizer.anonymize(df)
