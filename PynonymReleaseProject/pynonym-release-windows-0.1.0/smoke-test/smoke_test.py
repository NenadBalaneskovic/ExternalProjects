# smoke_test.py
# Windows-kompatibler Smoke-Test für pynonym

import pandas as pd

from pynonym import anonymize_text
from pynonym.tables import (
    TableAnonymizationConfig,
    anonymize_dataframe,
)


print("=== Smoke Test: Text-Anonymisierung ===")

text = "Angela Merkel traf Olaf Scholz in Berlin."
result = anonymize_text(text)

print("Original:", text)
print("Anonymisiert:", result)
print()


print("=== Smoke Test: Tabellen-Anonymisierung ===")

df = pd.DataFrame({
    "Name": ["Angela Merkel", "Olaf Scholz", "Karl Lauterbach"],
    "Stadt": ["Berlin", "Hamburg", "Köln"],
    "Diagnose": ["A", "B", "A"],
})

config = TableAnonymizationConfig(
    quasi_identifiers=["Stadt"],
    sensitive_attributes=["Diagnose"],
    pseudonymize_columns=["Name"],
    k=2,
    l=2,
    t=0.2,
    language="de",
    seed=42,
)

df_anonym = anonymize_dataframe(df, config)

print("Original-DF:")
print(df)
print()

print("Anonymisiertes DF:")
print(df_anonym)
print()

print("=== Privacy-Metriken (unter Windows deaktiviert) ===")
print("k-Anonymität:", df_anonym.attrs.get("k_anonymity"))
print("l-Diversität:", df_anonym.attrs.get("l_diversity"))
print("t-Closeness:", df_anonym.attrs.get("t_closeness"))
print()

print("=== Smoke Test abgeschlossen ===")
