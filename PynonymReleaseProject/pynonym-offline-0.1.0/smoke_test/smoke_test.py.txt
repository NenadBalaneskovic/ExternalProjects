import pandas as pd
import pynonym
from pynonym import PynonymConfig
from pynonym.text import anonymize_text
from pynonym.tables import TableAnonymizationConfig, anonymize_dataframe
print("=== 1. Imports erfolgreich ===")
# 2. spaCy Test
import spacy
nlp = spacy.load("de_core_news_md")
print("spaCy Modell geladen:", nlp.meta["name"])
# 3. Text-Anonymisierung
cfg = PynonymConfig(language="de", seed=42)
text = "Angela Merkel traf Olaf Scholz in Berlin."
anon = anonymize_text(text, config=cfg)
print("\n=== 2. Text Anonymisierung ===")
print("Original:", text)
print("Anonymisiert:", anon)
# 4. Tabellen-Anonymisierung
df = pd.DataFrame({
    "Name": ["Angela Merkel", "Olaf Scholz", "Karl Lauterbach"],
    "Stadt": ["Berlin", "Hamburg", "Köln"],
    "Diagnose": ["A", "B", "A"]
})
tcfg = TableAnonymizationConfig(
    pseudonymize_columns=["Name"],
    quasi_identifiers=["Stadt"],
    sensitive_attributes=["Diagnose"],
    seed=42,
    k=2,
    l=1,
    t=0.5
)
df_anon = anonymize_dataframe(df, config=tcfg)
print("\n=== 3. Tabellen Anonymisierung ===")
print("Original DF:")
print(df)
print("\nAnonymisiertes DF:")
print(df_anon)
# 5. Privacy-Metriken
print("\n=== 4. Privacy Metriken ===")
print(df_anon.attrs)
# 6. Determinismus
cfg2 = PynonymConfig(language="de", seed=42)
anon2 = anonymize_text(text, config=cfg2)
print("\n=== 5. Determinismus Test ===")
print("Deterministisch:", anon == anon2)
print("\n=== Smoke Test abgeschlossen ===")
