# mlflow_local_runner/src/examples/example_script.py
"""
example_script.py – Beispielskript für MLflow Local Runner

Dieses Skript:
- liest das Dataset über die ENV-Variable DATASET_PATH
- trainiert ein einfaches Modell (RandomForestClassifier)
- berechnet Accuracy und F1-Score
- gibt Modellrepräsentation und Metriken über stdout aus
  (für den Runner, der die Marker MODEL_READY / METRICS_READY erkennt)
"""

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier


def main():
    # ---------------------------------------------------------
    # 1. Dataset laden
    # ---------------------------------------------------------
    dataset_path = os.environ.get("DATASET_PATH")
    if not dataset_path:
        raise RuntimeError("DATASET_PATH ist nicht gesetzt.")

    df = pd.read_csv(dataset_path)

    # Annahme: letzte Spalte ist das Target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # ---------------------------------------------------------
    # 2. Train/Test Split
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------------------------------------
    # 3. Modell trainieren
    # ---------------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 4. Vorhersagen & Metriken
    # ---------------------------------------------------------
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted"))
    }

    # ---------------------------------------------------------
    # 5. Modellrepräsentation ausgeben
    # ---------------------------------------------------------
    print("MODEL_READY")
    print(str(model))

    # ---------------------------------------------------------
    # 6. Metriken ausgeben
    # ---------------------------------------------------------
    print("METRICS_READY")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
