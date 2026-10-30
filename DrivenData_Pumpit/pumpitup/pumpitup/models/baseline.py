from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from pumpitup.data.loader import PumpDataLoader
from pumpitup.features.engineer import FeatureEngineer


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
def load_data(data_dir="data"):
    data_dir = Path(data_dir)

    loader = PumpDataLoader()
    loader.load_raw_csvs(
        train_values=str(data_dir / "TrainingSetValues.csv"),
        train_labels=str(data_dir / "TrainingSetLabels.csv"),
        test_values=str(data_dir / "TestSetValues.csv"),
    )

    train_df = loader.get_training_dataframe()
    loader.close()

    X_train = train_df.drop(columns=["status_group"])
    y_train = train_df["status_group"]

    # Encode labels for XGBoost and CatBoost
    label_map = {
        "functional": 0,
        "functional needs repair": 1,
        "non functional": 2
    }
    y_train = y_train.map(label_map)

    return X_train, y_train


# ---------------------------------------------------------
# Build pipelines for sklearn-compatible models
# ---------------------------------------------------------
def build_rf():
    return Pipeline([
        ("fe", FeatureEngineer()),
        ("model", RandomForestClassifier(
            n_estimators=400,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        )),
    ])


def build_et():
    return Pipeline([
        ("fe", FeatureEngineer()),
        ("model", ExtraTreesClassifier(
            n_estimators=400,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        )),
    ])


def build_lgbm():
    return Pipeline([
        ("fe", FeatureEngineer()),
        ("model", LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def build_xgb():
    return Pipeline([
        ("fe", FeatureEngineer()),
        ("model", XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )),
    ])


# ---------------------------------------------------------
# Manual CatBoost evaluation (NO PIPELINE)
# ---------------------------------------------------------
def evaluate_catboost(X, y, n_splits=5):
    print("\nEvaluating CatBoost (manual CV)...")

    fe = FeatureEngineer()
    X_fe = fe.fit_transform(X, y)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in cv.split(X_fe, y):
        X_train_fold, X_test_fold = X_fe.iloc[train_idx], X_fe.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        model = CatBoostClassifier(
            iterations=600,
            learning_rate=0.05,
            depth=8,
            loss_function="MultiClass",
            random_seed=42,
            verbose=False,
        )

        model.fit(X_train_fold, y_train_fold)

        preds = model.predict(X_test_fold)

        # FIX: ensure preds is 1-D
        if preds.ndim > 1:
            preds = np.argmax(preds, axis=1)

        preds = preds.astype(int)

        scores.append((preds == y_test_fold.values).mean())

    scores = np.array(scores)
    print(f"CatBoost CV accuracy: mean={scores.mean():.4f}, std={scores.std():.4f}")
    return scores


# ---------------------------------------------------------
# Standard evaluation for sklearn-compatible models
# ---------------------------------------------------------
def evaluate_pipeline(name, pipeline, X, y, n_splits=5):
    print(f"\nEvaluating {name}...")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    print(f"{name} CV accuracy: mean={scores.mean():.4f}, std={scores.std():.4f}")
    return scores


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    X_train, y_train = load_data()

    models = {
        "RandomForest": build_rf(),
        "ExtraTrees": build_et(),
        "LightGBM": build_lgbm(),
        "XGBoost": build_xgb(),
    }

    for name, model in models.items():
        evaluate_pipeline(name, model, X_train, y_train)

    evaluate_catboost(X_train, y_train)


if __name__ == "__main__":
    main()
