import optuna
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from pumpitup.data.loader import PumpDataLoader
from pumpitup.features.engineer import FeatureEngineer
from pumpitup.experiments.tracker import ExperimentTracker
from pumpitup.models.utils import save_json


# ---------------------------------------------------------
# Load training data (with label encoding)
# ---------------------------------------------------------
def load_training_data(data_dir="data"):
    data_dir = Path(data_dir)

    loader = PumpDataLoader()
    loader.load_raw_csvs(
        train_values=str(data_dir / "TrainingSetValues.csv"),
        train_labels=str(data_dir / "TrainingSetLabels.csv"),
        test_values=str(data_dir / "TestSetValues.csv"),
    )

    train_df = loader.get_training_dataframe()
    loader.close()

    X = train_df.drop(columns=["status_group"])
    y = train_df["status_group"]

    label_map = {
        "functional": 0,
        "functional needs repair": 1,
        "non functional": 2,
    }
    y = y.map(label_map)

    return X, y


# ---------------------------------------------------------
# Build base models with Optuna parameters (RF, ET, LGBM, XGB)
# ---------------------------------------------------------
def build_models(trial):
    models = {}

    models["rf"] = Pipeline([
        ("fe", FeatureEngineer()),
        ("model", RandomForestClassifier(
            n_estimators=trial.suggest_int("rf_n_estimators", 200, 800),
            max_depth=trial.suggest_int("rf_max_depth", 8, 40),
            min_samples_split=trial.suggest_int("rf_min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 1, 5),
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )),
    ])

    models["et"] = Pipeline([
        ("fe", FeatureEngineer()),
        ("model", ExtraTreesClassifier(
            n_estimators=trial.suggest_int("et_n_estimators", 200, 800),
            max_depth=trial.suggest_int("et_max_depth", 8, 40),
            min_samples_split=trial.suggest_int("et_min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("et_min_samples_leaf", 1, 5),
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )),
    ])

    models["lgbm"] = Pipeline([
        ("fe", FeatureEngineer()),
        ("model", LGBMClassifier(
            n_estimators=trial.suggest_int("lgbm_n_estimators", 300, 800),
            learning_rate=trial.suggest_float("lgbm_learning_rate", 0.01, 0.2),
            num_leaves=trial.suggest_int("lgbm_num_leaves", 32, 256),
            subsample=trial.suggest_float("lgbm_subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("lgbm_colsample", 0.6, 1.0),
            min_child_samples=trial.suggest_int("lgbm_min_child", 5, 50),
            class_weight="balanced",
            boosting_type="goss",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    models["xgb"] = Pipeline([
        ("fe", FeatureEngineer()),
        ("model", XGBClassifier(
            n_estimators=trial.suggest_int("xgb_n_estimators", 300, 800),
            learning_rate=trial.suggest_float("xgb_learning_rate", 0.01, 0.2),
            max_depth=trial.suggest_int("xgb_max_depth", 4, 12),
            subsample=trial.suggest_float("xgb_subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("xgb_colsample", 0.6, 1.0),
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            max_bin=128,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    return models


# ---------------------------------------------------------
# Optuna objective (with faster CV)
# ---------------------------------------------------------
def objective(trial):
    X, y = load_training_data()

    models = build_models(trial)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    classes = np.unique(y)
    n_classes = len(classes)
    y_encoded = y.values

    oof_matrix = np.zeros((len(X), len(models) * n_classes))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        col_offset = 0

        for name, model in models.items():
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_val)

            oof_matrix[val_idx, col_offset:col_offset + n_classes] = proba
            col_offset += n_classes

    meta_params = {
        "n_estimators": trial.suggest_int("meta_n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("meta_learning_rate", 0.01, 0.2),
        "num_leaves": trial.suggest_int("meta_num_leaves", 32, 256),
        "random_state": 42,
        "boosting_type": "goss",
    }

    meta_model = LGBMClassifier(**meta_params)
    meta_model.fit(oof_matrix, y_encoded)

    preds = meta_model.predict(oof_matrix)
    acc = accuracy_score(y_encoded, preds)

    return acc


# ---------------------------------------------------------
# Run Optuna study (with pruning and fewer trials)
# ---------------------------------------------------------
def main():
    tracker = ExperimentTracker()

    study = optuna.create_study(
        direction="maximize",
        study_name="pumpitup_full_stack_opt",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )

    study.optimize(objective, n_trials=10, timeout=600)

    print("\n==============================")
    print(" Fast Full Stack Optuna Finished ")
    print("==============================")
    print(f"Best CV accuracy: {study.best_value:.5f}")
    print("Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    save_json(study.best_params, "best_full_stack_params.json")

    tracker.log(
        name="full_stack_optuna_fast",
        params=study.best_params,
        cv_score=study.best_value,
        notes="Fast full stack tuning (RF/ET/LGBM/XGB, pruned, 3-fold, 10 trials)",
    )

    print("\nSaved best_full_stack_params.json")


if __name__ == "__main__":
    main()
