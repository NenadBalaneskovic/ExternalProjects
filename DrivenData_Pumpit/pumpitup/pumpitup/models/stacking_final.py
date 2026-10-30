import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from pumpitup.data.loader import PumpDataLoader
from pumpitup.features.engineer import FeatureEngineer
from pumpitup.experiments.tracker import ExperimentTracker


# ---------------------------------------------------------
# Load tuned parameters
# ---------------------------------------------------------
def load_params(path="best_full_stack_params.json"):
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------
# Load data (with label encoding)
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
    test_df = loader.get_test_dataframe()
    loader.close()

    X_train = train_df.drop(columns=["status_group"])
    y_train_str = train_df["status_group"]
    X_test = test_df

    # TRUE string labels
    classes = np.array([
        "functional",
        "functional needs repair",
        "non functional"
    ])

    # Encode labels for model training
    label_map = {
        "functional": 0,
        "functional needs repair": 1,
        "non functional": 2,
    }
    y_train = y_train_str.map(label_map).values

    return X_train, y_train, X_test, classes


# ---------------------------------------------------------
# Build tuned base models (RF, ET, LGBM, XGB only)
# ---------------------------------------------------------
def build_tuned_models(params):
    models = {}

    models["rf"] = Pipeline([
        ("fe", FeatureEngineer()),
        ("model", RandomForestClassifier(
            n_estimators=params["rf_n_estimators"],
            max_depth=params["rf_max_depth"],
            min_samples_split=params["rf_min_samples_split"],
            min_samples_leaf=params["rf_min_samples_leaf"],
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )),
    ])

    models["et"] = Pipeline([
        ("fe", FeatureEngineer()),
        ("model", ExtraTreesClassifier(
            n_estimators=params["et_n_estimators"],
            max_depth=params["et_max_depth"],
            min_samples_split=params["et_min_samples_split"],
            min_samples_leaf=params["et_min_samples_leaf"],
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )),
    ])

    models["lgbm"] = Pipeline([
        ("fe", FeatureEngineer()),
        ("model", LGBMClassifier(
            n_estimators=params["lgbm_n_estimators"],
            learning_rate=params["lgbm_learning_rate"],
            num_leaves=params["lgbm_num_leaves"],
            subsample=params["lgbm_subsample"],
            colsample_bytree=params["lgbm_colsample"],
            min_child_samples=params["lgbm_min_child"],
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    models["xgb"] = Pipeline([
        ("fe", FeatureEngineer()),
        ("model", XGBClassifier(
            n_estimators=params["xgb_n_estimators"],
            learning_rate=params["xgb_learning_rate"],
            max_depth=params["xgb_max_depth"],
            subsample=params["xgb_subsample"],
            colsample_bytree=params["xgb_colsample"],
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    return models


# ---------------------------------------------------------
# Build CatBoost manually (NOT in Pipeline)
# ---------------------------------------------------------
def build_catboost(params):
    return CatBoostClassifier(
        iterations=params.get("cat_iterations", 800),
        learning_rate=params.get("cat_learning_rate", 0.1),
        depth=params.get("cat_depth", 6),
        loss_function="MultiClass",
        random_seed=42,
        verbose=False,
    )


# ---------------------------------------------------------
# Build tuned meta-model
# ---------------------------------------------------------
def build_meta_model(params):
    return LGBMClassifier(
        n_estimators=params["meta_n_estimators"],
        learning_rate=params["meta_learning_rate"],
        num_leaves=params["meta_num_leaves"],
        random_state=42,
    )


# ---------------------------------------------------------
# Train stacking ensemble
# ---------------------------------------------------------
def train_stacking(models, cat_model, meta_model, X, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    classes = np.unique(y)
    n_classes = len(classes)

    # 4 sklearn models + 1 CatBoost = 5 models
    oof_matrix = np.zeros((len(X), 5 * n_classes))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"\nFold {fold + 1}/{n_splits}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        col_offset = 0

        # sklearn models
        for name, model in models.items():
            print(f"  Training base model: {name}")
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_val)
            oof_matrix[val_idx, col_offset:col_offset + n_classes] = proba
            col_offset += n_classes

        # CatBoost manually
        print("  Training base model: cat")
        fe = FeatureEngineer()
        fe.fit(X_train, y_train)
        X_train_cat = fe.transform(X_train)
        X_val_cat = fe.transform(X_val)

        cat_model.fit(X_train_cat, y_train)
        proba_cat = cat_model.predict_proba(X_val_cat)
        oof_matrix[val_idx, col_offset:col_offset + n_classes] = proba_cat
        col_offset += n_classes

        # Meta-model fold accuracy
        meta_model.fit(oof_matrix[val_idx], y_val)
        preds_fold = meta_model.predict(oof_matrix[val_idx])
        acc_fold = accuracy_score(y_val, preds_fold)
        print(f"  Meta-model fold accuracy: {acc_fold:.4f}")

    # Fit meta-model on full OOF
    meta_model.fit(oof_matrix, y)

    # Retrain sklearn models on full data
    for name, model in models.items():
        print(f"Retraining base model on full data: {name}")
        model.fit(X, y)

    # Retrain CatBoost on full data
    fe_full = FeatureEngineer()
    fe_full.fit(X, y)
    X_full_cat = fe_full.transform(X)
    cat_model.fit(X_full_cat, y)

    return models, cat_model, meta_model, fe_full


# ---------------------------------------------------------
# Predict test set
# ---------------------------------------------------------
def predict_test(models, cat_model, meta_model, classes, fe_cat, X_test):
    n_classes = len(classes)
    meta_features = np.zeros((len(X_test), 5 * n_classes))

    col_offset = 0

    # sklearn models
    for name, model in models.items():
        proba = model.predict_proba(X_test)
        meta_features[:, col_offset:col_offset + n_classes] = proba
        col_offset += n_classes

    # CatBoost manually
    X_test_cat = fe_cat.transform(X_test)
    proba_cat = cat_model.predict_proba(X_test_cat)
    meta_features[:, col_offset:col_offset + n_classes] = proba_cat

    meta_preds = meta_model.predict(meta_features)

    # DECODE integers → string labels
    labels = pd.Categorical.from_codes(meta_preds, categories=classes)
    return labels


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    params = load_params()
    tracker = ExperimentTracker()

    X_train, y_train, X_test, classes = load_data()

    models = build_tuned_models(params)
    cat_model = build_catboost(params)
    meta_model = build_meta_model(params)

    # IMPORTANT: ignore encoded classes returned by train_stacking
    models, cat_model, meta_model, fe_cat = train_stacking(
        models, cat_model, meta_model, X_train, y_train
    )

    # Use TRUE string labels here
    test_preds = predict_test(models, cat_model, meta_model, classes, fe_cat, X_test)

    submission = pd.DataFrame({
        "id": X_test["id"],
        "status_group": test_preds
    })

    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")

    tracker.log(
        name="stacking_final",
        params=params,
        cv_score=float("nan"),
        notes="Final stacked model submission with manual CatBoost",
    )


if __name__ == "__main__":
    main()
