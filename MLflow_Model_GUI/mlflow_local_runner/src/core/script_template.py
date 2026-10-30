# mlflow_local_runner/src/core/script_template.py
"""
Erweitertes Nutzer-Skript-Template für den MLflow Local Runner (tokenfreie Version)

Option A: Modell-Logging + Registry aktiv
"""

import os
import json
import warnings
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score

# MLflow
import mlflow
import mlflow.sklearn


# ---------------------------------------------------------
# GLOBAL: ALLE WARNUNGEN UNTERDRÜCKEN
# ---------------------------------------------------------
warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def load_data(dataset_path: str):
    df = pd.read_csv(dataset_path, sep=None, engine="python")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

def build_preprocessing(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )


# ---------------------------------------------------------
# MODEL SELECTION
# ---------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


def get_model(model_type: str):
    models = {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "svm": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    }

    if XGBClassifier:
        models["xgboost"] = XGBClassifier(n_estimators=200, learning_rate=0.1)

    if LGBMClassifier:
        models["lightgbm"] = LGBMClassifier(n_estimators=200, learning_rate=0.1)

    if CatBoostClassifier:
        models["catboost"] = CatBoostClassifier(
            iterations=200, learning_rate=0.1, depth=6, verbose=False
        )

    if model_type not in models:
        raise ValueError(f"Unbekannter Modelltyp: {model_type}")

    return models[model_type]


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

def apply_feature_engineering(pipeline, use_pca=False):
    if use_pca:
        pipeline.steps.append(("pca", PCA(n_components=10)))
    return pipeline


# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------

def train_model(X_train, y_train, model_type="random_forest", tuning=False, use_pca=False):
    preprocessor = build_preprocessing(X_train)
    model = get_model(model_type)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline = apply_feature_engineering(pipeline, use_pca)

    if tuning and model_type == "random_forest":
        pipeline = GridSearchCV(
            pipeline,
            {"model__n_estimators": [100, 200], "model__max_depth": [None, 10, 20]},
            cv=3
        )

    pipeline.fit(X_train, y_train)
    return pipeline


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted"))
    }


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":
    dataset_path = os.environ["DATASET_PATH"]
    model_type = os.environ.get("MODEL_TYPE", "random_forest")
    tuning = os.environ.get("TUNING", "false").lower() == "true"
    use_pca = os.environ.get("USE_PCA", "false").lower() == "true"

    X_train, X_test, y_train, y_test = load_data(dataset_path)
    model = train_model(X_train, y_train, model_type, tuning, use_pca)
    metrics = evaluate_model(model, X_test, y_test)

    # ---------------------------------------------------------
    # OPTION A: Modell loggen + direkt in Registry eintragen
    # ---------------------------------------------------------
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",                      # Run-Artifact-Name
        registered_model_name="local_runner_model"  # Registry-Name
    )

    # GUI liest diese Marker aus stdout
    print("MODEL_READY")
    print(model)
    print("METRICS_READY")
    print(json.dumps(metrics))
