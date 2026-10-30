import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def encode_labels(y):
    """
    Encode status_group labels into integer codes.
    Returns encoded labels and the fitted LabelEncoder.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le


def decode_labels(encoded, label_encoder):
    """
    Convert integer predictions back to original labels.
    """
    return label_encoder.inverse_transform(encoded)


def concat_probas(proba_list):
    """
    Concatenate probability matrices from multiple models.
    Input: list of arrays shaped (n_samples, n_classes)
    Output: array shaped (n_samples, n_models * n_classes)
    """
    return np.hstack(proba_list)


def save_json(obj, path):
    """
    Save dictionary or list as JSON.
    """
    path = Path(path)
    path.write_text(json.dumps(obj, indent=2))


def load_json(path):
    """
    Load JSON file into Python object.
    """
    path = Path(path)
    return json.loads(path.read_text())


def hash_params(params):
    """
    Create a reproducible hash for a parameter dictionary.
    Useful for experiment tracking and leaderboard comparison.
    """
    s = json.dumps(params, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def ensure_dir(path):
    """
    Ensure directory exists.
    """
    Path(path).mkdir(parents=True, exist_ok=True)