"""
convert_sklearn.py
------------------
scikit-learn backend converter for ONNX Model Generator.

This converter:
- Dynamically loads a Python module containing a scikit-learn model
- Extracts a fitted estimator (user must define it)
- Converts the estimator into ONNX using skl2onnx
- Writes ONNX model + metadata into the output folder

Expected user model structure:
- The entry point module must define a fitted scikit-learn estimator
  named `model`, or a function `get_model()` returning one.

Example user model file (model.py):

    from sklearn.linear_model import LogisticRegression
    import numpy as np

    X = np.random.randn(100, 4)
    y = (X[:, 0] > 0).astype(int)

    model = LogisticRegression().fit(X, y)

"""

import os
import sys
import importlib.util
import json
import traceback

import numpy as np
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


# ---------------------------------------------------------------------------
# Helper: dynamic module loader
# ---------------------------------------------------------------------------
def load_python_module(module_path):
    """
    Load a Python module dynamically from a given file path.
    Returns the loaded module object.
    """
    module_name = os.path.splitext(os.path.basename(module_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        raise RuntimeError(f"Failed to load module {module_name}: {e}")


# ---------------------------------------------------------------------------
# Helper: extract scikit-learn model
# ---------------------------------------------------------------------------
def extract_sklearn_model(module):
    """
    Extract a scikit-learn estimator from the loaded module.

    Supported patterns:
    - module.model  (fitted estimator)
    - module.get_model()  (returns fitted estimator)
    """

    if hasattr(module, "model"):
        return module.model

    if hasattr(module, "get_model"):
        return module.get_model()

    raise RuntimeError(
        "No scikit-learn model found. Expected `model` or `get_model()`."
    )


# ---------------------------------------------------------------------------
# Helper: write metadata.json
# ---------------------------------------------------------------------------
def write_metadata(output_folder, entry_point):
    metadata = {
        "backend": "sklearn",
        "entry_point": entry_point,
        "type": "onnx",
        "description": "scikit-learn model exported via skl2onnx."
    }

    with open(os.path.join(output_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------
def convert(model_folder, entry_point, output_folder, log_callback=print):
    """
    Convert a scikit-learn model into ONNX format.

    Parameters:
    - model_folder: path to the folder containing Python model files
    - entry_point: Python file to load (e.g., "model.py")
    - output_folder: where to write ONNX and metadata
    - log_callback: function for logging (GUI or PowerShell)

    Returns:
    - Path to the generated ONNX file
    """

    try:
        log_callback("> scikit-learn backend: loading Python module...")
        module_path = os.path.join(model_folder, entry_point)

        module = load_python_module(module_path)
        log_callback(f"> Loaded module: {entry_point}")

        log_callback("> Extracting scikit-learn estimator...")
        estimator = extract_sklearn_model(module)
        log_callback(f"> Found estimator: {type(estimator).__name__}")

        # Determine input shape (simple heuristic)
        # User may override this later with config files
        log_callback("> Determining input shape...")
        if hasattr(estimator, "n_features_in_"):
            n_features = estimator.n_features_in_
        else:
            n_features = 1  # fallback

        initial_type = [("input", FloatTensorType([None, n_features]))]

        log_callback("> Converting estimator to ONNX...")
        onnx_model = convert_sklearn(estimator, initial_types=initial_type)

        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        onnx_path = os.path.join(output_folder, "model.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        log_callback(f"> ONNX model written to: {onnx_path}")

        log_callback("> Writing metadata.json...")
        write_metadata(output_folder, entry_point)

        log_callback("> scikit-learn ONNX conversion completed successfully.")
        return onnx_path

    except Exception as e:
        log_callback("> ERROR during scikit-learn conversion:")
        log_callback(str(e))
        log_callback(traceback.format_exc())
        raise
