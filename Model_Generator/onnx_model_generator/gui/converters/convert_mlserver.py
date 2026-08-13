"""
convert_mlserver.py
-------------------
MLServer backend converter for ONNX Model Generator.

This converter:
- Loads a Python model module dynamically
- Builds or loads an ONNX model
- Writes MLServer-compatible metadata
- Prepares the output folder for MLServer deployment

MLServer expects:
- model.onnx
- model-settings.json
- optional: custom Python runtime (not used here)
"""

import os
import sys
import importlib.util
import onnx
from onnx import helper, TensorProto
import json
import traceback


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
# Helper: build minimal ONNX graph (placeholder)
# ---------------------------------------------------------------------------
def build_minimal_onnx_graph():
    """
    Build a minimal ONNX graph with a single Identity node.
    MLServer will load this ONNX model normally.

    Replace this with your algorithmic model logic.
    """

    # Define input tensor
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])

    # Define output tensor
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])

    # Identity node
    node = helper.make_node(
        "Identity",
        inputs=["input"],
        outputs=["output"]
    )

    # Build graph
    graph = helper.make_graph(
        [node],
        "MLServerCustomGraph",
        [input_tensor],
        [output_tensor]
    )

    # Build model
    model = helper.make_model(graph)
    return model


# ---------------------------------------------------------------------------
# Helper: write MLServer model-settings.json
# ---------------------------------------------------------------------------
def write_mlserver_settings(output_folder, model_name="custom-model"):
    """
    Write MLServer-compatible model-settings.json file.
    """

    settings = {
        "name": model_name,
        "version": "1.0.0",
        "platform": "onnx",
        "inputs": [
            {
                "name": "input",
                "datatype": "FP32",
                "shape": [1]
            }
        ],
        "outputs": [
            {
                "name": "output",
                "datatype": "FP32",
                "shape": [1]
            }
        ]
    }

    with open(os.path.join(output_folder, "model-settings.json"), "w") as f:
        json.dump(settings, f, indent=4)


# ---------------------------------------------------------------------------
# Helper: write metadata.json
# ---------------------------------------------------------------------------
def write_metadata(output_folder, entry_point):
    metadata = {
        "backend": "mlserver",
        "entry_point": entry_point,
        "type": "onnx",
        "description": "MLServer-compatible ONNX model."
    }

    with open(os.path.join(output_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------
def convert(model_folder, entry_point, output_folder, log_callback=print):
    """
    Convert a Python model into MLServer-compatible ONNX format.

    Parameters:
    - model_folder: path to the folder containing Python model files
    - entry_point: Python file to load (e.g., "model.py")
    - output_folder: where to write ONNX and metadata
    - log_callback: function for logging (GUI or PowerShell)

    Returns:
    - Path to the generated ONNX file
    """

    try:
        log_callback("> MLServer backend: loading Python module...")
        module_path = os.path.join(model_folder, entry_point)

        module = load_python_module(module_path)
        log_callback(f"> Loaded module: {entry_point}")

        # Optional: call module.main() if it exists
        if hasattr(module, "main"):
            log_callback("> Calling module.main()...")
            module.main()
        else:
            log_callback("> No main() function found. Skipping execution.")

        log_callback("> Building ONNX graph...")
        model = build_minimal_onnx_graph()

        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Write ONNX model
        onnx_path = os.path.join(output_folder, "model.onnx")
        onnx.save(model, onnx_path)
        log_callback(f"> ONNX model written to: {onnx_path}")

        # Write MLServer settings
        log_callback("> Writing MLServer model-settings.json...")
        write_mlserver_settings(output_folder)

        # Write metadata.json
        log_callback("> Writing metadata.json...")
        write_metadata(output_folder, entry_point)

        log_callback("> MLServer ONNX conversion completed successfully.")
        return onnx_path

    except Exception as e:
        log_callback("> ERROR during MLServer conversion:")
        log_callback(str(e))
        log_callback(traceback.format_exc())
        raise
