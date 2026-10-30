"""
convert_custom.py
-----------------
Custom backend converter for algorithmic Python models that do not use
Torch, TensorFlow, or scikit-learn.

This converter:
- Dynamically loads a Python module from the model folder
- Calls a user-defined function or class
- Wraps the output into a minimal ONNX graph
- Writes the ONNX model to the output folder
- Writes metadata.json and conversion.log

This is a minimal prototype. The ONNX graph is intentionally simple
and should be extended depending on the model structure.
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
# Helper: build minimal ONNX graph
# ---------------------------------------------------------------------------
def build_minimal_onnx_graph():
    """
    Build a minimal ONNX graph with a single identity node.
    This is a placeholder. Replace with your algorithmic model logic.
    """

    # Define input tensor
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])

    # Define output tensor
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])

    # Define a simple identity node
    node = helper.make_node(
        "Identity",
        inputs=["input"],
        outputs=["output"]
    )

    # Build graph
    graph = helper.make_graph(
        [node],
        "CustomModelGraph",
        [input_tensor],
        [output_tensor]
    )

    # Build model
    model = helper.make_model(graph)
    return model


# ---------------------------------------------------------------------------
# Helper: write metadata
# ---------------------------------------------------------------------------
def write_metadata(output_folder, backend_name, entry_point):
    metadata = {
        "backend": backend_name,
        "entry_point": entry_point,
        "type": "custom_python",
        "description": "Algorithmic model exported via custom ONNX converter."
    }

    with open(os.path.join(output_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------
def convert(model_folder, entry_point, output_folder, log_callback=print):
    """
    Convert a custom Python model into ONNX format.

    Parameters:
    - model_folder: path to the folder containing Python model files
    - entry_point: Python file to load (e.g., "model.py")
    - output_folder: where to write ONNX and metadata
    - log_callback: function for logging (GUI or PowerShell)

    Returns:
    - Path to the generated ONNX file
    """

    try:
        log_callback("> Custom backend: loading Python module...")
        module_path = os.path.join(model_folder, entry_point)

        module = load_python_module(module_path)
        log_callback(f"> Loaded module: {entry_point}")

        # Placeholder: call a function if it exists
        if hasattr(module, "main"):
            log_callback("> Calling module.main()...")
            module.main()
        else:
            log_callback("> No main() function found. Skipping execution.")

        log_callback("> Building ONNX graph...")
        model = build_minimal_onnx_graph()

        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        onnx_path = os.path.join(output_folder, "model.onnx")
        onnx.save(model, onnx_path)

        log_callback(f"> ONNX model written to: {onnx_path}")

        log_callback("> Writing metadata...")
        write_metadata(output_folder, "custom", entry_point)

        log_callback("> Custom ONNX conversion completed successfully.")
        return onnx_path

    except Exception as e:
        log_callback("> ERROR during custom conversion:")
        log_callback(str(e))
        log_callback(traceback.format_exc())
        raise