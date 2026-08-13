"""
convert_triton.py
-----------------
Triton backend converter for ONNX Model Generator.

This converter:
- Dynamically loads a Python module containing a model
- Extracts a torch.nn.Module or builds a minimal ONNX graph
- Creates a Triton-compatible model repository structure:
      model_repository/<model_name>/1/model.onnx
      model_repository/<model_name>/config.pbtxt
      metadata.json

Supported user model patterns:
- module.model  (torch.nn.Module)
- module.get_model()  (returns torch.nn.Module)
- module.get_dummy_input()  (optional)
- fallback: minimal ONNX graph

"""

import os
import sys
import importlib.util
import json
import traceback

import torch
import torch.onnx
import onnx
from onnx import helper, TensorProto


# ---------------------------------------------------------------------------
# Helper: dynamic module loader
# ---------------------------------------------------------------------------
def load_python_module(module_path):
    module_name = os.path.splitext(os.path.basename(module_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        raise RuntimeError(f"Failed to load module {module_name}: {e}")


# ---------------------------------------------------------------------------
# Helper: extract torch model (optional)
# ---------------------------------------------------------------------------
def extract_torch_model(module):
    if hasattr(module, "model") and isinstance(module.model, torch.nn.Module):
        return module.model

    if hasattr(module, "get_model"):
        m = module.get_model()
        if isinstance(m, torch.nn.Module):
            return m

    return None  # fallback to ONNX-only mode


# ---------------------------------------------------------------------------
# Helper: extract dummy input
# ---------------------------------------------------------------------------
def extract_dummy_input(module):
    if hasattr(module, "get_dummy_input"):
        dummy = module.get_dummy_input()
        if isinstance(dummy, torch.Tensor):
            return dummy

    return torch.randn(1, 1)


# ---------------------------------------------------------------------------
# Helper: build minimal ONNX graph (fallback)
# ---------------------------------------------------------------------------
def build_minimal_onnx_graph():
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])

    node = helper.make_node("Identity", inputs=["input"], outputs=["output"])

    graph = helper.make_graph(
        [node],
        "TritonFallbackGraph",
        [input_tensor],
        [output_tensor]
    )

    return helper.make_model(graph)


# ---------------------------------------------------------------------------
# Helper: write Triton config.pbtxt
# ---------------------------------------------------------------------------
def write_triton_config(model_folder, model_name="custom_model"):
    config = f"""
name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: 0

input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [1]
  }}
]

output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [1]
  }}
]
"""

    with open(os.path.join(model_folder, "config.pbtxt"), "w") as f:
        f.write(config)


# ---------------------------------------------------------------------------
# Helper: write metadata.json
# ---------------------------------------------------------------------------
def write_metadata(output_folder, entry_point):
    metadata = {
        "backend": "triton",
        "entry_point": entry_point,
        "type": "onnx",
        "description": "Triton Inference Server ONNX model."
    }

    with open(os.path.join(output_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------
def convert(model_folder, entry_point, output_folder, log_callback=print):
    """
    Convert a Python model into Triton-compatible ONNX format.

    Triton model repository structure:
        output_folder/
            model_repository/
                <model_name>/
                    1/
                        model.onnx
                    config.pbtxt
            metadata.json

    Parameters:
    - model_folder: path to the folder containing Python model files
    - entry_point: Python file to load (e.g., "model.py")
    - output_folder: where to write Triton model repository
    - log_callback: function for logging (GUI or PowerShell)

    Returns:
    - Path to the generated ONNX file
    """

    try:
        log_callback("> Triton backend: loading Python module...")
        module_path = os.path.join(model_folder, entry_point)

        module = load_python_module(module_path)
        log_callback(f"> Loaded module: {entry_point}")

        # Try to extract a torch model
        log_callback("> Extracting torch model (if available)...")
        model = extract_torch_model(module)

        # Prepare Triton model repository structure
        model_name = "custom_model"
        repo_root = os.path.join(output_folder, "model_repository", model_name)
        version_folder = os.path.join(repo_root, "1")

        os.makedirs(version_folder, exist_ok=True)

        onnx_path = os.path.join(version_folder, "model.onnx")

        if model is not None:
            log_callback("> Torch model found. Exporting to ONNX...")

            dummy_input = extract_dummy_input(module)
            log_callback(f"> Dummy input shape: {tuple(dummy_input.shape)}")

            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"]
            )

            log_callback(f"> ONNX model written to: {onnx_path}")

        else:
            log_callback("> No torch model found. Using fallback ONNX graph...")
            fallback_model = build_minimal_onnx_graph()
            onnx.save(fallback_model, onnx_path)
            log_callback(f"> Fallback ONNX model written to: {onnx_path}")

        # Write Triton config
        log_callback("> Writing Triton config.pbtxt...")
        write_triton_config(repo_root, model_name)

        # Write metadata.json
        log_callback("> Writing metadata.json...")
        write_metadata(output_folder, entry_point)

        log_callback("> Triton ONNX conversion completed successfully.")
        return onnx_path

    except Exception as e:
        log_callback("> ERROR during Triton conversion:")
        log_callback(str(e))
        log_callback(traceback.format_exc())
        raise
