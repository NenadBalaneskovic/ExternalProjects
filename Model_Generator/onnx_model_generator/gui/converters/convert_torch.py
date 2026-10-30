"""
convert_torch.py
----------------
Torch backend converter for ONNX Model Generator.

This converter:
- Dynamically loads a Python module containing a torch.nn.Module
- Extracts a model instance (user must define it)
- Creates a dummy input tensor (unless user provides one)
- Exports the model to ONNX using torch.onnx.export
- Writes ONNX model + metadata into the output folder
"""

import os
import sys
import importlib.util
import json
import traceback

import torch
import torch.onnx


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
# Helper: extract torch model
# ---------------------------------------------------------------------------
def extract_torch_model(module):
    if hasattr(module, "model"):
        model = module.model
        if isinstance(model, torch.nn.Module):
            return model

    if hasattr(module, "get_model"):
        model = module.get_model()
        if isinstance(model, torch.nn.Module):
            return model

    raise RuntimeError("No torch.nn.Module found. Expected `model` or `get_model()`.")


# ---------------------------------------------------------------------------
# Helper: extract dummy input (robust version)
# ---------------------------------------------------------------------------
def extract_dummy_input(module, model):
    """
    Extract a dummy input tensor for ONNX export.

    Priority:
    1. module.get_dummy_input()
    2. infer from model (Linear, Conv, etc.)
    3. fallback: torch.randn(1, 1)
    """

    # 1. User-defined dummy input
    if hasattr(module, "get_dummy_input"):
        dummy = module.get_dummy_input()
        if isinstance(dummy, torch.Tensor):
            return dummy

    # 2. Infer input shape from ANY Linear layer in the model
    try:
        for submodule in model.modules():
            if isinstance(submodule, torch.nn.Linear):
                return torch.randn(1, submodule.in_features)

        # Conv2d fallback
        for submodule in model.modules():
            if isinstance(submodule, torch.nn.Conv2d):
                return torch.randn(1, submodule.in_channels, 64, 64)

    except Exception:
        pass

    # 3. Fallback
    return torch.randn(1, 1)


# ---------------------------------------------------------------------------
# Helper: write metadata.json
# ---------------------------------------------------------------------------
def write_metadata(output_folder, entry_point):
    metadata = {
        "backend": "torch",
        "entry_point": entry_point,
        "type": "onnx",
        "description": "PyTorch model exported via torch.onnx.export."
    }

    with open(os.path.join(output_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------
def convert(model_folder, entry_point, output_folder, log_callback=print):
    try:
        log_callback("> Torch backend: loading Python module...")
        module_path = os.path.join(model_folder, entry_point)

        module = load_python_module(module_path)
        log_callback(f"> Loaded module: {entry_point}")

        log_callback("> Extracting torch.nn.Module...")
        model = extract_torch_model(module)
        log_callback(f"> Found model: {type(model).__name__}")

        log_callback("> Extracting dummy input tensor...")
        dummy_input = extract_dummy_input(module, model)
        log_callback(f"> Dummy input shape: {tuple(dummy_input.shape)}")

        os.makedirs(output_folder, exist_ok=True)
        onnx_path = os.path.join(output_folder, "model.onnx")

        log_callback("> Exporting model to ONNX...")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,   # ⭐ FIX: avoid Gemm downgrade crash
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"]
        )

        log_callback(f"> ONNX model written to: {onnx_path}")

        log_callback("> Writing metadata.json...")
        write_metadata(output_folder, entry_point)

        log_callback("> Torch ONNX conversion completed successfully.")
        return onnx_path

    except Exception as e:
        log_callback("> ERROR during Torch conversion:")
        log_callback(str(e))
        log_callback(traceback.format_exc())
        raise
