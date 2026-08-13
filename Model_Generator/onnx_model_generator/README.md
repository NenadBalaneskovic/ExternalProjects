# **ONNX Model Generator — User Manual**

## **1. Overview**
The ONNX Model Generator is a single‑window GUI application that converts Python‑based machine‑learning or algorithmic models into ONNX format.  
It supports multiple backends:

- Torch  
- scikit‑learn  
- Custom Python  
- MLServer  
- Triton  

The GUI provides:

- Model folder selection  
- Automatic detection of Python files  
- Backend selection  
- Environment validation  
- ONNX model generation  
- Live log console  
- Progress bar  
- Integration with PowerShell scripts  

---

## **2. Folder Structure**
Your project must follow this structure:

```
onnx_model_generator/
│
├── gui/
│   ├── main_gui.py
│   ├── converters/
│   │   ├── convert_torch.py
│   │   ├── convert_sklearn.py
│   │   ├── convert_custom.py
│   │   ├── convert_mlserver.py
│   │   └── convert_triton.py
│   └── assets/
│       └── darkblue3_theme.json
│
├── scripts/
│   ├── Generate-ONNXModel.ps1
│   ├── Validate-Environment.ps1
│   └── Install-Dependencies.ps1
│
├── output/
│
└── requirements.txt
```

---

## **3. Installation**

### **3.1 Install Python dependencies**
From the project root:

```
pip install -r requirements.txt
```

### **3.2 Install system dependencies**
The GUI expects:

- Python ≥ 3.10  
- PowerShell (Windows built‑in)  
- Git  
- Podman (optional, for container workflows)

To install missing Python packages:

```
powershell -ExecutionPolicy Bypass -File scripts/Install-Dependencies.ps1
```

---

## **4. Starting the GUI**

### **Option A — Run from command line**
Navigate to the `gui` folder:

```
cd onnx_model_generator/gui
python main_gui.py
```

### **Option B — Double‑click**
You can also double‑click `main_gui.py` if `.py` files are associated with Python.

---

## **5. Using the GUI**

### **5.1 Select the model folder**
Click **Browse** next to *Model Folder* and choose the directory containing your Python model files.

The GUI will automatically:

- Scan the folder  
- Detect `.py` files  
- Populate the *Entry Point* dropdown  
- Display detected files  

### **5.2 Choose the entry point**
Select the Python file that contains your model:

- `model.py`  
- `pipeline.py`  
- `generators.py`  
- or any other `.py` file  

### **5.3 Select a backend**
Choose one of the supported backends:

- **Torch** — for `torch.nn.Module` models  
- **scikit‑learn** — for fitted sklearn estimators  
- **Custom Python** — for algorithmic models  
- **MLServer** — for ONNX models served via MLServer  
- **Triton** — for Triton Inference Server repositories  

### **5.4 Select output folder**
Choose where the ONNX model and metadata should be written.

### **5.5 Validate environment**
Click **Validate Environment**.

The GUI will:

- Run `Validate-Environment.ps1`  
- Stream logs into the console  
- Update the progress bar  
- Show PASS/FAIL status  

### **5.6 Generate ONNX model**
Click **Generate ONNX Model**.

The GUI will:

- Run `Generate-ONNXModel.ps1`  
- Pass model folder, entry point, backend, and output folder  
- Stream logs live  
- Update progress bar  
- Write ONNX model + metadata  

---

## **6. Backend Requirements**

### **Torch backend**
Your entry point must define:

```python
model = MyModel()
```

or:

```python
def get_model():
    return MyModel()
```

Optional:

```python
def get_dummy_input():
    return torch.randn(1, 4)
```

---

### **scikit‑learn backend**
Your entry point must define:

```python
model = fitted_estimator
```

or:

```python
def get_model():
    return fitted_estimator
```

---

### **Custom Python backend**
Your entry point may define:

```python
def main():
    # optional preprocessing
```

A minimal ONNX graph is generated automatically.

---

### **MLServer backend**
Produces:

```
model.onnx
model-settings.json
metadata.json
```

---

### **Triton backend**
Produces a Triton model repository:

```
model_repository/
    custom_model/
        1/
            model.onnx
        config.pbtxt
metadata.json
```

---

## **7. Troubleshooting**

### **PowerShell not found**
Ensure you are running on Windows.

### **Python not found**
Add Python to PATH or reinstall Python ≥ 3.10.

### **Missing dependencies**
Run:

```
powershell -ExecutionPolicy Bypass -File scripts/Install-Dependencies.ps1
```

### **No Python files detected**
Ensure your model folder contains `.py` files.

### **Backend errors**
Check the log console for detailed messages.

---

## **8. Support**
For questions, improvements, or backend extensions, contact the project maintainer or open an issue in the repository.

---
