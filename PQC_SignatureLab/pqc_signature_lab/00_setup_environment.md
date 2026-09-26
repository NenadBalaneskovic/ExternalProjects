# 🧩 **00_setup_environment.ipynb — Full Content**

## 🎯 **Result: Notebook 00 Is Fully Complete**

This notebook:

- checks all required packages  
- generates `utils.py`  
- generates `pqc_libs.json`  
- generates `benchmarks.json`  
- is Fedora‑compatible  
- is robust and minimalistic  
- forms the foundation for all other notebooks  
- fits perfectly into our hybrid architecture  


```python
## 📌 Section 1 — Imports & Helpers
import os
import importlib
import json
from datetime import datetime

def exists(path):
    return os.path.exists(path)

def check_package(pkg):
    try:
        importlib.import_module(pkg)
        print(f"✓ {pkg} found")
        return True
    except ImportError:
        print(f"✗ {pkg} missing — install via terminal:")
        print(f"    pip install {pkg}")
        return False
```


```python
import sys
!{sys.executable} -m pip install cmake git+https://github.com/open-quantum-safe/liboqs-python.git
!{sys.executable} -m pip install fastapi uvicorn
```

    Collecting git+https://github.com/open-quantum-safe/liboqs-python.git
      Cloning https://github.com/open-quantum-safe/liboqs-python.git to /tmp/pip-req-build-m59c8opq
      Running command git clone --filter=blob:none --quiet https://github.com/open-quantum-safe/liboqs-python.git /tmp/pip-req-build-m59c8opq
      Resolved https://github.com/open-quantum-safe/liboqs-python.git to commit cbf1788acdecf5f54c3b29900c7e404d618616cb
      Installing build dependencies ... [?25ldone
    [?25h  Getting requirements to build wheel ... [?25ldone
    [?25h  Preparing metadata (pyproject.toml) ... [?25ldone
    [?25hRequirement already satisfied: cmake in /home/nenadbalaneskovic/.venv/lib64/python3.12/site-packages (4.4.3)
    Collecting fastapi
      Downloading fastapi-0.141.1-py3-none-any.whl.metadata (27 kB)
    Collecting uvicorn
      Downloading uvicorn-0.54.0-py3-none-any.whl.metadata (6.6 kB)
    Collecting starlette>=0.46.0 (from fastapi)
      Downloading starlette-1.7.0-py3-none-any.whl.metadata (6.6 kB)
    Requirement already satisfied: pydantic>=2.9.0 in /home/nenadbalaneskovic/.venv/lib64/python3.12/site-packages (from fastapi) (2.13.4)
    Requirement already satisfied: typing-extensions>=4.8.0 in /home/nenadbalaneskovic/.venv/lib64/python3.12/site-packages (from fastapi) (4.16.0)
    Requirement already satisfied: typing-inspection>=0.4.2 in /home/nenadbalaneskovic/.venv/lib64/python3.12/site-packages (from fastapi) (0.4.4)
    Requirement already satisfied: annotated-doc>=0.0.2 in /home/nenadbalaneskovic/.venv/lib64/python3.12/site-packages (from fastapi) (0.0.5)
    Requirement already satisfied: click>=7.0 in /home/nenadbalaneskovic/.venv/lib64/python3.12/site-packages (from uvicorn) (8.4.2)
    Requirement already satisfied: h11>=0.8 in /home/nenadbalaneskovic/.venv/lib64/python3.12/site-packages (from uvicorn) (0.16.0)
    Requirement already satisfied: annotated-types>=0.6.0 in /home/nenadbalaneskovic/.venv/lib64/python3.12/site-packages (from pydantic>=2.9.0->fastapi) (0.8.0)
    Requirement already satisfied: pydantic-core==2.46.4 in /home/nenadbalaneskovic/.venv/lib64/python3.12/site-packages (from pydantic>=2.9.0->fastapi) (2.46.4)
    Requirement already satisfied: anyio<5,>=4.0.0 in /home/nenadbalaneskovic/.venv/lib64/python3.12/site-packages (from starlette>=0.46.0->fastapi) (4.14.2)
    Requirement already satisfied: idna>=2.8 in /home/nenadbalaneskovic/.venv/lib64/python3.12/site-packages (from anyio<5,>=4.0.0->starlette>=0.46.0->fastapi) (3.19)
    Downloading fastapi-0.141.1-py3-none-any.whl (131 kB)
    Downloading uvicorn-0.54.0-py3-none-any.whl (87 kB)
    Downloading starlette-1.7.0-py3-none-any.whl (78 kB)
    Installing collected packages: uvicorn, starlette, fastapi
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3/3[0m [fastapi]m2/3[0m [fastapi]
    [1A[2KSuccessfully installed fastapi-0.141.1 starlette-1.7.0 uvicorn-0.54.0


Execute in jupyter terminal:
```bash
sudo dnf install -y cmake gcc gcc-c++ ninja-build git openssl-devel make
```


```python
## 📌 Section 2 — Package Check

required = [
    "oqs",          # PQC algorithms
    "cryptography", # ECDSA
    "pandas",       # CSV handling
    "matplotlib",   # plotting
    "fastapi",      # service demo
    "uvicorn",      # service runner
    "aiohttp",      # client
    "requests"      # HTTP client
]

print("=== Package Check ===")
missing = [pkg for pkg in required if not check_package(pkg)]

if missing:
    print("\nMissing packages detected.")
    print("Install them in your Jupyter terminal:")
    print("pip install " + " ".join(missing))
else:
    print("\nAll required packages available.")
```

    === Package Check ===
    ✓ oqs found
    ✓ cryptography found
    ✓ pandas found
    ✓ matplotlib found
    ✓ fastapi found
    ✓ uvicorn found
    ✓ aiohttp found
    ✓ requests found
    
    All required packages available.


Execute in a terminal:
```bash
echo "$HOME/_oqs/lib64" | sudo tee /etc/ld.so.conf.d/liboqs.conf
sudo ldconfig
ldconfig -p | grep liboqs
```


```python
import oqs
print(oqs.oqs_version())
```

    0.16.0



```python
## 📌 Section 3 — Generate `utils.py` Module  
# This module is imported by all other notebooks.

utils_code = """
import time
import matplotlib.pyplot as plt

def timer(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start

def plot_bar(labels, values, title, path):
    plt.figure(figsize=(6,4))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel("Value")
    plt.savefig(path)
    plt.close()

def plot_line(labels, values, title, path):
    plt.figure(figsize=(6,4))
    plt.plot(labels, values, marker='o')
    plt.title(title)
    plt.ylabel("Value")
    plt.savefig(path)
    plt.close()
"""

os.makedirs("../modules", exist_ok=True)

with open("../modules/utils.py", "w") as f:
    f.write(utils_code)

print("✓ Created modules/utils.py")

```

    ✓ Created modules/utils.py



```python
## 📌 Section 4 — PQC Library Check & JSON Export

import oqs

info = {
    "timestamp": datetime.now().isoformat(),
    "oqs_version": oqs.oqs_version(),
    "oqs_python_version": oqs.oqs_python_version(),
    "available_algorithms": {
        "signatures": oqs.get_enabled_sig_mechanisms(),
        "kem": oqs.get_enabled_kem_mechanisms()
    }
}

os.makedirs("../data", exist_ok=True)

with open("../data/pqc_libs.json", "w") as f:
    json.dump(info, f, indent=2)

print("✓ Created data/pqc_libs.json")
```

    ✓ Created data/pqc_libs.json



```python
## 📌 Section 5 — Mini‑Benchmark (Keygen / Sign / Verify)

import sys, os
sys.path.append(os.path.abspath(".."))

import pandas as pd
import oqs
from modules.utils import timer

def benchmark_signature(alg):
    msg = b"benchmark"

    with oqs.Signature(alg) as sig:
        # Keygen (returns the public key)
        public_key, t_keygen = timer(sig.generate_keypair)

        # Sign
        signature, t_sign = timer(sig.sign, msg)

        # Verify
        _, t_verify = timer(sig.verify, msg, signature, public_key)

    return {
        "algorithm": alg,
        "keygen": t_keygen,
        "sign": t_sign,
        "verify": t_verify
    }

enabled_sigs = oqs.get_enabled_sig_mechanisms()

algorithms = ["ML-DSA-65", "Falcon-512"]

results = []
for alg in algorithms:
    if alg not in enabled_sigs:
        raise ValueError(
            f"'{alg}' is not enabled in this liboqs build.\n"
            f"Available signature mechanisms:\n{enabled_sigs}"
        )
    print(f"Running benchmark for {alg}...")
    results.append(benchmark_signature(alg))

df = pd.DataFrame(results)
os.makedirs("../data", exist_ok=True)
df.to_json("../data/benchmarks.json", orient="records", indent=2)

print("✓ Created data/benchmarks.json")
```

    Running benchmark for ML-DSA-65...
    Running benchmark for Falcon-512...
    ✓ Created data/benchmarks.json



```python
## 📌 Section 6 — Completion Message

print("\n=== Setup Complete ===")
print("utils.py, pqc_libs.json, benchmarks.json generated.")
print("You can now continue with Notebook 01.")
```

    
    === Setup Complete ===
    utils.py, pqc_libs.json, benchmarks.json generated.
    You can now continue with Notebook 01.



```python

```
