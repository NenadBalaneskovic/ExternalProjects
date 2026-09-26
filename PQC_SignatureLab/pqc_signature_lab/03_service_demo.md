# 🧩 **03_service_demo.ipynb — Full Content**


```python
## 📌 Section 1 — Setup Check

import importlib
import os

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

required = ["fastapi", "uvicorn", "requests", "oqs", "cryptography"]

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
    ✓ fastapi found
    ✓ uvicorn found
    ✓ requests found
    liboqs-python faulthandler is disabled
    ✓ oqs found
    ✓ cryptography found
    
    All required packages available.



```python
## 📌 Section 2 — Imports

import sys
sys.path.append("../modules")

import crypto_agility
import utils

import requests
import json
import time
import threading
```


```python
## 📌 Section 3 — Define Mini‑Service (FastAPI)

import time
from datetime import datetime
from fastapi import FastAPI, Request

app = FastAPI()

# In-memory request log (resets when the service restarts)
request_log = []

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    request_log.append({
        "timestamp": datetime.now().isoformat(),
        "method": request.method,
        "path": request.url.path,
        "query_params": dict(request.query_params),
        "status_code": response.status_code,
        "duration_sec": round(duration, 6)
    })

    return response


@app.get("/")
def root():
    return {
        "message": "PQC signature service is running",
        "endpoints": ["/sign", "/verify", "/requests"],
        "docs": "/docs"
    }


@app.get("/sign")
def sign_endpoint(msg: str, alg: str):
    signature = crypto_agility.sign(msg.encode(), alg)
    return {
        "algorithm": alg,
        "signature_length": len(signature)
    }


@app.get("/verify")
def verify_endpoint(msg: str, alg: str):
    signature = crypto_agility.sign(msg.encode(), alg)
    valid = crypto_agility.verify(msg.encode(), signature, alg)
    return {
        "algorithm": alg,
        "valid": valid
    }


@app.get("/requests")
def get_requests(limit: int = 50):
    """Return the most recent logged requests (default: last 50)."""
    return {
        "total_requests": len(request_log),
        "showing": min(limit, len(request_log)),
        "requests": request_log[-limit:]
    }
```


```python
## 📌 Section 4 — Start Service in Background Thread

# FastAPI + Uvicorn runs in a background thread so the notebook continues executing.

import uvicorn

def run_service():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

thread = threading.Thread(target=run_service, daemon=True)
thread.start()

print("✓ Mini-Service started at http://127.0.0.1:8000")
time.sleep(1)

```

    ✓ Mini-Service started at http://127.0.0.1:8000



```python
## 📌 Section 5 — Execute Client Requests

algorithms = ["ecdsa", "dilithium3", "falcon512"]
logs = []

print("=== Running Service Requests ===")

for alg in algorithms:
    url = f"http://127.0.0.1:8000/sign?msg=test&alg={alg}"

    start = time.perf_counter()
    r = requests.get(url)
    end = time.perf_counter()

    latency = end - start
    data = r.json()

    logs.append({
        "algorithm": alg,
        "latency": latency,
        "signature_length": data["signature_length"]
    })

    print(f"→ {alg}: {latency:.6f} sec")
```

    === Running Service Requests ===
    → ecdsa: 0.018362 sec
    → dilithium3: 0.006029 sec
    → falcon512: 0.005262 sec



```python
## 📌 Section 6 — Save Artifacts

os.makedirs("../data", exist_ok=True)

with open("../data/service_logs.json", "w") as f:
    json.dump(logs, f, indent=2)

print("✓ Saved service_logs.json")
```

    ✓ Saved service_logs.json



```python
## 📌 Section 7 — Generate Latency Plot

import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(logs)

os.makedirs("../plots", exist_ok=True)

plt.figure(figsize=(6,4))
plt.plot(df["algorithm"], df["latency"], marker="o")
plt.title("Service Latency per Algorithm")
plt.ylabel("Latency (sec)")
plt.savefig("../plots/service_latency.png")
plt.close()

print("✓ Created service_latency.png")
```

    ✓ Created service_latency.png



```python
## 📌 Section 8 — Completion Message

print("\n=== Mini-Service Demo Complete ===")
print("Generated: service_logs.json")
print("Generated plot: service_latency.png")
print("Continue with Notebook 99 (Presentation).")
```

    
    === Mini-Service Demo Complete ===
    Generated: service_logs.json
    Generated plot: service_latency.png
    Continue with Notebook 99 (Presentation).



```python

```
