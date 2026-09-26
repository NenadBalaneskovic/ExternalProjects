# 🧩 **02_crypto_agility.ipynb — Full Content**


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

required = ["oqs", "cryptography", "pandas", "matplotlib"]

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
    liboqs-python faulthandler is disabled
    ✓ oqs found
    ✓ cryptography found
    ✓ pandas found
    ✓ matplotlib found
    
    All required packages available.



```python
## 📌 Section 2 — Imports

import oqs
import pandas as pd
import json
import matplotlib.pyplot as plt

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes

import sys
sys.path.append("../modules")
import utils
```


```python
## 📌 Section 3 — Create `crypto_agility.py` Module (If Missing)

import os
os.remove("../modules/crypto_agility.py")

module_path = "../modules/crypto_agility.py"

crypto_agility_code = """
import oqs
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes

# Generate one keypair per algorithm, once at import time, and reuse it
# for both sign() and verify() -- otherwise each call would use a fresh,
# unrelated random key and verification would always fail.
_ecdsa_private_key = ec.generate_private_key(ec.SECP256R1())
_ecdsa_public_key = _ecdsa_private_key.public_key()

_dilithium_sig = oqs.Signature("ML-DSA-65")
_dilithium_pk = _dilithium_sig.generate_keypair()

_falcon_sig = oqs.Signature("Falcon-512")
_falcon_pk = _falcon_sig.generate_keypair()


def sign(msg, alg):
    if alg == "ecdsa":
        return _ecdsa_private_key.sign(msg, ec.ECDSA(hashes.SHA256()))

    if alg == "dilithium3":
        return _dilithium_sig.sign(msg)

    if alg == "falcon512":
        return _falcon_sig.sign(msg)

    raise ValueError("Unknown algorithm")


def verify(msg, signature, alg):
    if alg == "ecdsa":
        _ecdsa_public_key.verify(signature, msg, ec.ECDSA(hashes.SHA256()))
        return True

    if alg == "dilithium3":
        return _dilithium_sig.verify(msg, signature, _dilithium_pk)

    if alg == "falcon512":
        return _falcon_sig.verify(msg, signature, _falcon_pk)

    raise ValueError("Unknown algorithm")
"""

# Always (re)write the module here so this cell can be used to apply fixes
# by simply re-running it, rather than only writing when the file is missing.
with open(module_path, "w") as f:
    f.write(crypto_agility_code)
print("✓ Created/updated crypto_agility.py")
```

    ✓ Created/updated crypto_agility.py



```python
## 📌 Section 4 — Import the Module

import crypto_agility
```


```python
## 📌 Section 5 — Crypto‑Agility Switching Tests

algorithms = ["ecdsa", "dilithium3", "falcon512"]
msg = b"Agility Test"

results = []

print("=== Running Crypto-Agility Tests ===")
for alg in algorithms:
    print(f"→ {alg}")

    # Sign
    signature, t_sign = utils.timer(crypto_agility.sign, msg, alg)

    # Verify
    _, t_verify = utils.timer(crypto_agility.verify, msg, signature, alg)

    results.append({
        "algorithm": alg,
        "sign_time": t_sign,
        "verify_time": t_verify,
        "signature_length": len(signature)
    })

df = pd.DataFrame(results)
```

    === Running Crypto-Agility Tests ===
    → ecdsa
    → dilithium3
    → falcon512



```python
## 📌 Section 6 — Save Artifacts

os.makedirs("../data", exist_ok=True)

with open("../data/agility_tests.json", "w") as f:
    json.dump(results, f, indent=2)

print("✓ Saved agility_tests.json")

```

    ✓ Saved agility_tests.json



```python
## 📌 Section 7 — Generate Plots

os.makedirs("../plots", exist_ok=True)

### Agility Matrix (Signature Sizes)

utils.plot_bar(
    df["algorithm"],
    df["signature_length"],
    "Crypto-Agility: Signature Sizes",
    "../plots/agility_matrix.png"
)

print("✓ Created agility_matrix.png")

### Algorithm Switch Cost (Sign Time)

utils.plot_line(
    df["algorithm"],
    df["sign_time"],
    "Algorithm Switch Cost (Sign Time)",
    "../plots/algorithm_switch_cost.png"
)

print("✓ Created algorithm_switch_cost.png")
```

    ✓ Created agility_matrix.png
    ✓ Created algorithm_switch_cost.png



```python
## 📌 Section 8 — Completion Message

print("\n=== Crypto-Agility Layer Complete ===")
print("Generated: agility_tests.json")
print("Generated plots: agility_matrix.png, algorithm_switch_cost.png")
print("Continue with Notebook 03.")
```

    
    === Crypto-Agility Layer Complete ===
    Generated: agility_tests.json
    Generated plots: agility_matrix.png, algorithm_switch_cost.png
    Continue with Notebook 03.



```python

```
