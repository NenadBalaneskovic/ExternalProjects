# 🧩 **01_signature_lab.ipynb — Full Content**


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
import pickle
import matplotlib.pyplot as plt

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes

import sys
sys.path.append("../modules")
import utils
```


```python
## 📌 Section 3 — Signature Functions

### **ECDSA‑P256**

def sign_ecdsa(msg):
    key = ec.generate_private_key(ec.SECP256R1())
    signature = key.sign(msg, ec.ECDSA(hashes.SHA256()))
    return signature, key.public_key()

### **Dilithium3**

def sign_dilithium(msg):
    with oqs.Signature("Dilithium3") as sig:
        pk = sig.generate_keypair()
        signature = sig.sign(msg)
        return signature, pk

### **Falcon512**

def sign_falcon(msg):
    with oqs.Signature("Falcon-512") as sig:
        pk = sig.generate_keypair()
        signature = sig.sign(msg)
        return signature, pk
```


```python
## 📌 Section 4 — Measurements (Keygen / Sign / Verify)

def benchmark_signature(alg):
    msg = b"Hello PQC"

    if alg == "ecdsa":
        # Keygen (generate the key ONCE, reuse for sign+verify)
        private_key, t_keygen = utils.timer(
            lambda: ec.generate_private_key(ec.SECP256R1())
        )
        pubkey = private_key.public_key()

        # Sign (with the SAME private key)
        signature, t_sign = utils.timer(
            lambda: private_key.sign(msg, ec.ECDSA(hashes.SHA256()))
        )

        # Verify
        _, t_verify = utils.timer(
            lambda: pubkey.verify(signature, msg, ec.ECDSA(hashes.SHA256()))
        )

    elif alg == "dilithium3":
        with oqs.Signature("ML-DSA-65") as sig:
            # Keygen
            pk, t_keygen = utils.timer(sig.generate_keypair)
            # Sign
            signature, t_sign = utils.timer(sig.sign, msg)
            # Verify
            _, t_verify = utils.timer(sig.verify, msg, signature, pk)

    elif alg == "falcon512":
        with oqs.Signature("Falcon-512") as sig:
            # Keygen
            pk, t_keygen = utils.timer(sig.generate_keypair)
            # Sign
            signature, t_sign = utils.timer(sig.sign, msg)
            # Verify
            _, t_verify = utils.timer(sig.verify, msg, signature, pk)

    return {
        "algorithm": alg,
        "keygen": t_keygen,
        "sign": t_sign,
        "verify": t_verify,
        "signature_length": len(signature)
    }
```


```python
## 📌 Section 5 — Run Benchmarks

import oqs
print(oqs.get_enabled_sig_mechanisms())

algorithms = ["ecdsa", "dilithium3", "falcon512"]
results = []

print("=== Running Signature Benchmarks ===")
for alg in algorithms:
    print(f"→ {alg}")
    results.append(benchmark_signature(alg))

df = pd.DataFrame(results)
```

    ('ML-DSA-44', 'ML-DSA-65', 'ML-DSA-87', 'ML-DSA-44-extmu', 'ML-DSA-65-extmu', 'ML-DSA-87-extmu', 'Falcon-512', 'Falcon-1024', 'Falcon-padded-512', 'Falcon-padded-1024', 'MAYO-1', 'MAYO-2', 'MAYO-3', 'MAYO-5', 'cross-rsdp-128-balanced', 'cross-rsdp-128-fast', 'cross-rsdp-128-small', 'cross-rsdp-192-balanced', 'cross-rsdp-192-fast', 'cross-rsdp-192-small', 'cross-rsdp-256-balanced', 'cross-rsdp-256-fast', 'cross-rsdp-256-small', 'cross-rsdpg-128-balanced', 'cross-rsdpg-128-fast', 'cross-rsdpg-128-small', 'cross-rsdpg-192-balanced', 'cross-rsdpg-192-fast', 'cross-rsdpg-192-small', 'cross-rsdpg-256-balanced', 'cross-rsdpg-256-fast', 'cross-rsdpg-256-small', 'OV-Is', 'OV-Ip', 'OV-III', 'OV-V', 'OV-Is-pkc', 'OV-Ip-pkc', 'OV-III-pkc', 'OV-V-pkc', 'OV-Is-pkc-skc', 'OV-Ip-pkc-skc', 'OV-III-pkc-skc', 'OV-V-pkc-skc', 'SNOVA_I_K', 'SNOVA_I_K_AES', 'SNOVA_I_B', 'SNOVA_I_B_AES', 'SNOVA_I_S', 'SNOVA_I_S_AES', 'SNOVA_III_K', 'SNOVA_III_K_AES', 'SNOVA_III_B', 'SNOVA_III_B_AES', 'SNOVA_III_S', 'SNOVA_III_S_AES', 'SNOVA_V_K', 'SNOVA_V_K_AES', 'SNOVA_V_B', 'SNOVA_V_B_AES', 'SNOVA_V_S', 'SNOVA_V_S_AES', 'mqom3_cat1_gf16_fast_ct', 'mqom3_cat1_gf16_fast_ot', 'mqom3_cat1_gf16_short_ct', 'mqom3_cat1_gf16_short_ot', 'mqom3_cat1_gf2_shorter_ct', 'mqom3_cat1_gf2_shorter_ot', 'mqom3_cat3_gf16_fast_ct', 'mqom3_cat3_gf16_fast_ot', 'mqom3_cat3_gf16_short_ct', 'mqom3_cat3_gf16_short_ot', 'mqom3_cat3_gf2_shorter_ct', 'mqom3_cat3_gf2_shorter_ot', 'mqom3_cat5_gf16_fast_ct', 'mqom3_cat5_gf16_fast_ot', 'mqom3_cat5_gf16_short_ct', 'mqom3_cat5_gf16_short_ot', 'mqom3_cat5_gf2_shorter_ct', 'mqom3_cat5_gf2_shorter_ot', 'SLH_DSA_PURE_SHA2_128S', 'SLH_DSA_PURE_SHA2_128F', 'SLH_DSA_PURE_SHA2_192S', 'SLH_DSA_PURE_SHA2_192F', 'SLH_DSA_PURE_SHA2_256S', 'SLH_DSA_PURE_SHA2_256F', 'SLH_DSA_PURE_SHAKE_128S', 'SLH_DSA_PURE_SHAKE_128F', 'SLH_DSA_PURE_SHAKE_192S', 'SLH_DSA_PURE_SHAKE_192F', 'SLH_DSA_PURE_SHAKE_256S', 'SLH_DSA_PURE_SHAKE_256F', 'SLH_DSA_SHA2_224_PREHASH_SHA2_128S', 'SLH_DSA_SHA2_224_PREHASH_SHA2_128F', 'SLH_DSA_SHA2_224_PREHASH_SHA2_192S', 'SLH_DSA_SHA2_224_PREHASH_SHA2_192F', 'SLH_DSA_SHA2_224_PREHASH_SHA2_256S', 'SLH_DSA_SHA2_224_PREHASH_SHA2_256F', 'SLH_DSA_SHA2_224_PREHASH_SHAKE_128S', 'SLH_DSA_SHA2_224_PREHASH_SHAKE_128F', 'SLH_DSA_SHA2_224_PREHASH_SHAKE_192S', 'SLH_DSA_SHA2_224_PREHASH_SHAKE_192F', 'SLH_DSA_SHA2_224_PREHASH_SHAKE_256S', 'SLH_DSA_SHA2_224_PREHASH_SHAKE_256F', 'SLH_DSA_SHA2_256_PREHASH_SHA2_128S', 'SLH_DSA_SHA2_256_PREHASH_SHA2_128F', 'SLH_DSA_SHA2_256_PREHASH_SHA2_192S', 'SLH_DSA_SHA2_256_PREHASH_SHA2_192F', 'SLH_DSA_SHA2_256_PREHASH_SHA2_256S', 'SLH_DSA_SHA2_256_PREHASH_SHA2_256F', 'SLH_DSA_SHA2_256_PREHASH_SHAKE_128S', 'SLH_DSA_SHA2_256_PREHASH_SHAKE_128F', 'SLH_DSA_SHA2_256_PREHASH_SHAKE_192S', 'SLH_DSA_SHA2_256_PREHASH_SHAKE_192F', 'SLH_DSA_SHA2_256_PREHASH_SHAKE_256S', 'SLH_DSA_SHA2_256_PREHASH_SHAKE_256F', 'SLH_DSA_SHA2_384_PREHASH_SHA2_128S', 'SLH_DSA_SHA2_384_PREHASH_SHA2_128F', 'SLH_DSA_SHA2_384_PREHASH_SHA2_192S', 'SLH_DSA_SHA2_384_PREHASH_SHA2_192F', 'SLH_DSA_SHA2_384_PREHASH_SHA2_256S', 'SLH_DSA_SHA2_384_PREHASH_SHA2_256F', 'SLH_DSA_SHA2_384_PREHASH_SHAKE_128S', 'SLH_DSA_SHA2_384_PREHASH_SHAKE_128F', 'SLH_DSA_SHA2_384_PREHASH_SHAKE_192S', 'SLH_DSA_SHA2_384_PREHASH_SHAKE_192F', 'SLH_DSA_SHA2_384_PREHASH_SHAKE_256S', 'SLH_DSA_SHA2_384_PREHASH_SHAKE_256F', 'SLH_DSA_SHA2_512_PREHASH_SHA2_128S', 'SLH_DSA_SHA2_512_PREHASH_SHA2_128F', 'SLH_DSA_SHA2_512_PREHASH_SHA2_192S', 'SLH_DSA_SHA2_512_PREHASH_SHA2_192F', 'SLH_DSA_SHA2_512_PREHASH_SHA2_256S', 'SLH_DSA_SHA2_512_PREHASH_SHA2_256F', 'SLH_DSA_SHA2_512_PREHASH_SHAKE_128S', 'SLH_DSA_SHA2_512_PREHASH_SHAKE_128F', 'SLH_DSA_SHA2_512_PREHASH_SHAKE_192S', 'SLH_DSA_SHA2_512_PREHASH_SHAKE_192F', 'SLH_DSA_SHA2_512_PREHASH_SHAKE_256S', 'SLH_DSA_SHA2_512_PREHASH_SHAKE_256F', 'SLH_DSA_SHA2_512_224_PREHASH_SHA2_128S', 'SLH_DSA_SHA2_512_224_PREHASH_SHA2_128F', 'SLH_DSA_SHA2_512_224_PREHASH_SHA2_192S', 'SLH_DSA_SHA2_512_224_PREHASH_SHA2_192F', 'SLH_DSA_SHA2_512_224_PREHASH_SHA2_256S', 'SLH_DSA_SHA2_512_224_PREHASH_SHA2_256F', 'SLH_DSA_SHA2_512_224_PREHASH_SHAKE_128S', 'SLH_DSA_SHA2_512_224_PREHASH_SHAKE_128F', 'SLH_DSA_SHA2_512_224_PREHASH_SHAKE_192S', 'SLH_DSA_SHA2_512_224_PREHASH_SHAKE_192F', 'SLH_DSA_SHA2_512_224_PREHASH_SHAKE_256S', 'SLH_DSA_SHA2_512_224_PREHASH_SHAKE_256F', 'SLH_DSA_SHA2_512_256_PREHASH_SHA2_128S', 'SLH_DSA_SHA2_512_256_PREHASH_SHA2_128F', 'SLH_DSA_SHA2_512_256_PREHASH_SHA2_192S', 'SLH_DSA_SHA2_512_256_PREHASH_SHA2_192F', 'SLH_DSA_SHA2_512_256_PREHASH_SHA2_256S', 'SLH_DSA_SHA2_512_256_PREHASH_SHA2_256F', 'SLH_DSA_SHA2_512_256_PREHASH_SHAKE_128S', 'SLH_DSA_SHA2_512_256_PREHASH_SHAKE_128F', 'SLH_DSA_SHA2_512_256_PREHASH_SHAKE_192S', 'SLH_DSA_SHA2_512_256_PREHASH_SHAKE_192F', 'SLH_DSA_SHA2_512_256_PREHASH_SHAKE_256S', 'SLH_DSA_SHA2_512_256_PREHASH_SHAKE_256F', 'SLH_DSA_SHA3_224_PREHASH_SHA2_128S', 'SLH_DSA_SHA3_224_PREHASH_SHA2_128F', 'SLH_DSA_SHA3_224_PREHASH_SHA2_192S', 'SLH_DSA_SHA3_224_PREHASH_SHA2_192F', 'SLH_DSA_SHA3_224_PREHASH_SHA2_256S', 'SLH_DSA_SHA3_224_PREHASH_SHA2_256F', 'SLH_DSA_SHA3_224_PREHASH_SHAKE_128S', 'SLH_DSA_SHA3_224_PREHASH_SHAKE_128F', 'SLH_DSA_SHA3_224_PREHASH_SHAKE_192S', 'SLH_DSA_SHA3_224_PREHASH_SHAKE_192F', 'SLH_DSA_SHA3_224_PREHASH_SHAKE_256S', 'SLH_DSA_SHA3_224_PREHASH_SHAKE_256F', 'SLH_DSA_SHA3_256_PREHASH_SHA2_128S', 'SLH_DSA_SHA3_256_PREHASH_SHA2_128F', 'SLH_DSA_SHA3_256_PREHASH_SHA2_192S', 'SLH_DSA_SHA3_256_PREHASH_SHA2_192F', 'SLH_DSA_SHA3_256_PREHASH_SHA2_256S', 'SLH_DSA_SHA3_256_PREHASH_SHA2_256F', 'SLH_DSA_SHA3_256_PREHASH_SHAKE_128S', 'SLH_DSA_SHA3_256_PREHASH_SHAKE_128F', 'SLH_DSA_SHA3_256_PREHASH_SHAKE_192S', 'SLH_DSA_SHA3_256_PREHASH_SHAKE_192F', 'SLH_DSA_SHA3_256_PREHASH_SHAKE_256S', 'SLH_DSA_SHA3_256_PREHASH_SHAKE_256F', 'SLH_DSA_SHA3_384_PREHASH_SHA2_128S', 'SLH_DSA_SHA3_384_PREHASH_SHA2_128F', 'SLH_DSA_SHA3_384_PREHASH_SHA2_192S', 'SLH_DSA_SHA3_384_PREHASH_SHA2_192F', 'SLH_DSA_SHA3_384_PREHASH_SHA2_256S', 'SLH_DSA_SHA3_384_PREHASH_SHA2_256F', 'SLH_DSA_SHA3_384_PREHASH_SHAKE_128S', 'SLH_DSA_SHA3_384_PREHASH_SHAKE_128F', 'SLH_DSA_SHA3_384_PREHASH_SHAKE_192S', 'SLH_DSA_SHA3_384_PREHASH_SHAKE_192F', 'SLH_DSA_SHA3_384_PREHASH_SHAKE_256S', 'SLH_DSA_SHA3_384_PREHASH_SHAKE_256F', 'SLH_DSA_SHA3_512_PREHASH_SHA2_128S', 'SLH_DSA_SHA3_512_PREHASH_SHA2_128F', 'SLH_DSA_SHA3_512_PREHASH_SHA2_192S', 'SLH_DSA_SHA3_512_PREHASH_SHA2_192F', 'SLH_DSA_SHA3_512_PREHASH_SHA2_256S', 'SLH_DSA_SHA3_512_PREHASH_SHA2_256F', 'SLH_DSA_SHA3_512_PREHASH_SHAKE_128S', 'SLH_DSA_SHA3_512_PREHASH_SHAKE_128F', 'SLH_DSA_SHA3_512_PREHASH_SHAKE_192S', 'SLH_DSA_SHA3_512_PREHASH_SHAKE_192F', 'SLH_DSA_SHA3_512_PREHASH_SHAKE_256S', 'SLH_DSA_SHA3_512_PREHASH_SHAKE_256F', 'SLH_DSA_SHAKE_128_PREHASH_SHA2_128S', 'SLH_DSA_SHAKE_128_PREHASH_SHA2_128F', 'SLH_DSA_SHAKE_128_PREHASH_SHA2_192S', 'SLH_DSA_SHAKE_128_PREHASH_SHA2_192F', 'SLH_DSA_SHAKE_128_PREHASH_SHA2_256S', 'SLH_DSA_SHAKE_128_PREHASH_SHA2_256F', 'SLH_DSA_SHAKE_128_PREHASH_SHAKE_128S', 'SLH_DSA_SHAKE_128_PREHASH_SHAKE_128F', 'SLH_DSA_SHAKE_128_PREHASH_SHAKE_192S', 'SLH_DSA_SHAKE_128_PREHASH_SHAKE_192F', 'SLH_DSA_SHAKE_128_PREHASH_SHAKE_256S', 'SLH_DSA_SHAKE_128_PREHASH_SHAKE_256F', 'SLH_DSA_SHAKE_256_PREHASH_SHA2_128S', 'SLH_DSA_SHAKE_256_PREHASH_SHA2_128F', 'SLH_DSA_SHAKE_256_PREHASH_SHA2_192S', 'SLH_DSA_SHAKE_256_PREHASH_SHA2_192F', 'SLH_DSA_SHAKE_256_PREHASH_SHA2_256S', 'SLH_DSA_SHAKE_256_PREHASH_SHA2_256F', 'SLH_DSA_SHAKE_256_PREHASH_SHAKE_128S', 'SLH_DSA_SHAKE_256_PREHASH_SHAKE_128F', 'SLH_DSA_SHAKE_256_PREHASH_SHAKE_192S', 'SLH_DSA_SHAKE_256_PREHASH_SHAKE_192F', 'SLH_DSA_SHAKE_256_PREHASH_SHAKE_256S', 'SLH_DSA_SHAKE_256_PREHASH_SHAKE_256F')
    === Running Signature Benchmarks ===
    → ecdsa
    → dilithium3
    → falcon512



```python
## 📌 Section 6 — Save Artifacts

### Signatures (Pickle)

os.makedirs("../data", exist_ok=True)

with open("../data/signatures.pkl", "wb") as f:
    pickle.dump(results, f)

print("✓ Saved signatures.pkl")

### Signature Sizes

df_sizes = df[["algorithm", "signature_length"]]
df_sizes.to_csv("../data/sizes.csv", index=False)

print("✓ Saved sizes.csv")

### Timings

df_timings = df[["algorithm", "keygen", "sign", "verify"]]
df_timings.to_csv("../data/timings.csv", index=False)

print("✓ Saved timings.csv")
```

    ✓ Saved signatures.pkl
    ✓ Saved sizes.csv
    ✓ Saved timings.csv



```python
## 📌 Section 7 — Generate Plots

os.makedirs("../plots", exist_ok=True)

### Signature Sizes

utils.plot_bar(
    df_sizes["algorithm"],
    df_sizes["signature_length"],
    "Signature Sizes",
    "../plots/signature_sizes.png"
)

print("✓ Created signature_sizes.png")

### Verification Times

utils.plot_line(
    df_timings["algorithm"],
    df_timings["verify"],
    "Verification Times",
    "../plots/verification_times.png"
)

print("✓ Created verification_times.png")

### Keygen Times

utils.plot_line(
    df_timings["algorithm"],
    df_timings["keygen"],
    "Keygen Times",
    "../plots/key_sizes.png"
)

print("✓ Created key_sizes.png")

```

    ✓ Created signature_sizes.png
    ✓ Created verification_times.png
    ✓ Created key_sizes.png



```python
## 📌 Section 8 — Completion Message

print("\n=== Signature Lab Complete ===")
print("Generated: signatures.pkl, sizes.csv, timings.csv")
print("Generated plots: signature_sizes.png, verification_times.png, key_sizes.png")
print("Continue with Notebook 02.")
```

    
    === Signature Lab Complete ===
    Generated: signatures.pkl, sizes.csv, timings.csv
    Generated plots: signature_sizes.png, verification_times.png, key_sizes.png
    Continue with Notebook 02.



```python

```
