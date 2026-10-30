# PQC Signature Lab & Crypto-Agility Demo

This project demonstrates practical Post-Quantum Cryptography (PQC) on a local Fedora Linux system using Python, Jupyter Notebooks,
and the Open Quantum Safe (OQS) library.

It consists of:
- 4 modular development notebooks
- 1 presentation notebook
- 2 Python modules
- automatically generated artifacts and plots

---

## 📁 Project Structure

pqc_signature_lab/
│
├── notebooks/
│   ├── 00_setup_environment.ipynb
│   ├── 01_signature_lab.ipynb
│   ├── 02_crypto_agility.ipynb
│   ├── 03_service_demo.ipynb
│   └── 99_presentation.ipynb
│
├── modules/
│   ├── utils.py
│   └── crypto_agility.py
│
├── data/
│   ├── pqc_libs.json
│   ├── signatures.pkl
│   ├── sizes.csv
│   ├── timings.csv
│   ├── agility_tests.json
│   └── service_logs.json
│
├── plots/
│   ├── signature_sizes.png
│   ├── verification_times.png
│   ├── key_sizes.png
│   ├── agility_matrix.png
│   └── service_latency.png
│
└── requirements.txt

---

## 📘 Notebooks

### **00_setup_environment.ipynb**
- Checks required packages  
- Generates `utils.py`  
- Generates `pqc_libs.json`  
- Runs mini benchmarks  

### **01_signature_lab.ipynb**
- Generates signatures (ECDSA, Dilithium3, Falcon512)  
- Measures keygen/sign/verify  
- Saves artifacts (`signatures.pkl`, `sizes.csv`, `timings.csv`)  
- Produces plots  

### **02_crypto_agility.ipynb**
- Generates `crypto_agility.py`  
- Runs algorithm switching tests  
- Measures agility costs  
- Produces plots  

### **03_service_demo.ipynb**
- Starts mini FastAPI service  
- Executes client requests  
- Measures latency  
- Produces `service_logs.json` and plot  

### **99_presentation.ipynb**
- Presentation notebook  
- Loads all artifacts  
- Generates missing artifacts (fallback)  
- Runs live demos  

---

## 🔧 Modules

### **utils.py**
- Timer  
- Bar/line plots  
- Helper functions  

### **crypto_agility.py**
Unified signature API:
- `sign(msg, alg)`
- `verify(msg, signature, alg)`

Supported algorithms:
- `ecdsa`
- `dilithium3`
- `falcon512`

---

## 🧪 Installation (Fedora)

### System packages

```bash
sudo dnf install python3-devel python3-pip gcc openssl-devel
sudo dnf install liboqs liboqs-devel
```

### Python packages

```bash
pip install -r requirements.txt
```

---

## 🎯 Project Goals

This project demonstrates:

- Classical vs PQC signatures  
- Performance comparisons  
- Crypto‑Agility (algorithm switching)  
- PQC service demo  
- Fully reproducible PQC experiments  

Ideal for presentations, workshops, and team discussions on PQC migration.

---

## 📜 License

MIT License (optional)
```