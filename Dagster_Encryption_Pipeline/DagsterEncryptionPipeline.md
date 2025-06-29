# 1. 🚀 Project Introduction: Dagster's Crypto-Pipeline

## Objective  
The purpose of a Dagster Encryption Pipeline is to test and showcase dagsters DAG-chaining capabilities,
resulting in a self-contained encryption-decryption pipeline which, 1) connects different encryption methods
of  a plain text prior to, 2) decrypting (and thus reversing the performed chain of encryption operations 
performed on an) encrypted message.  
 
This project aims at designing a encryption-decryption pipeline via Dagsterr's website that would support users in 
**generating robustly encrpted messages** from a well-defined plain text input 
(see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/GaugeStudeBalanced/GaugeStudy.md#8--references) 1 - 3 below).

# 2. 🔐 Encryption-Decryption Dagster Pipeline

This project implements a secure and modular encryption-decryption pipeline using [Dagster](https://dagster.io/). It orchestrates a hybrid cryptographic workflow combining:

- **Vigenère Cipher** for classical character-level obfuscation
- **AES-256 (CBC mode)** for symmetric encryption of structured data
- **RSA-OAEP** for secure asymmetric key wrapping

The pipeline defines two primary Dagster ops:

- `encrypt_op`: Encrypts plaintext input using Vigenère, AES, and RSA layers in sequence
- `decrypt_op`: Reverses the process to retrieve the original plaintext

The `encrypt_main()` and `decrypt_main()` functions are pure, testable Python routines that can be invoked independently or orchestrated via Dagster. This makes the system both CI-friendly and scalable.

Whether you're securing short messages or experimenting with layered encryption in a data workflow context, this pipeline is a clean, educational foundation.

## 2.1 Structure of the Encryption Pipeline

Let us proceed layer by layer — from old-school ciphers to modern cryptographic standards — and unpack how each is applied algorithmically, mathematically, and in your Python implementation.

### 🔠 Vigenère Cipher

**Mathematical Concept**:
The Vigenère cipher is a polyalphabetic substitution cipher that shifts each character by an amount determined by a repeating keyword. Given:
- plaintext letter \( P_i \)
- keyword letter \( K_i \)
- alphabet size \( 26 \)

The encryption function is:
\[
C_i = (P_i + K_i) \mod 26
\]
Decryption is simply:
\[
P_i = (C_i - K_i + 26) \mod 26
\]

**Pythonic Implementation**:
In your pipeline, this is done using `ord()` for character-to-index mapping and `cycle()` from `itertools` to repeat the key:

```python
for c, k in zip(plaintext.lower(), cycle(key.lower())):
    if c in alphabet:
        shifted = (ord(c) - ord('a') + ord(k) - ord('a')) % 26
        enc.append(chr(ord('a') + shifted))
```

It’s efficient, readable, and makes classical encryption elegantly Pythonic.

### 🔐 AES-256 (CBC mode)

**Mathematical Foundation**:
AES (Advanced Encryption Standard) is a symmetric-key block cipher based on the Rijndael algorithm. Key concepts:
- Operates on 128-bit blocks
- Uses 14 rounds for AES-256 (with 256-bit keys)
- Each round involves byte substitution (S-box), row shifting, column mixing, and key addition

CBC (Cipher Block Chaining) mode adds security by XOR’ing each block with the previous ciphertext block (with a unique IV for the first one):
\[
C_i = E_K(P_i \oplus C_{i-1})
\]

**Pythonic Application**:
You're using `cryptography`'s `Cipher` object and PKCS7 padding to align with 128-bit blocks:

```python
padder = padding.PKCS7(128).padder()
cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
```

The AES logic is clean and leverages Python’s object-oriented cryptographic primitives for security and clarity.

### 🔑 RSA-OAEP

**Mathematical Structure**:
RSA encryption is built on the difficulty of factoring large numbers:
\[
C = M^e \mod n
\quad \text{and} \quad
M = C^d \mod n
\]

OAEP (Optimal Asymmetric Encryption Padding) adds a randomized mask to the message to prevent deterministic output:
- Uses MGF1 (Mask Generation Function)
- Applies two-layered masking with a hash algorithm (e.g., SHA-256)

**Pythonic Mechanics**:
You use OAEP with SHA-256 for padding during key wrapping (wrapping AES key with RSA):

```python
public_key.encrypt(
    aes_key,
    asympadding.OAEP(
        mgf=asympadding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None
    )
)
```

This adheres to modern best practices, prevents padding oracle attacks, and ensures non-repetitive ciphertexts.

### Conclusion

Each layer you’ve built plays a distinct role:
- Vigenère: human-readable obfuscation
- AES-256-CBC: high-speed bulk data encryption
- RSA-OAEP: secure key distribution

Together, they give your pipeline both **depth** and **defense in layers** — elegant, robust, and cryptographically sound.

## 2.2 Elaboration on pythonic implementation details

**Layering multiple encryption techniques** is a legitimate way to increase confidentiality, especially when combining reversible 
(symmetric/asymmetric ciphers) with non-reversible (hashing or key derivation) methods. Let’s break down a pipeline concept:

### 🔐 Step 1: Vigenère or Caesar (Lightweight Substitution)
A first-pass classical cipher adds basic scrambling and deters casual eyes. We’ve already built this. An example class:  

```python
class VigenereCipher:
    def __init__(self, keyword: str):
        self.keyword = keyword.lower()
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'

    def shift_char(self, c, key_c, encode=True):
        if c not in self.alphabet:
            return c
        shift = self.alphabet.index(key_c)
        if not encode:
            shift = -shift
        return self.alphabet[(self.alphabet.index(c) + shift) % 26]

    def transform(self, text: str, encode=True):
        text = text.lower()
        result = []
        key_len = len(self.keyword)
        for i, c in enumerate(text):
            key_c = self.keyword[i % key_len]
            result.append(self.shift_char(c, key_c, encode))
        return ''.join(result)

    def encrypt(self, text: str) -> str:
        return self.transform(text, True)

    def decrypt(self, text: str) -> str:
        return self.transform(text, False)

if __name__ == "__main__":
    keyword = input("Enter encryption keyword: ").strip()
    cipher = VigenereCipher(keyword)

    plaintext = input("Enter text to encrypt: ").strip()
    encrypted = cipher.encrypt(plaintext)
    print(f"Encrypted text: {encrypted}")

    decrypted = cipher.decrypt(encrypted)
    print(f"Decrypted back: {decrypted}")
    ```


### 🔁 Step 2: Symmetric Encryption (AES or ChaCha20)
Use a **modern symmetric cipher** like:
- **AES** (Advanced Encryption Standard) — widely adopted, secure, hardware-accelerated
- **ChaCha20** — faster in software and more resistant to timing attacks

🛠 We would need a secure key and an Initialization Vector (IV) — both must be handled carefully.


### 🔑 Step 3: Asymmetric Encryption (RSA or ECC)
Wrap the **AES/ChaCha key using RSA** (public-key crypto). That way:
- Only the intended recipient (with the private key) can unwrap the symmetric key.
- This hybrid approach is how HTTPS and encrypted messaging apps work.


### 🔂 Step 4: Hashing + Signature (Optional but Powerful)
- **Hash (SHA-256)** the original text for **integrity checking**
- Sign the hash using a private key (e.g., **RSA signature** or **ECDSA**) to verify authenticity


### 🔁 Bonus: Polymorphic Layer
A **polymorphic cipher** randomly alters its encoding strategy per encryption session — changing keys, routes, or even algorithm subsets. It’s excellent for:
- Detecting tampering or replay
- Obfuscation-based defense (especially in malware counter-analysis)

---

### 📦 Example Pipeline Flow

```plaintext
[Plaintext]
   ↓ Vigenère (keyword obfuscation)
[Step 1 Output]
   ↓ AES (symmetric block encryption)
[Step 2 Output]
   ↓ RSA public key encryption (protect AES key)
[Encrypted + EncryptedKey]
   ↓ Hash + Sign (SHA-256 + RSA signature)
[Final Output Package]
```

## 2.3 Pythonic Encryption code

We are venturing into thrilling territory: a **modular, secure cryptographic pipeline** with CLI orchestration. 
Think of it like building your own personal encryption factory. Here’s a blueprint you can iterate on, using Python’s
 `cryptography` library (or `PyCryptodome`, if preferred). In the following we lay out key stages and suggest how to 
 wire them into a mini-pipeline or even a Dagster-lite setup.

### 🔐 Stage 1: Vigenère Cipher (Custom Classical Layer)

```python
def vigenere_encrypt(plaintext: str, key: str) -> str:
    from itertools import cycle
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    enc = []
    for c, k in zip(plaintext.lower(), cycle(key.lower())):
        if c in alphabet:
            shifted = (ord(c) - ord('a') + ord(k) - ord('a')) % 26
            enc.append(chr(ord('a') + shifted))
        else:
            enc.append(c)
    return ''.join(enc)
```

### 🔐 Stage 2: AES Encryption (from `cryptography`)

```python
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import os

def aes_encrypt(data: bytes, key: bytes) -> tuple[bytes, bytes, bytes]:
    iv = os.urandom(16)
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(data) + padder.finalize()
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ct = encryptor.update(padded_data) + encryptor.finalize()
    return ct, iv, key
```

### 🔐 Stage 3: RSA Wrapper (Encrypting AES Key)

```python
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asympadding
from cryptography.hazmat.primitives import serialization, hashes

def rsa_encrypt_key(aes_key: bytes, public_key) -> bytes:
    return public_key.encrypt(
        aes_key,
        asympadding.OAEP(
            mgf=asympadding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
```

### 🔐 Stage 4: SHA-256 Hashing + Signing (Optional)

```python
from cryptography.hazmat.primitives.asymmetric import padding as asypad
from cryptography.hazmat.primitives import hashes

def sign_data(private_key, message: bytes) -> bytes:
    return private_key.sign(
        message,
        asypad.PSS(
            mgf=asypad.MGF1(hashes.SHA256()),
            salt_length=asypad.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
```

### ⚙️ CLI Pipeline Script (Think: Entry Point)

```python
import argparse

def main():
    parser = argparse.ArgumentParser(description="Secure Encryption Pipeline")
    parser.add_argument("text", help="Text to encrypt")
    parser.add_argument("--vkey", required=True, help="Vigenère keyword")
    args = parser.parse_args()

    # Step 1: Vigenère
    v_encrypted = vigenere_encrypt(args.text, args.vkey)

    # Step 2: AES
    aes_key = os.urandom(32)
    aes_ct, iv, key = aes_encrypt(v_encrypted.encode(), aes_key)

    # Step 3: RSA
    # Load/generate RSA keys here
    # rsa_ct = rsa_encrypt_key(aes_key, public_key)

    print("Encrypted Text:", aes_ct.hex())

if __name__ == "__main__":
    main()
```

### 🚀 Orchestrating With a Free Dagster-Alternative

If Dagster is too heavy or too hosted, we may consider:
- **Prefect (open-source Core edition)** — beautiful CLI-first workflow engine
- **Airflow (via Docker)** — heavier but still usable locally
- **Plain `typer`/`argparse` + task runner** — lightweight and shell-friendly

Or even write a *custom* YAML + Python orchestrator that loads YAML pipeline definitions and runs each stage.

## 2.4 Pythonic Decryption code

Below is a **complete, reverse decryption pipeline** that unwinds every step of our original encryption process: RSA → AES → Vigenère.  
 
This version includes:

- RSA decryption of the AES key
- AES decryption of the Vigenère-encrypted content
- Vigenère decryption of the final plaintext
- CLI arguments and clear structure to match your encryption pipeline

We can paste this into a single Python file or cell:

```python
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding, serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding as asympadding
from cryptography.hazmat.backends import default_backend
from itertools import cycle
import argparse

# Step 1: Vigenère Decryption
def vigenere_decrypt(ciphertext: str, key: str) -> str:
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    plaintext = []
    for c, k in zip(ciphertext.lower(), cycle(key.lower())):
        if c in alphabet:
            shift = (ord(c) - ord(k)) % 26
            plaintext.append(chr(ord('a') + shift))
        else:
            plaintext.append(c)
    return ''.join(plaintext)

# Step 2: AES Decryption
def aes_decrypt(ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    return unpadder.update(padded_data) + unpadder.finalize()

# Step 3: RSA Key Decryption
def rsa_decrypt_key(encrypted_key: bytes, private_key_path: str) -> bytes:
    with open(private_key_path, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,
            backend=default_backend()
        )
    return private_key.decrypt(
        encrypted_key,
        asympadding.OAEP(
            mgf=asympadding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

# Main CLI logic
def main():
    parser = argparse.ArgumentParser(description="Secure Decryption Pipeline")
    parser.add_argument("ciphertext", help="AES ciphertext as hex string")
    parser.add_argument("--iv", required=True, help="AES initialization vector (hex)")
    parser.add_argument("--rsa_key", required=True, help="Path to private RSA PEM key")
    parser.add_argument("--enc_key", required=True, help="RSA-encrypted AES key (hex)")
    parser.add_argument("--vkey", required=True, help="Vigenère keyword used at encryption")
    args = parser.parse_args()

    # Step 1: Decrypt AES key with RSA
    encrypted_key = bytes.fromhex(args.enc_key)
    aes_key = rsa_decrypt_key(encrypted_key, args.rsa_key)

    # Step 2: Decrypt AES-encrypted Vigenère text
    ciphertext = bytes.fromhex(args.ciphertext)
    iv = bytes.fromhex(args.iv)
    vigenere_encrypted = aes_decrypt(ciphertext, aes_key, iv).decode()

    # Step 3: Decrypt Vigenère to get plaintext
    original_text = vigenere_decrypt(vigenere_encrypted, args.vkey)

    print("\n✅ Decrypted Plaintext:", original_text)

if __name__ == "__main__":
    main()
```

---

### 🛠️ Sample Usage (Terminal):  

The pipeline expects:
- the RSA-encrypted AES key (`--enc_key`)
- the AES ciphertext (`--ciphertext`) and IV (`--iv`) as hex
- the RSA private key file path
- the Vigenère keyword  

```bash
python decrypt_pipeline.py \
  "62c99f..." \
  --iv "3f3ec2..." \
  --rsa_key private_key.pem \
  --enc_key "45ff19..." \
  --vkey lemon
``` 

# 3. Elaborate Pythonic implementation of the encryption-decryption pipeline with dagster

## 3.1 Packages to be installed

````bash
!pip install cryptography
!pip install pycryptodome
!pip install typer rich
!pip install dagster dagster-webserver

# Optional: for Jupyter support
!pip install dagster[jupyter]

# Optional: for filesystem and database IO management
!pip install dagster-io

# Optional: include example integrations
!pip install dagster pandas requests
````
## 3.2 Generate private and public keys

````python
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import os

# Ensure the rsa/ directory exists
os.makedirs("rsa", exist_ok=True)

# Generate private key
private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

# Save private key
with open("root/rsa/private.pem", "wb") as f:
    f.write(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
    )

# Save public key
public_key = private_key.public_key()
with open("root/rsa/public.pem", "wb") as f:
    f.write(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
````  

Remark: These keys have to be generated simultaneously within the same folder in order for the 
pipeline to operate flawlessly.

## 3.3 Encryption pipeline

The encryption pipeline code may look as follows:

````python
def vigenere_encrypt(plaintext: str, key: str) -> str:
    from itertools import cycle
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    enc = []
    for c, k in zip(plaintext.lower(), cycle(key.lower())):
        if c in alphabet:
            shifted = (ord(c) - ord('a') + ord(k) - ord('a')) % 26
            enc.append(chr(ord('a') + shifted))
        else:
            enc.append(c)
    return ''.join(enc)

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import os

def aes_encrypt(data: bytes, key: bytes) -> tuple[bytes, bytes, bytes]:
    iv = os.urandom(16)
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(data) + padder.finalize()
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ct = encryptor.update(padded_data) + encryptor.finalize()
    return ct, iv, key

from cryptography.hazmat.primitives.asymmetric import rsa, padding as asympadding
from cryptography.hazmat.primitives import serialization, hashes

def rsa_encrypt_key(aes_key: bytes, public_key) -> bytes:
    return public_key.encrypt(
        aes_key,
        asympadding.OAEP(
            mgf=asympadding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

from cryptography.hazmat.primitives.asymmetric import padding as asypad
from cryptography.hazmat.primitives import hashes

def sign_data(private_key, message: bytes) -> bytes:
    return private_key.sign(
        message,
        asypad.PSS(
            mgf=asypad.MGF1(hashes.SHA256()),
            salt_length=asypad.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

import argparse

def main():
    parser = argparse.ArgumentParser(description="Secure Encryption Pipeline")
    parser.add_argument("text", help="Text to encrypt")
    parser.add_argument("--vkey", required=True, help="Vigenère keyword")
    args = parser.parse_args()

    # Step 1: Vigenère
    v_encrypted = vigenere_encrypt(args.text, args.vkey)

    # Step 2: AES
    aes_key = os.urandom(32)
    aes_ct, iv, key = aes_encrypt(v_encrypted.encode(), aes_key)

    # Step 3: RSA
    # Load/generate RSA keys here
    # rsa_ct = rsa_encrypt_key(aes_key, public_key)

    print("Encrypted Text:", aes_ct.hex())

if __name__ == "__main__":
    main()
````

### Example usage (bash)

Here’s an example of how one could use the encryption pipeline from a terminal, along with a demonstration of what happens at each step using mock data:

#### ✅ Step 1: Save the Script

Save your pipeline code into a Python file, for example:

```
secure_pipeline.py
```

#### ✅ Step 2: Run It From CLI

```bash
python secure_pipeline.py "This is top secret text." --vkey lemon
```

This triggers:

1. **Vigenère Encryption** of the plaintext using the keyword `lemon`  
   Output (example): `aohu wp hsd fvuvy vica.`

2. **AES Encryption** of the result (binary output encoded as hex)  
   Output (example): `afcb0192b8eaeffa1e3435a4da02c976f65411a7f71af5a56ce10833e6785021`

3. **RSA Encryption** (optional — in your code it’s currently commented out)

4. **Digital Signing** of the message (optional — defined but not yet called)

#### 🔐 Example Output (simplified)

```plaintext
Encrypted Text: afcb0192b8eaeffa1e3435a4da02c976f65411a7f71af5a56ce10833e6785021
```

#### 💡 Notes for Extension

- To make the pipeline fully functional, we should:
  - Load or generate a public/private RSA key pair
  - Implement `rsa_encrypt_key(aes_key, public_key)` and print/store the result
  - Store or transmit the IV and the RSA-encrypted AES key securely
  - Optionally sign the AES ciphertext with `sign_data(...)`
  

## 3.4 Decryption pipeline

The decryption pipeline code may look as follows: 

````python
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding, serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding as asympadding
from cryptography.hazmat.backends import default_backend
from itertools import cycle
import argparse

# Step 1: Vigenère Decryption
def vigenere_decrypt(ciphertext: str, key: str) -> str:
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    plaintext = []
    for c, k in zip(ciphertext.lower(), cycle(key.lower())):
        if c in alphabet:
            shift = (ord(c) - ord(k)) % 26
            plaintext.append(chr(ord('a') + shift))
        else:
            plaintext.append(c)
    return ''.join(plaintext)

# Step 2: AES Decryption
def aes_decrypt(ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    return unpadder.update(padded_data) + unpadder.finalize()

# Step 3: RSA Key Decryption
def rsa_decrypt_key(encrypted_key: bytes, private_key_path: str) -> bytes:
    with open(private_key_path, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,
            backend=default_backend()
        )
    return private_key.decrypt(
        encrypted_key,
        asympadding.OAEP(
            mgf=asympadding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

# Main CLI logic
def main():
    parser = argparse.ArgumentParser(description="Secure Decryption Pipeline")
    parser.add_argument("ciphertext", help="AES ciphertext as hex string")
    parser.add_argument("--iv", required=True, help="AES initialization vector (hex)")
    parser.add_argument("--rsa_key", required=True, help="Path to private RSA PEM key")
    parser.add_argument("--enc_key", required=True, help="RSA-encrypted AES key (hex)")
    parser.add_argument("--vkey", required=True, help="Vigenère keyword used at encryption")
    args = parser.parse_args()

    # Step 1: Decrypt AES key with RSA
    encrypted_key = bytes.fromhex(args.enc_key)
    aes_key = rsa_decrypt_key(encrypted_key, args.rsa_key)

    # Step 2: Decrypt AES-encrypted Vigenère text
    ciphertext = bytes.fromhex(args.ciphertext)
    iv = bytes.fromhex(args.iv)
    vigenere_encrypted = aes_decrypt(ciphertext, aes_key, iv).decode()

    # Step 3: Decrypt Vigenère to get plaintext
    original_text = vigenere_decrypt(vigenere_encrypted, args.vkey)

    print("\n✅ Decrypted Plaintext:", original_text)

if __name__ == "__main__":
    main()
````

### Example usage (bash)

```bash
python decrypt_pipeline.py \
  "62c99f..." \
  --iv "3f3ec2..." \
  --rsa_key private_key.pem \
  --enc_key "45ff19..." \
  --vkey lemon
```

# 4. dagster pipeline orchestration

We are ready to bring in Dagster to orchestrate your encryption and decryption flows, here's how one can wire it all together. 
We’ll treat `secure_pipeline.py` and `decrypt_pipeline.py` as modular components and then use Dagster to manage the flow, 
parameters, and runtime configuration.  

Dagster isn’t just a tool for orchestrating data pipelines — it’s a framework that encourages a thoughtful, *composable*, and 
*observable* approach to building reliable workflows.

### 🧠 Dagster’s Core Philosophy

At its heart, Dagster is built around three transformative ideas:

1. **Software-defined assets (SDAs)**:
   - Instead of thinking in terms of abstract steps or jobs, Dagster encourages you to define the *data itself* as first-class citizens.
   - You define assets and how they're built — this shifts the focus from *how to run tasks* to *what data artifacts are produced and how they depend on each other*.

2. **Declarative over imperative thinking**:
   - Traditional pipeline tools often define “what to do step by step.” Dagster, on the other hand, lets you declare the relationships and let the system figure out optimal execution.
   - Dependencies (inputs and outputs) are tracked explicitly.

3. **Observability, testing, and ergonomics from day one**:
   - Dagster treats reliability, logging, introspection, and local testing as core concerns — not afterthoughts.
   - You can dry-run, inspect intermediate values, retry failed steps, visualize asset lineage, and more — all with powerful tooling like Dagit.

### ⚙️ How Dagster Works

Your pipeline is made of:

- **@asset or @op**: The individual units of computation. Think of an `@op` as a function that’s trackable, testable, and observable. `@asset` declares higher-level data dependencies.

- **Graphs / Jobs**: You group `ops` or `assets` into logical workflows using `@job`.

- **Dagit**: The browser-based UI for running, inspecting, and debugging pipelines. It offers a beautiful DAG view of your workflows and interactive logs.

- **IO Managers / Resources**: Pluggable hooks to handle where your data goes (memory, filesystem, database), and how you connect to external services (like S3, Spark, PostgreSQL, etc.).

### 🌟 What Makes Dagster Different

- **Dev-first & CI-friendly**: Local workflows mirror production; it plays nicely with Git.
- **Context-aware execution**: Each op receives a structured context (e.g. logs, config, resources) rather than relying on global state.
- **Robust type system**: You can define schemas, validation, and structured inputs/outputs — it’s like the joy of Python typing, but for pipeline data.
- **Incremental builds**: It supports re-materializing only the assets that are out-of-date — like `make` for data pipelines.

In short, Dagster reimagines what a data orchestration tool should be: *introspective*, *testable*, and *built for humans who like clean code and confidence in deployment*.

## 4.1 📦 Step 1: Project Structure

```
project_root/
│
├── pipelines/
│   ├── secure_pipeline.py        # contains encrypt logic
│   ├── decrypt_pipeline.py       # contains decrypt logic
│
├── dagster_pipelines/
│   ├── __init__.py
│   ├── jobs.py                   # Dagster job definitions
│   └── ops.py                    # Dagster steps
│
├── workspace.yaml
└── dagster.yaml
```


## 4.2 ⚙️ Step 2: Define Dagster Ops (ops.py)

```python
from dagster import op
from pipelines import secure_pipeline, decrypt_pipeline

@op
def encrypt_op(context, text: str, vkey: str):
    return secure_pipeline.encrypt_main(text, vkey)  # create `encrypt_main()` in secure_pipeline.py

@op
def decrypt_op(context, encrypted_data: dict, vkey: str, rsa_key_path: str):
    return decrypt_pipeline.decrypt_main(
        ciphertext=encrypted_data['ciphertext'],
        iv=encrypted_data['iv'],
        enc_key=encrypted_data['enc_key'],
        priv_key_path=rsa_key_path,
        vkey=vkey
    )
```


## 4.3 🔧 Step 3: Define the Dagster Job (jobs.py)

```python
from dagster import job
from .ops import encrypt_op, decrypt_op

@job
def encryption_decryption_job():
    decrypted = decrypt_op(
        encrypted_data=encrypt_op(),
        vkey="lemon",
        rsa_key_path="rsa_keys/private_key.pem"
    )
```

We can also parameterize the job with `@configurable` or use Dagster’s config schema if you want to expose CLI/graph inputs.


## 4.4 🔌 Step 4: Entry Point (CLI or UI)

From the terminal:

```bash
dagster dev   # launches Dagster UI
```

Or from a Python CLI:

```bash
from dagster_pipelines.jobs import encryption_decryption_job
encryption_decryption_job.execute_in_process()
```


## 4.5 🧩 How our secure_pipeline.py and decrypt_pipeline.py Should Look

Each should expose a callable like this:

```python
# secure_pipeline.py
def encrypt_main(text: str, vkey: str) -> dict:
    # return {
    #   "ciphertext": <hex>,
    #   "iv": <hex>,
    #   "enc_key": <hex>
    # }
```

```python
# decrypt_pipeline.py
def decrypt_main(ciphertext, iv, enc_key, priv_key_path, vkey) -> str:
    # return plaintext
```


## 4.6 🚀 Ready to Run

This turns our CLI tools into reusable data pipeline steps. Add logging, tests, or run schedules and we on our way 
to production-grade cryptography orchestration.

# 5. Pipeline execution

To get up and running with **Dagster**, we will want to install its core package along with any extras you plan to use 
(like AWS, dbt, Airbyte, etc). But for a basic local development setup with the web UI, here’s what we need:

```bash
!pip install dagster dagster-webserver
```

This gives us:
- `dagster`: core library for defining jobs, ops, resources, etc.
- `dagster-webserver`: the lightweight UI you can launch with `dagster dev`

If we are using extras like Jupyter, SQLAlchemy, or cloud integrations, you can layer in specific extras:

```bash
# Optional: for Jupyter support
!pip install dagster[jupyter]

# Optional: for filesystem and database IO management
!pip install dagster-io

# Optional: include example integrations
!pip install dagster pandas requests
```

Once installed, we launch our dev server like this in a terminal:

```bash
dagster dev
```

And our UI will be live at [http://localhost:3000](http://localhost:3000) 🎛️  

## 5.1 Step 1: package installation

````bash
!pip install cryptography
!pip install pycryptodome
!pip install typer rich
!pip install dagster dagster-webserver

# Optional: for Jupyter support
!pip install dagster[jupyter]

# Optional: for filesystem and database IO management
!pip install dagster-io

# Optional: include example integrations
!pip install dagster pandas requests
````

## 5.2 Step 2: Generate appropriate folder structure

### Folder Structure, part 1
![Two-factor balanced Gauge Study results](https://github.com/NenadBalaneskovic/ExternalProjects/blob/100f6c03a6d8c9b7298ec33a88608186b949083d/GaugeStudeBalanced/two_factor_gauge_green_corrupt.PNG)

### Py and yaml pipeline files

1. A ![csv file](https://github.com/NenadBalaneskovic/ExternalProjects/blob/1cdd76715308f6a7970d5a0852e0adc286666f31/GaugeStudeBalanced/results.csv) of extracted Gauge Study parameters.
2. A ![pdf report](https://github.com/NenadBalaneskovic/ExternalProjects/blob/6d3669798335f1ee4f07293aad3b1bfdd020a035/GaugeStudeBalanced/report_2_factor_GaugeStudy.pdf) of obtained Gauge Study results.

## 5.3 Step 3: Generate public and private keys

````python
`from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import os

# Ensure the rsa/ directory exists
os.makedirs("rsa", exist_ok=True)

# Generate private key
private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

# Save private key
with open("root/rsa/private.pem", "wb") as f:
    f.write(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
    )

# Save public key
public_key = private_key.public_key()
with open("root/rsa/public.pem", "wb") as f:
    f.write(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
````

## 5.4 Step 4: Execute the dagster UI

````python
import os
print(os.getcwd()) # Verify the existence of our current working directory

# Chnage to the pwd location
os.chdir("C:\\Users\\balan\\OneDrive\\Desktop\\EncryptionPipeline\\root")

# Execute the dagster-pipeline
!dagster dev -w workspace.yaml
````

## 5.5 Step 5: Open our dagster UI in a browser, run the pipeline and monitor its progress

Our UI will be live at [http://localhost:3000](http://localhost:3000) 🎛️  


# 6. Resume - dagster's visual tour

Let us visualize how our encryption-decryption pipeline fits into Dagster’s elegant DAG (Directed Acyclic Graph) philosophy.

## 6.1 🔄 The DAG View of Our Crypto Pipeline

Our Dagster job can be visualized as a simple, two-op DAG:

```
          ┌─────────────┐
          │ encrypt_op  │
          └─────┬───────┘
                │
                ▼
          ┌─────────────┐
          │ decrypt_op  │
          └─────────────┘
```

## 6.2 🧩 Op Breakdown

- **`encrypt_op`**
  - Input: plaintext string and Vigenère key
  - Flow:
    1. Applies the Vigenère cipher
    2. Generates a random AES-256 key
    3. Encrypts the Vigenère text with AES in CBC mode
    4. Encrypts the AES key using an RSA public key (OAEP)
  - Output: dictionary containing:
    - ciphertext
    - initialization vector (IV)
    - RSA-encrypted AES key

- **`decrypt_op`**
  - Input: the dictionary above, private RSA key path, and Vigenère key
  - Flow:
    1. Decrypts the AES key using RSA private key
    2. Uses the AES key + IV to decrypt the ciphertext into Vigenère text
    3. Applies Vigenère decryption to recover original plaintext
  - Output: restored message (string)

## 6.3 🌐 How Dagster Makes This Better

- **Visual traceability**: In Dagit, we see this pipeline clearly mapped, so you know exactly what data flows where.
- **Logs and retries**: If, say, the decryption step fails (e.g., due to mismatched keys), only `decrypt_op` needs to be retried — `encrypt_op`'s output can be reused.
- **Dagster context**: Each op can log intermediate results using `context.log`, making debugging crypto internals much easier.
- **Unit testable**: Because each op wraps around `encrypt_main()` and `decrypt_main()`, you can test them in isolation or as a job.

## 6.4 🛡️ Crypto Encryption-Decryption Pipeline — Powered by Dagster (a resume)

### Overview

This project implements a modular, multi-layered encryption and decryption pipeline using **Python** and **Dagster**. It combines three cryptographic layers — a classical Vigenère cipher, industry-standard AES-256 in CBC mode, and RSA-OAEP key encapsulation — into a single orchestrated workflow. Dagster's type-safe, observable framework powers clean data lineage, easy debugging, and end-to-end reproducibility.

### Pipeline Components

1. **Vigenère Cipher (Python-based Classic Cipher)**
   - Applies polyalphabetic substitution at the character level
   - Adds lightweight obfuscation prior to cryptographic encryption
   - Fully reversible for decryption step

2. **AES-256 (CBC Mode)**
   - Uses a 256-bit randomly generated key for symmetric encryption
   - Ensures confidentiality with PKCS7 padding and secure IV generation
   - Implemented using the `cryptography` library for robustness

3. **RSA-OAEP**
   - Encrypts the AES key using an RSA public key
   - Decrypts it using the matching private key
   - Guarantees secure key exchange and resistance to chosen ciphertext attacks

### Dagster Integration

- **@op Decorators**:
  - `encrypt_op`: Wraps the encryption routine
  - `decrypt_op`: Wraps the decryption logic with error-safe RSA and AES reversal

- **@job Composition**:
  - `encryption_decryption_job`: A sequential DAG that securely encrypts data and then verifies its integrity by decrypting it

- **Dagit Support**:
  - Real-time DAG visualization
  - Execution logs and intermediate values
  - Easy testing and local development

### Use Cases

- Educational tool for layered cryptography
- Modular base for secure communication pipelines
- Blueprint for adding encryption to ETL workflows or asset versioning

✅ **All stored results can be accessed via the following links:**  
1. A ![csv file](https://github.com/NenadBalaneskovic/ExternalProjects/blob/1cdd76715308f6a7970d5a0852e0adc286666f31/GaugeStudeBalanced/results.csv) of extracted Gauge Study parameters.
2. A ![pdf report](https://github.com/NenadBalaneskovic/ExternalProjects/blob/6d3669798335f1ee4f07293aad3b1bfdd020a035/GaugeStudeBalanced/report_2_factor_GaugeStudy.pdf) of obtained Gauge Study results.

# 7. 📚 References
1. C. Paar, J. Pelzl: "__Understanding Cryptography__", Springer (2010); H. Delfs, H. Knebl: "__Introduction to Cryptography__", 3rd Ed. Springer (2015); J. Katz, Y. lindell: "__Introduction to Modern Cryptography__", 2nd Ed, CRC Press (2015); 
O. Goldreich: "__Foundations of Cryptography__", Cambridge University Press (2008); J. P. Aumasson: "__Serious Cryptography__", no starch press (2018).
2. [![Jupyter Notebook | English](https://img.shields.io/badge/Jupyter%20Notebook-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/103e40d82c83aec2ef483998e961f03cc7e22826/GaugeStudeBalanced/GaugeStudyGUI.ipynb)
3. [![Forecasting Report | English](https://img.shields.io/badge/GaugeStudy%20Report-English-yellowblue?logoColor=blue&labelColor=red)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/4e3ee63c691c9482f70fe836c43d6173f98cb53b/GaugeStudeBalanced/GaugeStudyReport.pdf) 
4. A. Meister , T. Sonar: "__Numerik__", 1st Ed. Springer-Spektrum (2019); S. Chapra, R. Canale: "__Numerical Methods for Engineers__", Mcgraw-Hill, 6th Edition (2010). 
5. J. Kilty, A. M. McAllister: "__Mathematical Modeling and Applied Calculus__", 1st Ed. Oxford University Press (2018).
6. U. Kockelkorn: "__Statistik für Anwender__", 1st Ed. Springer (2012), s. chapters 7 - 8.
7. Robert H. Shumway, David S. Stoffer: "__Time Series Analysis and Its Applications with R Examples__", Springer (2011).
8. Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor: "__An Introduction to Statistical Learning with Applications in Python__", Springer (2023).
9. Cornelis W. Oosterlee, Lech A. Grzelak: "__Mathematical Modeling and Computation in Finance with Exercises and Python and MATLAB Computer Codes__", World Scientific (2020).
10. Richard Szeliski: "__Computer Vision - Algorithms and Applications__", Springer (2022).
11. Anthony Scopatz, Kathryn D. Huff: "__Effective Computation in Physics - Field Guide to Research with Python__", O'Reilly Media (2015).
12. Alex Gezerlis: "__Numerical Methods in Physics with Python__", Cambridge University Press (2020).
13. Gary Hutson, Matt Jackson: "__Graph Data Modeling in Python. A practical guide__", Packt-Publishing (2023).
14. Hagen Kleinert: "__Path Integrals in Quantum Mechanics, Statistics, Polymer Physics, and Financial Markets__", 5th Edition, World Scientific Publishing Company (2009).
15. Peter Richmond, Jurgen Mimkes, Stefan Hutzler: "__Econophysics and Physical Economics__", Oxford University Press (2013).
16. A. Coryn , L. Bailer Jones: "__Practical Bayesian Inference A Primer for Physical Scientists__", Cambridge University Press (2017).
17. Avram Sidi: "__Practical Extrapolation Methods - Theory and Applications__", Cambridge university Press (2003).
18. Volker Ziemann: "__Physics and Finance__", Springer (2021).
19. Zhi-Hua Zhou: "__Ensemble methods, foundations and algorithms__", CRC Press (2012).
20. B. S. Everitt, et al.: "__Cluster analysis__", Wiley (2011).
21. Lior Rokach, Oded Maimon: "__Data Mining With Decision Trees - Theory and Applications__", World Scientific (2015).
22. Bernhard Schölkopf, Alexander J. Smola: "__Learning with kernels - support vector machines, regularization, optimization and beyond__", MIT Press (2009).
23. Johan A. K. Suykens: "__Regularization, Optimization, Kernels, and Support Vector Machines__", CRC Press (2014).
24. Sarah Depaoli: "__Bayesian Structural Equation Modeling__", Guilford Press (2021).
25. Rex B. Kline: "__Principles and Practice of Structural Equation Modeling__", Guilford Press (2023).
26. Ekaterina Kochmar: "__Getting Started with Natural Language Processing__", Manning (2022).
27. Jakub Langr, Vladimir Bok: "__GANs in Action__", Computer Vision Lead at Founders Factory (2019).
28. David Foster: "__Generative Deep Learning__", O'Reilly(2023).
29. Rowel Atienza: "__Advanced Deep Learning with Keras: Applying GANs and other new deep learning algorithms to the real world__", Packt Publishing (2018).
30. Josh Kalin: "__Generative Adversarial Networks Cookbook__", Packt Publishing (2018).  
31. Thomas Haslwanter: "__Hands-on Signal Analysis with Python: An Introduction__", Springer (2021).
32. Jose Unpingco: "__Python for Signal Processing__", Springer (2023).
33. R. K. Burdick, C. M. Borror, D. C. Montgomery: "__Design and Analysis of Gauge R&R Studies__", 1st Ed. SIAM (2005); 
S. H. Derakhshan , C. V. Deutsch: "__Numerical Integration of Bivariate Gaussian Distribution__", Paper 405, CCG Anual Report 13 (2011).
