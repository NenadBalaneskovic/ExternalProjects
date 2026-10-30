"""
post_quantum_crypto.py
----------------------

Post-quantum encryption layer for fractional–fuzzy QKD.

Depends on:
    - Crypto (PyCryptodome)
    - pqcrypto (Kyber)
    - numpy
    - qec_adaptive.py (for final key K)
    - qrng.py (for entropy tests)
    - fractional_dynamics.py (indirectly)

Features:
    - HKDF-SHA3-256 key derivation
    - AES-256-GCM authenticated encryption/decryption
    - Kyber hybrid encryption:
        * Kyber keypair generation
        * Wrap Kyber secret key using AES-256 derived from fractional-QKD
        * Unwrap and verify correctness
"""

import os
import numpy as np

from Crypto.Protocol.KDF import HKDF
from Crypto.Hash import SHA3_256
from Crypto.Cipher import AES

import pqcrypto.kem.kyber512 as kyber


# ---------------------------------------------------------------------------
# HKDF Key Derivation
# ---------------------------------------------------------------------------

def derive_key(K_bits, key_len=32):
    """
    Derive a cryptographic key from bitstring using HKDF-SHA3-256.

    Parameters:
        K_bits  → numpy array of bits (0/1)
        key_len → output key length in bytes (default 32 for AES-256)

    Returns:
        key (bytes)
    """
    bitstring = "".join(str(b) for b in K_bits).encode()
    return HKDF(master=bitstring,
                key_len=key_len,
                salt=b"fractional-qkd",
                hashmod=SHA3_256)


# ---------------------------------------------------------------------------
# AES-256-GCM Encryption / Decryption
# ---------------------------------------------------------------------------

def aes_encrypt(key: bytes, plaintext: str):
    """
    AES-256-GCM authenticated encryption.

    Returns:
        nonce, ciphertext, tag
    """
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode())
    return cipher.nonce, ciphertext, tag


def aes_decrypt(key: bytes, nonce: bytes, ciphertext: bytes, tag: bytes):
    """
    AES-256-GCM authenticated decryption.

    Returns:
        plaintext (str)
    """
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag).decode()


# ---------------------------------------------------------------------------
# Kyber Hybrid Encryption
# ---------------------------------------------------------------------------

def kyber_generate_keypair():
    """
    Generate Kyber512 keypair.
    Returns:
        pk, sk
    """
    return kyber.generate_keypair()


def kyber_wrap_secret(aes_key: bytes, sk: bytes):
    """
    Wrap Kyber secret key using AES-256-GCM.

    Returns:
        nonce, ciphertext, tag
    """
    return aes_encrypt(aes_key, sk.hex())


def kyber_unwrap_secret(aes_key: bytes, nonce: bytes, ct: bytes, tag: bytes):
    """
    Unwrap Kyber secret key using AES-256-GCM.

    Returns:
        sk (bytes)
    """
    sk_hex = aes_decrypt(aes_key, nonce, ct, tag)
    return bytes.fromhex(sk_hex)


# ---------------------------------------------------------------------------
# Full Hybrid Demo
# ---------------------------------------------------------------------------

def demo_hybrid_encryption(K_bits):
    """
    Demonstrates full hybrid encryption:

        1. Derive AES-256 key from fractional-QKD bits
        2. Generate Kyber keypair
        3. Wrap Kyber secret key using AES-256
        4. Unwrap and verify correctness

    Returns:
        dict with keys:
            aes_key, pk, sk, sk_recovered, success
    """

    # Step 1: derive AES-256 key
    aes_key = derive_key(K_bits, key_len=32)

    # Step 2: Kyber keypair
    pk, sk = kyber_generate_keypair()

    # Step 3: wrap secret key
    nonce, ct, tag = kyber_wrap_secret(aes_key, sk)

    # Step 4: unwrap
    sk_recovered = kyber_unwrap_secret(aes_key, nonce, ct, tag)

    return {
        "aes_key": aes_key,
        "pk": pk,
        "sk": sk,
        "sk_recovered": sk_recovered,
        "success": (sk_recovered == sk)
    }
