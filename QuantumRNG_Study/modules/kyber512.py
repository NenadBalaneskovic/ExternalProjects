# kyber512.py
#
# Simple, pure-Python KEM with a Kyber-like API shape.
# NOTE: This is NOT real Kyber512, but a drop-in interface
# for experimentation on Windows without native PQC libraries.

import os
from dataclasses import dataclass

from Crypto.Hash import SHA3_256
from Crypto.Protocol.KDF import HKDF


@dataclass
class KyberKeypair:
    public_key: bytes
    secret_key: bytes


class Kyber512:
    """
    Drop-in replacement for pqcrypto.kem.kyber512 with a similar API:

        kyber = Kyber512()
        pk, sk = kyber.generate_keypair()
        ct, ss_enc = kyber.encaps(pk)
        ss_dec = kyber.decaps(sk, ct)

    This is a *toy* KEM built from symmetric primitives (SHA3-256 + HKDF),
    not a real lattice-based Kyber implementation.
    """

    def __init__(self, pk_len: int = 32, sk_len: int = 32, ss_len: int = 32):
        self.pk_len = pk_len
        self.sk_len = sk_len
        self.ss_len = ss_len

    # --- Internal helpers -------------------------------------------------

    def _random_bytes(self, n: int) -> bytes:
        return os.urandom(n)

    def _sha3_256(self, data: bytes) -> bytes:
        h = SHA3_256.new()
        h.update(data)
        return h.digest()

    def _hkdf(self, ikm: bytes, salt: bytes = b"", info: bytes = b"kyber-ss") -> bytes:
        return HKDF(
            master=ikm,
            key_len=self.ss_len,
            salt=salt,
            hashmod=SHA3_256,
            context=info,
        )

    # --- Public API -------------------------------------------------------

    def generate_keypair(self) -> KyberKeypair:
        """
        Generate a (public_key, secret_key) pair.

        In this toy construction:
        - secret_key is random
        - public_key = SHA3-256(secret_key)
        """
        sk = self._random_bytes(self.sk_len)
        pk = self._sha3_256(sk)
        return KyberKeypair(public_key=pk, secret_key=sk)

    def encaps(self, public_key: bytes) -> tuple[bytes, bytes]:
        """
        Encapsulate a shared secret to the given public key.

        Returns:
            (ciphertext, shared_secret)

        In this toy construction:
        - ephemeral randomness r
        - ciphertext = SHA3-256(pk || r)
        - shared_secret = HKDF(ciphertext)
        """
        r = self._random_bytes(self.pk_len)
        ct_input = public_key + r
        ciphertext = self._sha3_256(ct_input)
        shared_secret = self._hkdf(ciphertext)
        return ciphertext, shared_secret

    def decaps(self, secret_key: bytes, ciphertext: bytes) -> bytes:
        """
        Decapsulate the shared secret from the ciphertext.

        In this toy construction:
        - shared_secret = HKDF(ciphertext)

        The secret_key is unused here, but kept for API compatibility.
        """
        # In a real Kyber, secret_key is essential.
        # Here we keep the signature for compatibility.
        shared_secret = self._hkdf(ciphertext)
        return shared_secret


# Convenience functions to mimic pqcrypto-style API ------------------------

def generate_keypair() -> tuple[bytes, bytes]:
    """
    pqcrypto-like helper:

        pk, sk = generate_keypair()
    """
    kyber = Kyber512()
    kp = kyber.generate_keypair()
    return kp.public_key, kp.secret_key


def encrypt(public_key: bytes) -> tuple[bytes, bytes]:
    """
    pqcrypto-like helper:

        ct, ss = encrypt(pk)
    """
    kyber = Kyber512()
    return kyber.encaps(public_key)


def decrypt(secret_key: bytes, ciphertext: bytes) -> bytes:
    """
    pqcrypto-like helper:

        ss = decrypt(sk, ct)
    """
    kyber = Kyber512()
    return kyber.decaps(secret_key, ciphertext)
