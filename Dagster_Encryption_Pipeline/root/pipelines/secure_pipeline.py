from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding, serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding as asympadding
from cryptography.hazmat.backends import default_backend
import os
from itertools import cycle

# Step 1: Vigenère Encryption
def vigenere_encrypt(plaintext: str, key: str) -> str:
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    enc = []
    for c, k in zip(plaintext.lower(), cycle(key.lower())):
        if c in alphabet:
            shifted = (ord(c) - ord('a') + ord(k) - ord('a')) % 26
            enc.append(chr(ord('a') + shifted))
        else:
            enc.append(c)
    return ''.join(enc)

# Step 2: AES Encryption
def aes_encrypt(data: bytes, key: bytes) -> tuple[bytes, bytes]:
    iv = os.urandom(16)
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(data) + padder.finalize()
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ct = encryptor.update(padded_data) + encryptor.finalize()
    return ct, iv

# Step 3: RSA Encryption
def rsa_encrypt_key(aes_key: bytes, public_key) -> bytes:
    return public_key.encrypt(
        aes_key,
        asympadding.OAEP(
            mgf=asympadding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )

# Main function used by your Dagster op
def encrypt_main(text: str, vkey: str) -> dict:
    # Step 1: Vigenère encryption
    v_encrypted = vigenere_encrypt(text, vkey)

    # Step 2: AES encryption
    aes_key = os.urandom(32)  # AES-256
    ciphertext, iv = aes_encrypt(v_encrypted.encode(), aes_key)

    # Step 3: Load RSA public key and encrypt AES key
    with open("rsa/public.pem", "rb") as f:
        public_key = serialization.load_pem_public_key(f.read())

    encrypted_key = rsa_encrypt_key(aes_key, public_key)

    # Return encrypted bundle
    return {
        "ciphertext": ciphertext,
        "iv": iv,
        "enc_key": encrypted_key
    }



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