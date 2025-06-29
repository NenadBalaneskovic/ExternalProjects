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
    #context.log.info(f"Private key size: {private_key.key_size // 8} bytes")
    return private_key.decrypt(
        encrypted_key,
        asympadding.OAEP(
            mgf=asympadding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
def decrypt_main(ciphertext: bytes, iv: bytes, enc_key: bytes, priv_key_path: str, vkey: str) -> str:
    # Step 1: Decrypt AES key with RSA
    aes_key = rsa_decrypt_key(enc_key, priv_key_path)

    # Step 2: Decrypt AES-encrypted Vigenère text
    vigenere_encrypted = aes_decrypt(ciphertext, aes_key, iv).decode()

    # Step 3: Decrypt Vigenère to get plaintext
    return vigenere_decrypt(vigenere_encrypted, vkey)


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
