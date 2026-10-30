from dagster import job
from dagster_pipelines.ops import encrypt_op, decrypt_op, provide_key, provide_text, provide_rsa_key_path

@job
def encryption_decryption_job():
    key = provide_key()
    text = provide_text()
    rsa_path = provide_rsa_key_path()
    encrypted = encrypt_op(text=text, vkey=key)
    decrypted = decrypt_op(encrypted_data=encrypted, vkey=key, rsa_key_path=rsa_path)
