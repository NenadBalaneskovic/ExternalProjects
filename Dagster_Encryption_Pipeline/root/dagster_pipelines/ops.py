from dagster import op, job
from pipelines import secure_pipeline, decrypt_pipeline

@op
def provide_rsa_key_path() -> str:
    return "rsa/private.pem"

@op
def provide_text() -> str:
    return "Top secret message"

@op
def provide_key() -> str:
    return "lemon"

@op
def encrypt_op(text: str, vkey: str):
    return secure_pipeline.encrypt_main(text, vkey)

@op
def decrypt_op(context, encrypted_data: dict, vkey: str, rsa_key_path: str):
    enc_key = encrypted_data["enc_key"]
    context.log.info(f"Encrypted key length: {len(enc_key)} bytes")

    return decrypt_pipeline.decrypt_main(
        ciphertext=encrypted_data["ciphertext"],
        iv=encrypted_data["iv"],
        enc_key=enc_key,
        priv_key_path=rsa_key_path,
        vkey=vkey
    )


