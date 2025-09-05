import os
from dotenv import load_dotenv
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from io import BytesIO
from s3_bucket_utils import upload_fileobj_to_s3


load_dotenv(override=True)


def _get_key_and_iv(key, salt, iv_size, key_size):
    backend = default_backend()
    kdf = Scrypt(
        salt=salt,
        length=iv_size + key_size,
        n=2**14,
        r=8,
        p=1,
        backend=backend
    )
    derived = kdf.derive(key)
    return derived[:key_size], derived[key_size:]


def encrypt_file(file_content):
    secret_key = os.environ["ENCRYPTION_KEY"].encode("utf-8")
    salt = os.urandom(16)
    key_size = 32
    iv_size = 16
    key, iv = _get_key_and_iv(secret_key, salt, iv_size, key_size)

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(file_content) + padder.finalize()

    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    return salt + iv + encrypted_data


def decrypt_file(encrypted_file_content):
    secret_key = os.environ["ENCRYPTION_KEY"].encode("utf-8")
    salt = encrypted_file_content[:16]
    iv_size = 16
    key_size = 32
    key, iv = _get_key_and_iv(secret_key, salt, iv_size, key_size)

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_data = decryptor.update(encrypted_file_content[16 + iv_size:]) + decryptor.finalize()

    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    unpadded_data = unpadder.update(decrypted_data) + unpadder.finalize()
    return unpadded_data


if __name__ == "__main__":
    with open("./test_files/Test_pdf.pdf", "rb") as file:
        file_content = file.read()
        
    encrypted_file_content = encrypt_file(file_content)
    
    # try:
    #     file_name = "Test_pdf.pdf"
    #     s3_par_url = upload_fileobj_to_s3(BytesIO(file_content), file_name)
    #     print(s3_par_url)
    # except Exception as e:
    #     print(e)
    with open("./test_files/Test_pdf_encrypted.pdf", "wb") as file:
        file.write(encrypted_file_content)
    
    # Decryption
    with open("./test_files/Test_pdf_encrypted.pdf", "rb") as f:
        encrypted_file_content = f.read()
    decrypted_file_content = decrypt_file(encrypted_file_content)
    with open("./test_files/Test_pdf_decrypted.pdf", "wb") as file:
        file.write(decrypted_file_content)


# Encrypted File URL: https://abcd-bot-s3.s3.amazonaws.com/Test_pdf_encrypted.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAUSQEC34JD3N7I32R%2F20240627%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20240627T063950Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=ac4d521b8861308aeea38bd802a2c882f2d1ec98c2263f7e1de8a3b8b9360559

# Original File S3 URL: https://abcd-bot-s3.s3.amazonaws.com/Test_pdf.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAUSQEC34JD3N7I32R%2F20240627%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20240627T064125Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=41be6920e13b6042c7c68bb10f402d8d66eb10a6e3bda0ed694cf4c91aeade2a

# Encryption/Decryption Key: BKwMwhQieVGlM2Kf7mHA_f0IdpYAECKJ7UPw5EeFpIQ=