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
import os
import csv
from urllib.parse import urlparse
import requests
#from s3_bucket_utils import upload_fileobj_to_s3


load_dotenv(override=True)

secret_key = "BKwMwhQieVGlM2Kf7mHA_f0IdpYAECKJ7UPw5EeFpIQ="

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
    #secret_key = os.environ["ENCRYPTION_KEY"].encode("utf-8")
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
    #secret_key = os.environ["ENCRYPTION_KEY"].encode("utf-8")
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


def download_and_decrypt_files(urls):
    print("Starting download and decryption process...")
    
    # Create directories if they don't exist
    os.makedirs("encrypted_files", exist_ok=True)
    os.makedirs("decrypted_files", exist_ok=True)
    
    for i, url in enumerate(urls, 1):
        print(f"Processing file {i} of {len(urls)}...")
        
        # Extract filename from URL
        filename = os.path.basename(urlparse(url).path)
        
        # Download file
        response = requests.get(url)
        if response.status_code == 200:
            encrypted_content = response.content
            encrypted_path = os.path.join("encrypted_files", filename)
            
            # Save encrypted file
            with open(encrypted_path, "wb") as f:
                f.write(encrypted_content)
            print(f"Downloaded and saved encrypted file: {filename}")
            
            # Decrypt file
            decrypted_content = decrypt_file(encrypted_content)
            decrypted_path = os.path.join("decrypted_files", filename)
            
            # Save decrypted file
            with open(decrypted_path, "wb") as f:
                f.write(decrypted_content)
            print(f"Decrypted and saved file: {filename}")
        else:
            print(f"Failed to download file from URL: {url}")
    
    print("Download and decryption process completed.")


def process_csv(csv_path):
    print(f"Reading CSV file: {csv_path}")
    urls = []
    
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if 'encrypted_file_urls' in row:
                urls.append(row['encrypted_file_urls'])
    
    print(f"Found {len(urls)} URLs in the CSV file.")
    return urls


if __name__ == "__main__":
    
    # urls = [
    #     "https://abcd-bot-s3.s3.amazonaws.com/Test_pdf_encrypted.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAUSQEC34JD3N7I32R%2F20240627%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20240627T063950Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=ac4d521b8861308aeea38bd802a2c882f2d1ec98c2263f7e1de8a3b8b9360559"
    #     # We can add more URLs as needed
    # ]
    
    csv_path = "encrypted_file_urls.csv"
    
    urls = process_csv(csv_path)
    if urls:
        download_and_decrypt_files(urls)
    else:
        print("No URLs found in the CSV file. Exiting.")
    
    download_and_decrypt_files(urls)
    
    
# Encrypted File URL: https://abcd-bot-s3.s3.amazonaws.com/Test_pdf_encrypted.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAUSQEC34JD3N7I32R%2F20240627%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20240627T063950Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=ac4d521b8861308aeea38bd802a2c882f2d1ec98c2263f7e1de8a3b8b9360559