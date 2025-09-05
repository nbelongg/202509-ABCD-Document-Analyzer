import boto3
import os
from dotenv import load_dotenv
from botocore.client import Config


load_dotenv(override=True)


aws_s3_bucket_access_key_id = os.getenv("belongg_aws_s3_bucket_access_key_id")
aws_s3_bucket_secret_access_key = os.getenv("belongg_aws_s3_bucket_secret_access_key") 
s3_bucket_name = os.getenv("belongg_s3_bucket_name")
aws_region = os.getenv("belongg_aws_region")


def convert_showcase_to_s3_img_filename(title):
    valid_filename = title.replace(" ", "_")
    valid_filename = valid_filename.replace(":", "").replace(";", "")
    valid_filename += ".jpg"
    return valid_filename


def upload_fileobj_to_s3(fileobj, object_name):
    par_url=""
    try:
        if object_exists(object_name):
            par_url = get_file_par_url(object_name)        
        else:
            s3_client = boto3.client(
                's3',
                region_name=aws_region,
                aws_access_key_id=aws_s3_bucket_access_key_id,
                aws_secret_access_key=aws_s3_bucket_secret_access_key,
                config=Config(signature_version='s3v4')
            )
            s3_client.upload_fileobj(fileobj,s3_bucket_name,object_name)
            par_url = get_file_par_url(object_name)
    except Exception as e:
        print(f"Upload failed: {e}")
    return par_url
    

def get_file_par_url(file_key):
    presigned_url = ""
    try:
        if not object_exists(file_key):
            return None
        s3_client = boto3.client(
            's3',
            region_name=aws_region,
            aws_access_key_id=aws_s3_bucket_access_key_id,
            aws_secret_access_key=aws_s3_bucket_secret_access_key,
            config=Config(signature_version='s3v4')
        )
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': s3_bucket_name,
            'Key': file_key},
            ExpiresIn=604800
        )
    except Exception as e:
        print(f"Error generating presigned URL: {e}")
    return  presigned_url


def object_exists(object_key):
    s3_client = boto3.client(
        's3',
        region_name=aws_region,
        aws_access_key_id=aws_s3_bucket_access_key_id,
        aws_secret_access_key=aws_s3_bucket_secret_access_key,
        config=Config(signature_version='s3v4')
    )
    try:
        s3_client.head_object(Bucket=s3_bucket_name, Key=object_key)
        return True
    except Exception as e:
        print(f"Error while checking if file exists: {str(e)}")
    return False