import time
import os
from dotenv import load_dotenv
from openai import OpenAI
from app.email_service import EmailService
from datetime import datetime
import pandas as pd
import json
from openai import OpenAI
import pandas as pd
from app.api_db_utils import log_analysis_request
from logger import api_logger as logger


load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("openai_api_key"),
    organization=os.getenv("openai_organization"),
)


def process_file(content):
    if content is None:
        return None
    # Define the directory to store processed files
    processed_dir = "processed_files"
    os.makedirs(processed_dir, exist_ok=True)

    # Create a unique filename with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    processed_file_path = f"{processed_dir}/processed_{timestamp}.txt"
    logger.info(processed_file_path)

    # Save the processed content to the new file
    with open(processed_file_path, "wb") as f:
        f.write(content)  # Replace with actual processing logic

    return processed_file_path


def save_responses_to_csv(response_dicts, batch_id, output_folder="processed_files"):
    # Create a dictionary to hold responses grouped by prompt prefix
    grouped_responses = {}

    # Group responses by prompt prefix
    for response in response_dicts:
        custom_id = response["custom_id"]
        prefix, file_name = custom_id.split("_", 1)  # Extract the prefix and file name
        prefix = f"{prefix}"
        if prefix not in grouped_responses:
            grouped_responses[prefix] = {}

        # Extract the content of the response
        content = response["response"]["body"]["choices"][0]["message"]["content"]
        grouped_responses[prefix][file_name] = content

        # Log the response in the database
        log_data = {
            "batch_id": batch_id,
            "filename": file_name,
            "prefix": prefix,
            "input_tokens": response["response"]["body"]["usage"]["prompt_tokens"],
            "output_tokens": response["response"]["body"]["usage"]["completion_tokens"],
            "total_tokens": response["response"]["body"]["usage"]["total_tokens"],
            "input": response["response"]["body"]["choices"][0]["message"]["content"],
        }
        log_analysis_request(log_data)

    # Create a DataFrame from the grouped responses
    df = pd.DataFrame.from_dict(grouped_responses, orient="index").transpose()

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define the CSV file path
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    csv_file_path = os.path.join(output_folder, f"pdf_analysis_{timestamp}.csv")

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=True, index_label="File Name")

    return csv_file_path


def retrieve_batch_results(batch_id, email):
    start_time = time.time()
    while True:
        batch_details = client.batches.retrieve(batch_id)
        batch_status = batch_details.status

        if batch_status == "completed":
            end_time = time.time()
            time_taken = end_time - start_time
            logger.info(f"Time taken to finish batch: {batch_id}, is {time_taken}")
            file_response = client.files.content(batch_details.output_file_id)
            response_text = file_response.text
            json_objects = response_text.strip().split("\n")
            response_dicts = [json.loads(obj) for obj in json_objects if obj.strip()]
            processed_file_path = save_responses_to_csv(response_dicts, batch_id)
            logger.info("Sending email notification to {}".format(email))
            send_email_notification(recipient_email=email, attachments=processed_file_path)
            logger.info("Email notification sent successfully")
            break
        elif batch_status == "in_progress":
            time.sleep(10)
            continue
        else:
            handle_failure(batch_id=batch_id, recipient_email=email, status=batch_status)
            break


def send_email_notification(recipient_email, attachments=None):
    email_service = EmailService()
    email_service.send_email(recipient_email, attachments)


def handle_failure(batch_id, recipient_email, status):
    logger.error(f"Batch {batch_id} failed with status: {status}")
    try:
        email_service = EmailService()
        email_service.send_failure_notification(recipient_email, batch_id, status)
        logger.info(f"Failure notification sent for batch {batch_id}")
    except Exception as e:
        logger.error(f"Failed to send failure notification for batch {batch_id}: {e}")
