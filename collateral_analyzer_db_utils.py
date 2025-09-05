import mysql.connector
import json
import traceback
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv

client = OpenAI(
    api_key=os.getenv("openai_api_key"),
    organization=os.getenv("openai_organization"),
)
load_dotenv()

host = os.getenv("DB_HOST")
database = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")

LOG_FILE_PATH = "temp_files/batch_processing_log.json"

def write_log_to_json(log_data, log_file=LOG_FILE_PATH):
    """Writes log data to a JSON file."""
    
    # Check if the file exists; if not, create an empty list
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as file:
            try:
                logs = json.load(file)  # Load existing logs
            except json.JSONDecodeError:
                logs = []  # If file is empty or invalid, initialize as an empty list
    else:
        logs = []  # If file does not exist, start with an empty list

    # Append the new log entry
    logs.append(log_data)

    # Write back to JSON file
    with open(log_file, "w", encoding="utf-8") as file:
        json.dump(logs, file, indent=4, ensure_ascii=False)

    print(f"Log entry saved successfully to {log_file}")
    
# Function to Fetch and Store Results
def fetch_and_store_results(batch_id, response_dicts):
    """Retrieve batch processing results and store in DB."""
    # conn = mysql.connector.connect(
    #         host=host,
    #         database=database,
    #         user=user,
    #         password=password
    #     )
    # cur = conn.cursor()

    grouped_responses = {}
    for response in response_dicts:
        print(response)
        custom_id = response["custom_id"]
        ids = custom_id.split("_")
        print(ids)
        file_id = ids[1]
        section_name = ids[2]
        campaign_id = ids[0]
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
            "file_id": file_id,
            "section_name": section_name,
            "status": "completed",
            "campaign_id": campaign_id,
            "output": content,
            "input_tokens": response["response"]["body"]["usage"]["prompt_tokens"],
            "output_tokens": response["response"]["body"]["usage"]["completion_tokens"],
            "total_tokens": response["response"]["body"]["usage"]["total_tokens"],
            "input": response["response"]["body"]["choices"][0]["message"]["content"],
        }

        write_log_to_json(log_data)
        # Store in PostgreSQL
    #     query = """
    #     INSERT INTO collateral_analysis (file_id, campaign_id, section_name, content, status, created_at)
    #     VALUES (%s, %s, %s, %s, 'completed', NOW());
    #     """
    #     cur.execute(query, (file_id, campaign_id, section_name, json.dumps(content)))

    # conn.commit()
    # cur.close()
    # conn.close()