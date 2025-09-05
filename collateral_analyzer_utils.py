import base64
import json
import io
import uuid
from openai import OpenAI
import os
from fastapi import FastAPI, UploadFile, File, Form
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
from typing import List
import traceback
import tempfile
import requests

client = OpenAI(
    api_key=os.getenv("openai_api_key"),
    organization=os.getenv("openai_organization"),
)

import json
import base64
from pdf2image import convert_from_path
from collateral_analyzer_db_utils import fetch_and_store_results
import os
import shutil
import time
from logger import api_logger as logger

class JSONLRequestGenerator:
    def __init__(self, file_paths, jsonl_file_name, prompts, campaign_id):
        self.file_paths = file_paths
        self.jsonl_file_name = jsonl_file_name
        self.prompts = prompts
        self.campaign_id = campaign_id
        self.temp_files_dir = os.path.join(os.getcwd(), "temp_files")
        os.makedirs(self.temp_files_dir, exist_ok=True)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def convert_pdf_to_images(self, pdf_path):
        temp_dir = os.path.join(self.temp_files_dir, f"pdf_images_{int(time.time())}")
        os.makedirs(temp_dir, exist_ok=True)
        images = convert_from_path(pdf_path, output_folder=temp_dir)
        image_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(temp_dir, f"page_{i + 1}.jpg")
            image.save(image_path, "JPEG")
            image_paths.append(image_path)
        return image_paths, temp_dir

    def generate(self):
        jsonl_file_path = os.path.join(self.temp_files_dir, self.jsonl_file_name)
        with open(jsonl_file_path, "w") as jsonl_file:
            for file_path in self.file_paths:
                file_id = str(uuid.uuid4())
                file_name = os.path.basename(file_path)
                if file_path.lower().endswith(".pdf"):
                    image_paths, temp_dir = self.convert_pdf_to_images(file_path)
                    base64_images = [self.encode_image(image_path) for image_path in image_paths]
                    # shutil.rmtree(temp_dir)
                else:
                    base64_images = [self.encode_image(file_path)]

                for prompt in self.prompts:
                    prompt_title = prompt["prompt_title"]
                    prompt_text = prompt["prompt"]
                    section = prompt["section_name"]
                    content = [{"type": "text", "text": prompt_text}] + [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        for base64_image in base64_images
                    ]

                    request = {
                        "custom_id": f"{self.campaign_id}_{file_id}_{section}_{file_name}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-4o-mini",
                            "messages": [{"role": "user", "content": content}],
                            "max_tokens": 300,
                        }
                    }
                    jsonl_file.write(json.dumps(request) + "\n")

        return jsonl_file_path

class BatchProcessing:
    def __init__(self, client):
        self.client = client

    def generate_analysis(self, file):
        """Generate analysis by submitting a batch job with the specified file."""
        batch_input_file_id = self._create_batch_input_file(file)
        batch_id = self._create_batch(batch_input_file_id)
        return batch_id.id

    def _create_batch_input_file(self, file):
        """Create a batch input file and return its ID."""
        with open(file, "rb") as f:
            batch_input_file = self.client.files.create(file=f, purpose="batch")
        return batch_input_file.id

    def _create_batch(self, input_file_id):
        """Create a batch job for processing."""
        batch = self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "Collateral Analysis job"},
        )
        return batch

    def check_batch_status(self, batch_id):
        """Check the status of a batch job."""
        batch_status = self.client.batches.retrieve(batch_id)
        return batch_status.status, batch_status.output_file_id

    def retrieve_batch_results(self, file_id):
        """Retrieve the results of a completed batch job."""
        file_response = self.client.files.content(file_id)
        responses = []
        try:
            json_data = json.loads(file_response.text)
            response = json_data["response"]["body"]["choices"][0]["message"]["content"]
            return response, json_data
        except json.JSONDecodeError:
            json_strings = file_response.text.strip().split("\n")

            # Parse each JSON string and store in a list
            parsed_data = [json.loads(js) for js in json_strings]
            responses = [item["response"]["body"]["choices"][0]["message"]["content"] for item in parsed_data]

            return responses, parsed_data

    def cancel_batch(self, batch_id):
        """Cancel a running batch job."""
        return self.client.batches.cancel(batch_id)
    
def submit_for_image_analysis(uploaded_files, selected_prompts, campaign_id):
    generator = JSONLRequestGenerator(uploaded_files, "image_analysis.jsonl", selected_prompts, campaign_id)
    if uploaded_files and selected_prompts:
            # Spinner for creating batch requests
            print("Creating batch requests...")
            try:
                # Generate messages and write to a JSONL file
                generator = JSONLRequestGenerator(uploaded_files, "collateral_analysis.jsonl", selected_prompts, campaign_id)
                jsonl_file_path = generator.generate()
            except Exception as e:
                traceback.print_exc()

            # Spinner for sending batch request
            logger.info("Sending batch request...")
            try:
                # Start batch processing and post to the backend
                batch_id = BatchProcessing(client).generate_analysis(jsonl_file_path)
                print(batch_id)
                url = f"http://127.0.0.1:8001/retrieve/{str(batch_id)}"
                response = requests.post(url=url)
                print(response.status_code)
                logger.info(f"Batch processing started with ID: {batch_id}")
            except Exception as e:
                logger.error(f"Error sending batch request: {e}")
                traceback.print_exc()
    else:
        logger.error("Please upload files and select at least one prompt before submitting.")
        
# def convert_pdf_to_base64_images(pdf_bytes):
#     """Converts a multi-page PDF into a list of base64-encoded images."""
#     images = convert_from_bytes(pdf_bytes)
#     encoded_images = []
    
#     for img in images:
#         buffer = io.BytesIO()
#         img.save(buffer, format="JPEG")
#         encoded_images.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
    
#     return encoded_images
async def convert_pdf_to_base64_images(pdf_file):
    """Converts a multi-page PDF (saved in temp dir) to base64-encoded images."""
    
    # Create a temporary directory to store the PDF
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_pdf_path = os.path.join(temp_dir, "temp.pdf")

        # Save the uploaded PDF file to the temp directory
        with open(temp_pdf_path, "wb") as f:
            f.write(await pdf_file.read())

        # Convert the PDF pages to images
        images = convert_from_path(temp_pdf_path)

        encoded_images = []
        for img in images:
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")  # Save as JPEG for efficiency
            encoded_images.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))  # Encode to Base64

    return encoded_images  # Returns a list of base64 images


# Function to Convert Image to Base64
def encode_image(image_bytes):
    """Encodes a single image file as base64."""
    return base64.b64encode(image_bytes.read()).decode("utf-8")

# Function to Submit Batch API Request
def submit_batch_request(requests):
    """Submit batch requests to OpenAI."""
    response = client.batches.create(requests=requests)
    return response["id"]

def retrieve_batch_results(batch_id):
    start_time = time.time()
    while True:
        batch_details = client.batches.retrieve(batch_id)
        print(batch_details)
        batch_status = batch_details.status
        print(batch_status)
        if batch_status == "completed":
            end_time = time.time()
            time_taken = end_time - start_time
            logger.info(f"Time taken to finish batch: {batch_id}, is {time_taken}")
            file_response = client.files.content(batch_details.output_file_id)
            response_text = file_response.text
            json_objects = response_text.strip().split("\n")
            response_dicts = [json.loads(obj) for obj in json_objects if obj.strip()]
            fetch_and_store_results(batch_id, response_dicts)
            # file_response = client.files.content(batch_details.output_file_id)
            # response_text = file_response.text
            # json_objects = response_text.strip().split("\n")
            # response_dicts = [json.loads(obj) for obj in json_objects if obj.strip()]
            logger.info("Files Processed Successfully")
            break
        elif batch_status == "in_progress":
            time.sleep(10)
            continue
        else:
            print("Failed")
            break