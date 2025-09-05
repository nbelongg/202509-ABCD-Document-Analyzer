import json
import base64
from pdf2image import convert_from_path
import os
import shutil
import time


class JSONLRequestGenerator:
    def __init__(self, file_paths, jsonl_file_name, prompts):
        self.file_paths = file_paths
        self.jsonl_file_name = jsonl_file_name
        self.prompts = prompts
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
                file_name = os.path.basename(file_path)
                if file_path.lower().endswith(".pdf"):
                    image_paths, temp_dir = self.convert_pdf_to_images(file_path)
                    base64_images = [self.encode_image(image_path) for image_path in image_paths]
                    shutil.rmtree(temp_dir)
                else:
                    base64_images = [self.encode_image(file_path)]

                for prompt in self.prompts:
                    prompt_title = prompt["prompt_title"]
                    prompt_text = prompt["prompt"]
                    content = [{"type": "text", "text": prompt_text}] + [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        for base64_image in base64_images
                    ]

                    request = {
                        "custom_id": f"{prompt_title}_{file_name}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-4o-mini",
                            "messages": [{"role": "user", "content": content}],
                            "max_tokens": 300,
                        },
                    }
                    jsonl_file.write(json.dumps(request) + "\n")

        return jsonl_file_path
