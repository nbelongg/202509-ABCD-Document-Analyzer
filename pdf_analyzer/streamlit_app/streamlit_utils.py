import streamlit as st
import uuid
from openai import OpenAI
from streamlit_app.db_utils import get_prompts, get_image_prompts, log_analysis_request
import requests
import os
from dotenv import load_dotenv
import json
from typing import Union
from llama_parse import LlamaParse
import PyPDF2
import nest_asyncio
from datetime import datetime
import traceback
import asyncio
from pypdf import PdfReader
from streamlit_app.prompts_manager import prompt_update_ui
from streamlit_app.batch_request_generator import JSONLRequestGenerator
from logger import streamlit_logger as logger
from pathlib import Path

nest_asyncio.apply()
load_dotenv(override=True)

llama_parser_api = os.getenv("llama_parser_api")
client = OpenAI(
    api_key=os.getenv("openai_api_key"),
    organization=os.getenv("openai_organization"),
)


def pdf_analyzer_ui():
    # Input field for batch processing status check
    batch_id = st.sidebar.text_input("Check batch processing Status")
    batch_id = batch_id.strip()

    # Button to trigger status check
    if st.sidebar.button("Check Status"):
        status, output_file_id = BatchProcessing(client).check_batch_status(str(batch_id))
        if status == "completed":
            st.sidebar.success("Successfully processed")
        elif status == "in_progress":
            st.sidebar.write("In progress")
        elif status == "cancelled":
            st.sidebar.error("Cancelled")
        elif status == "failed":
            st.sidebar.error("Failed to process")

    if st.session_state.get("email", "") == "nirat@belongg.net":
        # Show "Update Prompts" button only when not in prompt update UI
        if not st.session_state.get("show_prompt_update", False):
            if st.sidebar.button("Update Prompts"):
                st.session_state.show_prompt_update = True
                st.rerun()

        # Show prompt update UI if the button was clicked
        if st.session_state.get("show_prompt_update", False):
            prompt_update_ui(st)
            # Add a "Return to Home" button
            if st.sidebar.button("Return to Home"):
                st.session_state.show_prompt_update = False
                st.rerun()
        else:
            pdf_prompt_selection()
    else:
        pdf_prompt_selection()


def handle_user_inputs():
    # Dynamic title based on analysis type
    analysis_type = st.selectbox("Select Analysis Type", options=["PDF Analysis", "Image Analysis"], index=0)

    st.title(f"{analysis_type}")

    if analysis_type == "PDF Analysis":
        # PDF analysis form
        with st.form("pdf_analysis_form"):
            # Input for user email
            user_email = st.text_input("Enter your Email", value=st.session_state.get("email", ""))

            # File uploader for PDF files
            uploaded_files = st.file_uploader(
                "Upload a PDF file(max 100 pdfs)", type=["pdf"], accept_multiple_files=True
            )

            # Multiselect to choose prompts
            selected_prompts = st.multiselect(
                "Select prompts for analysis",
                options=st.session_state.get("prompt_options", []),
            )

            use_llama_parse = st.checkbox("Use Llama Parse for Parsing", value=True)

            # Add a submit button to the form
            submit_button = st.form_submit_button("Submit")

        if submit_button:
            st.session_state["user_email"] = user_email
            st.session_state["uploaded_files"] = uploaded_files
            st.session_state["selected_prompts"] = selected_prompts
            st.session_state["use_llama_parse"] = use_llama_parse

            prompts_dict = {}
            for prompt, prompt_title, prompt_content in st.session_state["prompts"]:
                if prompt_title in selected_prompts:
                    prompts_dict[prompt_title] = prompt_content

            st.session_state["prompts_dict"] = prompts_dict
            print(prompts_dict.keys())

            submit_for_analysis(uploaded_files, selected_prompts)

    else:
        image_prompts = st.session_state.get("image_prompts", {})
        options = [prompt["prompt_title"] for prompt in image_prompts]
        print(image_prompts)
        # Image analysis form
        with st.form("image_analysis_form"):
            # Input for user email
            user_email = st.text_input("Enter your Email", value=st.session_state.get("email", ""))

            # File uploader for images and PDFs
            uploaded_files = st.file_uploader(
                "Upload files (max 10 files)", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True
            )

            if uploaded_files and len(uploaded_files) > 10:
                st.error("Maximum 10 files allowed")
                return

            # Multiselect to choose prompts
            selected_prompt_titles = st.multiselect(
                "Select prompts for analysis",
                options=options,
            )

            # Add a submit button to the form
            submit_button = st.form_submit_button("Submit")

        if submit_button:
            st.session_state["user_email"] = user_email
            image_prompts = st.session_state["image_prompts"]
            selected_prompts = [prompt for prompt in image_prompts if prompt["prompt_title"] in selected_prompt_titles]
            print(f"\nSelected Prompts: {selected_prompts}\n")

            # Create temporary directory
            temp_dir = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True)

            try:
                # Save uploaded files to temp directory
                temp_files = []
                for uploaded_file in uploaded_files:
                    temp_path = temp_dir / uploaded_file.name
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    temp_files.append(str(temp_path))

                # Pass temp file paths to analysis function
                submit_for_image_analysis(temp_files, selected_prompts)

            finally:
                # Cleanup: Delete all temporary files
                for temp_file in temp_files:
                    Path(temp_file).unlink(missing_ok=True)
                temp_dir.rmdir()

    return


def pdf_prompt_selection():
    if "prompts" not in st.session_state:
        st.session_state["prompts"] = get_prompts()

    if "image_prompts" not in st.session_state:
        st.session_state["image_prompts"] = get_image_prompts()

    prompts = st.session_state["prompts"]
    image_prompts = st.session_state["image_prompts"]

    # Fetch prompts only if not already in session state
    if "prompts_dict" not in st.session_state:
        prompt_options = []
        prompts_dict = {}
        if prompts:
            for prompt, prompt_title, prompt_content in prompts:
                prompt_options.append(prompt_title)
                prompts_dict[prompt] = prompt_content
        else:
            print("No prompts and titles found or an error occurred.")
        st.session_state["prompts_dict"] = prompts_dict
        st.session_state["prompt_options"] = prompt_options

    handle_user_inputs()


def submit_for_analysis(uploaded_files, selected_prompts):
    if uploaded_files and selected_prompts:
        if "user_email" not in st.session_state or not st.session_state["user_email"]:
            st.error("Please provide your email address for Analysis.")
        else:
            with st.expander("Selected Prompts"):
                prompts_dict = st.session_state["prompts_dict"]
                for i, prompt_title in enumerate(prompts_dict):
                    prompt = prompts_dict[prompt_title]
                    st.text(f"Prompt {i+1}: {prompt_title}")
                    st.json({prompt_title: prompt})

            pdf_file_names = []
            pdf_file_content = []
            combined_contents = []
            system_messages = []
            user_messages = {}

            with st.spinner("Processing PDF files..."):
                try:
                    for uploaded_file in uploaded_files:
                        file_type = uploaded_file.type
                        file_name = uploaded_file.name
                        pdf_file_names.append(file_name)

                        # Extract file content
                        file_content = get_file_text(
                            uploaded_file, file_type, file_name, st.session_state.use_llama_parse
                        )
                        pdf_file_content.append(file_content)
                        user_messages[file_name] = file_content

                        # Prepare combined content for display
                        combined_contents.append(
                            {
                                "file_name": file_name,
                                "file_content": file_content,
                                "filesize": f"{uploaded_file.size / (1024 * 1024):.2f} MB",
                            }
                        )
                except Exception as e:
                    st.error(f"Error processing PDF files: {e}")
                    traceback.print_exc()

            # Display all file contents after processing
            with st.expander("All Pdf File Content"):
                for i, content in enumerate(combined_contents):
                    st.text(f"Pdf {i+1}: {content['file_name']}")
                    st.json(content)

            # Add selected prompts to system messages
            for prompt_title, prompt in st.session_state["prompts_dict"].items():
                system_messages.append(prompt)

            # Set model name and initialize the JSONL creator
            model_name = "gpt-4o"  # Model used for analysis
            jsonl_creator = JsonFileOperator(
                model_name=model_name,
                user_messages=user_messages,
                system_messages=system_messages,
                prompt_dict=st.session_state["prompts_dict"],
            )

            # Spinner for creating batch requests
            with st.spinner("Creating batch requests..."):
                try:
                    # Generate messages and write to a JSONL file
                    jsonl_creator.message_list = jsonl_creator.create_messages_from_roles()
                    uploaded_file_names = [uploaded_file.name for uploaded_file in uploaded_files]
                    jsonl_creator.message_list = jsonl_creator.create_message_list(uploaded_file_names)
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    request_file_path = jsonl_creator.write_jsonl_file(
                        filename=f"input_files/{st.session_state['user_email']}_input_{timestamp}.jsonl"
                    )

                    # Expander to show generated message list
                    with st.expander("Analysis requests"):
                        st.json(jsonl_creator.message_list)
                except Exception as e:
                    st.error(f"Error creating batch requests: {e}")
                    traceback.print_exc()

            # Spinner for sending batch request
            with st.spinner("Sending batch request..."):
                try:
                    # Start batch processing and post to the backend
                    batch_id = BatchProcessing(client).generate_analysis(request_file_path)
                    st.session_state["batch_id"] = batch_id
                    url = f"http://127.0.0.1:8000/retrieve/{str(batch_id)}"
                    response = requests.post(url=url, data={"email": str(st.session_state["user_email"])})
                    print(response.status_code)
                    st.success(f"Batch processing started with ID: {batch_id}")
                except Exception as e:
                    st.error(f"Error sending batch request: {e}")
                    traceback.print_exc()

            log_analysis_request(
                st.session_state["email"],
                st.session_state["user_email"],
                batch_id,
                pdf_file_names,
                pdf_file_content,
                st.session_state["prompts_dict"].keys(),
                model_name,
                st.session_state["use_llama_parse"],
            )
    else:
        st.error("Please upload files and select at least one prompt before submitting.")


def submit_for_image_analysis(uploaded_files, selected_prompts):
    generator = JSONLRequestGenerator(uploaded_files, "image_analysis.jsonl", selected_prompts)
    if uploaded_files and selected_prompts:
        if "user_email" not in st.session_state or not st.session_state["user_email"]:
            st.error("Please provide your email address for Analysis.")
        else:
            with st.expander("Selected Prompts"):
                for i, prompt in enumerate(selected_prompts):
                    st.text(f"Prompt {i+1}: {prompt["prompt_title"]}")
                    st.json({prompt["prompt_title"]: prompt["prompt"]})

            # Spinner for creating batch requests
            with st.spinner("Creating batch requests..."):
                try:
                    # Generate messages and write to a JSONL file
                    generator = JSONLRequestGenerator(uploaded_files, "image_analysis.jsonl", selected_prompts)
                    jsonl_file_path = generator.generate()
                except Exception as e:
                    st.error(f"Error creating batch requests: {e}")
                    traceback.print_exc()

            # Spinner for sending batch request
            with st.spinner("Sending batch request..."):
                try:
                    # Start batch processing and post to the backend
                    batch_id = BatchProcessing(client).generate_analysis(jsonl_file_path)
                    print(batch_id)
                    st.session_state["batch_id"] = batch_id
                    url = f"http://127.0.0.1:8001/retrieve/{str(batch_id)}"
                    response = requests.post(url=url, data={"email": str(st.session_state["user_email"])})
                    print(response.status_code)
                    st.success(f"Batch processing started with ID: {batch_id}")
                except Exception as e:
                    st.error(f"Error sending batch request: {e}")
                    traceback.print_exc()
    else:
        st.error("Please upload files and select at least one prompt before submitting.")


def read_pdf_to_plain_text(bytes_data):
    content = ""
    pdf_reader = PyPDF2.PdfReader(bytes_data)
    num_pages = len(pdf_reader.pages)
    for i in range(num_pages):
        page = pdf_reader.pages[i]
        content += page.extract_text()
    return content


def read_with_llama_parse(file_path, result_type="text"):
    """Reads file using LlamaParse and returns the parsed text based on the specified result type."""
    # logger.info(f"Reading file with LlamaParse: {file_path}")
    print(f"reading file with LlamaParse: {file_path}")
    parser = LlamaParse(api_key=os.getenv("LLAMA_CLOUD_API_KEY"), result_type=result_type)
    documents = parser.load_data(file_path)
    print(documents)
    doc_text = ""
    for doc in documents:
        doc_text = doc_text + doc.text
    return doc_text


def get_file_text(bytes_data, file_type, filename, use_llama_parse):
    # logger.info(f"fileType: {file_type}")
    # logger.info(f"Reading file with type: {file_type} and filename: {filename}")

    # Reset the buffer position to the start
    bytes_data.seek(0)
    file_content = bytes_data.getvalue()
    # file_path = save_uploaded_file(bytes_data, filename)
    print(f"Reading file with type: {file_type} and filename: {filename}")
    try:
        # Handle different file types
        if file_type == "application/pdf":
            content = (
                read_pdf_to_plain_text(bytes_data)
                if not use_llama_parse
                else extract_text_llama_parse(file_content, filename)
            )
            content = content.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        else:
            raise ValueError("Unsupported file type.")

        return content
    except Exception as e:
        print(f"Error reading file: {e}")
        st.error(f"Error reading file: {e}")
    finally:
        # If a temporary file was created for llama_parse, it's deleted
        if use_llama_parse and os.path.exists(filename):
            os.remove(filename)


def extract_text_llama_parse(uploaded_file, file_name):
    file_path = save_uploaded_file(uploaded_file, file_name)
    text = ""
    try:
        # Run LlamaParse with a 30-second timeout
        document = asyncio.run(
            asyncio.wait_for(LlamaParse(api_key=llama_parser_api, result_type="text").aload_data(file_path), timeout=30)
        )
        text = "".join(chunk.text for chunk in document)
    except asyncio.TimeoutError:
        logger.error(f"Timeout occurred while processing file: {file_name}")
        text = extract_text_pypdf(file_path)
    except Exception as e:
        logger.error(f"Error processing file with LlamaParse: {file_name}, Error: {e}")
        traceback.print_exc()
        text = extract_text_pypdf(file_path)
    finally:
        try:
            delete_uploaded_file(file_path)
        except Exception as e:
            logger.error(f"Error deleting file: {file_path}, Error: {e}")
    return text


def extract_text_pypdf(file_path):
    text = ""
    try:
        logger.info("Extracting text from PDF using PyPDF...")
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text()
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error extracting text from PDF: {e}")
        print(f"Error extracting text from PDF: {e}")
    return text


def save_uploaded_file(file_content: Union[bytes, bytearray], filename: str) -> str:
    try:
        file_name = str(uuid.uuid4()) + os.path.splitext(filename)[-1]
        directory = "input_files"
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, file_name)

        with open(file_path, "wb") as buffer:
            buffer.write(file_content)

        return file_path
    except Exception as e:
        print(f"Error saving file: {e}")
        return ""


def delete_uploaded_file(file_path: str) -> None:
    try:
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")
    except Exception as e:
        print(f"Error deleting file: {e}")


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
            metadata={"description": "PDF Analysis job"},
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


class JsonFileOperator:
    def __init__(
        self, model_name, user_messages, system_messages, prompt_dict, filename="batchinput.jsonl", max_tokens=4000
    ):
        self.model_name = model_name
        self.user_messages = user_messages
        self.system_messages = system_messages
        self.prompt_dict = prompt_dict
        self.filename = filename
        self.max_tokens = max_tokens
        self.message_list = []

    @staticmethod
    def generate_custom_id(system_message, pdf_name, prompt_dict):
        # Get the prompt title corresponding to the system message
        prompt_title = [title for title, message in prompt_dict.items() if message == system_message]
        prompt_title = prompt_title[0] if prompt_title else "unknown_prompt"
        return f"{prompt_title}_{pdf_name}"

    def create_messages_from_roles(self):
        messages = []
        for system_content in self.system_messages:
            for user_content in self.user_messages:
                message = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]
                messages.append(message)
        return messages

    def create_message_list(self, uploaded_file_names):
        message_list = []

        for system_message in self.system_messages:
            for uploaded_file_name, user_message in self.user_messages.items():
                custom_id = self.generate_custom_id(
                    system_message, uploaded_file_name, self.prompt_dict
                )  # Pass prompts_dict
                message_list.append(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.model_name,
                            "messages": [
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": user_message},
                            ],
                            "max_tokens": self.max_tokens,
                        },
                    }
                )

        return message_list

    def write_jsonl_file(self, filename):
        with open(filename, "w") as f:
            for item in self.message_list:
                f.write(json.dumps(item) + "\n")
        return filename

    @staticmethod
    def jsonl_to_dict(jsonl_string):
        dict_list = []
        for line in jsonl_string.strip().split("\n"):
            if line:  # Check if the line is not empty
                dict_list.append(json.loads(line))
        return dict_list
