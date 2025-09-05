from pdfminer.high_level import extract_text
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse
import os, uuid
from typing import Union
from pypdf import PdfReader
import io
import nest_asyncio
import asyncio
from concurrent.futures import TimeoutError
import traceback
from logger import streamlit_logger as logger

nest_asyncio.apply()

llama_parser_api = os.environ["LLAMA_CLOUD_API_KEY"]


def get_pdf_chunks_list(uploaded_file, chunk_size):
    text = extract_text_from_pdf(uploaded_file)
    # print(text)
    # print(len(text))
    # return split_text_into_chunks(text, chunk_size)
    return split_docs(text, chunk_size)


def extract_text_from_pdf(uploaded_file):
    with io.BytesIO() as buffer:
        buffer.write(uploaded_file.read())
        buffer.seek(0)
        return extract_text(buffer)
    
    
def split_docs(text, chunk_size, chunk_overlap=500):
    #print("size od chunk is ",chunk_size)
    #print("*"*50)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_text(text)
    #print("length of docs is ",len(docs))
    return docs
    

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
   

def extract_text_llama_parse_old(uploaded_file, file_name):
   file_path = save_uploaded_file(uploaded_file, file_name)
   document = LlamaParse(api_key=llama_parser_api, result_type="text").load_data(file_path)
   delete_uploaded_file(file_path)
   text = ""
   for chunk in document:
       text += chunk.text
   return text


def extract_text_llama_parse(uploaded_file, file_name):
    logger.info("Extracting text from PDF using LlamaParse...")
    file_path = save_uploaded_file(uploaded_file, file_name)
    try:
        # Run LlamaParse with a 10-second timeout
        logger.info("Running LlamaParse with a 30-second timeout...")
        document = asyncio.run(asyncio.wait_for(
            LlamaParse(api_key=llama_parser_api, result_type="text").aload_data(file_path),
            timeout=30
        ))
        text = "".join(chunk.text for chunk in document)
    except Exception as e:
        # Fallback to extract_text_from_pdf if LlamaParse times out
        logger.info(f"LlamaParse timed out: {e}")
        text = extract_text_pypdf(file_path)
    finally:
        delete_uploaded_file(file_path)
    return text


async def extract_text_llama_parse_async(uploaded_file, file_name):
   file_path = save_uploaded_file(uploaded_file, file_name)
   document = await LlamaParse(api_key=llama_parser_api, result_type="text").aload_data(file_path)
   delete_uploaded_file(file_path)
   return document[0].text


def save_uploaded_file(file_content: Union[bytes, bytearray], filename: str) -> str:
    try:
        file_name = str(uuid.uuid4()) + os.path.splitext(filename)[-1]
        directory = "temp_files"
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




