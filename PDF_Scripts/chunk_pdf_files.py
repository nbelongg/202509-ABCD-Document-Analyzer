import os
import zipfile
import openai
import pinecone
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import concurrent.futures
import glob
from pdfminer.high_level import extract_text
from langchain.docstore.document import Document
import pickle
import shutil


def is_zipfile(file_path):
    try:
        with zipfile.ZipFile(file_path) as zf:
            return True
    except zipfile.BadZipFile:
        return False

def unzip_folder(zip_path, output_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)


def split_docs(documents, chunk_size=4000, chunk_overlap=300):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_text(documents)
    return docs

def process_pdf(pdf_path,pdf_chunk_dir="pdf_chunks",processed_pdfs_folder = "processed_pdfs"):
    
    metadata=[]
    docs=[]
    docObjs=[]
    content = extract_text(pdf_path)
    

    if not os.path.exists(processed_pdfs_folder):
      os.makedirs(processed_pdfs_folder)

    filename=os.path.basename(pdf_path)
    
    temp={'filename':filename}
    temp['21June1000Files'] = True
    
    chunks = split_docs(content)
    num = len(chunks)
    data = [temp for _ in range(num)]
    metadata.extend(data)
    docs.extend(chunks)
    

    print(f"Processed: {filename}")
    print("PDF content length")
    print(len(content))
    print("MetaData:")
    print(len(metadata))
    print("Docs:")
    print(len(docs))

    for j in range(0,num):
      doc_data = docs[j]
      doc_info = metadata[j]
      doc_obj=Document(page_content=doc_data,metadata=doc_info)
      docObjs.append(doc_obj)
    
    print(docObjs)
    print(len(docObjs))
    
    
    if not os.path.exists(pdf_chunk_dir):
      os.makedirs(pdf_chunk_dir)

    picklFile=os.path.join(pdf_chunk_dir, f"{filename}.pkl")
    print(picklFile)
    with open(picklFile, 'wb') as file:
      pickle.dump(docObjs, file)

    # Move the file to the destination folder
    shutil.move(pdf_path, processed_pdfs_folder)

def get_pdf_files(root_directory):
    pdf_files = []
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files


if __name__ == "__main__":
    
    zip_file = "ML_Model_2106_1.zip"
    if is_zipfile(zip_file):
    
        output_folder = 'original_pdfs'
        unzip_folder(zip_file, output_folder)
        
        pdf_files = get_pdf_files(output_folder)
        num_workers = 5

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_pdf, pdf_file) for pdf_file in pdf_files]
            concurrent.futures.wait(futures)
        
    else:
        print("The file is not a zip file.")
