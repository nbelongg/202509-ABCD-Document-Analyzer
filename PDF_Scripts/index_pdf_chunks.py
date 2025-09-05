from langchain.embeddings import SentenceTransformerEmbeddings
import pinecone 
from langchain.vectorstores import Pinecone
import concurrent.futures
import os
import glob
import sys
from langchain.docstore.document import Document
import pickle
import shutil
from dotenv import load_dotenv
load_dotenv()
sys.path.append('.')
sys.path.append('..')


pinecone_api_key = os.getenv("pinecone_api_key")
pinecone_env = os.getenv("pinecone_env")

pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)




index_name = "open-source-index"

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def add_data_to_pinecone(pikl_file,processed_folder = "processed_pikl_files"):

    if not os.path.exists(processed_folder):
      os.makedirs(processed_folder)

    with open(pikl_file, 'rb') as file:
      pdf_chunks = pickle.load(file)
    
    filename=os.path.basename(pikl_file)

    print(f"Processed chunk file: {filename}")
    print("PDF content length")
    print(len(pdf_chunks))
    
    openai_index = Pinecone.from_documents(pdf_chunks, embeddings, index_name=index_name)
    
    # Move the file to the destination folder
    shutil.move(pikl_file, processed_folder)

def get_pdf_pikl_files(root_directory):
    pikl_files = []
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.lower().endswith('.pkl'):
                print(file)
                pikl_files.append(os.path.join(root, file))
    return pikl_files


if __name__ == "__main__":

    pikl_folder = 'pdf_chunks'

    if os.path.exists(pikl_folder):
        pikl_files = get_pdf_pikl_files(pikl_folder)
        num_workers = 20
        print(len(pikl_files))

        for pikl_file in pikl_files:
            add_data_to_pinecone(pikl_file)
        #with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            
        #    futures = [executor.submit(add_data_to_pinecone, pikl_file) for pikl_file in pikl_files]
            
        #    concurrent.futures.wait(futures)
