
import pinecone 
import concurrent.futures
import os
import glob
import sys
import pickle
import shutil
from dotenv import load_dotenv
load_dotenv()
sys.path.append('.')
sys.path.append('..')
import csv
import ast
from sentence_transformers import SentenceTransformer,util
model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone_api_key = os.getenv("pinecone_api_key")
pinecone_env = os.getenv("pinecone_env")

pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)


index_name = "open-source-meaningful"
index = pinecone.Index(index_name)



if __name__ == "__main__":

    file_path = "../Data/set_4_pred.csv"
    csv_file_path = "set_4_pred.csv_indexed.csv"
    
    totvect = index.describe_index_stats()['total_vector_count']
    
    
    
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader, None)
            
            i=0
            data_list = []
            for row in csv_reader:
                
                try:
                    validity=int(row[2])
                    if validity==1:
                        chunk=row[0]
                        metdata=ast.literal_eval(row[1])
                        metdata["text"]=chunk
                        metdata["valid_chunk"]=True
                        data_list.append((
                            str(i+totvect),
                            model.encode(chunk).tolist(),
                            metdata
                        ))
                        if len(data_list) == 5:    
                            print(str(i+totvect))
                            index.upsert(vectors=data_list)
                            
                            for row in data_list:
                                idx=row[0]
                                chunk=row[2]["text"]
                                filename=row[2]["filename"]
                                validity=row[2]["valid_chunk"]
                                writer.writerow([idx,chunk,filename,validity])
                            
                            data_list = []
                        i=i+1
            
                
                except ValueError as e:
                    print("Error:", e)
            
            if data_list != []:
                print(str(i+totvect))
                index.upsert(vectors=data_list)
                for row in data_list:
                    idx=row[0]
                    chunk=row[2]["text"]
                    filename=row[2]["filename"]
                    validity=row[2]["valid_chunk"]
                    writer.writerow([idx,chunk,filename,validity])
                data_list = []   

   