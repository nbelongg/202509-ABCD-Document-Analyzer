import sys
import os
from dotenv import load_dotenv
sys.path.append('.')
sys.path.append('..')
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langsmith import traceable
import warnings
from logger import api_logger as logger
from dotenv import load_dotenv

load_dotenv(override=True)

warnings.filterwarnings("ignore")

openai_organization = os.getenv("openai_organization")
openai_api_key = os.getenv("openai_api_key")

pinecone_api_key = os.getenv("pinecone_api_key")
belongg_api_key = "7f7cee85-3509-46fa-8d2a-7917d16cbeee"
pinecone_env = os.getenv("pinecone_env")
load_dotenv(override=True)

warnings.filterwarnings("ignore")

openai_organization = os.getenv("openai_organization")
openai_api_key = os.getenv("openai_api_key")

pinecone_api_key = os.getenv("pinecone_api_key")
belongg_api_key = "7f7cee85-3509-46fa-8d2a-7917d16cbeee"
pinecone_env = os.getenv("pinecone_env")

logger.info("Connecting to Pinecone...")
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
logger.info("Connected to Pinecone.")


def find_match(input, top_k, filter=None):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    input_em = model.encode(input).tolist()
    index = pc.Index('open-source-meaningful')
    
    result = index.query(vector=input_em, 
                         filter=filter,
                         top_k=top_k, 
                         includeMetadata=True)
    
    matches = result['matches']
    
    result_dict = {}
    result_dict['all_context'] = ''
    for i in range(min(top_k, len(matches))):
        pdf_name=""
        
        if 'filename' in matches[i]['metadata']:
            pdf_name=matches[i]['metadata']['filename']
        elif 'PDF Title' in matches[i]['metadata']:
            pdf_name=matches[i]['metadata']['PDF Title']

        if pdf_name:
            meta = os.path.basename(str(pdf_name))
            context = matches[i]['metadata']['text']
            result_dict[f'meta_{i+1}'] = meta
            result_dict[f'context_{i+1}'] = context
            result_dict['all_context'] += context + "\n"

    return result_dict


@traceable(tags=["pinecone-similarity-search"])
def find_match_retriever(input, top_k, filter=None):
    index_name = "open-source-meaningful"
    
    logger.info("Loading Sentence Transformer model...")
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    logger.info("Connecting to Pinecone...")
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embedding_function,
        pinecone_api_key=pinecone_api_key,
    )
    logger.info("Connected to Pinecone.")
    
    logger.info(f"Finding {top_k} matches in Pinecone...")
    result = vectorstore.similarity_search(input, k=top_k, filter=filter)
    logger.info("Matches found successfully.")
    
    result_dict = {}
    result_dict['all_context'] = ''
    for i, res in enumerate(result):
        metadata = res.metadata
        if 'filename' in metadata:
            pdf_name=metadata['filename']
        elif 'PDF Title' in metadata:
            pdf_name=metadata['PDF Title']
        
        meta = os.path.basename(str(pdf_name))
        result_dict[f'meta_{i+1}'] = meta
        result_dict[f'context_{i+1}'] = res.page_content
        result_dict['all_context'] += res.page_content + "\n\n"
        
    return result_dict


def extract_unique_chunks_langchain(query, top_k, multiplier=3, filter=None):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(index_name="open-source-meaningful", embedding=embeddings, pinecone_api_key=pinecone_api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k * multiplier, "filter": filter})
    docs = retriever.invoke(query)
    
    used = {}
    
    result_dict = {'all_context': ''}
    
    for i, doc in enumerate(docs):
        pdf_name = doc.metadata.get('PDF Title') or doc.metadata.get('filename')
        pdf_name = os.path.basename(pdf_name) if pdf_name else None
        
        if pdf_name is not None and pdf_name not in used:
            relevant_chunk = {
                'meta': pdf_name,
                'url': doc.metadata.get('URL', ''),
                'publication_year': doc.metadata.get('Publication Year', ''),
                'reference': doc.metadata.get('Reference (in APA 7 format)', ''),
                'country': doc.metadata.get('Country', ''),
                'author': doc.metadata.get('Author', ''),
                'chunk_id': doc.metadata.get('chunk_id', ''),
                'context': doc.page_content,
                'metadata': doc.metadata
            }
            
            used[pdf_name] = 1 
            
            result_dict['all_context'] += relevant_chunk['context'] + "\n"
            
            for key, value in relevant_chunk.items():
                result_dict[f'{key}_{len(used)}'] = value

            if len(used) == top_k:
                break

    return result_dict


def extract_unique_showcase_chunks(input, top_k, filter=None):
    # Extract the relevant chunks from the results based on top_k * multiplier.
    pc_belongg = Pinecone(api_key=belongg_api_key)
    multiplier = 3
    model = SentenceTransformer('all-MiniLM-L6-v2')
    input_em = model.encode(input).tolist()
    index = pc_belongg.Index("open-source-v2")
    
    temp_filter = {"Showcase": {"$eq": True}}
    
    if filter:
        temp_filter.update(filter)

    result_dict = index.query(vector=input_em, 
                         filter=temp_filter,
                         top_k=top_k, 
                         includeMetadata=True)
    
    # Initialize a dictionary to keep track of used PDFs.
    used = {}
    
    # Initialize a list to store relevant chunks.
    relevant_chunks = []
    matches = result_dict["matches"]
    
    # Iterate over the results to find unique chunks.
    for i in range(0, len(matches)):
        match_metadata=matches[i]['metadata']
        pdf_url = match_metadata.get('URL', None)  # Get the PDF name for the current chunk.
        
        # Check if the PDF name is not None and if it has not been used yet.
        if pdf_url is not None and pdf_url not in used:
            relevant_chunk = {
                'pdf_url': pdf_url,
                'organisation': match_metadata.get('Organisation',None),  # Include the URL.
                'showcase_title': match_metadata.get('Showcase Title', None),  # Include the Publication Year.
                'description': match_metadata.get('Brief Description', None),  # Include the Reference.
                'organisation_logo': match_metadata.get('Organisation Logo', None),  # Include the Country.
                'showcase_image': match_metadata.get('Showcase Image', None)  # Include the Author.
            }
            relevant_chunks.append(relevant_chunk)
            used[pdf_url] = 1  # Mark the PDF as used.
        

            # If we have collected enough unique chunks, break the loop.
            if len(relevant_chunks) == top_k:
                break

    return relevant_chunks


def find_match_expert(input,top_k,filter=None):
    pc_belongg = Pinecone(api_key=belongg_api_key)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    input_em = model.encode(input).tolist()
    index = pc_belongg.Index("open-source-v2")

    result = index.query(vector=input_em, 
                         top_k=top_k, 
                         filter={
                            "knowledge_object": {"$eq": True},
                            "Expert": {"$eq": True}
                         },
                         includeMetadata=True)
    expert_matches=[]
    expert_chunks=[]
    if 'matches' in result:
        matches = result['matches']

        
        for i in range(min(top_k, len(matches))):       
            expert_details={"name": matches[i]['metadata']['Name'],
                            "email": matches[i]['metadata']['Email'],
                            "short_description": matches[i]['metadata']['short Description'],
                            "description": matches[i]['metadata']['Description'],
                            "photo": matches[i]['metadata']['Photo']}
            expert_matches.append(expert_details)

    return expert_matches,expert_chunks


def extract_unique_chunks(input, top_k, multiplier, filter=None):
    # Extract the relevant chunks from the results based on top_k * multiplier.
    result_dict = find_match_retriever(input, top_k * multiplier, filter)
    
    used = {}
    relevant_chunks = []
    
    unique_dict = {'all_context': ''}
    
    # Iterate over the results to find unique chunks.
    for i in range(0, top_k * multiplier + 1):
        pdf_name = result_dict.get(f'meta_{i}', None)  # Get the PDF name for the current chunk.
        
        if pdf_name is not None and not pdf_name.endswith('.pdf'):
            pdf_name += '.pdf'
            
        # Check if the PDF name is not None and if it has not been used yet.
        if pdf_name is not None and pdf_name not in used:
            
            relevant_chunk = {
                'meta': pdf_name,
                'context': result_dict.get(f'context_{i}', None)  # Include the context.
            }
            relevant_chunks.append(relevant_chunk)
            used[pdf_name] = 1  # Mark the PDF as used.
            
            # Add the context to 'all_context'.
            unique_dict['all_context'] += relevant_chunk['context'] + "\n"
            
            # Add all relevant fields to unique_dict.
            for key, value in relevant_chunk.items():
                unique_dict[f'{key}_{len(relevant_chunks)}'] = value

            # If we have collected enough unique chunks, break the loop.
            if len(relevant_chunks) == top_k:
                break

    return unique_dict


def get_pinecone_analyzer_filters():
    
    c1_filter=None
    c2_filter= {"MBS": {"$eq": True},"GPP": {"$eq": True}}
    c3_filter= {"LC": {"$eq": True},"IID": {"$eq": True}}
    c4_filter= {"SDSC": {"$eq": True}}
    c5_filter= {"CSS": {"$eq": True}}
    filters={"C1(Universal corpus)":c1_filter,"C2(MBS and GPP)":c2_filter,"C3(LC and IID)":c3_filter,"C4(SDSC)":c4_filter,"C5(CSS)":c5_filter}
    
    return filters