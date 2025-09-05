import mysql.connector
import json
import traceback
import os
from dotenv import load_dotenv

load_dotenv()

host = os.getenv("DB_HOST")
database = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")

def get_analyzer_config(prompt_label, doc_type):
    conn = None
    customization_prompt = ""
    base_prompt = ""
    wisdom_1 = ""
    wisdom_2 = ""
    section_title = ""
    chunks = 0
    prompt_corpus = ""
    prompt_examples = ""
    dependencies = ""
    which_chunks = ""
    wisdom_received = ""
    llm_flow = ""
    llm = ""
    model = ""
    label_for_output = ""
    show_on_frontend = ""

    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        cursor = conn.cursor()

        try:
            retrieval_query = """
                SELECT customization_prompt, base_prompt, wisdom_1, wisdom_2, chunks, corpus_id, section_title, examples, 
                       dependencies, which_chunks, wisdom_received, LLM_Flow, LLM, Model, label_for_output, show_on_frontend
                FROM analyzer_custom_prompts
                WHERE prompt_label = %s AND doc_type = %s
                ORDER BY created_at DESC
                LIMIT 1
            """
            cursor.execute(retrieval_query, (prompt_label, doc_type))
            result = cursor.fetchone()

            if result:
                (customization_prompt, base_prompt, wisdom_1, wisdom_2, chunks, prompt_corpus, section_title, 
                 prompt_examples, dependencies, which_chunks, wisdom_received, llm_flow, llm, 
                 model, label_for_output, show_on_frontend) = result

                if prompt_examples:
                    prompt_examples = json.loads(prompt_examples)
        except Exception as e:
            print(f"An error occurred during the query execution: {str(e)}")
            traceback.print_exc()
        finally:
            cursor.close()
    except Exception as e:
        print(f"An error occurred during the database connection: {str(e)}")
        traceback.print_exc()
    finally:
        if conn is not None:
            conn.close()

    return (customization_prompt, base_prompt, wisdom_1, wisdom_2, chunks, prompt_corpus, section_title, 
            prompt_examples, dependencies, which_chunks, wisdom_received, llm_flow, llm, 
            model, label_for_output, show_on_frontend)