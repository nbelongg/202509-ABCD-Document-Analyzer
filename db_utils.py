import os
from dotenv import load_dotenv
import mysql.connector
from fastapi import HTTPException
from datetime import timezone, datetime
import traceback
import re
from custom_exception import CustomException, CustomDBException
import s3_bucket_utils
import api_utils, common_utils
import json
from typing import List, Optional
import concurrent.futures


load_dotenv(override=True)


host=os.getenv("mysql_host")
database=os.getenv("mysql_database")
user=os.getenv("mysql_user")
password=os.getenv("mysql_password")


def connect_to_mysql():
    try:
        connection = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        return connection
    except Exception as err:
        traceback.print_exc()
        #raise HTTPException(status_code=500, detail="Error connecting to the database.")
    

def get_prompt_corpus(prompt_label: str) -> str:
    prompt_corpus_mapping = {
        "P1": "C1(Universal corpus)",
        "P2": "C2(MBS and GPP)",
        "P3": "C3(LC and IID)",
        "P4": "C4(SDSC)",
        "P5": "C5(CSS)"
    }

    return prompt_corpus_mapping.get(prompt_label, "Invalid Prompt Label")


def get_current_gpt_config():
    conn=None
    prompt_string=""
    temperature=0
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
                                SELECT prompt_string, temperature
                                FROM gptconfig
                                ORDER BY created_at DESC
                                LIMIT 1
                            """

            cursor.execute(retrieval_query)
            result = cursor.fetchone()

            if result:
                prompt_string, temperature = result
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

    return prompt_string, temperature


def get_current_analyzer_gpt_config(prompt_label, doc_type):
    conn=None
    customization_prompt=""
    base_prompt=""
    wisdom_1=""
    wisdom_2=""
    section_title=""
    chunks=0
    prompt_corpus=""
    prompt_examples=""
    additional_dependencies=""
    which_chunks=""
    wisdom_received=""
    llm_flow=""
    llm=""
    model=""
    label_for_output=""
    show_on_frontend=""
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
                                SELECT customization_prompt,base_prompt,wisdom_1, wisdom_2,chunks, corpus_id, section_title,examples, dependencies,which_chunks,wisdom_received,
                                LLM_Flow, LLM, Model, label_for_output, show_on_frontend
                                FROM analyzer_prompts
                                WHERE prompt_label = %s AND doc_type=%s
                                ORDER BY created_at DESC
                                LIMIT 1
                            """
            
            cursor.execute(retrieval_query, (prompt_label,doc_type,))
            result = cursor.fetchone()

            if result:
                customization_prompt,base_prompt,wisdom_1, wisdom_2,chunks,prompt_corpus,section_title,prompt_examples,additional_dependencies,which_chunks,wisdom_received,llm_flow, llm, model, label_for_output, show_on_frontend = result
                
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

    return customization_prompt,base_prompt,wisdom_1, wisdom_2, chunks,prompt_corpus,section_title,prompt_examples, additional_dependencies,which_chunks,wisdom_received,llm_flow,llm,model,label_for_output,show_on_frontend


def get_custom_analyzer_gpt_config(prompt_label, nature_of_document):
    conn=None
    prompt_for_customization=""
    prompt_string=""
    chunks=15
    prompt_corpus=""
    prompt_examples=""
    try:
        # conn = mysql.connector.connect(
        #     host=host,
        #     database=database,
        #     user=user,
        #     password=password
        # )
        conn = connect_to_mysql()
        cursor = conn.cursor()

        try:
            query = """
                        SELECT customization_prompt, base_prompt, chunks, prompt_corpus, examples 
                        FROM analyzer_custom_prompts 
                        WHERE prompt_label = %s AND doc_type = %s
                        ORDER BY created_at
                        LIMIT 1
                    """
            cursor.execute(query, (prompt_label, nature_of_document))
            result = cursor.fetchone()
            return result
            if result:
                prompt_for_customization, prompt_string, chunks, prompt_corpus, prompt_examples = result
                prompt_corpus = get_prompt_corpus(prompt_label)
                
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
    return prompt_for_customization, prompt_string, chunks, prompt_corpus, prompt_examples


def get_current_analyzer_gpt_prompts(prompt_labels):
    prompts_configurations = {}

    try:
        for prompt_label in prompt_labels:
            try:
                prompt_for_customization, prompt_string, chunks,prompt_corpus, prompt_examples = get_current_analyzer_gpt_config(prompt_label)
                prompts_configurations[prompt_label] = (prompt_for_customization, prompt_string, chunks, prompt_corpus, prompt_examples)                
            except Exception as e:
                print(f"An error occurred during the query execution for label {prompt_label}: {str(e)}")
                traceback.print_exc()

    except Exception as e:
        print(f"An error occurred during the database connection: {str(e)}")
        traceback.print_exc()

    return prompts_configurations


def get_prompt_label_flows(prompt_labels, doc_type):
    conn=None
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        cursor = conn.cursor(dictionary=True)

        try:
            prompt_data = {label: {} for label in prompt_labels}
            prompt_flow_dependencies = {}

            query = """
                SELECT `prompt_label`, `base_prompt`, `customization_prompt`, `customize_prompt_based_on`, 
                `send_along_customised_prompt`, `wisdom_1`, `wisdom_2`, `prompt_corpus`, 
                `chunks`, `dependencies`, `which_chunks`, `wisdom_received`,
                `LLM_Flow`, `LLM`, `Model`, `label_for_output`, `show_on_frontend`
                FROM `analyzer_prompts`
                WHERE ({}) AND `doc_type` = %s
            """

            like_conditions = " OR ".join([f"`prompt_label` LIKE '{label}%'" for label in prompt_labels])

            cursor.execute(query.format(like_conditions), (doc_type,))

            results = cursor.fetchall()

            for row in results:
                main_label = row['prompt_label'].split('.')[0]
                sub_label = row['prompt_label']
                if main_label in prompt_data:
                    if sub_label == main_label:
                        continue
                    prompt_data[main_label][sub_label] = {
                        'base_prompt': row['base_prompt'],
                        'customization_prompt': row['customization_prompt'],
                        'customize_prompt_based_on': row['customize_prompt_based_on'].split(',') if row['customize_prompt_based_on'] else [],
                        'send_along_customised_prompt': row['send_along_customised_prompt'].split(',') if row['send_along_customised_prompt'] else [],
                        'wisdom_1': row['wisdom_1'],
                        'wisdom_2': row['wisdom_2'],
                        'prompt_corpus': row['prompt_corpus'],
                        'chunks': row['chunks'],
                        'additional_dependencies': [item.strip() for item in row['dependencies'].split(',')] if row['dependencies'] else [],
                        'which_chunks': [item.strip() for item in row['which_chunks'].split(',')] if row['which_chunks'] else [],
                        'wisdom_received': [item.strip() for item in row['wisdom_received'].split(',')] if row['wisdom_received'] else [],
                        'llm_flow': row['LLM_Flow'],
                        'llm': row['LLM'] if row['LLM'] else "ChatGPT",
                        'model': row['Model'] if row['Model'] else "gpt-4o-2024-08-06",
                        'label_for_output': row['label_for_output'],
                        'show_on_frontend': row['show_on_frontend']
                    }
            
            prompt_flow_dependencies[sub_label] = [item.strip() for item in row['additional_dependencies'].split(',')] if row['additional_dependencies'] else []
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
    return prompt_data, prompt_flow_dependencies


def get_custom_analyzer_gpt_prompts(prompt_labels, nature_of_document, user_role):
    prompts_configurations = {}
    try:
        for prompt_label in prompt_labels:
            try:
                print(f"Fetching Base and Customization Prompts for {prompt_label}!")
                prompt_string, prompt_for_customization, chunks, prompt_corpus, prompt_examples, wisdom_1, wisdom_2 = api_utils.get_prompts(prompt_label, nature_of_document)
                if common_utils.has_placeholder(prompt_string, "user_role"):
                    prompt_string = common_utils.replace_placeholder(prompt_string, "user_role", user_role)
                prompts_configurations[prompt_label] = (prompt_for_customization, prompt_string, chunks, prompt_corpus, prompt_examples, wisdom_1, wisdom_2) 
            except Exception as e:
                print(f"An error occurred during the query execution for label {prompt_label}: {str(e)}")
                raise e
        return prompts_configurations
    except Exception as e:
        print(f"An error occurred during the database connection: {str(e)}")
        raise e
    

def get_analyzer_prompts_multithreaded(prompt_labels, doc_type, user_role):
    prompts_configurations = {}
    prompts_section_titles = {}
    prompt_dependencies = {}
    prompt_label_corpus_mapping = {"P1":"C1(Inclusion 101)","P2":"C2(Unique Needs & experiences and journeys)","P3":"C3(Barriers)","P4":"C4(Solutions & Case Studies)","P5":"C5(Toolkits and methodologies)","P6":"C6(Partners & Experts)","P7":"C7(Policy & Compliance)","P8":"C8(People speak)","P9":"C9(Risks including AI-risks)"}
    
    def fetch_prompt(label):
        """Fetch prompts for a given label"""
        try:
            customization_prompt,base_prompt,wisdom_1,wisdom_2,chunks,prompt_corpus,section_title,prompt_examples, dependencies,which_chunks,wisdom_received,llm_flow,llm,model,label_for_output,show_on_frontend = get_current_analyzer_gpt_config(label,doc_type)
            
            if(customization_prompt == '' and base_prompt == ''):
                        return -1
            if user_role:
                if(common_utils.has_placeholder(base_prompt, "user_role")):
                        base_prompt = common_utils.replace_placeholder(base_prompt,"user_role", user_role)
                        
                if(common_utils.has_placeholder(customization_prompt, "user_role")):
                        customization_prompt = common_utils.replace_placeholder(customization_prompt,"user_role", user_role)
            
            if(common_utils.has_placeholder(customization_prompt, "Document.Summary")):
                    customization_prompt = common_utils.replace_placeholder(customization_prompt,"Document.Summary", "summary of the document")
            if(common_utils.has_placeholder(customization_prompt, "Document.Full")):
                    customization_prompt = common_utils.replace_placeholder(customization_prompt,"Document.Full","full document text")

            if not prompt_corpus:
                prompt_corpus = prompt_label_corpus_mapping[label]
            
            if not section_title:
                section_title = ""
            
            if not dependencies or dependencies == "":
                dependencies = []
            else:
                dependencies = [item.strip() for item in dependencies.split(",")] 
            
            if not wisdom_received or wisdom_received == "":
                wisdom_received = []
            else:
                wisdom_received = [item.strip() for item in wisdom_received.split(",")]

            if not which_chunks or which_chunks == "":
                which_chunks = []
            else:
                which_chunks = [item.strip() for item in which_chunks.split(",")]
            
            if not llm or llm == "":
                llm = "ChatGPT"
            
            if not model or model == "":
                model = "gpt-4o-2024-08-06"
            
            prompts_configurations[label]=(customization_prompt,base_prompt, wisdom_1, wisdom_2,chunks, prompt_corpus, prompt_examples,which_chunks,wisdom_received,llm_flow,llm,model,label_for_output,show_on_frontend)
            prompts_section_titles[label] = section_title
            prompt_dependencies[label] = dependencies  
        except Exception as e:
            print(f"An error occurred during the query execution for label {label}: {str(e)}")

    # Use ThreadPoolExecutor to run multiple fetches concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(fetch_prompt, prompt_labels)

    return prompts_configurations, prompts_section_titles, prompt_dependencies


def get_custom_analyzer_gpt_prompts_multithreaded_original(prompt_labels, nature_of_document, user_role):
    print("Using multithreading to fetch prompts!")
    prompts_configurations = {}
    wisdom = {}
    prompts_section_title = {}
    dependency_graph = {}
    def fetch_prompt(label):
        """Fetch prompts for a given label"""
        try:
            prompt_string, prompt_for_customization, chunks, prompt_corpus, prompt_examples, wisdom_1, wisdom_2, section_title, dependencies = get_analyzer_config(label, nature_of_document)
            if common_utils.has_placeholder(prompt_string, "user_role"):
                prompt_string = common_utils.replace_placeholder(prompt_string, "user_role", user_role)
            prompts_configurations[label] = (prompt_for_customization, prompt_string, chunks, prompt_corpus, prompt_examples)
            prompts_section_title[label] = section_title
            wisdom[label] = (wisdom_1, wisdom_2)
            dependency_graph[label] = dependencies if dependencies else []
        except Exception as e:
            print(f"An error occurred during the query execution for label {label}: {str(e)}")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(fetch_prompt, prompt_labels)

    return prompts_configurations, wisdom, prompts_section_title, dependency_graph


def get_current_analyzer_custom_partition_prompts(nature_of_document, user_role, organization_id):
    try:
        base_prompt, customization_prompt, chunks, section_title = api_utils.get_custom_prompts(nature_of_document, organization_id)
            
        prompts_configurations = {}

        if base_prompt and customization_prompt and chunks:
            if common_utils.has_placeholder(base_prompt, "user_role"):
                base_prompt = common_utils.replace_placeholder(base_prompt, "user_role", user_role)
            if common_utils.has_placeholder(customization_prompt, "user_role"):
                customization_prompt = common_utils.replace_placeholder(customization_prompt, "user_role", user_role)

            prompts_configurations["P_Custom"] = (customization_prompt, base_prompt, chunks, "", "")

        return prompts_configurations, section_title
    except Exception as e:
        traceback.print_exc()
        raise e


def get_evaluator_prompts(partition_type=None, doc_type=None, organization_id=None):
    try:
        # Attempt to fetch evaluator prompts from the database
        evaluator_prompts = []
        if partition_type:
            for partition in partition_type:
                evaluator_prompt = get_all_evaluator_prompts_from_db(paritition_type=partition,doc_type=doc_type, organization_id=organization_id)
                evaluator_prompts.extend(evaluator_prompt)
        else:
            evaluator_prompts = get_all_evaluator_prompts_from_db(doc_type=doc_type, organization_id=organization_id)
        # Check if the fetched prompts are empty
        if not evaluator_prompts:
            raise ValueError("No evaluator prompts found for the given Organization ID.")

        prompts_configurations = {}
        for prompt in evaluator_prompts:
            prompt_label = prompt.get("prompt_label", "Unknown Label")
            base_prompt = prompt.get("base_prompt", "No base prompt provided")
            customization_prompt = prompt.get("customization_prompt", "No customization prompt provided")
            chunks = prompt.get("chunks", [])
            prompt_examples = None
            
            prompts_configurations[prompt_label] = {
                "base_prompt": base_prompt,
                "customization_prompt": customization_prompt,
                "chunks": chunks,
                "prompt_examples": prompt_examples,
            }
        
        return prompts_configurations

    except Exception as e:
        # If an error occurs, raise a descriptive error message
        raise RuntimeError("An error occurred while fetching evaluator prompts: " + str(e))


def get_evaluator_prompts_multithreaded(partition_type=None, doc_type=None, organization_id=None, org_guideline_id=None, st=None):
    prompts_configurations = {}
    wisdom = {}
    evaluator_section_titles = {}
    evaluator_dependency_graph = {}

    # Function to fetch evaluator prompts concurrently
    def fetch_prompts(partition=None):
        try:
            if partition:
                evaluator_prompt = get_all_evaluator_prompts_from_db(paritition_type=partition, doc_type=doc_type, organization_id=organization_id, org_guideline_id=org_guideline_id, st=st)
            else:
                evaluator_prompt = get_all_evaluator_prompts_from_db(doc_type=doc_type, organization_id=organization_id, org_guideline_id=org_guideline_id, st=st)

            if not evaluator_prompt:
                raise ValueError("No evaluator prompts found for the given Organization ID.")

            for prompt in evaluator_prompt:
                prompt_label = prompt.get("prompt_label", "Unknown Label")
                base_prompt = prompt.get("base_prompt", "No base prompt provided")
                customization_prompt = prompt.get("customization_prompt", "No customization prompt provided")
                chunks = prompt.get("chunks", [])
                dependencies = prompt.get("dependencies", "").split(',') if prompt.get("dependencies") else []
                
                prompt_examples = None
                
                wisdom1 = prompt.get("wisdom_1", None)
                wisdom2 = prompt.get("wisdom_2", None)
                if prompt_label != "Unknown Label":
                    wisdom[prompt_label] = (wisdom1, wisdom2)
                    
                section_title = prompt.get("section_title", None)
                evaluator_section_titles[prompt_label] = section_title
                evaluator_dependency_graph[prompt_label] = dependencies if dependencies else []
                
                prompts_configurations[prompt_label] = {
                    "base_prompt": base_prompt,
                    "customization_prompt": customization_prompt,
                    "chunks": chunks,
                    "prompt_examples": prompt_examples,
                }
            print(f"Evaluator Dependencies: {evaluator_dependency_graph}")
        except Exception as e:
            raise RuntimeError("An error occurred while fetching evaluator prompts: " + str(e))

    # Use ThreadPoolExecutor to run fetch_prompts concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        if partition_type:
            executor.map(fetch_prompts, partition_type)
        else:
            fetch_prompts()  # If no partition, run synchronously

    return prompts_configurations, wisdom, evaluator_section_titles, evaluator_dependency_graph


def retrieve_prompts(prompt_labels, nature_of_document, organization_id, org_guideline_id, logger):

    def get_analyzer_prompts():
        try:
            return get_custom_analyzer_gpt_prompts_multithreaded(prompt_labels, nature_of_document, None)
        except Exception as e:
            logger.error("Error retrieving analyzer prompts", exc_info=True)
            raise HTTPException(status_code=500, detail="Error connecting to the database. Please try again later.")

    def get_evaluator_prompts():
        try:
            return get_evaluator_prompts_multithreaded(
                doc_type=nature_of_document,
                organization_id=organization_id,
                org_guideline_id=org_guideline_id,
            )
        except Exception as e:
            logger.error("Error retrieving evaluator prompts", exc_info=True)
            raise HTTPException(status_code=404, detail=f"No prompts found for the organization_id: {organization_id}")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        analyzer_future = executor.submit(get_analyzer_prompts)
        evaluator_future = executor.submit(get_evaluator_prompts)

        selected_prompts, wisdom, prompts_section_title, dependency_graph = analyzer_future.result()
        selected_evaluator_prompts, evaluator_wisdom, evaluator_section_titles, evaluator_dependency_graph = evaluator_future.result()

    return selected_prompts, wisdom, prompts_section_title, dependency_graph, selected_evaluator_prompts, evaluator_wisdom, evaluator_section_titles, evaluator_dependency_graph


def get_all_evaluator_prompts_from_db(paritition_type=None, doc_type=None, organization_id=None, org_guideline_id=None, st=None):
    conn = None
    prompts = []
    
    try:
        # Establish a connection to the database
        conn = connect_to_mysql()
        cursor = conn.cursor(dictionary=True)
        
        query = "SELECT * FROM analyzer_custom_evaluator_prompts"
        params = []

        if doc_type:
            query += " WHERE doc_type = %s"
            params.append(doc_type)
        if organization_id:
            if not doc_type:
                query += " WHERE"
            else:
                query += " AND"
            query += " organization_id = %s"
            params.append(organization_id)
        if paritition_type:
            if not doc_type and not organization_id:
                query += " WHERE"
            else:
                query += " AND"
            query += " prompt_label = %s"
            params.append(paritition_type)
        if org_guideline_id is not None:
            print(f"Fetching prompts for org_guideline_id: {org_guideline_id}")
            if not doc_type and not organization_id and not paritition_type:
                query += " WHERE"
            else:
                query += " AND"
            query += " org_guideline_id = %s"
            params.append(org_guideline_id)

        cursor.execute(query, params)
        prompts = cursor.fetchall()
        
        if not prompts:
            raise ValueError(f"No evaluator prompts found for the given Organization ID: {organization_id}.")
    except Exception as e:
        return None
    finally:
        if conn:
            conn.close()

    return prompts


def fetch_org_guidelines():
    conn = connect_to_mysql()
    try:
        cursor = conn.cursor(dictionary=True)
        
        query = """
            SELECT organization_id, GROUP_CONCAT(DISTINCT org_guideline_id) AS guideline_ids
            FROM analyzer_custom_evaluator_prompts
            GROUP BY organization_id;
        """
        cursor.execute(query)

        org_guidelines = {}

        rows = cursor.fetchall()

        for row in rows:
            org_guideline_ids = {}
            org_id = row['organization_id']
            if row['guideline_ids']:
                guideline_ids = row['guideline_ids'].split(',')
            else:
                guideline_ids = []
            org_guideline_ids["org_guideline_ids"] = guideline_ids
            org_guidelines[org_id] = org_guideline_ids

        return {"organization_ids": org_guidelines}
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        cursor.close()
        conn.close()


def set_analyzer_gpt_config(prompt_for_customization, prompt_label, prompt, chunks, prompt_corpus, prompt_examples):
    conn=None
    updated=False
    
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        cursor = conn.cursor()

        try:
            
            insertion_query = """
                                INSERT INTO analyzer_prompt
                                (prompt_for_customization,prompt_string, prompt_label,chunks,prompt_corpus,examples)
                                VALUES
                                (%s,%s, %s, %s, %s,%s)
                            """

            data = (prompt_for_customization,prompt, prompt_label, chunks, prompt_corpus, prompt_examples)
            cursor.execute(insertion_query, data)
            conn.commit()
            updated=True
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

    return updated


def set_summary_prompt(summary_prompt):
    conn=None
    updated=False
    
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        cursor = conn.cursor()

        try:
            
            insertion_query = """
                                INSERT INTO summary_prompt
                                (summary_prompt)
                                VALUES
                                (%s)
                            """
            
            data = (summary_prompt,)
            
            cursor.execute(insertion_query, data)
            conn.commit()
            updated=True
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

    return updated


def get_current_summary_prompt():
    conn=None
    summary_prompt=""
    
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
                                SELECT summary_prompt
                                FROM summary_prompt
                                ORDER BY created_at DESC
                                LIMIT 1
                            """
            
            cursor.execute(retrieval_query)
            result = cursor.fetchone()

            if result:
                summary_prompt = result[0]
                
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

    return summary_prompt


def get_proposal_summary_prompt(doc_type):
    conn=None
    proposal_summary_prompt=""
    
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
                SELECT proposal_prompt
                FROM analyzer_proposal_summary_prompts
                WHERE doc_type = %s
                ORDER BY created_at DESC
                LIMIT 1
            """
            
            cursor.execute(retrieval_query, (doc_type,))
            result = cursor.fetchone()

            if result:
                proposal_summary_prompt = result[0]
                
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

    return proposal_summary_prompt


def get_comments_summary_prompt(doc_type):
    conn=None
    comments_summary_prompt=""
    
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
                SELECT summary_prompt
                FROM analyzer_comments_summary_prompts
                WHERE doc_type = %s
                ORDER BY created_at DESC
                LIMIT 1
            """
            
            cursor.execute(retrieval_query, (doc_type,))
            result = cursor.fetchone()

            if result:
                comments_summary_prompt = result[0]
                
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

    return comments_summary_prompt


def set_gpt_config(prompt,temperature):
    conn=None
    updated=False
    
    
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        cursor = conn.cursor()

        try:
            
            insertion_query = """
                                INSERT INTO gptconfig
                                (prompt_string, temperature)
                                VALUES
                                (%s, %s)
                            """

            data = (prompt, temperature)
            cursor.execute(insertion_query, data)
            conn.commit()
            updated=True
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

    return updated


def log_chat_response(user_id,user_name,user_email,session_id,query,contextData,response,response_id,source=None):
    conn=None
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        cursor = conn.cursor()

        try:
            feedback = None
            feedback_note = None
            
            dt = datetime.now(timezone.utc)
            utc_timestamp = dt.timestamp()
            created_at = datetime.utcfromtimestamp(utc_timestamp).strftime('%Y-%m-%dT%H:%M:%S%Z')
            modified_at= created_at
            
            print(created_at)

            insertion_query = """
                        INSERT INTO abcdchatdata
                        (user_id, user_name, user_email, session_id, query,response_id, gpt_response, feedback, feedback_note, context, created_at, modified_at, source)
                        VALUES
                        (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s)
                    """

            data = (user_id, user_name, user_email, session_id, query, response_id, response, feedback, feedback_note, str(contextData), created_at, modified_at, source)
            cursor.execute(insertion_query, data)

            conn.commit()
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


def log_analysis_response(
    user_id: Optional[str],
    user_name: Optional[str],
    session_id: Optional[str],
    pdf_name: Optional[str],
    text: str,
    nature_of_document: str,
    prompt_labels: List[str],
    summary_text: str,
    generated_prompts: dict,
    pinecone_filters: dict,
    generated_analyze_comments: dict,
    analyze_context_used: dict,
    time_taken: float,
    organization_id: Optional[str] = None,
    tokens_counter=None
    ):
    conn=None
    cursor=None
    
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        cursor = conn.cursor()

        user_objective_str = None
        interest_topics_str = None
        prompt_labels_str = ','.join(prompt_labels) if prompt_labels else None
        generated_prompts_json = json.dumps(generated_prompts)
        pinecone_filters_json = json.dumps(pinecone_filters)
        generated_analyze_comments_json = json.dumps(generated_analyze_comments)
        analyze_context_used_json = json.dumps(analyze_context_used)
        
        total_input_tokens = common_utils.total_tokens_count(tokens_counter[0])
        total_output_tokens = common_utils.total_tokens_count(tokens_counter[1])
        total_tokens = total_input_tokens + total_output_tokens
        tokens_counter[0]["total_tokens"] = total_input_tokens
        tokens_counter[1]["total_tokens"] = total_output_tokens
        total_tokens = {"total_tokens": total_tokens}
        total_tokens = json.dumps(total_tokens)
        
        input_tokens = None
        output_tokens = None
        if tokens_counter is not None:
            input_tokens = json.dumps(tokens_counter[0])
            output_tokens = json.dumps(tokens_counter[1])
        
        insert_query = """
        INSERT INTO analyzer_data(
            user_id, user_name, session_id, pdf_name, text, nature_of_document, 
            user_objective, interest_topics, prompt_labels, summary_text, 
            generated_prompts, pinecone_filters, generated_analyze_comments,analyze_context_used,time_taken, organization_id, input_tokens, output_tokens, total_tokens
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        cursor.execute(insert_query, (
            user_id, user_name, session_id, pdf_name, text, nature_of_document, 
            user_objective_str, interest_topics_str, prompt_labels_str, summary_text, 
            generated_prompts_json, pinecone_filters_json, generated_analyze_comments_json,analyze_context_used_json,time_taken, organization_id, input_tokens, output_tokens, total_tokens
        ))
        
        conn.commit()
        
    except Exception as err:
        print(f"Error: {err}")
        traceback.print_exc() 
    finally:
        if cursor is not None:
            cursor.close()
            
        if conn is not None:
            conn.close()
    

def get_analysis_sessions(user_id):
    conn=None
    cursor=None
    final_session_data = []
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        cursor = conn.cursor(dictionary=True)
        analysis_query = "SELECT session_title, pdf_name, text, session_id, created_at FROM analyzer_data WHERE user_id = %s"
        cursor.execute(analysis_query, (user_id,))
        
        session_data = cursor.fetchall()
        
        final_session_data=[]
        for record in session_data:
            temp={}
            pdf_name = record['pdf_name']
            text = record['text']
            session_title = record['session_title']
            temp['session_id'] = record['session_id']
            temp['created_at'] = record['created_at']
            if session_title:
                temp['data']=session_title
            elif '.txt' in pdf_name: 
                pattern = r'^(.*?)(?:\n\s*\n|$)'
                match = re.search(pattern, text, re.DOTALL)
                temp['data']=match.group(1)[:200]
            else:
                temp['data']=pdf_name
            final_session_data.append(temp)
   
    except Exception as e:
        print(f"An error occurred during the database connection: {str(e)}")
        traceback.print_exc()

    finally:
        if cursor is not None:
            cursor.close()    
        
        if conn is not None:
            conn.close()

    return final_session_data
 
 
def get_custom_evaluator_sessions(user_id):
    try:
        conn = connect_to_mysql()
        cursor = conn.cursor(dictionary=True)
        query = "SELECT user_id, session_id, session_title, proposal_pdf_name, proposal_text, created_at FROM analyzer_custom_evaluator_data WHERE user_id = %s"
        cursor.execute(query, (user_id,))
        
        results = cursor.fetchall()
        
        response = []
        
        for result in results:
            temp = {}
            temp['session_id'] = result['session_id']
            temp['created_at'] = result['created_at']
            session_title = result['session_title']
            pdf_name = result['proposal_pdf_name']
            text = result['proposal_text']
            if session_title != "none":
                temp['data'] = session_title
            elif '.txt' in pdf_name: 
                pattern = r'^(.*?)(?:\n\s*\n|$)'
                match = re.search(pattern, text, re.DOTALL)
                temp['data']=match.group(1)[:200]
            else:
                temp['data']=pdf_name
            response.append(temp)
        
        if not results:
            raise HTTPException(status_code=404, detail="No sessions found for this user.")
        
        return response
    except Exception as general_err:
        raise general_err
    finally:
        if conn and conn.is_connected():
            conn.close()
        
    
def get_analysis_conversations(user_id,session_id):
    conversation=[]
    conn=None
    cursor=None
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )

        cursor = conn.cursor(dictionary=True)

        conversation_query = """
                            SELECT query,gpt_response,section,created_at,response_id
                            FROM analyze_chat
                            WHERE user_id = %s AND session_id = %s
                            ORDER BY created_at ASC;
                            """

        cursor.execute(conversation_query, (user_id,session_id,))
        results = cursor.fetchall()

        conversation = []
        for result in results:
            user_dict = {"timestamp":result['created_at'],"role": "user", "content": result['query'],"section":result['section']}
            assistant_dict = {"timestamp":result['created_at'],"role": "assistant", "content": result['gpt_response'], "response_id":result['response_id']}
            
            conversation.append(user_dict)
            conversation.append(assistant_dict)
                    
        return conversation
    except Exception as e:
        print(f"DB error: {str(e)}")
        traceback.print_exc()
        raise e
    finally:
        if cursor is not None:
            cursor.close()
            
        if conn is not None:
            conn.close()


def get_analyzer_conversations(user_id, session_ids):
    conversations = {}
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )

        cursor = conn.cursor(dictionary=True)

        # Create a string of placeholders for the IN clause
        placeholders = ', '.join(['%s'] * len(session_ids))
        
        conversation_query = f"""
                            SELECT query, gpt_response, section, created_at, response_id, session_id
                            FROM analyze_chat
                            WHERE user_id = %s AND session_id IN ({placeholders})
                            ORDER BY session_id, created_at ASC;
                            """

        # Combine user_id and session_ids for the query parameters
        query_params = [user_id] + session_ids
        cursor.execute(conversation_query, tuple(query_params))
        results = cursor.fetchall()

        # Process results and organize by session_id
        for result in results:
            session_id = result['session_id']
            if session_id not in conversations:
                conversations[session_id] = []
            
            user_dict = {
                "timestamp": result['created_at'],
                "role": "user",
                "content": result['query'],
                "section": result['section']
            }
            assistant_dict = {
                "timestamp": result['created_at'],
                "role": "assistant",
                "content": result['gpt_response'],
                "response_id": result['response_id']
            }
            
            conversations[session_id].extend([user_dict, assistant_dict])

        return conversations
    except Exception as e:
        print(f"DB error: {str(e)}")
        traceback.print_exc()
        raise e
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()
            

def get_analysis_data(user_id,session_id):
    conn=None
    cursor=None
    session_data = {}
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        cursor = conn.cursor(dictionary=True)
        analysis_query = "SELECT user_id, pdf_name, text, generated_analyze_comments, created_at, session_id FROM analyzer_data WHERE user_id = %s and session_id = %s"

        cursor.execute(analysis_query, (user_id, session_id))
         
        results = cursor.fetchall()
        
        conversations = get_analysis_conversations(user_id,session_id)
        
        for row in results:
            
            session_data['user_id']=row['user_id']
            session_data['generated_analyze_comments']=json.loads(row['generated_analyze_comments'])
            session_data['conversations']=conversations
            session_data['created_at']=row['created_at']
            session_data['session_id']=row['session_id']
            session_data['s3_par_url']=""
            pdf_name = row['pdf_name']
                
            if not pdf_name:
                session_data['text'] = row['text']
            else:
                session_data['s3_par_url']=s3_bucket_utils.get_file_par_url(pdf_name)
                                
        
    except Exception as e:
        print(f"An error occurred during the database connection: {str(e)}")
        traceback.print_exc()

    finally:
        if cursor is not None:
            cursor.close()    
        
        if conn is not None:
            conn.close()

    return session_data


def get_analyzer_session_data(user_id, number_of_sessions=None, session_ids=None):
    conn = None
    cursor = None
    session_data = {}
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        cursor = conn.cursor(dictionary=True)
        
        # Fetch all sessions for the user
        query = """
        SELECT user_id, pdf_name, text, generated_analyze_comments, created_at, session_id 
        FROM analyzer_data 
        WHERE user_id = %s 
        """
        query = """
        SELECT user_id, pdf_name, text, generated_analyze_comments, created_at, session_id 
        FROM analyzer_data 
        """
        if user_id:
            query += "WHERE user_id = %s "
            params = (user_id,)
            if number_of_sessions:
                query += f"LIMIT {number_of_sessions}"
        elif session_ids:
            placeholders = ', '.join(['%s'] * len(session_ids))
            query += f"WHERE session_id IN ({placeholders}) "
            params = tuple(session_ids)
        else:
            raise ValueError("Either user_id or session_ids must be provided")

        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # if number_of_sessions:
        #     query += f" LIMIT {number_of_sessions}"

        # cursor.execute(query, (user_id,))
        # results = cursor.fetchall()
        
        if not results:
            return []  # Return an empty list if no sessions found

        # Extract session_ids
        session_ids = [row['session_id'] for row in results]
        
        # Fetch conversations for all sessions
        conversations = get_analyzer_conversations(user_id, session_ids)
        
        # Prepare the final response
        final_response = []
        for row in results:
            session_id = row['session_id']
            session_data = {
                'user_id': row['user_id'],
                'generated_analyze_comments': json.loads(row['generated_analyze_comments']),
                'conversations': conversations.get(session_id, []),
                'created_at': row['created_at'],
                'session_id': session_id,
                's3_par_url': ""
            }
            
            pdf_name = row['pdf_name']
            if not pdf_name:
                session_data['text'] = row['text']
            else:
                session_data['s3_par_url'] = s3_bucket_utils.get_file_par_url(pdf_name)
            
            final_response.append(session_data)

        return final_response

    except Exception as e:
        print(f"An error occurred during the database connection: {str(e)}")
        traceback.print_exc()
        return []  # Return an empty list in case of error
    finally:
        if cursor is not None:
            cursor.close()    
        if conn is not None:
            conn.close()


def get_analysis_section_data(user_id, session_id, section):
    conn=None
    cursor=None
    
    section_comments = ""
    context_data = ""
    
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )

        cursor = conn.cursor(dictionary=True)

        analysis_query = "SELECT generated_analyze_comments,analyze_context_used FROM analyzer_data WHERE user_id = %s and session_id = %s"

        cursor.execute(analysis_query, (user_id, session_id))
        
        results = cursor.fetchall()
        
        for row in results:
            generated_analyze_comments=json.loads(row['generated_analyze_comments'])
            analyze_context_used=json.loads(row['analyze_context_used'])
            
            if section=="P0":
                p0_summary = generated_analyze_comments["P0"]
                section_comments = section_comments+ "\n\n" + p0_summary
            
            elif section:
                if section in generated_analyze_comments:
                    section_comments = "\n".join(comment["comment"] for comment in generated_analyze_comments[section]["analyze_comments"])
                    context_data = analyze_context_used[section]
                else:
                    available_sections = list(generated_analyze_comments.keys())
                    error_message = f"Section '{section}' not found in generated analyze comments. Available sections: {available_sections}"
                    raise HTTPException(status_code=404, detail=error_message)
            else:
                for section, section_data in generated_analyze_comments.items():
                    if section == "P0":
                        continue
                    section_comments += "\n\n".join(comment["comment"] for comment in section_data["analyze_comments"])
                    
                for section, section_data in analyze_context_used.items():
                    context_data += section_data + "\n\n"
                    
    except Exception as e:
        print(f"An error occurred during the database connection: {str(e)}")
        traceback.print_exc()
        raise e
    finally:
        if cursor is not None:
            cursor.close()    
        
        if conn is not None:
            conn.close()

    return section_comments,context_data


def log_response_feedback(user_id,response_id,feedback,feedback_note):
    
    conn=None
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        
        cursor = conn.cursor()
        
       
        dt = datetime.now(timezone.utc)
        utc_timestamp = dt.timestamp()
        modified_at = datetime.utcfromtimestamp(utc_timestamp).strftime('%Y-%m-%dT%H:%M:%S%Z')
        
        query = """
                    UPDATE abcdchatdata
                    SET feedback = %s, feedback_note = %s, modified_at = %s
                    WHERE user_id = %s AND response_id = %s
                """

        data = (feedback, feedback_note,modified_at,user_id, response_id)
        cursor.execute(query, data)
        conn.commit()
        
        return {"message": f"Updated successfully"} 
     
    except Exception as e:
        print(f"An error occurred during the query execution: {str(e)}")
        traceback.print_exc()
        raise e
    finally:
        if conn is not None:
            conn.close()


def get_associated_prompt_label(doc_type):
    prompt_labels = []
    try:
        conn = connect_to_mysql()
        cursor = conn.cursor()

        try: 
            query = """
            SELECT prompt_label
            FROM analyzer_prompts
            WHERE doc_type = %s;
            """
            cursor.execute(query, (doc_type,))

            result = cursor.fetchall()
            prompt_labels = [row[0] for row in result]

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
    return prompt_labels


def db_exists():
    table_exists=None
    conn=None
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )

        cursor = conn.cursor()
        
        try:
            table_name = 'abcdchatdata'
            
            cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
            table_exists = cursor.fetchone()
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
    
    
    return table_exists


def create_db():
    if db_exists():
       return
   
    conn=None
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        cursor = conn.cursor()
        try:
            create_table = """CREATE TABLE abcdchatdata (user_id VARCHAR(255),
                                                        user_name VARCHAR(255),
                                                        user_email VARCHAR(255),
                                                        session_id VARCHAR(255),
                                                        query VARCHAR(255),
                                                        response_id VARCHAR(255),
                                                        gpt_response TEXT,
                                                        feedback BOOLEAN,
                                                        feedback_note TEXT,
                                                        context TEXT,
                                                        created_at TIMESTAMP,
                                                        modified_at TIMESTAMP)
                            """
            cursor.execute(create_table)
            conn.commit()
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


def get_user_data(user_id, source=None):
    
    session_data = {}
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )

        cursor = conn.cursor(dictionary=True)

        # Adjust condition based on source
        if source == "WA":
            condition = "AND source = 'WA'"
        else:
            condition = "AND source is NULL"

        session_data["user_id"] = user_id
        
        # Fetch the latest session ID
        session_query = f"""
                            SELECT session_id 
                            FROM abcdchatdata 
                            WHERE user_id = %s {condition}
                            ORDER BY modified_at DESC 
                            LIMIT 1;
                            """
        print("Session Query:", session_query)
        cursor.execute(session_query, (user_id,))
        result = cursor.fetchone()

        if result:
            session_id = result['session_id']
            conversation = []

            # Fetch conversation for the latest session
            conversation_query = """
                                SELECT query, gpt_response, context, response_id, created_at
                                FROM abcdchatdata
                                WHERE user_id = %s AND session_id = %s
                                ORDER BY modified_at ASC;
                                """
            print(conversation_query % (user_id, session_id))
            cursor.execute(conversation_query, (user_id, session_id))
            results = cursor.fetchall()

            for result in results:
                user_dict = {"timestamp": result['created_at'], "role": "user", "content": result['query']}
                context_data = eval(result['context'])
                sources_info = ""
                if "contextInfo" in context_data and "sources" in context_data:
                    sources_info = context_data["sources"]

                assistant_dict = {"timestamp": result['created_at'], "role": "assistant", "content": result['gpt_response'], "response_id": result['response_id'], "sources": sources_info}
                conversation.append(user_dict)
                conversation.append(assistant_dict)

            session_data["session_id"] = session_id
            session_data["conversation"] = conversation
        else:
            print("No session found for this user.")

    except Exception as e:
        print(f"An error occurred during the database connection: {str(e)}")
        traceback.print_exc()
    finally:
        if cursor is not None:
            cursor.close()
        
        if conn is not None:
            conn.close()

    return session_data


def get_user_sessions(user_id, source=None):
    
    sessions_data={}
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )

        cursor = conn.cursor(dictionary=True)

        if source == "WA":
            condition = "AND source = 'WA'"
        else:
            condition = "AND source is NULL"

    
        session_query = f"""
                            SELECT distinct(session_id)
                            FROM abcdchatdata 
                            WHERE user_id = %s {condition}
                            ORDER BY modified_at DESC;
                            """
        
        print("Session Query:", session_query)
        cursor.execute(session_query, (user_id,))
        results = cursor.fetchall()

        if results:
            sessions = [result['session_id'] for result in results]
            sessions_data["user_id"]=user_id
            sessions_data["sessions"]=sessions
        else:
            print("No session found for this user.")    
     
    except Exception as e:
        print(f"An error occurred during the database connection: {str(e)}")
        traceback.print_exc()
    finally:
        if cursor is not None:
            cursor.close()
        
        if conn is not None:
            conn.close()

    return sessions_data


def get_session_data(session_id):
    conversation=[]
    conn=None
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )

        cursor = conn.cursor(dictionary=True)

        try:
            
            conversation_query = """
                                SELECT query, gpt_response, response_id, context,created_at
                                FROM abcdchatdata
                                WHERE session_id = %s
                                ORDER BY modified_at ASC;
                                """
            print(conversation_query % (session_id))  # print the conversation query

            cursor.execute(conversation_query, (session_id,))
            results = cursor.fetchall()

            conversation = []
            for result in results:
                user_dict = {"timestamp":result['created_at'],"role": "user", "content": result['query']}
                
                
                context_data=eval(result['context'])
                
                #context_info=context_data
                sources_info=""
                if "contextInfo" in context_data and "sources" in context_data:
                    #context_info=context_data["contextInfo"]
                    sources_info=context_data["sources"]
                
                assistant_dict = {"timestamp":result['created_at'],"role": "assistant", "content": result['gpt_response'],"response_id":result['response_id'],"sources":sources_info}
                  
                conversation.append(user_dict)
                conversation.append(assistant_dict)
                
            print(conversation)
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

    return conversation


def get_user_conversations(user_id,session_id,conBuffWindow=3):
    
    conversation_string=""
    conn=None
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )

        cursor = conn.cursor()

        try:
            query = f"""
                        SELECT query, gpt_response
                        FROM abcdchatdata
                        WHERE user_id = %s
                        AND session_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s;
                    """
        
            cursor.execute(query,(user_id,session_id,conBuffWindow))

            results = cursor.fetchall()

            for i in range(len(results)-1, -1, -1):
                row=results[i]
                user_msg, bot_message = row
                conversation_string+=f"Human: {user_msg}"+"\n"
                conversation_string+=f"Bot: {bot_message}"+"\n"
                print("Conversation String:")
                print(conversation_string)
        
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
            
    return conversation_string


def log_analyze_section_feedback(user_id, session_id, section, response_id, feedback, feedback_note):
    
    conn=None
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        
        cursor = conn.cursor(dictionary=True)
        
        dt = datetime.now(timezone.utc)
        utc_timestamp = dt.timestamp()
        modified_at = datetime.utcfromtimestamp(utc_timestamp).strftime('%Y-%m-%dT%H:%M:%S%Z')
        
        if section:
            query = f"SELECT user_id, session_id, feedback FROM analyzer_data WHERE user_id = %s AND session_id = %s"
        
            cursor.execute(query,(user_id,session_id))
            rows = cursor.fetchall()
        
            if rows:
    
                result = rows[0]
            
                feedback_data = {
                    section : {"feedback": feedback, "feedback_note": str(feedback_note)}
                }
            
                feedback=result["feedback"]
            
                if feedback:
                    existing_feedback = json.loads(feedback)
                    existing_feedback.update(feedback_data)
                    feedback_data=existing_feedback
                # print("-"*50)
                # print(f"feedback data: {feedback_data}")
                # print("-"*50)
                feedback_data=json.dumps(feedback_data)
            
                query = """
                        UPDATE analyzer_data
                        SET feedback = %s, modified_at = %s
                        WHERE user_id = %s AND session_id = %s
                    """

                data = (feedback_data, modified_at, user_id, session_id)
                cursor.execute(query, data)
                conn.commit()      
                
                return {"message":"Added feedback successfully"}

            else:
                raise CustomException("Invalid user_id or session_id")

        elif response_id:
            
            check_query = """
                SELECT 1 FROM analyze_chat
                WHERE user_id = %s AND response_id = %s AND session_id = %s
            """
            
            cursor.execute(check_query, (user_id, response_id, session_id))
            result = cursor.fetchone()

            if not result:    
                raise CustomException("Invalid user_id, session_id or response_id")

            query = """
                        UPDATE analyze_chat
                        SET feedback = %s, feedback_note = %s, modified_at = %s
                        WHERE user_id = %s AND response_id = %s AND session_id = %s
                    """

            data = (feedback, feedback_note,modified_at,user_id,response_id,session_id)
            
            cursor.execute(query, data)
            conn.commit()
            
            return {"message":"Added feedback successfully"}
    
    except CustomException as excep:
        print(f"Invalid data: {str(excep)}")
        traceback.print_exc()
        raise excep
    except Exception as e:
        print(f"An error occurred while inserting feedback data: {str(e)}")
        traceback.print_exc()
        raise e
    finally:
        if conn is not None:
            conn.close()


def log_analyze_followup_response(user_id,session_id,query,response_id,response,context,section):
    conn=None
    cursor=None
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        cursor = conn.cursor()

        insertion_query = """
                    INSERT INTO analyze_chat
                    (user_id,session_id,query,response_id,gpt_response,context,section)
                    VALUES
                    (%s, %s, %s, %s, %s, %s, %s)
                """

        data = (user_id,session_id,query,response_id,response,context,section)
        cursor.execute(insertion_query, data)

        conn.commit()
        
    except Exception as e:
        print(f"DB error: {str(e)}")
        traceback.print_exc()
    finally:
        if cursor is not None:
            cursor.close()
        
        if conn is not None:
            conn.close()


def get_user_analyzer_conversations(user_id, session_id, section, conBuffWindow=4):
    conversation_string=""
    conn=None
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )

        cursor = conn.cursor()
        
        query = f"""
                    SELECT query, gpt_response
                    FROM analyze_chat
                    WHERE user_id = %s 
                    AND session_id = %s
                    AND section = %s
                    ORDER BY created_at DESC
                    LIMIT %s;
                """
    
        cursor.execute(query,(user_id,session_id,section,conBuffWindow))

        results = cursor.fetchall()

        for i in range(len(results)-1, -1, -1):
            row=results[i]
            user_msg, bot_message = row
            conversation_string+=f"Human: {user_msg}"+"\n"
            conversation_string+=f"Bot: {bot_message}"+"\n"
            print("Conversation String:")
            print(conversation_string)

    except Exception as e:
        print(f"An error occurred during the database connection: {str(e)}")
        traceback.print_exc()
    finally:
        if conn is not None:
            conn.close()
            
    return conversation_string


def add_session_title(user_id,session_id,session_title):
    conn=None
    cursor=None
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        cursor = conn.cursor()
        
        check_query = """
                SELECT 1 FROM analyzer_data
                WHERE user_id = %s AND  session_id = %s
            """
            
        cursor.execute(check_query, (user_id, session_id))
        result = cursor.fetchone()

        if not result:    
            raise CustomException("Invalid user_id or session_id")

        query = """
                    UPDATE analyzer_data
                    SET session_title = %s
                    WHERE user_id = %s AND session_id = %s
                """

        data = (session_title,user_id,session_id)
        cursor.execute(query, data)
        conn.commit()
        return {"message":"Added session title successfully"}
    
    except CustomException as exec:
        print(f"Invalid data: {str(exec)}")
        traceback.print_exc()
        raise exec
    except Exception as e:
        print(f"DB error: {str(e)}")
        traceback.print_exc()
        raise e
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


def add_custom_evaluator_session_title(user_id, session_id, session_title):
    conn=None
    cursor=None
    try:
        conn = connect_to_mysql()
        cursor = conn.cursor()
        check_query = """
                SELECT 1 FROM analyzer_custom_evaluator_data
                WHERE user_id = %s AND  session_id = %s
            """ 
        cursor.execute(check_query, (user_id, session_id))
        result = cursor.fetchone()
        if not result:    
            raise CustomException("Invalid user_id or session_id")
        
        query = """
                    UPDATE analyzer_custom_evaluator_data
                    SET session_title = %s
                    WHERE user_id = %s AND session_id = %s
                """
                
        data = (session_title,user_id,session_id)
        cursor.execute(query, data)
        conn.commit()
        return {"message":f"Added session title for session_id: {session_id}"}
    except CustomException as exec:
        print(f"Invalid data: {str(exec)}")
        traceback.print_exc()
        raise exec
    except Exception as e:
        print(f"DB error: {str(e)}")
        traceback.print_exc()
        raise e
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


def get_custom_evaluator_conversations(user_id,session_id):
    conversation=[]
    conn=None
    cursor=None
    try:
        conn = connect_to_mysql()

        cursor = conn.cursor(dictionary=True)

        conversation_query = """
                            SELECT query,gpt_response,section,created_at,response_id
                            FROM analyzer_custom_evaluator_chat
                            WHERE user_id = %s AND session_id = %s
                            ORDER BY created_at ASC;
                            """
        print(conversation_query % (user_id,session_id))  # print the conversation query

        cursor.execute(conversation_query, (user_id,session_id,))
        results = cursor.fetchall()

        conversation = []
        for result in results:
            user_dict = {"timestamp":result['created_at'],"role": "user", "content": result['query'],"section":result['section']}
            assistant_dict = {"timestamp":result['created_at'],"role": "assistant", "content": result['gpt_response'], "response_id":result['response_id']}
            
            conversation.append(user_dict)
            conversation.append(assistant_dict)
            
        print(conversation)
        
    except Exception as e:
        print(f"DB error: {str(e)}")
        traceback.print_exc()

    finally:
        if cursor is not None:
            cursor.close()
            
        if conn is not None:
            conn.close()

    return conversation


def get_custom_evaluator_conversations_new(user_id, session_ids):
    conversations = {}
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )

        cursor = conn.cursor(dictionary=True)

        # Create a string of placeholders for the IN clause
        placeholders = ', '.join(['%s'] * len(session_ids))
        
        conversation_query = f"""
                            SELECT query, gpt_response, section, created_at, response_id, session_id
                            FROM analyzer_custom_evaluator_chat
                            WHERE user_id = %s AND session_id IN ({placeholders})
                            ORDER BY session_id, created_at ASC;
                            """

        # Combine user_id and session_ids for the query parameters
        query_params = [user_id] + session_ids
        cursor.execute(conversation_query, tuple(query_params))
        results = cursor.fetchall()

        # Process results and organize by session_id
        for result in results:
            session_id = result['session_id']
            if session_id not in conversations:
                conversations[session_id] = []
            
            user_dict = {
                "timestamp": result['created_at'],
                "role": "user",
                "content": result['query'],
                "section": result['section']
            }
            assistant_dict = {
                "timestamp": result['created_at'],
                "role": "assistant",
                "content": result['gpt_response'],
                "response_id": result['response_id']
            }
            
            conversations[session_id].extend([user_dict, assistant_dict])

        return conversations
    except Exception as e:
        print(f"DB error: {str(e)}")
        traceback.print_exc()
        raise e
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


def get_custom_evaluator_data(user_id, session_id):
    conn=None
    cursor=None
    session_data = {}
    try:
        conn = connect_to_mysql()
        cursor = conn.cursor(dictionary=True)

        analysis_query = "SELECT user_id, proposal_pdf_name, proposal_text, tor_pdf_name, tor_text, generated_evaluator_comments, created_at, session_id FROM analyzer_custom_evaluator_data WHERE user_id = %s and session_id = %s"

        cursor.execute(analysis_query, (user_id, session_id))
         
        results = cursor.fetchall()
        if results:
            conversations = get_custom_evaluator_conversations(user_id,session_id)
            for row in results:
                generated_evaluator_comments = json.loads(row['generated_evaluator_comments'])
                session_data['user_id']=row['user_id']
                session_data['generated_evaluator_comments']=json.loads(row['generated_evaluator_comments'])
                session_data['conversations']=conversations
                session_data['created_at']=row['created_at']
                session_data['session_id']=row['session_id']
                
                session_data['proposal_pdf_name']=row['proposal_pdf_name']
                session_data['tor_pdf_name']=row['tor_pdf_name']
                session_data['proposal_text']=None
                session_data['tor_text']=None
                session_data['s3_par_url']=generated_evaluator_comments.get("s3_par_url", None)
                
                proposal_pdf_name = row['proposal_pdf_name']
                tor_pdf_name = row['tor_pdf_name']

                if not proposal_pdf_name:
                    session_data['proposal_text'] = row['proposal_text']
                if not tor_pdf_name:
                    session_data['tor_text'] = row['tor_text']
                if proposal_pdf_name:
                    session_data['s3_par_url']=s3_bucket_utils.get_file_par_url(proposal_pdf_name)
        else:
            raise CustomDBException("Invalid user_id, session_id or section.")
    except CustomDBException as exec:
        print(f"Invalid user_id or session_id: {str(exec)}")
        traceback.print_exc()
        raise exec
    except Exception as e:
        print(f"An error occurred during the database connection: {str(e)}")
        traceback.print_exc()
        raise e
    finally:
        if cursor is not None:
            cursor.close()    
        if conn is not None:
            conn.close()

    return session_data


def get_custom_evaluator_session_data(user_id, number_of_sessions=None, session_ids=None):
    conn = None
    cursor = None
    session_data = {}
    try:
        conn = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        cursor = conn.cursor(dictionary=True)
        
        # Fetch all sessions for the user
        query = """
        SELECT user_id, pdf_name, text, generated_analyze_comments, created_at, session_id 
        FROM analyzer_data 
        WHERE user_id = %s 
        """
        query = """
        SELECT user_id, proposal_pdf_name, proposal_text, tor_pdf_name, tor_text, generated_evaluator_comments, created_at, session_id 
        FROM analyzer_custom_evaluator_data
        """
        if user_id:
            query += "WHERE user_id = %s "
            params = (user_id,)
            query += " ORDER BY created_at DESC "  # Moved ORDER BY here
            if number_of_sessions:
                query += f"LIMIT {number_of_sessions} "
        elif session_ids:
            placeholders = ', '.join(['%s'] * len(session_ids))
            query += f"WHERE session_id IN ({placeholders}) "
            params = tuple(session_ids)
            query += " ORDER BY created_at DESC "  # Moved ORDER BY here
        else:
            raise ValueError("Either user_id or session_ids must be provided")
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        if not results:
            return []  # Return an empty list if no sessions found

        # Extract session_ids
        session_ids = [row['session_id'] for row in results]
        
        # Fetch conversations for all sessions
        conversations = get_custom_evaluator_conversations_new(user_id, session_ids)
        
        # Prepare the final response
        final_response = []
        for row in results:
            session_id = row['session_id']
            session_data = {
                'user_id': row['user_id'],
                'generated_evaluator_comments': json.loads(row['generated_evaluator_comments']),
                'conversations': conversations.get(session_id, []),
                'created_at': row['created_at'],
                'session_id': session_id,
                's3_par_url': "",
                "proposal_pdf_name": row['proposal_pdf_name'],
                "tor_pdf_name": row['tor_pdf_name'],
                "proposal_text": None,
                "tor_text": None
            }
            
            proposal_pdf_name = row['proposal_pdf_name']
            tor_pdf_name = row['tor_pdf_name']
            
            if not proposal_pdf_name:
                session_data['proposal_text'] = row['proposal_text']
            if not tor_pdf_name:
                session_data['tor_text'] = row['tor_text']
            if proposal_pdf_name:
                session_data['s3_par_url']=s3_bucket_utils.get_file_par_url(proposal_pdf_name)
            
            final_response.append(session_data)

        return final_response

    except Exception as e:
        print(f"An error occurred during the database connection: {str(e)}")
        traceback.print_exc()
        return []  # Return an empty list in case of error
    finally:
        if cursor is not None:
            cursor.close()    
        if conn is not None:
            conn.close()


def log_analyzer_custom_evaluator_data(
    user_id:Optional[str], 
    user_name:Optional[str],
    session_id:str, 
    proposal_pdf_name:str, 
    proposal_text:str,
    proposal_summary_text:str,
    nature_of_document:str,
    organization_id:str,
    tor_pdf_name:str, 
    tor_text:str,
    tor_summary_text:str, 
    generated_analyze_prompts:dict,
    generated_analyze_comments:dict,
    analyze_context_used:dict,
    generated_evaluator_prompts:dict,
    generated_evaluator_comments:dict, 
    time_taken: float,
    tokens_counter: dict
    ):
    conn=None
    cursor=None
    try:
        conn = connect_to_mysql()
        cursor = conn.cursor()

        insert_query = """INSERT INTO analyzer_custom_evaluator_data (
        user_id, user_name, session_id, proposal_pdf_name, proposal_text, proposal_summary_text, 
        nature_of_document, organization_id, tor_pdf_name, tor_text, tor_summary_text, 
        generated_analyze_prompts, generated_analyze_comments, analyze_context_used, 
        generated_evaluator_prompts, generated_evaluator_comments, time_taken, input_tokens, output_tokens, total_tokens
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """

        generated_analyze_prompts_json = json.dumps(generated_analyze_prompts)
        generated_analyze_comments_json = json.dumps(generated_analyze_comments)
        analyze_context_used_json = json.dumps(analyze_context_used)
        generated_evaluator_prompts_json = json.dumps(generated_evaluator_prompts)
        generated_evaluator_comments_json = json.dumps(generated_evaluator_comments)
        
        total_input_tokens = common_utils.total_tokens_count(tokens_counter[0])
        total_output_tokens = common_utils.total_tokens_count(tokens_counter[1])
        total_tokens = total_input_tokens + total_output_tokens
        tokens_counter[0]["total_tokens"] = total_input_tokens
        tokens_counter[1]["total_tokens"] = total_output_tokens
        total_tokens = {"total_tokens": total_tokens}
        total_tokens = json.dumps(total_tokens)
        
        input_tokens = None
        output_tokens = None
        if tokens_counter is not None:
            input_tokens = json.dumps(tokens_counter[0])
            output_tokens = json.dumps(tokens_counter[1])

        data_tuple = (
        user_id, user_name, session_id, proposal_pdf_name, proposal_text, proposal_summary_text,
        nature_of_document, organization_id, tor_pdf_name, tor_text, tor_summary_text,
        generated_analyze_prompts_json, generated_analyze_comments_json, analyze_context_used_json,
        generated_evaluator_prompts_json, generated_evaluator_comments_json, time_taken, input_tokens, output_tokens, total_tokens)

        cursor.execute(insert_query, data_tuple)
        conn.commit()
        print("Data logged successfully.")
    except Exception as err:
        #print(f"Error: {err}")
        traceback.print_exc() 
    finally:
        if cursor is not None:
            cursor.close()

        if conn is not None:
            conn.close()


def log_custom_evaluator_section_feedback(user_id,session_id,section,response_id,feedback,feedback_note):
    conn=None
    try:
        conn = connect_to_mysql()
        
        cursor = conn.cursor(dictionary=True)
        
        dt = datetime.now(timezone.utc)
        utc_timestamp = dt.timestamp()
        modified_at = datetime.utcfromtimestamp(utc_timestamp).strftime('%Y-%m-%dT%H:%M:%S%Z')
        
        if section:
        
            query = f"SELECT user_id, session_id, feedback FROM analyzer_custom_evaluator_data WHERE user_id = %s AND session_id = %s"
        
            cursor.execute(query,(user_id,session_id))
            rows = cursor.fetchall()
        
        
            if rows:
    
                result = rows[0]
            
                feedback_data = {
                    section : {"feedback": feedback, "feedback_note": str(feedback_note)}
                }
            
                feedback=result["feedback"]
            
                if feedback:
                    existing_feedback = json.loads(feedback)
                    existing_feedback.update(feedback_data)
                    feedback_data=existing_feedback

                print(feedback_data)
            
                feedback_data=json.dumps(feedback_data)
            
                query = """
                        UPDATE analyzer_custom_evaluator_data
                        SET feedback = %s
                        WHERE user_id = %s AND session_id = %s
                    """

                data = (feedback_data,user_id,session_id)
                cursor.execute(query, data)
                conn.commit()  
    
                
                return {"message":"Added feedback successfully"}
                
            else:
                raise CustomDBException("Invalid user_id or session_id")

        elif response_id:
            
            check_query = """
                SELECT 1 FROM analyzer_custom_evaluator_chat
                WHERE user_id = %s AND response_id = %s AND session_id = %s
            """
            
            cursor.execute(check_query, (user_id, response_id, session_id))
            result = cursor.fetchone()

            if not result:    
                raise CustomDBException("Invalid user_id, session_id or response_id")
            
            query = """
                        UPDATE analyzer_custom_evaluator_chat
                        SET feedback = %s, feedback_note = %s, modified_at = %s
                        WHERE user_id = %s AND response_id = %s AND session_id = %s
                    """

            data = (feedback, feedback_note,modified_at,user_id,response_id,session_id)
            
            cursor.execute(query, data)
            conn.commit()
            
            return {"message":"Added feedback successfully"}
    
    except CustomDBException as excep:
        print(f"Invalid data: {str(excep)}")
        traceback.print_exc()
        raise excep       
    except Exception as e:
        print(f"An error occurred while inserting feedback data: {str(e)}")
        traceback.print_exc()
        raise e
    finally:
        if conn is not None:
            conn.close()


def get_custom_evaluator_section_data(user_id,session_id,section):
    conn=None
    cursor=None
    section_comments = ""
    context_data = ""
    try:
        conn = connect_to_mysql()
        cursor = conn.cursor(dictionary=True)

        analysis_query = "SELECT generated_evaluator_comments FROM analyzer_custom_evaluator_data WHERE user_id = %s and session_id = %s"

        cursor.execute(analysis_query, (user_id, session_id))
        results = cursor.fetchall()
        
        if results:
            for row in results:
                generated_analyze_comments=json.loads(row['generated_evaluator_comments'])
                if section:
                    if section in generated_analyze_comments:
                        if section == "P_External":
                            section_comments = "\n".join(comment for comment in generated_analyze_comments[section]["comment"])
                        else:
                            section_comments = "\n".join(comment for comment in generated_analyze_comments[section])
                    else:
                        available_sections = list(generated_analyze_comments.keys())
                        error_message = f"Section {section} not found in generated evaluator comments. Available sections {available_sections}"
                        raise HTTPException(status_code=404, detail=error_message)
                else:
                    for section,section_data in generated_analyze_comments.items():
                        section_comments += "\n\n".join(comment for comment in section_data)
        else:
            raise CustomDBException("Invalid user_id, session_id or section")
    except CustomDBException as exec:
        print(f"Invalid user_id or session_id: {str(exec)}")
        traceback.print_exc()
        raise exec
    except Exception as e:
        print(f"An error occurred during the database connection: {str(e)}")
        traceback.print_exc()
        raise e
    finally:
        if cursor is not None:
            cursor.close()    
        if conn is not None:
            conn.close()
    return section_comments


def get_user_evaluator_conversations(user_id,session_id,section,conBuffWindow=4):
    conversation_string=""
    conn=None
    try:
        conn = connect_to_mysql()
        cursor = conn.cursor()
        query = f"""
                    SELECT query, gpt_response
                    FROM analyzer_custom_evaluator_chat
                    WHERE user_id = %s 
                    AND session_id = %s
                    AND section = %s
                    ORDER BY created_at DESC
                    LIMIT %s;
                """
        cursor.execute(query,(user_id,session_id,section,conBuffWindow))
        results = cursor.fetchall()
        for i in range(len(results)-1, -1, -1):
            row=results[i]
            user_msg, bot_message = row
            conversation_string+=f"Human: {user_msg}"+"\n"
            conversation_string+=f"Bot: {bot_message}"+"\n"
    except Exception as e:
        print(f"An error occurred during the database connection: {str(e)}")
        traceback.print_exc()
    finally:
        if conn is not None:
            conn.close()
    return conversation_string


def log_evaluator_followup_response(user_id,session_id,query,response_id,response,context,section):
    conn=None
    cursor=None
    try:
        conn = connect_to_mysql()
        cursor = conn.cursor()

        insertion_query = """
                    INSERT INTO analyzer_custom_evaluator_chat
                    (user_id,session_id,query,response_id,gpt_response,context,section)
                    VALUES
                    (%s, %s, %s, %s, %s, %s, %s)
                """

        data = (user_id,session_id,query,response_id,response,context,section)
        cursor.execute(insertion_query, data)

        conn.commit()
        
    except Exception as e:
        print(f"DB error: {str(e)}")
        traceback.print_exc()
    finally:
        if cursor is not None:
            cursor.close()
        
        if conn is not None:
            conn.close()


# Compare execution times
if __name__ == "__main__":
    prompt_string, prompt_for_customization, chunks, prompt_corpus, prompt_examples, wisdom_1, wisdom_2 = api_utils.get_prompts("P2", "Policy Document")
    print(f"Wisdom = {wisdom_1}, type = {type(wisdom_1)}")
    