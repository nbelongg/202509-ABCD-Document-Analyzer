import html
import time
import traceback
import uuid
from io import BytesIO
from fastapi import HTTPException
from datetime import datetime
import common_utils
import db_utils
import filemapping_utils
import gpt_utils
import openai
import pdf_utils
import pinecone_utils
import s3_bucket_utils
import pydantic_schemas
from custom_exception import CustomException
import os
import json
import nest_asyncio
import docx
import encrypter
import concurrent.futures
from functools import partial
import asyncio
from logger import api_logger as logger
from langsmith import traceable
from analyzer_utils import pzero_utils
from admin_apis import db_operations

#nest_asyncio.apply()

use_summary = True
use_example = True

pdf_mappings = filemapping_utils.get_pdf_mappings()

text_extraction = os.getenv("text_extraction")


def process_input(pdf_file, text_input, user_id, organization_id=None, text_extraction="PDF Reader"):
    start_time = time.perf_counter()
    text = ""
    file_name = ""
    s3_par_url = ""

    try:
        if pdf_file:
            file_name = pdf_file.filename
            if organization_id:
                organization_id = organization_id.strip()
                file_name = f"{organization_id}/{file_name}"

            file_content = pdf_file.file.read()

            if file_name.lower().endswith(".pdf"):
                logger.info(f"Processing PDF file: {file_name}")
                s3_par_url = s3_bucket_utils.upload_fileobj_to_s3(BytesIO(file_content), file_name)
                
                if text_extraction == "PDF Reader":
                    text = pdf_utils.extract_text_from_pdf(BytesIO(file_content))
                else:
                    logger.info("Using LlamaParse for text extraction...")
                    text = pdf_utils.extract_text_llama_parse(file_content, file_name)
                
            elif file_name.lower().endswith(".txt"):
                logger.info(f"Processing text file: {file_name}")
                text = file_content.decode("utf-8")
                s3_par_url = s3_bucket_utils.upload_fileobj_to_s3(BytesIO(file_content), file_name)
                
            elif file_name.lower().endswith(".docx"):
                logger.info(f"Processing docx file: {file_name}")
                doc = docx.Document(BytesIO(file_content))
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                s3_par_url = s3_bucket_utils.upload_fileobj_to_s3(BytesIO(file_content), file_name)
            
            else:
                logger.warning(f"Unsupported file type: {file_name}")
                return "", "", ""

        elif text_input:
            logger.info("Processing text input")
            text = html.escape(text_input)
            current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            file_name = f"{user_id}_{current_timestamp}.txt"
            file_directory = "s3_temp_files"

            os.makedirs(file_directory, exist_ok=True)

            file_path = os.path.join(file_directory, file_name)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(text)

            # Upload the text file to S3
            with open(file_path, 'rb') as file:
                s3_par_url = s3_bucket_utils.upload_fileobj_to_s3(file, file_name)

            # Remove the temporary file
            os.remove(file_path)

        else:
            logger.error("No input provided")
            raise HTTPException(status_code=400, detail="Either a file or text input must be provided.")

    except Exception as e:
        logger.error(f"Error processing input: {str(e)}")
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An error occurred while processing the input.")

    processing_time = time.perf_counter() - start_time
    logger.info(f"Input processed in {processing_time:.2f} seconds")

    return text, file_name, s3_par_url


@traceable(tags=["analyzer"])
def generate_analyze_comments(api_key, api_secret, user_id, user_name, pdf_file, text_input, nature_of_document, user_role, prompt_labels, organization_id=None, showcase_items=10):
    try:
        start_time = time.time()
        
        logger.info("Validating request...")
        if not common_utils.validate_request(api_secret=api_secret.strip(), api_key=api_key.strip()):
            raise HTTPException(status_code=400, detail="Invalid API secret or key")


        logger.info("Validating input...")
        if (pdf_file is None and text_input is None) or (pdf_file and text_input):
            raise HTTPException(status_code=400, detail="Either a file or text input must be provided.")


        logger.info("Starting analysis process...")
        session_id = str(uuid.uuid4())
        nature_of_document = nature_of_document.value
        user_role = user_role.value
        gpt4_model = pydantic_schemas.Model.o3
        tokens_counter = common_utils.get_analyzer_tokens_counter()
        
        
        prompt_labels = prompt_labels[0].split(',') if prompt_labels else ["P1", "P2", "P3", "P4", "P5"]
        common_utils.validate_prompt_labels(prompt_labels)
        
        
        logger.info("Getting Pinecone analyzer filters...")
        pinecone_analyze_filters = pinecone_utils.get_pinecone_analyzer_filters()
        
        
        logger.info("Processing input...")
        text, file_name, s3_par_url = process_input(pdf_file, text_input, user_id, organization_id)
        
        
        logger.info("Generating summary...")
        summary_text = gpt_utils.get_summary(text, gpt4_model, nature_of_document)
        
        
        logger.info("Getting prompt label flows...")
        prompt_label_flows, prompt_label_flows_dependencies = db_utils.get_prompt_label_flows(prompt_labels, nature_of_document)
        logger.info(f"Prompt label flows: {prompt_label_flows}")
        logger.info(f"Prompt label flows dependencies: {prompt_label_flows_dependencies}")
        
        
        logger.info("Getting custom analyzer prompts...")
        selected_prompts, prompt_section_titles, prompt_dependencies = db_utils.get_analyzer_prompts_multithreaded(prompt_labels, nature_of_document, user_role)
        
        
        logger.info("Filtering dependencies...")
        prompt_dependencies = common_utils.filter_dependencies(prompt_labels, prompt_dependencies)
        prompt_dependencies = common_utils.update_parent_dependencies(prompt_dependencies, prompt_label_flows_dependencies)
        logger.info(f"Prompt dependencies: {prompt_dependencies}")
        
        
        logger.info("Topological sorting...")
        selected_prompts_ordered = common_utils.topological_sort(prompt_dependencies)
        logger.info(f"Prompt order: {selected_prompts_ordered}")
        
        
        logger.info("Generating prompts...")
        generated_prompts, generate_prompt_input_token, generate_prompt_output_token = gpt_utils.generate_prompts(gpt4_model, nature_of_document, summary_text, text, use_example, selected_prompts)
        
        
        logger.info("Generating analyze comments...")
        generated_analyze_comments,analyze_context_used,prompt_label_flow,relevant_chunks,collected_data, analyze_comments_input_token,analyze_comments_output_token =gpt_utils.generate_analyze_comments_dependencies(text,nature_of_document,"",summary_text,pinecone_analyze_filters,gpt4_model,generated_prompts, organization_id, prompt_section_titles, prompt_dependencies,selected_prompts_ordered, prompt_label_flows)
        
        
        logger.info("Generating p0 summary...")
        p0_summary, summary_prompt = pzero_utils.get_pzero_summary(generated_analyze_comments, nature_of_document, gpt4_model, user_role)
        
        
        logger.info("Generating showcase chunks...")
        showcase = pinecone_utils.extract_unique_showcase_chunks(p0_summary, showcase_items)
        showcase = common_utils.convert_showcase_to_s3_urls(showcase)
        
        
        generated_analyze_comments["session_id"]=session_id
        generated_analyze_comments["pdf_name"] = file_name
        generated_analyze_comments["s3_par_url"]=s3_par_url
        generated_analyze_comments["P0"]=p0_summary
        generated_analyze_comments["showcase"]=showcase


        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Total time taken: {total_time:.2f} seconds")
        
        
        logger.info("Logging analysis response...")
        db_utils.log_analysis_response(
            user_id=user_id,
            user_name=user_name,
            session_id=session_id,
            pdf_name=file_name,
            text=text,
            nature_of_document=nature_of_document,
            prompt_labels=prompt_labels,
            summary_text=summary_text,
            generated_prompts=generated_prompts,
            pinecone_filters=pinecone_analyze_filters,
            generated_analyze_comments=generated_analyze_comments,
            analyze_context_used=analyze_context_used,
            time_taken=total_time,
            organization_id=organization_id,
            tokens_counter=tokens_counter
        )
        
        
        return generated_analyze_comments
    except openai.RateLimitError as e:
        print("OpenAI API ratelimit error: " + str(e))
        raise HTTPException(status_code=500, detail="OpenAI API request exceeded rate limit")
    except openai.APITimeoutError as e:
        print("OpenAI API Timeout error: " + str(e))
        raise HTTPException(status_code=500, detail="OpenAI API Timeout")
    except openai.APIConnectionError as e:
        print("OpenAI API Connection error: " + str(e))
        raise HTTPException(status_code=500, detail="OpenAI API Connection error")
    except openai.APIError as e:
        print("OpenAI API error: " + str(e))
        raise HTTPException(status_code=500, detail="OpenAI API Error") 
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")  


@traceable(tags=["evaluator"])
def generate_evaluator_comments(user_id, user_name, nature_of_document, organization_id, org_guideline_id, proposal_pdf_file, proposal_text_input, tor_pdf_file, tor_text_input):
    try:
        logger.info("Validating user input...")
        if (proposal_pdf_file is None and proposal_text_input is None) or (proposal_pdf_file and proposal_text_input):
            raise HTTPException(status_code=400, detail="Either a proposal_pdf_file or proposal_text_input must be provided.")
        if (tor_pdf_file and tor_text_input):
            raise HTTPException(status_code=400, detail="Both TOR Pdf and TOR Text Uploaded.")

        logger.info("Starting analysis process...")
        overall_start_time = time.time()
        
        session_id = str(uuid.uuid4())
        prompt_labels=["P1", "P2", "P3", "P4", "P5"]
        nature_of_document = nature_of_document.value
        gpt3_model = pydantic_schemas.Model.o3
        gpt4_model = pydantic_schemas.Model.o3
        summary_pdf_text_length = int(os.getenv("summary_pdf_text_length"))
        evaluator_tokens_counter = common_utils.get_evaluator_tokens_counter()
        
        prompt_labels=db_utils.get_associated_prompt_label(nature_of_document)

        pinecone_analyze_filters = pinecone_utils.get_pinecone_analyzer_filters()
        
        evaluator_prompts = db_operations.get_all_evaluator_prompts_from_db(doc_type=nature_of_document,organization_id=organization_id, org_guideline_id=org_guideline_id)
        
        prompts_configurations={}
        prompt_flows = {'P_Internal':{}}
        evaluator_prompt_labels = []
        evaluator_section_titles = {}
        evaluator_prompt_dependencies = {}
        for prompt in evaluator_prompts:        
            prompt_label = prompt.get("prompt_label")
            base_prompt = prompt.get("base_prompt")
            customization_prompt = prompt.get("customization_prompt")
            wisdom_1 = prompt.get("wisdom_1")
            wisdom_2 = prompt.get("wisdom_2")
            chunks = prompt.get("chunks")
            sec_title =  prompt.get("section_title", "")
            dependencies = prompt.get("additional_dependencies","")
            customize_prompt_based_on = prompt.get("customize_prompt_based_on","")
            send_along_customised_prompt = prompt.get("send_along_customised_prompt","")
            wisdom_received = prompt.get("wisdom_received","")
            llm_flow = prompt.get("LLM_Flow","")
            llm = prompt.get("LLM","")
            model = prompt.get("Model","")
            show_on_frontend = prompt.get("show_on_frontend","")
            label_for_output = prompt.get("label_for_output","")
            if wisdom_1 is None:
                wisdom_1 = ""
                
            if wisdom_2 is None:
                wisdom_2 = ""
            
            if sec_title is None:
                sec_title = ""
            if not dependencies or dependencies == "":
                dependencies = []
            else:
                dependencies = [item.strip() for item in dependencies.split(",")]
                
            if not wisdom_received or wisdom_received == "":
                wisdom_received = []
            else:
                wisdom_received = [item.strip() for item in wisdom_received.split(",")]
                
            if not customize_prompt_based_on or customize_prompt_based_on == "":
                customize_prompt_based_on = []
            else:
                customize_prompt_based_on = [item.strip() for item in customize_prompt_based_on.split(",")]
                
            if not send_along_customised_prompt or send_along_customised_prompt == "":
                send_along_customised_prompt = []
            else:
                send_along_customised_prompt = [item.strip() for item in send_along_customised_prompt.split(",")]
            
            if not llm or llm == "":
                llm = "ChatGPT"
            
            if not model or model == "":
                model = "gpt-4o-2024-08-06"
            
            if('.F1' in prompt_label or '.F2' in prompt_label):
                flow = {}
                flow[prompt_label]={"base_prompt":base_prompt,"customization_prompt":customization_prompt,"wisdom_1":wisdom_1, "wisdom_2": wisdom_2,"chunks":chunks,"wisdom_received":wisdom_received,"additional_dependencies":dependencies,"customize_prompt_based_on":customize_prompt_based_on,"send_along_customised_prompt":send_along_customised_prompt, "show_on_frontend":show_on_frontend,"label_for_output":label_for_output,"llm_flow":llm_flow,"llm":llm,"model":model}
                prompt_flows['P_Internal'].update(flow)
            else:
                if prompt_label == 'P_Internal':
                    evaluator_prompt_labels.append(prompt_label)
                    evaluator_section_titles[prompt_label] = sec_title
                    evaluator_prompt_dependencies[prompt_label] = dependencies
                    prompts_configurations[prompt_label]={"base_prompt":base_prompt,"customization_prompt":customization_prompt,"wisdom_1":wisdom_1, "wisdom_2": wisdom_2,"chunks":chunks,"customize_prompt_based_on":customize_prompt_based_on,"send_along_customised_prompt":send_along_customised_prompt,"wisdom_received":wisdom_received, "llm_flow":llm_flow,"llm":llm,"model":model, "show_on_frontend":show_on_frontend,"label_for_output":label_for_output}
                

        logger.info("Processing Proposal Pdf...")
        proposal_text, proposal_filename, s3_par_url = process_input(proposal_pdf_file, proposal_text_input, user_id)
        
        logger.info("Generating proposal summary...")
        proposal_summary_text = gpt_utils.get_summary(proposal_text, gpt4_model, nature_of_document)       

        evaluator_prompt_dependencies = common_utils.filter_dependencies(evaluator_prompt_labels, evaluator_prompt_dependencies)
        
        evaluator_prompts_ordered = common_utils.topological_sort(evaluator_prompt_dependencies)
        
        logger.info("Generating Evaluator Prompts")      
        generated_evaluator_prompts, evaluator_input_tokens, evaluator_output_tokens = gpt_utils.generate_evaluator_prompts(gpt4_model,nature_of_document,"","",prompts_configurations)
        generated_evaluator_prompts["P_Internal"]["wisdom_received"] = prompts_configurations["P_Internal"]["wisdom_received"]
        logger.info("Evaluator Prompts Generated Successfully")
        
        analyze_model = gpt4_model
        
        logger.info("Generating Evaluator Comments")
        generated_evaluator_comments, p_internal_context, p_internal_child_flows,evaluator_comments_input_tokens, evaluator_comments_output_tokens =gpt_utils.generate_evaluate_comments_multiflow(proposal_text,proposal_summary_text,nature_of_document,"",prompt_flows,analyze_model,generated_evaluator_prompts,evaluator_section_titles, evaluator_prompt_dependencies, evaluator_prompts_ordered)
        logger.info("Evaluator Comments Generated")
        
        generated_evaluator_comments["session_id"]=session_id
        generated_evaluator_comments["s3_par_url"]=s3_par_url
        
        total_time = time.time() - overall_start_time
        
        logger.info("Logging analysis response...")
        db_utils.log_analyzer_custom_evaluator_data(
            user_id=user_id,
            user_name=user_name,
            session_id=session_id,
            proposal_pdf_name=proposal_filename,
            proposal_summary_text=proposal_summary_text,
            proposal_text=proposal_text,
            nature_of_document=nature_of_document,
            organization_id=organization_id,
            tor_pdf_name="",
            tor_text="",
            tor_summary_text="",
            generated_analyze_prompts="",
            generated_analyze_comments="",
            analyze_context_used="",
            generated_evaluator_prompts=generated_evaluator_prompts,
            generated_evaluator_comments=generated_evaluator_comments,
            time_taken=total_time,
            tokens_counter=evaluator_tokens_counter
        )

        return generated_evaluator_comments
    except openai.RateLimitError as e:
        print("OpenAI API rate limit error: " + str(e))
        raise HTTPException(status_code=500, detail="OpenAI API request exceeded rate limit")
    except openai.APITimeoutError as e:
        print("OpenAI API Timeout error: " + str(e))
        raise HTTPException(status_code=500, detail="OpenAI API Timeout")
    except openai.APIConnectionError as e:
        print("OpenAI API Connection error: " + str(e))
        raise HTTPException(status_code=500, detail="OpenAI API Connection error")
    except openai.APIError as e:
        traceback.print_exc()
        print("OpenAI API error: " + str(e))
        raise HTTPException(status_code=500, detail="OpenAI API Error") 
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")


def get_analyze_sessions(user_id, api_key, api_secret):
    if not common_utils.validate_request(api_secret=api_secret.strip(),api_key=api_key.strip()):
        raise HTTPException(status_code=400, detail="Invalid API secret or key")
    
    if not db_utils.db_exists():
        raise HTTPException(status_code=500, detail="Internal DB error")
    
    session_data=db_utils.get_analysis_sessions(user_id.strip())

    return session_data


def get_evaluator_sessions(user_id):
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID must be provided.")
    try:
        session_data = db_utils.get_custom_evaluator_sessions(user_id)
        return session_data
    except Exception as e:
        raise e


def get_analyze_session_data(user_id, session_id, api_key, api_secret):
    if not common_utils.validate_request(api_secret=api_secret.strip(),api_key=api_key.strip()):
        raise HTTPException(status_code=400, detail="Invalid API secret or key")
    
    session_data=db_utils.get_analysis_data(user_id.strip(),session_id.strip())
    
    if not session_data:
        raise HTTPException(status_code=400, detail="Invalid user_id or session_id")
    
    return session_data


def get_analyze_session_with_user_id(user_id, number_of_sessions, session_ids):    
    logger.info(f"Received request for user_id: {user_id}, number_of_sessions: {number_of_sessions}, session_ids: {session_ids}")
    if not user_id and not session_ids:
        raise HTTPException(status_code=400, detail="Either user_id or session_ids must be provided.")
    
    if user_id and session_ids:
        raise HTTPException(status_code=400, detail="Both user_id and session_ids cannot be provided.")
    
    start_time = time.time()
    session_data=db_utils.get_analyzer_session_data(user_id, number_of_sessions, session_ids)
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total time taken to get session data: {total_time}")
    
    if not session_data:
        raise HTTPException(status_code=400, detail="Invalid user_id or session_id")
    
    return session_data

def generate_analyze_followup_response(user_id, session_id, query, section, api_key, api_secret):
    
    allowed_sections = ["P_Custom", "P0", "P1", "P2", "P3", "P4", "P5"]
    
    if section is not None and section not in allowed_sections:
        raise HTTPException(status_code=400, detail="Invalid section value. Allowed values are P_Custom, P0, P1, P2, P3, P4, P5.")
       
    if not common_utils.validate_request(api_secret=api_secret.strip(), api_key=api_key.strip()):
        raise HTTPException(status_code=400, detail="Invalid API secret or key")
    
    conversation_string = db_utils.get_user_analyzer_conversations(user_id, session_id, section)
    
    section_comments, section_context = db_utils.get_analysis_section_data(user_id, session_id, section)
    
    if section_comments:
        context = f"""{section_context}\n\n{section_comments}"""
        
        try:
            response=gpt_utils.get_section_query_response(query, context, conversation_string) 
        
            response_id=str(uuid.uuid4())
        
            db_utils.log_analyze_followup_response(user_id, session_id, query, response_id, response, context, section)
        
            return {"user_id": user_id,"session_id": session_id,"section":section,"query": query,"response": response,"response_id":response_id}
        except openai.RateLimitError as e:
            print("OpenAI API ratelimit error:"+str(e))
            raise HTTPException(status_code=500, detail="OpenAI API request exceeded rate limit")
        except openai.APITimeoutError as e:
            print("OpenAI API Timeout error:"+str(e))
            raise HTTPException(status_code=500, detail="OpenAI API Timeout")
        except openai.APIConnectionError as e:
            print("OpenAI API Connection error:"+str(e))
            raise HTTPException(status_code=500, detail="OpenAI API Connection error")
        except openai.APIError as e:
            print("OpenAI API error:"+str(e))
            raise HTTPException(status_code=500, detail="OpenAI API Error")
        except Exception as e:
            print("Exception:"+str(e))
            raise HTTPException(status_code=500, detail="Internal Error")
    else:
        raise HTTPException(status_code=400, detail="Invalid section, user id or session id")   


def generate_evaluator_followup_response(user_id, session_id, query, section):
    allowed_sections = ["P_Internal", "P_External", "P_Delta"]
    if section is not None and section not in allowed_sections:
        raise HTTPException(status_code=400, detail="Invalid section value. Allowed values are P_Internal, P_External, P_Delta.")
    try:
        section_comments=db_utils.get_custom_evaluator_section_data(user_id,session_id,section)
        conversation_string=db_utils.get_user_evaluator_conversations(user_id,session_id,section)
        
        if section_comments:
            context = f"""{section_comments}"""
            response=gpt_utils.get_analyzer_section_query_response(query, context, conversation_string) 
            response_id=str(uuid.uuid4())
            
            db_utils.log_evaluator_followup_response(user_id,session_id,query,response_id,response,context,section)
            
            return {"user_id": user_id,"session_id": session_id,"section":section,"query": query,"response": response,"response_id":response_id}
        else:
            raise HTTPException(status_code=400, detail="Invalid section, user id or session id")   
    except openai.RateLimitError as e:
        print("OpenAI API ratelimit error:"+str(e))
        raise HTTPException(status_code=500, detail="OpenAI API request exceeded rate limit")
    except openai.APITimeoutError as e:
        print("OpenAI API Timeout error:"+str(e))
        raise HTTPException(status_code=500, detail="OpenAI API Timeout")
    except openai.APIConnectionError as e:
        print("OpenAI API Connection error:"+str(e))
        raise HTTPException(status_code=500, detail="OpenAI API Connection error")
    except openai.APIError as e:
        print("OpenAI API error:"+str(e))
        raise HTTPException(status_code=500, detail="OpenAI API Error")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error connecting to database. Please try again later.") 


def add_analyze_section_feedback(user_id,session_id,feedback,feedback_note,response_id,section,api_key,api_secret):
    
    if not common_utils.validate_request(api_secret=api_secret.strip(),api_key=api_key.strip()):
        raise HTTPException(status_code=400, detail="Invalid API secret or key")
    
    if not response_id and not section:
        raise HTTPException(status_code=400, detail="Either of response_id or section is required for feedback")
    
    try:
        response=db_utils.log_analyze_section_feedback(user_id, session_id, section, response_id, feedback, feedback_note)
        return response
    except CustomException as exec:
        print(f"An error occurred while adding feedback: {str(exec)}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exec))
    except Exception as e:
        print(f"An error occurred while adding feedback: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Error while adding feedback")


def add_evaluator_section_feedback(user_id, session_id, feedback, feedback_note, response_id, section):
    if not response_id and not section:
        raise HTTPException(status_code=400, detail="Either of response_id or section is required for feedback")
    try:
        response = db_utils.log_custom_evaluator_section_feedback(user_id, session_id, section, response_id, feedback, feedback_note)
        return response
    except Exception as e:
        print(f"An error occurred while adding feedback: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Error while adding feedback")


def get_custom_evaluator_session_data_with_user_id(user_id, number_of_sessions, session_ids):    
    logger.info(f"Received request for user_id: {user_id}, number_of_sessions: {number_of_sessions}, session_ids: {session_ids}")
    if not user_id and not session_ids:
        raise HTTPException(status_code=400, detail="Either user_id or session_ids must be provided.")
    
    if user_id and session_ids:
        raise HTTPException(status_code=400, detail="Both user_id and session_ids cannot be provided.")
    
    start_time = time.time()
    session_data=db_utils.get_custom_evaluator_session_data(user_id, number_of_sessions, session_ids)
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total time taken to get session data: {total_time}")
    
    if not session_data:
        raise HTTPException(status_code=400, detail="Invalid user_id or session_id")
    
    return session_data


def add_analyze_session_title(user_id,session_id,session_title,api_key,api_secret):
    if not common_utils.validate_request(api_secret=api_secret.strip(),api_key=api_key.strip()):
        raise HTTPException(status_code=400, detail="Invalid API secret or key")
    
    try:
        response=db_utils.add_session_title(user_id,session_id,session_title)
        return response
    except CustomException as exec:
        print(f"An error occurred: {str(exec)}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exec))
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error. Could not change session title")
    
    
def add_custom_evaluator_session_title(user_id, session_id, session_title):
    try:
        response = db_utils.add_custom_evaluator_session_title(user_id, session_id, session_title)
        return response
    except Exception as e:
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    # Read a normal pdf file
    file_name = "Test_pdf_encrtpted.pdf"
    with open("./pdf_files/Test_pdf.pdf", "rb") as file:
        original_file_content = file.read()
        
    # Encrypt the file
    encrypted_file_content = encrypter.encrypt_file(original_file_content)
    s3_par_url = s3_bucket_utils.upload_fileobj_to_s3(BytesIO(encrypted_file_content), file_name)
    print(f"Encrypted file uploaded to S3: {s3_par_url}")
    
    with open("./pdf_files/Test_pdf_encrypted.pdf", "wb") as file:
        file.write(encrypted_file_content)
    
    # Decrypt the file
    # with open("./pdf_files/Test_pdf_encrypted.pdf", "rb") as file:
    #     encrypted_file_content = file.read()
    #     decrypted_file_content = encrypter.decrypt_file(encrypted_file_content)
    # with open("./pdf_files/Test_pdf_decrypted.pdf", "wb") as file:
    #     file.write(decrypted_file_content)
        
    # s3_par_url = s3_bucket_utils.upload_fileobj_to_s3(BytesIO(encrypted_file_content), file_name)
    # print(f"Decrypted file uploaded to S3: {s3_par_url}")
        
    
