import sys
sys.path.append('.')
sys.path.append('..')
from typing import List
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, Header, Depends, Request, HTTPException
from typing import Literal
import chatbot_handlers,custom_analyzer_handlers,pydantic_schemas
from db_utils import *
from common_utils import *
import db_utils
import traceback
import os
from dotenv import load_dotenv
from logger import api_logger as logger
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
import requests
from abcd_collateral_analyzer_api import collateral_router


load_dotenv(override=True)


API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
URL = os.getenv("URL")


def verify_api_credentials(api_key: str = Header(...), api_secret: str = Header(...)):
    if api_key != API_KEY or api_secret != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API key or API secret")
    return True


app = FastAPI()


@app.middleware("http")
async def log_response_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Request: {request.url.path} completed in {process_time:.4f} seconds")
    return response

@app.post("/feedback", tags=["Chatbot"])
def response_feedback(responseObj: pydantic_schemas.Response, api_key: str = Header(...), api_secret: str = Header(...)):
    try:
        return chatbot_handlers.add_feedback(responseObj,api_key,api_secret)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()        
        raise HTTPException(status_code=500, detail=f"Error in generating feedback {str(e)}")


@app.get("/get_last_session", tags=["Chatbot"])
def get_last_session(user_id: str,source: str = None,api_key: str = Header(...), api_secret: str = Header(...)):
    try:
        return chatbot_handlers.get_last_session(user_id,source,api_key,api_secret)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in getting last Session Details {str(e)}")


@app.get("/get_sessions", tags=["Chatbot"])
def get_sessions(user_id: str,source: str = None,api_key: str = Header(...), api_secret: str = Header(...)):
    try:
        return chatbot_handlers.get_sessions(user_id,source,api_key, api_secret)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in getting all Sessions  {str(e)}")


@app.get("/get_session_chat", tags=["Chatbot"])
def get_session_chat(session_id: str,api_key: str = Header(...), api_secret: str = Header(...)):
    try:
        return chatbot_handlers.get_session_chat(session_id,api_key,api_secret)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in getting Session chat  {str(e)}")   

    
@app.post("/get_answer", tags=["Chatbot"])
def generate_answer(chat_data: pydantic_schemas.Chat, api_key: str = Header(...), api_secret: str = Header(...)):
    try:
        return chatbot_handlers.generate_response(chat_data,api_key,api_secret)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in generating Answer  {str(e)}")      


@app.post("/analyze", tags=["Analyzer"], dependencies=[Depends(verify_api_credentials)])
def generate_analyze_comments(
        session_id: str = None,
        user_id: str = Form(...),
        user_name: Optional[str] = Form(None),
        pdf_file: UploadFile = File(None),
        text_input: Optional[str] = Form(None),
        nature_of_document: pydantic_schemas.newDocumentType = Form(pydantic_schemas.newDocumentType.Program_Design_Document),
        user_role: pydantic_schemas.UserRole = Form(pydantic_schemas.UserRole.Impact_Consultant),
        organization_id: Optional[str] = Form(None),
        showcase_items: Optional[int] = Form(10),
        prompt_labels: List[str] = Form(["P1", "P2", "P3", "P4", "P5"])):
    try:
        response = custom_analyzer_handlers.generate_analyze_comments(session_id, user_id, user_name, pdf_file, text_input, nature_of_document, user_role, prompt_labels, organization_id=organization_id, showcase_items=showcase_items)
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating Analyse Comments  {str(e)}")  
    

@app.post("/get_analyze_sessions", tags=["Analyzer"])
def get_analyze_sessions(user_id:str,api_key: str = Header(...), api_secret: str = Header(...)):
    try:
        return custom_analyzer_handlers.get_analyze_sessions(user_id,api_key,api_secret)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in getting  analyse sessions  {str(e)}")  


@app.post("/get_analyze_session_data", tags=["Analyzer"])
def get_analyze_session_data(user_id:str,session_id:str,api_key: str = Header(...), api_secret: str = Header(...)):
    try:
        return custom_analyzer_handlers.get_analyze_session_data(user_id,session_id,api_key,api_secret)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in getting analyse sessions details {str(e)}") 


@app.post("/get_analyze_session_with_user_id", tags=["Analyzer"], dependencies=[Depends(verify_api_credentials)])
def get_analyze_session_with_user_id(user_id: Optional[str] = None, number_of_sessions: Optional[int] = None, session_ids: List[str]=[]):
    return custom_analyzer_handlers.get_analyze_session_with_user_id(user_id, number_of_sessions, session_ids)


@app.post("/analyze_followup", tags=["Analyzer"])
def analyze_followup(
    user_id:str,
    session_id:str,
    query:str,
    section: Optional[str] = None,
    api_key: str = Header(...), 
    api_secret: str = Header(...)
    ):
    try: 
        return custom_analyzer_handlers.generate_analyze_followup_response(user_id, session_id, query, section, api_key, api_secret)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in generating followup response {str(e)}")


@app.post("/analyze_section_feedback", tags=["Analyzer"])
def analyze_section_feedback(
    user_id:str,
    session_id:str,
    feedback:bool,
    feedback_note: Optional[str] = None,
    response_id: Optional[str] = None,
    section: Optional[Literal["P_Custom", "P0", "P1", "P2", "P3", "P4", "P5"]] = None,
    api_key: str = Header(...), 
    api_secret: str = Header(...)
    ):
    try:
        return custom_analyzer_handlers.add_analyze_section_feedback(user_id, session_id, feedback, feedback_note, response_id, section, api_key, api_secret)       
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in getting getting analyse feedback details {str(e)}")   


@app.post("/add_session_title", tags=["Analyzer"])
def add_session_title(
    user_id:str,
    session_id:str,
    session_title:str,
    api_key: str = Header(...), 
    api_secret: str = Header(...)
    ):
    try:
        return custom_analyzer_handlers.add_analyze_session_title(user_id, session_id, session_title, api_key, api_secret)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error adding session title {str(e)}")


@app.post("/get_organization_id", tags=["Custom Evaluator"], dependencies=[Depends(verify_api_credentials)])
def get_organization_and_org_guideline_id():
    try:
        response = db_utils.fetch_org_guidelines()
        if response:
            return response
        else:
            raise HTTPException(status_code=404, detail="No organization_id found.")
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in fetching organization_id {str(e)}")


@app.post("/custom_evaluator",tags=["Custom Evaluator"], dependencies=[Depends(verify_api_credentials)])
def custom_evaluator(
    session_id: str = Form(None),
    user_id: str = Form(...),
    user_name: Optional[str] = Form(None),
    nature_of_document: pydantic_schemas.newDocumentType = Form(pydantic_schemas.newDocumentType.Program_Design_Document),
    organization_id: Optional[str] = Form(None),
    org_guideline_id: Optional[str] = Form(None),
    proposal_pdf_file: UploadFile = File(None),
    proposal_text_input: Optional[str] = Form(None),
    tor_pdf_file: UploadFile = File(None),
    tor_text_input: Optional[str] = Form(None)
    ):
    try:
        response = custom_analyzer_handlers.generate_evaluator_comments(session_id, user_id, user_name, nature_of_document, organization_id, org_guideline_id, proposal_pdf_file, proposal_text_input, tor_pdf_file, tor_text_input)
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in generating evaluator comments {str(e)}")


@app.post("/get_custom_evaluator_sessions",tags=["Custom Evaluator"], dependencies=[Depends(verify_api_credentials)])
def get_evaluator_sessions(
    user_id: str,
    ):
    try:
        sessions = custom_analyzer_handlers.get_evaluator_sessions(user_id.strip())
        return {"sessions":sessions}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in fetching organization_id {str(e)}")


@app.post("/get_custom_evaluator_session_data",tags=["Custom Evaluator"], dependencies=[Depends(verify_api_credentials)])
def get_custom_evaluator_session_data(
    user_id: str,
    session_id: str
    ):
    try:
        session_data = db_utils.get_custom_evaluator_data(user_id.strip(), session_id.strip())
        return session_data
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in fetching organization_id {str(e)}")


@app.post("/get_custom_evaluator_session_data_with_user_id",tags=["Custom Evaluator"], dependencies=[Depends(verify_api_credentials)])
def get_custom_evaluator_session_data_with_user_id(user_id: Optional[str] = None, number_of_sessions: Optional[int] = None, session_ids: List[str]=[]):
    return custom_analyzer_handlers.get_custom_evaluator_session_data_with_user_id(user_id, number_of_sessions, session_ids)

    
@app.post("/custom_evaluator_followup",tags=["Custom Evaluator"], dependencies=[Depends(verify_api_credentials)])
def custom_evaluator_followup(
    user_id:str,
    session_id:str,
    query:str,
    section: Optional[str] = None
    ):
    try:
        return custom_analyzer_handlers.generate_evaluator_followup_response(user_id, session_id, query, section)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in generating followup response {str(e)}")
    

@app.post("/custom_evaluator_section_feedback",tags=["Custom Evaluator"], dependencies=[Depends(verify_api_credentials)])
def custom_evaluator_section_feedback(
    user_id:str,
    session_id:str,
    feedback:bool,
    feedback_note: Optional[str] = None,
    response_id: Optional[str] = None,
    section: Optional[Literal["P_Internal", "P_External", "P_Delta"]] = None
    ):
    try:
        return custom_analyzer_handlers.add_evaluator_section_feedback(user_id, session_id, feedback, feedback_note, response_id, section) 
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in adding section feedback {str(e)}")
          

@app.post("/add_custom_evaluator_session_title",tags=["Custom Evaluator"], dependencies=[Depends(verify_api_credentials)])
def add_evaluator_session_title(
    user_id:str,
    session_id:str,
    session_title:str
    ):
    try:
        return custom_analyzer_handlers.add_custom_evaluator_session_title(user_id, session_id, session_title)
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in adding session title {str(e)}")


def call_analyzer_api(data, files):
    api_url = f"{URL}/analyze"
    headers = {
        "accept": "application/json",
        "api-key": "abcd_chatbot_prod",
        "api-secret": "e87a50be-4c74-4bfc-80b0-bbcca3e2a1bd",
    }
    response = requests.post(api_url, headers=headers, data=data, files=files)
    return response.json()


def call_evaluator_api(data, files):
    api_url = f"{URL}/custom_evaluator"
    headers = {
        "accept": "application/json",
        "api-key": "abcd_chatbot_prod",
        "api-secret": "e87a50be-4c74-4bfc-80b0-bbcca3e2a1bd",
    }
    response = requests.post(api_url, headers=headers, data=data, files=files)
    return response.json()


@app.post("/get_analyze_evaluate_comments", tags=["Analyzer-Evaluator"], dependencies=[Depends(verify_api_credentials)])
def get_analyze_evaluate_comments(
        session_id: str = None,
        user_id: str = Form(...),
        user_name: Optional[str] = Form(None),
        pdf_file: UploadFile = File(None),
        text_input: Optional[str] = Form(None),
        nature_of_document: pydantic_schemas.newDocumentType = Form(pydantic_schemas.newDocumentType.Program_Design_Document),
        analyzer_organization_id: Optional[str] = Form(None),
        showcase_items: Optional[int] = Form(10),
        evaluator_organization_id: Optional[str] = Form(None),
        org_guideline_id: Optional[str] = Form(None),
        proposal_pdf_file: UploadFile = File(None),
        proposal_text_input: Optional[str] = Form(None),
        tor_pdf_file: UploadFile = File(None),
        tor_text_input: Optional[str] = Form(None),
        prompt_labels: List[str] = Form(["P1", "P2", "P3", "P4", "P5"]),
        services: List[str] = Form(["analyzer", "evaluator"])):
    
    if not session_id:
        session_id = str(uuid.uuid4())
    
    if services:
        services = services[0].split(',')
        
    if not any([pdf_file, text_input, proposal_pdf_file, proposal_text_input, tor_pdf_file, tor_text_input]):
        logger.error(f"Session {session_id}: No input provided")
        raise HTTPException(status_code=400, detail="At least one input method (file or text) must be provided")
    
    logger.info(f"Starting request processing for session_id: {session_id}, user_id: {user_id}")
    logger.info(f"Requested services: {services}")
    
    # Prepare analyzer data and files
    analyzer_data = {
        "session_id": session_id,
        "user_id": user_id,
        "user_name": user_name,
        "text_input": text_input or "",
        "nature_of_document": nature_of_document.value,
        "prompt_labels": ",".join(prompt_labels)
    }
    
    analyzer_files = {}
    if pdf_file:
        analyzer_files["pdf_file"] = (pdf_file.filename, pdf_file.file, "application/pdf")
    
    # Prepare evaluator data and files
    evaluator_data = {
        "session_id": session_id,
        "user_id": user_id,
        "user_name": user_name,
        "nature_of_document": nature_of_document.value,
        "organization_id": evaluator_organization_id,
        "org_guideline_id": org_guideline_id,
        "proposal_text_input": proposal_text_input or "",
        "tor_text_input": tor_text_input or ""
    }
    
    evaluator_files = {}
    if proposal_pdf_file:
        evaluator_files["proposal_pdf_file"] = (proposal_pdf_file.filename, proposal_pdf_file.file, "application/pdf")
    if tor_pdf_file:
        evaluator_files["tor_pdf_file"] = (tor_pdf_file.filename, tor_pdf_file.file, "application/pdf")
    
    try:
        analyzer_response = None
        evaluator_response = None
        logger.info(f"Services {services}")
        logger.info(f"Session {session_id}: Starting API calls")
        with ThreadPoolExecutor(max_workers=2) as executor:
            analyzer_future = None
            evaluator_future = None
            
            if "analyzer" in services:
                analyzer_future = executor.submit(call_analyzer_api, analyzer_data, analyzer_files)
            if "evaluator" in services:
                evaluator_future = executor.submit(call_evaluator_api, evaluator_data, evaluator_files)
            
            # Increased timeout to 180 seconds (3 minutes)
            if analyzer_future:
                analyzer_response = analyzer_future.result(timeout=600)
            if evaluator_future:
                evaluator_response = evaluator_future.result(timeout=600)
        
        response = {
            "analyzer_response": analyzer_response if analyzer_response else None,
            "evaluator_response": evaluator_response if evaluator_response else None,
            "session_id": session_id,
            "status": "success" if analyzer_response and evaluator_response else "partial success" if analyzer_response or evaluator_response else "failure",
        }
        return response
    except TimeoutError:
        logger.error(f"Session {session_id}: Request timeout")
        raise HTTPException(status_code=504, detail="API request timeout")
    except Exception as e:
        logger.error(f"Session {session_id}: Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        logger.info(f"Session {session_id}: Cleaning up resources")
        # Clean up file handles
        for file in [pdf_file, proposal_pdf_file, tor_pdf_file]:
            if file:
                file.file.close()
        logger.info(f"Session {session_id}: Request handling completed")

app.include_router(collateral_router, tags=["Collateral-Analyzer"], dependencies=[Depends(verify_api_credentials)])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001, reload=True, log_level="info")