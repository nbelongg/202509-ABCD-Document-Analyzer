from fastapi import FastAPI, HTTPException, Query, Header, Form, Depends, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from typing import Optional, List
from dotenv import load_dotenv
import os
import db_operations
from models import PromptLabel, DocType, PromptUpdate, P0SummaryUpdate, P0ProposalSummary, CustomPromptUpdateRequest, EvaluatorPromptLabels, EvaluatorPromptUpdateRequest, TorPromptUpdateRequest

# Load environment variables from .env file
load_dotenv(override=True)

host=os.getenv("mysql_host")
database=os.getenv("mysql_database")
user=os.getenv("mysql_user")
password=os.getenv("mysql_password")

# Constants for API key and API secret
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# Verify API key and API secret
def verify_api_credentials(api_key: str = Header(...), api_secret: str = Header(...)):
    if api_key != API_KEY or api_secret != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API key or API secret")
    return True
 
               
fastapi_metadata = [
    {
        "name": "Analyzer Prompts",
        "description": "Allows to Get/Update the Base and Customization Prompt for various Prompt Labels and Document Type"   
    },
    {
        "name": "Analyzer Comment Summary Prompts",
        "description": "Allows to Get/Update the Summary Prompt for P0 Summary, for a particular Document Type"
    },
    {
        "name": "Proposal Summary Prompts",
        "description": "Allows to Get/Update the Proposal Summary Prompt for a particular Document Type used for summarizing Proposal PDF/Text"
    },
    {
        "name": "Organization Prompts",
        "description": "Allows to Get/Update the Base and Customization Prompt for various Prompt Labels based on Document Type and Organization"
    },
    {
        "name": "Evaluator Prompts",
        "description": "Allows to Get/Update the Base and Customization Prompt for Evaluator, for various Paritions based on Document Type and Organization"
    },
    {
        "name": "Tor Summary Prompts",
        "description": "Allows to Get/Update the TOR Summary Prompt for a particular Document Type used for summarizing TOR PDF/Text"
    }
]


app = FastAPI(openapi_tags=fastapi_metadata, dependencies=[Depends(verify_api_credentials)])


@app.get("/get_prompts", tags=["Analyzer Prompts"])
def get_prompts(
    prompt_label: str = None,
    doc_type: DocType = None
    ):
    try:
        prompts = db_operations.get_prompts_from_db(prompt_label, doc_type)
        return prompts
    except HTTPException as ex:
        raise ex
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching prompts: {ex}")

 
@app.put("/update_prompts", tags=["Analyzer Prompts"])
def update_prompts(
    prompt_label: str,
    prompt_update: List[PromptUpdate],
    ):
    try:
        if not prompt_update:
            raise HTTPException(status_code=400, detail="No data provided for update")
                
        updated_prompts = db_operations.update_prompts_in_db(prompt_label, prompt_update)
        return {"prompts": updated_prompts}
    except HTTPException as ex:
        raise ex
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"An error occurred while updating prompts: {ex}")


@app.delete("/delete_prompts", tags=["Analyzer Prompts"])
def delete_prompts(
    prompt_label: PromptLabel,
    doc_type: DocType
    ):
    try:
        deleted_prompts = db_operations.delete_prompts_from_db(prompt_label, doc_type)
        return {"prompts": deleted_prompts}
    except HTTPException as ex:
        raise ex
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"An error occurred while deleting prompts: {ex}")


@app.post("/get_corpus_id_analyzer_prompts", tags=["Analyzer Prompts"])
def get_corpus_id_analyzer_prompts():
    try:
        corpus_ids = db_operations.fetch_corpus_id_analyzer_prompts()
        return corpus_ids
    except HTTPException as ex:
        raise ex
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching prompts: {ex}")


@app.get("/get_analyzer_comments_summary_prompts/", dependencies=[Depends(verify_api_credentials)], tags=["Analyzer Comment Summary Prompts"])
def get_analyzer_summary_prompts(doc_type: Optional[DocType] = None):
    try:
        prompts = db_operations.get_all_summary_prompts_from_db(doc_type)
        if prompts:
            return prompts
        else:
            raise HTTPException(status_code=404, detail="No prompts found.")
    except HTTPException as ex:
        raise ex
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching summary prompts: {ex}")


@app.put("/update_analyzer_comments_summary_prompts/", dependencies=[Depends(verify_api_credentials)], tags=["Analyzer Comment Summary Prompts"])
def update_analyzer_summary_prompts(prompts: List[P0SummaryUpdate]):
    try:
        updated_prompts = db_operations.update_summary_prompts_in_db(prompts)
        return {"prompts": updated_prompts}
    except HTTPException as ex:
        raise ex
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"An error occurred while updating summary prompts: {ex}")   


@app.get("/get_analyzer_proposal_summary_prompts/", tags=["Proposal Summary Prompts"])
def get_analyzer_proposal_summary_prompts(doc_type: Optional[DocType] = None):
    try:
        prompts = db_operations.get_proposal_summary_prompts_from_db(doc_type)
        if prompts:
            return prompts
        else:
            raise HTTPException(status_code=404, detail="No prompts found.")
    except HTTPException as ex:
        raise ex
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching proposal summary prompts: {ex}")


@app.put("/update_analyzer_proposal_summary_prompts/", tags=["Proposal Summary Prompts"])
def update_analyzer_proposal_summary_prompts(prompts: List[P0ProposalSummary]):
    try:
        updated_prompts = db_operations.update_proposal_prompts_in_db(prompts)
        return {"prompts": updated_prompts}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"An error occurred while updating proposal prompts: {ex}")


@app.post("/get_custom_prompts", tags=["Organization Prompts"])
def get_custom_prompts(
        doc_type: DocType  = Form(None),
        organization_id: Optional[str] = Form(None), 
    ):
    prompts = db_operations.get_all_custom_prompts_from_db(doc_type, organization_id)
    if prompts:
        return {"prompts":prompts}
    else:
        raise HTTPException(status_code=404, detail="No prompts found.")


@app.put("/update_custom_prompts", tags=["Organization Prompts"])
async def update_custom_prompts(prompts_update: List[CustomPromptUpdateRequest],):
    updated_prompts = db_operations.update_custom_prompts_in_db(prompts_update, "P_Custom")
    if updated_prompts:
        return {"prompts": updated_prompts}
    else:
        raise HTTPException(status_code=404, detail="Update failed.")


@app.delete("/delete_custom_prompts", tags=["Organization Prompts"])
def delete_custom_prompts(
    doc_type: DocType,
    organization_id: str
    ):
    try:
        deleted_prompts = db_operations.delete_custom_prompts_from_db(doc_type, organization_id)
        return {"prompts": deleted_prompts}
    except HTTPException as ex:
        raise ex
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"An error occurred while deleting prompts: {ex}")


@app.post("/get_corpus_id_organization_prompts", tags=["Organization Prompts"])
def get_corpus_id_organization_prompts():
    try:
        corpus_ids = db_operations.fetch_corpus_id_organization_prompts()
        return corpus_ids
    except HTTPException as ex:
        raise ex
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching prompts: {ex}")


@app.post("/get_evaluator_prompts", tags=["Evaluator Prompts"])
def get_evaluator_prompts(
    partition_type: Optional[EvaluatorPromptLabels] = Form(None),
    doc_type: Optional[DocType]  = Form(None),
    organization_id: Optional[str] = Form(None),
    org_guideline_id: Optional[str] = Form(None) 
    ):
    prompts = db_operations.get_all_evaluator_prompts_from_db(partition_type, doc_type, organization_id, org_guideline_id)
        
    if prompts:
        return {"prompts":prompts}
    else:
        raise HTTPException(status_code=404, detail="No prompts found.")


@app.put("/update_evaluator_prompts", tags=["Evaluator Prompts"])
async def update_evaluator_prompts(prompts_update: List[EvaluatorPromptUpdateRequest]):

    updated = db_operations.update_evaluator_prompts_in_db(prompts_update)
    if updated:
        return {"prompts": updated}
    else:
        raise HTTPException(status_code=404, detail="Update failed.")


@app.delete("/delete_evaluator_prompts", tags=["Evaluator Prompts"])
def delete_evaluator_prompts(
    evaluator_prompt_label: EvaluatorPromptLabels,
    doc_type: DocType,
    organization_id: str,
    org_guideline_id: str
    ):
    
    deleted_prompts = db_operations.delete_evaluator_prompt(evaluator_prompt_label, doc_type, organization_id, org_guideline_id)
    return {"prompts": deleted_prompts}
    

@app.post("/get_tor_summary_prompts", tags=["Tor Summary Prompts"])
def get_tor_summary_prompts(
    doc_type: Optional[DocType]  = Form(None),
    organization_id: Optional[str] = Form(None)
    ):
    prompts = db_operations.get_tor_summary_prompts_from_db(doc_type, organization_id)
    if prompts:
        return {"prompts":prompts}
    else:
        raise HTTPException(status_code=404, detail="No prompts found.")


@app.put("/update_tor_summary_prompts", tags=["Tor Summary Prompts"])
async def update_tor_summary_prompts(prompts_update: List[TorPromptUpdateRequest]):
    
    updated = db_operations.update_tor_prompts_in_db(prompts_update)
    if updated:
        return {"prompts": updated}
    else:
        raise HTTPException(status_code=404, detail="Update failed.")