import sys
sys.path.append('.')
sys.path.append('..')
from typing import List
from typing import Optional
from fastapi import APIRouter,FastAPI, File, UploadFile, Form,  HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from enum import Enum
from typing import Literal
import chatbot_handlers,custom_analyzer_handlers,pydantic_schemas
from pydantic_schemas import (BehaviorGoalsEnum, EmotionsEnum, BehaviorGoalsResponse, EmotionsResponse,
                            TopicsEnum, BehaviorEnum,TopicsResponse, SubTopicsResponse, BehaviorResponse, SubBehaviorsResponse,
                            TargetAudienceEnum, TargetAudienceResponse, SettlementTypeEnum, SettlementTypeResponse,
                            LiteracyComfortLevelEnum, LiteracyComfortLevelResponse,SUB_TOPICS, SUB_BEHAVIORS)
from db_utils import *
from common_utils import *
from collateral_analyzer_utils import encode_image, convert_pdf_to_base64_images, submit_batch_request, retrieve_batch_results, submit_for_image_analysis
from collateral_analyzer_db_utils import fetch_and_store_results
import db_utils
import traceback
import os
from dotenv import load_dotenv
from logger import api_logger as logger
import time
import io
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import requests
import tempfile

load_dotenv(override=True)


app = FastAPI()
collateral_router = APIRouter(prefix="/api/v1")


@collateral_router.get("/get-behavior-goals", response_model=BehaviorGoalsResponse)
def get_behavior_goals():
    """
    Fetches a list of behavior goals.
    """
    return BehaviorGoalsResponse(values=[goal.value for goal in BehaviorGoalsEnum])

@collateral_router.get("/get-emotions", response_model=EmotionsResponse)
def get_emotions():
    """
    Fetches a list of available emotions.
    """
    return EmotionsResponse(values=[emotion.value for emotion in EmotionsEnum])

@collateral_router.get("/get-topics", response_model=TopicsResponse)
def get_topics():
    """
    Fetches a list of available main topics.
    """
    return TopicsResponse(values=[topic.value for topic in TopicsEnum])


@collateral_router.get("/get-sub-topics", response_model=SubTopicsResponse)
def get_sub_topics(topic: TopicsEnum = Query(..., description="The main topic for which sub-topics are required")):
    """
    Fetches sub-topics related to a given main topic.
    """
    if topic not in SUB_TOPICS:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    return SubTopicsResponse(topic=topic, values=SUB_TOPICS[topic])


@collateral_router.get("/get-behavior", response_model=BehaviorResponse)
def get_behavior():
    """
    Fetches a list of available behaviors.
    """
    return BehaviorResponse(values=[behavior.value for behavior in BehaviorEnum])

@collateral_router.get("/get-sub-behaviors", response_model=SubBehaviorsResponse)
def get_sub_behaviors(behavior: BehaviorEnum = Query(..., description="The main behavior for which sub-behaviors are required")):
    """
    Fetches sub-behaviors related to a given main behavior.
    """
    if behavior not in SUB_BEHAVIORS:
        raise HTTPException(status_code=404, detail="Behavior not found")

    return SubBehaviorsResponse(behavior=behavior, values=SUB_BEHAVIORS[behavior])

@collateral_router.get("/get-target-audience", response_model=TargetAudienceResponse)
def get_target_audience():
    """
    Fetches a list of target audience categories.
    """
    return TargetAudienceResponse(values=[audience.value for audience in TargetAudienceEnum])

@collateral_router.get("/get-settlement-type", response_model=SettlementTypeResponse)
def get_settlement_type():
    """
    Fetches a list of settlement types.
    """
    return SettlementTypeResponse(values=[settlement.value for settlement in SettlementTypeEnum])

@collateral_router.get("/get-literacy-comfort-level", response_model=LiteracyComfortLevelResponse)
def get_literacy_comfort_level():
    """
    Fetches a list of literacy & comfort levels.
    """
    return LiteracyComfortLevelResponse(values=[level.value for level in LiteracyComfortLevelEnum])


SECTION_ANALYSIS_PROMPT = """
Analyze the provided document for the section: {section_name}.
---
Framework: {framework}
Provide three structured sections:
1. What Works Well
2. Recommendations
3. Areas of Improvement
"""

MODEL = "gpt-4o"

AnalysisSections = Literal[
    "Audience Clarity",
    "Cultural Alignment",
    "Message Tone",
    "Visual Layout Imagery"
]

# Predefined Analysis Frameworks (Single-select)
AnalysisFramework = Literal[
    "Health Communication Framework",
    "Environmental Awareness Framework",
    "Education Outreach Framework"
]

TEMP_DIR = "temp_files"

# Ensure the temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

async def save_files_to_temp_dir(files: List[UploadFile]) -> List[str]:
    """Saves uploaded files to a persistent temp directory and returns their file paths."""
    file_paths = []

    for file in files:
        temp_file_path = os.path.join(TEMP_DIR, file.filename)

        # Save file asynchronously
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())  # Read file content and save

        file_paths.append(temp_file_path)

    return file_paths  # Return the list of saved file paths
    
@app.post("/analyze", summary="Analyze Files with Sections & Framework")
async def analyze_files(
    files: List[UploadFile] = File(..., description="Upload up to 10 PDFs or images"),
    analysis_sections: List[AnalysisSections] = Query(..., description="Select analysis sections"),
    analysis_framework: AnalysisFramework = Query(..., description="Select a single framework")
):
    """API to process uploaded PDFs/images and analyze them using OpenAI Vision."""
    
    batch_requests = []
    prompts = []
    campaign_id = str(uuid.uuid4())  # Unique campaign ID

    file_paths = await save_files_to_temp_dir(files)
    
    for sections in analysis_sections:
        prompt = SECTION_ANALYSIS_PROMPT.format(section_name=sections, framework=analysis_framework)
        prompts.append({"prompt_title": sections, "prompt": prompt, "section_name": sections})
    
    submit_for_image_analysis(file_paths, prompts, campaign_id)

    response = {
                "id": campaign_id,
                "status": "analyzing",
                "createdAt": datetime.utcnow(),
                "framework": {
                    "id": 1,
                    "frameworkName": "Health Communication Framework",
                    "description": "Framework for analyzing health-related communication materials"
                },
                "analysisSections": analysis_sections,
                "collaterals": [],
                "analyses": []
            }
    return response

@app.post("/retrieve/{batch_id}")
def retrieve_data(
    batch_id: str,
    background_tasks: BackgroundTasks
):
    logger.info(f"Retrieval for batch {batch_id} started")
    background_tasks.add_task(retrieve_batch_results, batch_id)
    return {"message": f"Retrieval for batch {batch_id} started"}
    
app.include_router(collateral_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, reload=True, log_level="info")