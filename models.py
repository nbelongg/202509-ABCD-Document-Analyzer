from fastapi import Form
from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum

class PromptLabel(str, Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"
    P5 = "P5"
    
class CaseInsensitiveEnum(str, Enum):
    @classmethod
    def _missing_(cls, value):
        value = value.lower()
        for member in cls:
            if member.lower() == value:
                return member
        return None

class DocType(CaseInsensitiveEnum):
    Policy_Document = "Policy Document"
    Investment_or_Grant_Proposal = "Investment or grant proposal"
    Research_Draft_or_Proposal = "Research draft or proposal"
    Program_Design_Document = "Program design Document"
    Strategy_Recommendations = "Strategy recommendations"
    Media_Article_or_Draft = "Media article or draft"
    School_or_College_Course_Outline = "School or college course outline"
    MEL_Approach = "MEL approach"
    Product_or_Service_Design = "Product or service design"
    
class PromptUpdate(BaseModel):
    base_prompt: str = None
    customization_prompt: str = None
    doc_type: DocType

class P0SummaryUpdate(BaseModel):
    doc_type: DocType
    summary_prompt: str
    
class P0ProposalSummary(BaseModel):
    doc_type: DocType
    proposal_prompt: str
    
class CustomPromptUpdateRequest(BaseModel):
    base_prompt: str = "Enter base Prompt..."
    customization_prompt: str = "Enter customization Prompt..."
    doc_type: DocType
    organization_id: str = "Enter Organization ID..."
    
