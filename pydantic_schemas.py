import sys
sys.path.append('.')
sys.path.append('..')
from pydantic import BaseModel,Field
from enum import Enum
from typing import Optional
import json
import os
from dotenv import load_dotenv

load_dotenv(override=True)

possible_models = json.loads(os.getenv("possible_models"))

class Model(str, Enum):
    gpt3 = possible_models["gpt-3.5"]
    gpt4 = possible_models["gpt-4"]
    gpt4o = "gpt-4o-mini"
    o3 = "o3"
    
    
class Chat(BaseModel):
    question: str
    user_id: str
    
    user_name: Optional[str] = None
    user_email: Optional[str] = None
    session_id: Optional[str] = None
    model: Optional[Model] = Model.gpt4
    source:Optional[str] = Field(default=None, pattern="^WA$")


class Response(BaseModel): 
    response_id: str
    user_id:str
    feedback:bool
    feedback_note: Optional[str] = None
    
    
class CaseInsensitiveEnum(str, Enum):
    @classmethod
    def _missing_(cls, value):
        value = value.lower()
        for member in cls:
            if member.lower() == value:
                return member
        return None
       
    
class DocumentType(CaseInsensitiveEnum):
    POLICY_DOCUMENT = "Policy Document"
    PROGRAM_DOCUMENT = "Program Document"
    INVESTMENT_DOCUMENT = "Investment Document"
    RESEARCH_PROPOSAL = "Research Proposal"
    STRATEGY_DOCUMENT =  "Strategy document"
    
class newDocumentType(CaseInsensitiveEnum):
    Program_Design_Document = "Program design Document"
    Policy_Document = "Policy Document"
    Investment_or_Grant_Proposal = "Investment or grant proposal"
    Research_Draft_or_Proposal = "Research draft or proposal"
    Strategy_Recommendations = "Strategy recommendations"
    Media_Article_or_Draft = "Media article or draft"
    School_or_College_Course_Outline = "School or college course outline"
    MEL_Approach = "MEL approach"
    Product_or_Service_Design = "Product or service design"
    
class UserRole(CaseInsensitiveEnum):
    Philanthropy_Program_Officer = "Philanthropy program officer"
    NGO_Leader = "NGO leader"
    Impact_Consultant = "Impact consultant"
    Impact_Investor = "Impact investor"
    Researcher = "Researcher"
    Journalist = "Journalist"
    Policy_Analyst = "Policy analyst"
    Bureaucrat = "Bureaucrat"
    Product_Manager = "Product manager"
    Social_Entrepreneur = "Social entrepreneur"
    Student = "Student"