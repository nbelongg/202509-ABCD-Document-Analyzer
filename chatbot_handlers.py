import sys
sys.path.append('.')
sys.path.append('..')
import common_utils
import db_utils
import gpt_utils
import pinecone_utils
from fastapi import HTTPException
import traceback
import uuid
import filemapping_utils

pdf_mappings = filemapping_utils.get_pdf_mappings()

def add_feedback(responseObj,api_key,api_secret):
    
    if not common_utils.validate_request(api_secret=api_secret.strip(),api_key=api_key.strip()):
        raise HTTPException(status_code=400, detail="Invalid API secret or key")
    
    if not db_utils.db_exists():
        raise HTTPException(status_code=500, detail="Internal DB error")
    
    db_utils.log_response_feedback(responseObj.user_id,responseObj.response_id,responseObj.feedback,responseObj.feedback_note)
    
    return {"message": "Added the feedback"}

def get_last_session(user_id,source,api_key,api_secret):
    
    if not common_utils.validate_request(api_secret=api_secret.strip(),api_key=api_key.strip()):
        raise HTTPException(status_code=400, detail="Invalid API secret or key")
    
    if source != "WA" and source is not None:
        raise HTTPException(status_code=400, detail="Invalid source value. Only 'WA' or empty string is allowed.")
    
    if not db_utils.db_exists():
        return {"message": "Response Table doesn't exist"}
    print(f"User_id: {user_id}, source: {source}  ")
    session_data=db_utils.get_user_data(user_id,source)

    if not session_data:
        return {"message": f"No sessions found for user_id: {user_id}"}

    return session_data


def get_sessions(user_id,source,api_key,api_secret):
    
    if not common_utils.validate_request(api_secret=api_secret.strip(),api_key=api_key.strip()):
        raise HTTPException(status_code=400, detail="Invalid API secret or key")
    
    if source != "WA" and source is not None:
        raise HTTPException(status_code=400, detail="Invalid source value. Only 'WA' or empty string is allowed.")
    
    
    if not db_utils.db_exists():
        raise HTTPException(status_code=500, detail="Internal DB error")
    
    print(f"User_id: {user_id}, source: {source}  ")
    
    session_data=db_utils.get_user_sessions(user_id=user_id,source=source)

    if not session_data:
        raise HTTPException(status_code=400, detail=f"No sessions found for user_id: {user_id}")
    
    return session_data


def get_session_chat(session_id,api_key,api_secret):
    
    if not common_utils.validate_request(api_secret=api_secret.strip(),api_key=api_key.strip()):
        raise HTTPException(status_code=400, detail="Invalid API secret or key")
    
    if not db_utils.db_exists():
        raise HTTPException(status_code=500, detail="Internal DB error")
    
    conversation=db_utils.get_session_data(session_id.strip())

    if not conversation:
        raise HTTPException(status_code=400, detail=f"No session found for session_id: {session_id}")
        
    return {"session_id": session_id,"conversation": conversation}
    
def generate_response(chat_data,api_key,api_secret):
    
    if not common_utils.validate_request(api_secret=api_secret.strip(),api_key=api_key.strip()):
        raise HTTPException(status_code=400, detail="Invalid API secret or key")
    
    if not db_utils.db_exists():
        raise HTTPException(status_code=500, detail="Internal DB error")
    
    query = chat_data.question
    user_id = chat_data.user_id
    user_name = chat_data.user_name
    user_email = chat_data.user_email
    session_id = chat_data.session_id
    model=chat_data.model.value
    source=chat_data.source
    
    print(f"User_id: {user_id}, source: {source}  ")
    conversation_string=""
    
    if session_id:
        conversation_string=db_utils.get_user_conversations(user_id,session_id)
    else:
        session_id = str(uuid.uuid4()) 
        
    requires_retrieval,refined_query = gpt_utils.query_refiner(conversation_string, query) 

    print("Final Query:")
    print(refined_query)
    
    
    context=""
    context_info=[]
    sources_info=[]

    if requires_retrieval:
        k=4
        result_dict=pinecone_utils.extract_unique_chunks(refined_query,top_k=k,multiplier=2)
        
        for i in range(k):
            try:
                pdf_name=result_dict[f'meta_{i+1}']
                context = {
                            "pdf_name": pdf_name,
                            "pdf_context": result_dict[f'context_{i+1}']
                        }
                context_info.append(context)

                pdf_info={}
            
                if pdf_name in pdf_mappings: 
                    pdf_info=pdf_mappings[pdf_name]

                if not pdf_info:
                    pdf_info={
                        'sno':'',
                        'title': '',
                        'author_organization': '',
                        'publication_year':'',
                        'link': '',
                        'pdf_title': ''
                    }
                    pdf_info["pdf_title"]=pdf_name
                    
                sources_info.append(pdf_info)
            except Exception as e:
                print(e)
                print("Error in extracting pdf name and context")
                continue
        sources_info = common_utils.get_unique_sources(sources_info)
        
        context = result_dict['all_context']
        
    response,_,within_knowledge_base=gpt_utils.get_response(model,conversation_string,query,context,source)
    
    response_id=str(uuid.uuid4())
    
    if within_knowledge_base==False:
        context_info=""
        sources_info=""

    context_data={"contextInfo": context_info, "sources":sources_info}
    
    
    try:
        db_utils.log_chat_response(user_id,user_name,user_email,session_id,query,context_data,response,response_id,source)
    except Exception as e:
        print(str(e))
        traceback.print_exc() 
    
    if source=="WA":
        response=common_utils.break_into_paragraphs(response)
    
    return {"user_id":user_id,"session_id":session_id,"response": response, "response_id":response_id, "contextInfo": context_info, "sources":sources_info}
