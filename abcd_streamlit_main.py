import sys
sys.path.append('.')
sys.path.append('..')
import streamlit as st
import filemapping_utils as pdfUtils 
from dotenv import load_dotenv
load_dotenv()
import chatbot_streamlit_utils
from streamlit_utils import analyzer_streamlit_utils, evaluator_streamlit_utils
import db_utils, api_utils

@st.cache_data
def get_pdf_mappings():
    pdf_mappings=pdfUtils.get_pdf_mappings()
    
    return pdf_mappings


def current_configuration_admin_panel(st, task, prompt_label, doc_type):
    
    if task == "chatbot":
        chatbot_streamlit_utils.chatbot_current_config_panel(st) 
    else:
        analyzer_streamlit_utils.analyzer_current_config_panel(st, prompt_label, doc_type)
        

def set_configuration(st,task,prompt,temperature,prompt_for_customization,prompt_label, doc_type,chunks,prompt_corpus,prompt_examples,summary_prompt,prompt_change_info_bar):
    
    if task == "chatbot":
        status = db_utils.set_gpt_config(prompt,temperature)
    else:
        status = analyzer_streamlit_utils.set_analyzer_config(st, prompt, prompt_for_customization, prompt_label, doc_type,chunks, prompt_corpus, prompt_examples, summary_prompt)

    if status:
        with prompt_change_info_bar:
            st.sidebar.success("Updated the prompt configuration")


def admin_panel(params,st):
    
    if "user" in params and params["user"][0] == "admin" and "password" in params and params["password"][0] == "admin123":
        is_admin = True
    else:
        is_admin = False

    if is_admin:
        prompt_change_info_bar = st.sidebar.empty()
        
        task = st.sidebar.selectbox("Select Task", ["chatbot", "analyzer"])
        
        prompt=""
        prompt_label=""
        temperature=0.0
        chunks=0
        prompt_corpus=""
        prompt_for_customization=""
        summary_prompt=""
        comments_summary_prompt = ""
        prompt_examples=""
        doc_type = "Policy Document"
        
        if task == "chatbot":
            prompt = st.sidebar.text_area("chatbot")
            temperature = st.sidebar.slider('Temperature',value=0.0,min_value= 0.0, max_value=1.0, step=0.1)   
        else:
            prompt_corpus, summary_prompt, prompt_label, doc_type, prompt_for_customization, prompt, prompt_examples, chunks, comments_summary_prompt = analyzer_streamlit_utils.analyzer_admin_panel(st)
           
        submit = st.sidebar.button("Submit", key="set_config")
        
        if submit:
            # First update the existing Prompts using API
            if summary_prompt != "":
                api_utils.update_analyzer_proposal_summary_prompts(doc_type, summary_prompt)
                print("Updated Proposal Summary Prompt!")
            if prompt != "" and prompt_for_customization != "":
                api_utils.update_prompts(prompt_label, doc_type, prompt, prompt_for_customization)
                print("Updated Base and Customization Prompts!")
            if comments_summary_prompt != "":
                api_utils.update_analyzer_comments_summary_prompts(doc_type, comments_summary_prompt)
                print("Updated Comments Summary Prompt!")
            
            # Second Retrieve Prompts using API
            summary_prompt = api_utils.get_analyzer_proposal_summary_prompts(doc_type)
            prompt, prompt_for_customization = api_utils.get_prompts(prompt_label, doc_type)
                
            set_configuration(st, task, prompt, temperature, prompt_for_customization, prompt_label, doc_type, chunks, prompt_corpus, prompt_examples, summary_prompt, prompt_change_info_bar)    

        current_configuration_admin_panel(st, task, prompt_label, doc_type)


def main_panel(st):
    st.subheader("ABCD ChatBot")

    selected_mode = st.selectbox('Select Mode', ('Chatbot', 'Analyzer', 'Evaluator'))

    if selected_mode == 'Chatbot':
        chatbot_streamlit_utils.chatbot_panel(st, pdf_mappings)
    elif selected_mode == 'Analyzer':
        analyzer_streamlit_utils.analyzer_panel(st, pdf_mappings)
    elif selected_mode == 'Evaluator':
        evaluator_streamlit_utils.evaluator_panel(st)
    


pdf_mappings=get_pdf_mappings()
try:
    qp = st.query_params
    params = {k: (v if isinstance(v, list) else [v]) for k, v in qp.items()}
except Exception:
    params = st.experimental_get_query_params()
admin_panel(params,st)
main_panel(st)


    