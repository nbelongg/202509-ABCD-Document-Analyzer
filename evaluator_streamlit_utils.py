import os
import json
import uuid
import time
from io import BytesIO
from dotenv import load_dotenv
import json
import gpt_utils, db_utils, common_utils, pinecone_utils, filemapping_utils
from db_utils import log_analyzer_custom_evaluator_data
from s3_bucket_utils import upload_fileobj_to_s3
import s3_bucket_utils
from pdfminer.high_level import extract_text
from pdf_utils import extract_text_llama_parse
from logger import setup_logger
import logging
from logger import streamlit_logger as logger
import streamlit as st


load_dotenv(override=True)


text_extraction = os.getenv("text_extraction")
possible_models = json.loads(os.getenv("possible_models"))
prompt_labels = ["P1", "P2", "P3", "P4", "P5"]
pdf_mappings = filemapping_utils.get_pdf_mappings()


def extract_text_from_pdf(uploaded_file):
    try:
        with BytesIO() as buffer:
            buffer.write(uploaded_file.read())
            buffer.seek(0)
            return extract_text(buffer)
    except Exception as e:
        st.error(f"Failed to extract text from the PDF: {e}")
        return None


def evaluator_prompts_panel(st):
    st.write("Upload Proposal PDF file:")
    proposal_file = st.file_uploader("Choose a file", type=["pdf"], key="proposal")

    st.write("Upload TOR PDF file:")
    tor_file = st.file_uploader("Choose a file", type=["pdf"], key="tor")

    proposal_content = ""
    proposal_pdf_name = ""

    tor_content = ""
    tor_pdf_name = ""

    if proposal_file:
        proposal_content = proposal_file.getvalue()
        proposal_pdf_name = proposal_file.name

    if tor_file:
        tor_content = tor_file.getvalue()
        tor_pdf_name = tor_file.name
        
    nature_options = [
        "Policy Document",
        "Investment or grant proposal",
        "Research draft or proposal",
        "Program design Document",
        "Strategy recommendations",
        "Media article or draft",
        "School or college course outline",
        "MEL approach",
        "Product or service design"
    ]
    
    nature_default = "Program design Document"

    nature = st.selectbox("Nature of Document", nature_options, index=nature_options.index(nature_default))

    organization_id = st.text_input(label="Enter Organization ID. Eg. UNICEF, BMGF")
    
    org_guideline_id = st.text_input(label="Enter Org Guideline ID.")
    org_guideline_id = org_guideline_id if org_guideline_id else None

    selected_prompts_labels = st.multiselect("Prompts", ["P_Internal", "P_External", "P_Delta"])

    selected_model_key = st.selectbox('Prompt Model', ('gpt-4','o3'))

    selected_model = possible_models[selected_model_key]

    submit = st.button("Generate Prompts")

    return proposal_content, proposal_pdf_name, tor_content, tor_pdf_name, nature, organization_id, org_guideline_id, selected_prompts_labels, selected_model, submit


def generate_evaluator_prompts(st, proposal_content, proposal_pdf_name, tor_content, tor_pdf_name, nature, organization_id, org_guideline_id, selected_prompt_labels, selected_model, identity=None):
    proposal_summary_text = ""
    proposal_text = ""
    tor_text = None
    tor_summary_text = None
    prompts_associated = {}

    summary_pdf_text_length = int(os.getenv("summary_pdf_text_length"))
    
    with st.spinner("Fetching Evaluator prompts..."):
        evaluator_prompts = db_utils.get_evaluator_prompts_multithreaded(partition_type=selected_prompt_labels, doc_type=nature, organization_id=organization_id, org_guideline_id=org_guideline_id, st=st)
        if not evaluator_prompts:
            st.error("No prompts found for the organization: " + organization_id)
            return -1
        
    with st.spinner("Uploading Proposal PDF to S3 bucket..."):
        if proposal_content:
            if text_extraction == "PDF Reader":
                proposal_text=extract_text_from_pdf(BytesIO(proposal_content))
            else:
                proposal_text=extract_text_llama_parse(proposal_content, proposal_pdf_name)
                
            if proposal_pdf_name not in st.session_state:
                s3_par_url = s3_bucket_utils.upload_fileobj_to_s3(BytesIO(proposal_content), proposal_pdf_name)
                st.session_state['proposal_pdf_url'] = s3_par_url
        else:
            st.session_state['proposal_pdf_url'] = ""

    with st.spinner("Reading TOR file..."):
        if tor_content:
            if text_extraction == "PDF Reader":
                tor_text = extract_text_from_pdf(BytesIO(tor_content))
            else:
                tor_text=extract_text_llama_parse(tor_content, tor_pdf_name)

    with st.spinner("Generating Proposal Summary..."):
        proposal_summary_text, proposal_summary_prompt = gpt_utils.get_summary_streamlit(proposal_text, selected_model, nature)
        prompts_associated["proposal_summary_prompt"] = proposal_summary_prompt
    
    if tor_text:
        with st.spinner("Generating TOR Summary..."):
            tor_summary_text, tor_summary_prompt = gpt_utils.get_tor_summary_streamlit(tor_text, selected_model, nature, organization_id)
            prompts_associated["tor_summary_prompt"] = tor_summary_prompt
        
    with st.spinner("Generating prompts..."):
        for key, value in evaluator_prompts.items():
            prompts_associated[key] = {
                "base_prompt": value["base_prompt"],
                "customization_prompt": value["customization_prompt"]
            }
        generated_evaluator_prompts = common_utils.generate_evaluator_prompts(selected_model, nature, tor_summary_text, evaluator_prompts)
        
        if selected_prompt_labels:
            if "P_Internal" in selected_prompt_labels:
                generated_evaluator_prompts["P_Internal"] = evaluator_prompts["P_Internal"]["base_prompt"]
        else:
            generated_evaluator_prompts["P_Internal"] = evaluator_prompts["P_Internal"]["base_prompt"]

    st.session_state['evaluator_prompts'] = prompts_associated
    st.session_state['generated_evaluator_prompts'] = generated_evaluator_prompts
    st.session_state['proposal_summary_text'] = proposal_summary_text
    st.session_state['proposal_text'] = proposal_text
    st.session_state['tor_summary_text'] = tor_summary_text
    st.session_state['tor_text'] = tor_text
    st.session_state['proposal_pdf_name'] = proposal_pdf_name
    st.session_state['tor_pdf_name'] = tor_pdf_name


def generate_evaluator_comments(st, nature, organization_id):
    
    proposal_pdf_name = st.session_state['proposal_pdf_name']
    tor_pdf_name = st.session_state['tor_pdf_name']
    
    used_analyze_prompts = st.session_state['analyze_prompts']
    generated_analyze_prompts = st.session_state['generated_analyze_prompts']
    wisdom = st.session_state['wisdom']
    
    used_evaluator_prompts = st.session_state['evaluator_prompts']
    generated_evaluator_prompts = st.session_state['generated_evaluator_prompts']
    
    proposal_pdf_url = st.session_state['proposal_pdf_url']
    proposal_summary_text = st.session_state['proposal_summary_text']
    proposal_text = st.session_state['proposal_text']
    tor_summary_text = st.session_state['tor_summary_text']
    
    tor_text = st.session_state['tor_text']
    
    with st.expander("Proposal S3 Url"):
        st.markdown(f'<a href="{proposal_pdf_url}" target="_blank">{proposal_pdf_url}</a>', unsafe_allow_html=True)
    if used_analyze_prompts:
        with st.expander("Used Analyzer Prompts"):
            st.json(used_analyze_prompts)
    with st.expander("Used Evaluator Prompts"):
        st.json(used_evaluator_prompts)
    with st.expander("Proposal Summary"):
        st.text(proposal_summary_text)
    with st.expander("TOR Summary"):
        st.text(tor_summary_text)
    if generated_analyze_prompts:
        with st.expander("Generated Analyze Prompts"):
            st.json(generated_analyze_prompts)
    with st.expander("Generated Evaluator Prompts"):
        st.json(generated_evaluator_prompts)
    
    model_key = st.selectbox('Evaluate Model', ('gpt-4', 'o3'))
    model = possible_models[model_key]
    evaluate_submit = st.button("Evaluate")
    
    if evaluate_submit:
        
        if generated_analyze_prompts:
            with st.spinner(f"Generating Analyze Comments"):
                pinecone_analyze_filters = pinecone_utils.get_pinecone_analyzer_filters()

                generated_analyze_comments, analyze_context_used = common_utils.generate_custom_analyze_comments(proposal_summary_text, pinecone_analyze_filters, pdf_mappings, model, generated_analyze_prompts, wisdom, nature)
                analyze_comments = common_utils.combine_comment(generated_analyze_comments)

                with st.expander("Generated Analyze Comments"):
                    st.code(analyze_comments)
        else:
            analyze_context_used = ""
            generated_analyze_comments = ""
            analyze_comments = ""
            
        with st.spinner(f"Generating Evaluate Comments"):
            generated_evaluator_comments, p_internal_context, p_external_context, p_delta_context = gpt_utils.generate_evaluator_comments(proposal_text, model, generated_evaluator_prompts, analyze_comments)
            
            if p_internal_context:
                with st.expander("P_Internal Context"):
                    st.json(p_internal_context)
                with st.expander("Generated P_Internal Comments"):
                    st.text(generated_evaluator_comments["P_Internal"])
            
            if p_external_context:
                with st.expander("P_External Context"):
                    st.json(p_external_context)
                with st.expander("Generated P_External Comments"):
                    st.text(generated_evaluator_comments["P_External"])
            
            if p_delta_context:
                with st.expander("P_Delta Context"):
                    st.json(p_delta_context)
                with st.expander("Generated P_Delta Comments"):
                    st.text(generated_evaluator_comments["P_Delta"])
            
            session_id = "st-"+str(uuid.uuid4()) 
            
            time_taken=None
            
            if 'start_time' in st.session_state:
                time_taken = time.time() - st.session_state['start_time']
                
            evaluator_tokens_counter = common_utils.get_evaluator_tokens_counter()
    
            db_utils.log_analyzer_custom_evaluator_data(
                user_id=None,
                user_name=None,
                session_id=session_id,
                proposal_pdf_name=proposal_pdf_name,
                proposal_summary_text=proposal_summary_text,
                proposal_text=proposal_text,
                nature_of_document=nature,
                organization_id=organization_id,
                tor_pdf_name=tor_pdf_name,
                tor_text=tor_text,
                tor_summary_text=tor_summary_text,
                generated_analyze_prompts=generated_analyze_prompts,
                generated_analyze_comments=generated_analyze_comments,
                analyze_context_used=analyze_context_used,
                generated_evaluator_prompts=generated_evaluator_prompts,
                generated_evaluator_comments=generated_evaluator_comments,
                time_taken=time_taken,
                tokens_counter=evaluator_tokens_counter
            )


def generate_analyzer_prompts(st, summary_text, nature, selected_prompts_labels, selected_model, use_example=False, user_role=None):
    prompts_associated = {}

    with st.spinner("Generating analyzer prompts..."):
        
        selected_prompts, wisdom = db_utils.get_custom_analyzer_gpt_prompts_multithreaded(prompt_labels, nature, None)

        for key, value in selected_prompts.items():
            base_prompt = value[1]
            customization_prompt = value[0]
        
        # Associate prompts with prompt labels
        prompts_associated[key] = {
            "base_prompt": base_prompt,
            "customization_prompt": customization_prompt
        }
        generated_analyze_prompts = common_utils.generate_prompts(selected_model, nature, summary_text, use_example, selected_prompts)

        st.session_state['analyze_prompts'] = prompts_associated
        st.session_state['generated_analyze_prompts'] = generated_analyze_prompts
        st.session_state['wisdom'] = wisdom


def evaluator_panel(st):
    proposal_content, proposal_pdf_name, tor_content, tor_pdf_name, nature, organization_id, org_guideline_id, selected_prompt_labels, selected_model, submit = evaluator_prompts_panel(st)

    if not proposal_content:
        st.info("Please upload a proposal PDF file.")
        return -1
    
    if (len(selected_prompt_labels) == 1) and selected_prompt_labels[0] == "P_Delta":
        st.info("P_Delta comments cannot be generated without P_Internal and P_External")
        return -1

    if (len(selected_prompt_labels) == 2) and "P_Delta" in selected_prompt_labels:
        st.info("P_Delta comments cannot be generated without both P_Internal and P_External")
        return -1

    if submit:
        st.session_state['start_time'] = time.time()
        generate_evaluator_prompts(st, proposal_content, proposal_pdf_name, tor_content, tor_pdf_name, nature, organization_id, org_guideline_id, selected_prompt_labels, selected_model)
        print(type(org_guideline_id))
        if 'proposal_summary_text' in st.session_state and "P_External" in selected_prompt_labels:
            summary_text = st.session_state['proposal_summary_text']
            generate_analyzer_prompts(st, summary_text, nature, prompt_labels, selected_model, organization_id)
        else:
            st.session_state["generated_analyze_prompts"] = ""
            st.session_state['analyze_prompts'] = ""

    if 'generated_analyze_prompts' in st.session_state and 'generated_evaluator_prompts' in st.session_state:
        generate_evaluator_comments(st, nature, organization_id)



