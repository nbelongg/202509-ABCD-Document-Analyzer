import json
import api_utils, common_utils, db_utils, pdf_utils, pinecone_utils, s3_bucket_utils
from analyzer_utils import pzero_utils
from gpt_utils import get_summary
import gpt_utils
from io import BytesIO
import time
import os
import docx
from dotenv import load_dotenv
from logger import streamlit_logger as logger
import traceback


load_dotenv(override=True)

text_extraction = os.getenv("text_extraction")

possible_models = json.loads(os.getenv("possible_models"))


def set_analyzer_config(st, prompt, prompt_for_customization, prompt_label, doc_type, chunks, prompt_corpus, prompt_examples, summary_prompt):
    
    status = db_utils.set_analyzer_gpt_config(prompt_for_customization, prompt_label, prompt, chunks, prompt_corpus, prompt_examples)
        
    if summary_prompt:
        last_summary_prompt=""
        if last_summary_prompt in st.session_state:
            last_summary_prompt=st.session_state["last_summary_prompt"]
            
        if last_summary_prompt.lower() != summary_prompt.lower():
            db_utils.set_summary_prompt(summary_prompt)
    
    return status


def analyzer_current_config_panel(st, prompt_label, doc_type):
    
    prompt_for_customization,current_prompt,chunks,prompt_corpus,prompt_examples = db_utils.get_current_analyzer_gpt_config(prompt_label)
        
    summary_prompt = db_utils.get_proposal_summary_prompt(doc_type)
    
    st.session_state["last_summary_prompt"]=summary_prompt
    
    request=""
    response=""
    
    if prompt_examples:
        examples= prompt_examples["examples"][0]
        request=examples["prompt_request"]
        response=examples["prompt_response"]
    
    st.sidebar.markdown(f"**Current Summary Prompt:**\n\n{str(summary_prompt)}")
    st.sidebar.markdown(f"**Current {prompt_label} Prompt Corpus:**\n\n{str(prompt_corpus)}")
    st.sidebar.markdown(f"**Current Prompt used for {prompt_label} customization:**\n\n{str(prompt_for_customization)}")
    st.sidebar.markdown(f"**Current {prompt_label} Base Prompt:**\n\n{str(current_prompt)}")
    st.sidebar.markdown(f"**Examples for {prompt_label} request and response:**\n\nRequest:\n{str(request)}\n\nResponse:\n{str(response)}")
    st.sidebar.markdown(f"**Current {prompt_label} chunks:**\n\n{str(chunks)}")


def analyzer_admin_panel(st):
    
    doc_type = st.sidebar.selectbox("Select Document Type", ["Policy Document", "Program Document", "Investment Document", "Research Proposal", "Strategy document"])

    
    summary_prompt=""
    
    pinecone_analyzer_filters = pinecone_utils.get_pinecone_analyzer_filters()
    
    if "summary_prompt" in st.session_state:
        summary_prompt=st.session_state["summary_prompt"]
                
    summary_prompt = st.sidebar.text_area(f"Prompt for Generating Summary of PDF with Document type as: {doc_type}:",summary_prompt)
    
    if summary_prompt:
        st.session_state["summary_prompt"] = summary_prompt
        
    prompt_corpus_mapping = {
        "P1": "C1(Universal corpus)",
        "P2": "C2(MBS and GPP)",
        "P3": "C3(LC and IID)",
        "P4": "C4(SDSC)",
        "P5": "C5(CSS)"
    }
        
    prompt_label = st.sidebar.selectbox("Select Prompt Label", list(prompt_corpus_mapping.keys()))
    
    corpuses = pinecone_analyzer_filters.keys()
    prompt_corpus = prompt_corpus_mapping[prompt_label]
    st.sidebar.write(f"Corpus for Prompt Label: {prompt_corpus}")
    # prompt_corpus = st.sidebar.selectbox("Select Corpus for Prompt Label", corpuses)
    
    
    prompt_for_customization = st.sidebar.text_area(f"Prompt to be used for {prompt_label}, {doc_type} customization:")

    prompt = st.sidebar.text_area(f"{prompt_label}, {doc_type} Base Prompt")
    
    comments_summary_prompt=""
    comments_summary_prompt = st.sidebar.text_area("Enter prompt to generate comments summary:")
    
    prompt_request = st.sidebar.text_area("Example for Prompt Generation Request:")
    
    prompt_response = st.sidebar.text_area("Example for Prompt Generation Response:")
    
    prompt_examples = json.dumps({"examples":[{"prompt_request":prompt_request,"prompt_response":prompt_response}]})
    
    k_options = list(range(1, 31))
    chunks = st.sidebar.selectbox("Select Number of Chunks (K)", k_options,index=5)     


    return prompt_corpus, summary_prompt, prompt_label, doc_type, prompt_for_customization, prompt,prompt_examples, chunks, comments_summary_prompt


def analyzer_prompts_panel(st):
    
    st.write("Upload a file for analyzer:")
    uploaded_file = st.file_uploader("Upload File", type=["pdf", "txt", "docx"])
    
    prompt_fields = {
        'P1': ['P1.F1', 'P1.F2', 'P1.F3'],
        'P2': ['P2.F1', 'P2.F2', 'P2.F3'],
        'P3': ['P3.F1', 'P3.F2', 'P3.F3'],
        'P4': ['P4.F1', 'P4.F2', 'P4.F3'],
        'P5': ['P5.F1', 'P5.F2', 'P5.F3'],
    }
    
    file_content = ""    
    pdf_name=""
    if uploaded_file:
        file_content = uploaded_file.getvalue()
        pdf_name = uploaded_file.name
    
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

    user_roles = [
        "Philanthropy program officer",
        "NGO leader",
        "Impact consultant",
        "Impact investor",
        "Researcher",
        "Journalist",
        "Policy analyst",
        "Bureaucrat",
        "Product manager",
        "Social entrepreneur",
        "Student"
    ]

    nature_default = "Program design Document"
    user_role_default = "Impact consultant"

    nature = st.selectbox("Nature of Document", nature_options, index=nature_options.index(nature_default))
    user_role = st.selectbox("User Role", user_roles, index=user_roles.index(user_role_default))
    organization_id = st.text_input(label = "Enter Organization ID. Eg. UNICEF, BMGF")

    selected_prompts_labels = st.multiselect("Prompts", ["P1" , "P2", "P3", "P4", "P5"])
    
    available_flows = []
    selected_flows = []

    for prompt in selected_prompts_labels:
        available_flows.extend(prompt_fields[prompt])

    if available_flows:
        selected_flows = st.multiselect('Select Fields', available_flows)
        
    selected_model_key = st.selectbox('Prompt Model',('o3'))
    
    selected_model = possible_models[selected_model_key]
    
    use_summary = st.selectbox("Use Summary of uploaded paper For Generating Customized Prompt", [True, False])
    
    use_example = st.selectbox("Use Examples For Generating Customized Prompt", [True, False])
    
    submit = st.button("Generate Prompts")
    
    return file_content, pdf_name, nature, user_role, organization_id, selected_prompts_labels, selected_flows, selected_model, use_summary, use_example, submit

   
def generate_analyzer_prompts(st, file_content, pdf_name, nature, user_role, selected_prompts_labels, selected_flows, selected_model, use_summary, use_example, organization_id=None):
    start_time = time.time()
    try:
        with st.spinner("Fetching prompts..."):            
            logger.info("Getting analyzer prompts...")
            selected_prompts, prompt_section_titles, prompt_dependencies = db_utils.get_analyzer_prompts_multithreaded(selected_prompts_labels, nature, user_role)
            
            logger.info("Getting prompt label flows...")
            final_flows = {}
            prompt_flows = db_utils.get_prompt_label_flows(selected_prompts_labels,nature)
            for prompt_label in selected_prompts_labels:
                final_flows[prompt_label] = {}
                for flow in prompt_flows[prompt_label].keys():
                    if flow in selected_flows:
                        final_flows[prompt_label][flow] =prompt_flows[prompt_label][flow]
                
            logger.info(f"#############{selected_prompts}")
            logger.info(f"#############{prompt_dependencies}")
            logger.info(f"#############{prompt_flows}")

            logger.info("Filtering dependencies...")
            prompt_dependencies = common_utils.filter_dependencies(selected_prompts_labels, prompt_dependencies)
            
            logger.info("Topological sorting...")
            selected_prompts_ordered = common_utils.topological_sort(prompt_dependencies)
            
            # if organization_id:
            #     try:
            #         logger.info("Getting custom analyzer partition prompts...")
            #         selected_custom_prompt, org_prompt_section_title = db_utils.get_current_analyzer_custom_partition_prompts(nature, user_role, organization_id)
            #         prompts_section_title["P_Custom"] = org_prompt_section_title
            #         selected_prompts.update(selected_custom_prompt)
            #         dependency_graph["P_Custom"] = []
            #         logger.info(f"Custom partition prompts fetched for organization {organization_id} in {time.time() - start_time:.2f} seconds.")
            #     except Exception as e:
            #         st.error(f"No prompts found for the organization: {organization_id}")
            #         logger.info(f"No prompts found for the organization: {organization_id}. Error: {e}")
            #         return -1
                
        with st.spinner("Processing Proposal File..."):
            logger.info("Extracting text from PDF...")
            extraction_start_time = time.time()
            text = ""
            if pdf_name.endswith(".docx"):
                doc = docx.Document(BytesIO(file_content))
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            elif pdf_name.endswith(".pdf"):
                if text_extraction == "PDF Reader":
                    text = pdf_utils.extract_text_from_pdf(BytesIO(file_content))
                else:
                    text = pdf_utils.extract_text_llama_parse(file_content, pdf_name)
            elif pdf_name.endswith(".txt"):
                text = file_content.decode("utf-8")
            logger.info(f"Text extracted from File in {time.time() - extraction_start_time:.2f} seconds.")
            
            if pdf_name not in st.session_state:
                s3_par_url = s3_bucket_utils.upload_fileobj_to_s3(BytesIO(file_content), pdf_name)
                st.session_state[pdf_name] = s3_par_url
                logger.info(f"File uploaded to S3 in {time.time() - extraction_start_time:.2f} seconds.")
                
        with st.spinner("Generating proposal summary..."):
            summary_text = ""
            proposal_summary_prompt = ""
            if use_summary:
                logger.info("Generating summary for text from PDF...")
                summary_start_time = time.time()
                summary_text = get_summary(text, selected_model, nature)
                proposal_summary_prompt = api_utils.get_analyzer_proposal_summary_prompts(nature)
                logger.info(f"Summary generated in {time.time() - summary_start_time:.2f} seconds.")
        
        with st.spinner("Generating prompts..."):
            prompt_generation_start_time = time.time()
            logger.info("Generating prompts...")
            generated_prompts, generate_prompt_input_token, generate_prompt_output_token = gpt_utils.generate_prompts(selected_model, nature, summary_text, text, use_example, selected_prompts)
            #generated_prompts = common_utils.generate_prompts(selected_model, nature, summary_text, use_example, selected_prompts)
            logger.info(f"Prompts generated in {time.time() - prompt_generation_start_time:.2f} seconds.")
        
        st.session_state['generated_prompts'] = generated_prompts
        st.session_state['selected_prompts'] = selected_prompts
        st.session_state['proposal_summary_prompt'] = proposal_summary_prompt
        st.session_state['summary_text'] = summary_text
        st.session_state['text'] = text
        st.session_state['pdf_name'] = pdf_name
        st.session_state['wisdom'] = ""
        st.session_state['prompts_section_title'] = prompt_section_titles
        st.session_state['dependency_graph'] = prompt_dependencies
        st.session_state['organization_id'] = organization_id
        st.session_state['prompt_section_titles'] = prompt_section_titles
        st.session_state['prompt_dependencies'] = prompt_dependencies
        st.session_state['selected_prompts_ordered'] = selected_prompts_ordered
        st.session_state['prompt_label_flows'] = prompt_flows
        st.session_state['user_role'] = user_role
        st.session_state['selected_flows'] = final_flows
        
        
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error generating prompts: {e}")
        st.error("An error occurred while generating prompts. Please try again.")
    logger.info(f"Total time taken for generating prompts: {time.time() - start_time:.2f} seconds.")
    
    
def generate_analyzer_comments(st, pdf_mappings, nature):
    try:
        pinecone_analyzer_filters = pinecone_utils.get_pinecone_analyzer_filters()
        generated_prompts = st.session_state['generated_prompts']
        generated_prompts = dict(sorted(generated_prompts.items()))
        selected_prompts = st.session_state['selected_prompts']
        base_customization_prompts = common_utils.extract_base_customization_prompts(selected_prompts)
        proposal_summary_prompt = st.session_state['proposal_summary_prompt']
        text = st.session_state['text']
        wisdom = st.session_state['wisdom']
        prompts_section_title = st.session_state['prompts_section_title']
        dependency_graph = st.session_state['dependency_graph']
        summary_text = st.session_state['summary_text']
        pdf_name = st.session_state['pdf_name']
        s3_par_url = st.session_state[pdf_name]
        organization_id = st.session_state['organization_id']
        prompt_section_titles = st.session_state['prompt_section_titles']
        prompt_dependencies = st.session_state['prompt_dependencies']
        selected_prompts_ordered = st.session_state['selected_prompts_ordered']
        prompt_label_flows = st.session_state['prompt_label_flows']
        user_role = st.session_state['user_role']
        selected_flows = st.session_state['selected_flows']
        
        with st.expander("File S3 Url"):
            st.markdown(f'<a href="{s3_par_url}" target="_blank">{s3_par_url}</a>', unsafe_allow_html=True)
            
        with st.expander("Selected Prompts"):
            st.json(base_customization_prompts)
        
        with st.expander("Selected Flows"):
            st.json(selected_flows)
            
        with st.expander("Summary"):
            st.text(summary_text)
            
        with st.expander("Generated Prompts"):
            st.json(generated_prompts)
        

        model_key = st.selectbox('Analyze Model', ('o3'))
        model = possible_models[model_key]
        
        analyze_submit = st.button("analyze")
        
        if analyze_submit:
            
            with st.spinner("Generating Analyze Comments..."):
                logger.info("Generating analyze comments...")
                generated_analyze_comments,analyze_context_used,prompt_label_flow,relevant_chunks,collected_data, analyze_comments_input_token,analyze_comments_output_token =gpt_utils.generate_analyze_comments_dependencies(text,nature,"",summary_text,pinecone_analyzer_filters,model,generated_prompts, organization_id, prompt_section_titles, prompt_dependencies,selected_prompts_ordered, prompt_label_flows)
                #common_utils.generate_analyze_comments_test(summary_text, pinecone_analyzer_filters, pdf_mappings, model, generated_prompts, wisdom, prompts_section_title, nature, st=st, dependency_graph=dependency_graph)
            
            with st.spinner("Generating p0 summary..."):
                logger.info("Generating p0 summary...")
                p0_summary, summary_prompt = pzero_utils.get_pzero_summary(generated_analyze_comments, nature, model, user_role)
                
            with st.spinner("Generating showcase chunks..."):
                logger.info("Generating showcase chunks...")
                showcase = pinecone_utils.extract_unique_showcase_chunks(p0_summary, 10)
                showcase = common_utils.convert_showcase_to_s3_urls(showcase)
                
            with st.expander("P0 Prompt"):
                st.code(summary_prompt)
            with st.expander("P0"):
                st.markdown(p0_summary)
                
            # with st.expander("Generated Analyze Comments"):
            #     st.write(generated_analyze_comments)
            
            s = set()
            for prompt_label, prompt_corpus, prompt_analyze_comments in collected_data:
                if prompt_label in s:
                    continue
                s.add(prompt_label)
                with st.expander(f"{prompt_label}  {prompt_corpus}:"):
                    st.write(prompt_analyze_comments)
            for key, value in prompt_label_flow.items():
                # Display F1 and F2 prompts
                for sub_key, sub_value in value["prompt_flow_prompts"].items():
                    with st.expander(f"{sub_key} Prompts"):
                        st.json(sub_value)
                        
                # Display F1 and F2 comments
                for sub_key, sub_value in value["prompt_flow_result"].items():
                    with st.expander(f"{sub_key} Comments"):
                        comment = sub_value["comments"]
                        if(isinstance(comment,str)):
                            comment = {"comments": [{comment}]}
                        st.json(comment)
                        
                for sub_key, sub_value in value["prompt_flow_contexts"].items():
                    with st.expander(f"{sub_key} Context"):
                        st.json(sub_value)
                
            with st.expander("Child Flows"):
                st.json(prompt_label_flow)
            
            with st.expander("Contexts Used"):
                st.json(analyze_context_used)
            
            with st.expander("Showcase"):
                st.json(showcase)
                    
            #showcase = pinecone_utils.extract_unique_showcase_chunks(P0_summary, 10)
            #showcase = common_utils.convert_showcase_to_s3_urls(showcase)
            
            # with st.expander("Showcase"):
            #     st.json(showcase)
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error generating analyzer comments: {e}")
        st.error("An error occurred while generating comments. Please try again.")


def analyzer_panel(st, pdf_mappings):
    file_content, pdf_name, nature, user_role, organization_id, selected_prompts_labels, selected_flows, selected_model, use_summary, use_example, submit = analyzer_prompts_panel(st)
    
    if not file_content:
        st.info("Please upload a proposal PDF file.")
        return -1
    
    if submit:
        logger.info("Generating prompts...")
        generate_analyzer_prompts(st, file_content, pdf_name, nature, user_role, selected_prompts_labels, selected_flows, selected_model, use_summary, use_example, organization_id)
        
    if 'generated_prompts' in st.session_state:
        generate_analyzer_comments(st, pdf_mappings, nature)
        
                
if __name__ == "__main__":
    start = time.time()
    response = db_utils.get_custom_analyzer_gpt_prompts_multithreaded(["P1", "P2", "P3", "P4", "P5"], "Policy Document", "User")
    finish = time.time()
    print(f"Time taken to get prompts: {finish-start}")
    print(response)