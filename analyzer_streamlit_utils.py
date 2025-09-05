import json
import api_utils, common_utils, db_utils, pdf_utils, pinecone_utils, s3_bucket_utils
from gpt_utils import get_summary
from io import BytesIO
import time
import os
from dotenv import load_dotenv

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
    
    st.write("Upload a PDF file for analyzer:")
    uploaded_file = st.file_uploader("Upload PDF File", type=["pdf"])
    
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
        
    selected_model_key = st.selectbox('Prompt Model',('o3'),index=1)
    
    selected_model = possible_models[selected_model_key]
    
    use_summary = st.selectbox("Use Summary of uploaded paper For Generating Customized Prompt", [True, False])
    
    use_example = st.selectbox("Use Examples For Generating Customized Prompt", [True, False])
    
    submit = st.button("Generate Prompts")
    
    return file_content, pdf_name, nature, user_role, organization_id, selected_prompts_labels, selected_model, use_summary, use_example, submit

   
def generate_analyzer_prompts(st, file_content, pdf_name, nature, user_role, selected_prompts_labels, selected_model, use_summary, use_example, organization_id=None):
    with st.spinner("Generating prompts..."):
        start_time = time.time()
        try:
            selected_prompts, wisdom = db_utils.get_custom_analyzer_gpt_prompts_multithreaded(selected_prompts_labels, nature, user_role)
            print(f"Custom analyzer GPT prompts fetched in {time.time() - start_time:.2f} seconds.")
            
            if organization_id:
                try:
                    print("Getting custom analyzer partition prompts...")
                    selected_custom_prompt = db_utils.get_current_analyzer_custom_partition_prompts(nature, user_role, organization_id)
                    selected_prompts.update(selected_custom_prompt)
                    print(f"Custom partition prompts fetched for organization {organization_id} in {time.time() - start_time:.2f} seconds.")
                except Exception as e:
                    st.error(f"No prompts found for the organization: {organization_id}")
                    print(f"No prompts found for the organization: {organization_id}. Error: {e}")
                    return -1
            
            print("Extracting text from PDF...")
            extraction_start_time = time.time()
            if text_extraction == "PDF Reader":
                text = pdf_utils.extract_text_from_pdf(BytesIO(file_content))
            else:
                text = pdf_utils.extract_text_llama_parse(file_content, pdf_name)
            print(f"Text extracted from PDF in {time.time() - extraction_start_time:.2f} seconds.")
            
            if pdf_name not in st.session_state:
                s3_par_url = s3_bucket_utils.upload_fileobj_to_s3(BytesIO(file_content), pdf_name)
                st.session_state[pdf_name] = s3_par_url
                print(f"PDF uploaded to S3 in {time.time() - extraction_start_time:.2f} seconds.")
            
            summary_text = ""
            proposal_summary_prompt = ""
            if use_summary:
                print("Generating summary for text from PDF...")
                summary_start_time = time.time()
                summary_text = get_summary(text, selected_model, nature)
                proposal_summary_prompt = api_utils.get_analyzer_proposal_summary_prompts(nature)
                print(f"Summary generated in {time.time() - summary_start_time:.2f} seconds.")
            
            prompt_generation_start_time = time.time()
            generated_prompts = common_utils.generate_prompts(selected_model, nature, summary_text, use_example, selected_prompts)
            print(f"Prompts generated in {time.time() - prompt_generation_start_time:.2f} seconds.")
            
            st.session_state['generated_prompts'] = generated_prompts
            st.session_state['selected_prompts'] = selected_prompts
            st.session_state['proposal_summary_prompt'] = proposal_summary_prompt
            st.session_state['summary_text'] = summary_text
            st.session_state['text'] = text
            st.session_state['pdf_name'] = pdf_name
            st.session_state['wisdom'] = wisdom
            
        except Exception as e:
            print(f"Error generating prompts: {e}")
            st.error("An error occurred while generating prompts. Please try again.")
        print(f"Total time taken for generating prompts: {time.time() - start_time:.2f} seconds.")

        
def generate_analyzer_comments(st, pdf_mappings, nature):
    pinecone_analyzer_filters = pinecone_utils.get_pinecone_analyzer_filters()
    generated_prompts = st.session_state['generated_prompts']
    generated_prompts = dict(sorted(generated_prompts.items()))
    selected_prompts = st.session_state['selected_prompts']
    base_customization_prompts = common_utils.extract_base_customization_prompts(selected_prompts)
    proposal_summary_prompt = st.session_state['proposal_summary_prompt']
    text = st.session_state['text']
    wisdom = st.session_state['wisdom']
    
    summary_text = st.session_state['summary_text']
    pdf_name = st.session_state['pdf_name']
    s3_par_url = st.session_state[pdf_name]
    
    with st.expander("File S3 Url"):
        st.markdown(f'<a href="{s3_par_url}" target="_blank">{s3_par_url}</a>', unsafe_allow_html=True)
        
    with st.expander("Summary"):
        st.text(summary_text)
        
    with st.expander("Generated Prompts"):
        st.json(generated_prompts)
    
    with st.expander(f"Proposal Summary Prompt for {nature}"):
        st.text(proposal_summary_prompt)
    
    with st.expander("Base Prompts and Customization Prompts"):
        st.json(base_customization_prompts)

    
    model_key = st.selectbox('Analyze Model', ('gpt-4', 'o3'), index=1)
    model = possible_models[model_key]
    
    analyze_submit = st.button("analyze")
    
    if analyze_submit:
        if text:
            with st.spinner(f"Generating Comments for corpus"):
                common_utils.generate_analyze_comments(summary_text, pinecone_analyzer_filters, pdf_mappings, model, generated_prompts, wisdom, nature, st)


def analyzer_panel(st, pdf_mappings):
    file_content, pdf_name, nature, user_role, organization_id, selected_prompts_labels, selected_model, use_summary, use_example, submit = analyzer_prompts_panel(st)
    
    if not file_content:
        st.info("Please upload a proposal PDF file.")
        return -1
    
    if submit:
        print("Generating prompts...")
        #generate_analyzer_prompts(st, file_content, pdf_name, nature, user_role, selected_prompts_labels, selected_model, use_summary, use_example, organization_id)
        
    if 'generated_prompts' in st.session_state:
        generate_analyzer_comments(st, pdf_mappings, nature)
        
                
if __name__ == "__main__":
    start = time.time()
    response = db_utils.get_custom_analyzer_gpt_prompts_multithreaded(["P1", "P2", "P3", "P4", "P5"], "Policy Document", "User")
    finish = time.time()
    print(f"Time taken to get prompts: {finish-start}")
    print(response)