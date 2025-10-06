import os
import openai
import json
import streamlit as st
import db_utils
import api_utils
from openai import OpenAI
import concurrent.futures
import common_utils
import time
import openai
from langsmith.wrappers import wrap_openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_anthropic import ChatAnthropic
import traceback
from logger import api_logger as logger
import pinecone_utils


load_dotenv(override=True)


api_key = os.getenv("openai_api_key")
claude_api_key = os.getenv("claude_api_key")
organization= os.getenv("openai_organization")

llm = ChatOpenAI(model="gpt-4o", api_key=api_key, openai_organization=organization, temperature=1)

client = openai.Client(api_key=api_key, organization=organization)

possible_models={"gpt-3.5":"gpt-3.5-turbo","gpt-4":"gpt-4o", "o3":"o3", "gpt-5":"gpt-5"}


def get_whatsapp_prompt():
    
    prompt="""Taking into account the context and your knowledge base, please provide a well-informed, truthful, and structured answer. 
            If possible, include specific examples and statistics to support your response. However, if there is insufficient information to provide a well-informed answer, please include a disclaimer.
            Please keep the response concise, within 1000 characters suitable for shorter chats. If the response exceeds 1000 characters, kindly break it into smaller paragraphs."""
    temperature=0
    
    return prompt,temperature


def get_gpt_config(source):


    if(source=="WA"):
        prompt,temperature=get_whatsapp_prompt()
    else:
        prompt,temperature = db_utils.get_current_gpt_config()

    return temperature,prompt


def query_refiner(conversation, query):
    
    system_role = """Given the following user Query and CONVERSATION LOG, 
    formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.
    The CONVERSATION LOG and Query are delimited with triple backticks. 
    Format the response as a valid json with keys "Refined Query" and "knowledge Base".Format the "knowledge Base" value as a boolean.
    If the query requires knowledge base lookup for answering, set "knowledge Base" as true else set it as false. One way to know whether knowledge base lookup is required is to check if the query is not greeting message.
    If no conversation log is available, set "Refined Query" as same Query provided. 
    """


    example_text = f"""\n\nCONVERSATION LOG:\n```
                    Human: hi
                    Bot: hi
                    Human: What is Washing with hope??
                    Bot: this is bot response```

                    Query: ```What is its meaning?```

                    """
    response_text = "{\"Refined Query\": \"What is the meaning or significance behind Washing with hope?\",\"knowledge Base\": true}"
    user_input= f"\n\nCONVERSATION LOG: \n```{conversation}```\n\nQuery: ```{query}```\n\n"

    response = client.chat.completions.create(
        model=possible_models["o3"],
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": example_text},
            {"role": "assistant", "content": response_text},
            {"role": "user", "content": user_input}
        ],
        response_format={"type": "json_object"}
    )

    
    gpt_response = response.choices[0].message.content
    
    # with st.expander("Refine Query Output:"):
    #     st.write("GPT Response:",gpt_response)
        
    try:
        response_dict = json.loads(gpt_response)
    except:
        response_dict = convert_response_to_valid_json(gpt_response)
    
    refined_query=response_dict["Refined Query"]
    
    requires_retrieval=response_dict["knowledge Base"]
    
    return (requires_retrieval,refined_query)


def convert_response_to_valid_json(response):
    
    syste_role = "Convert given text into valid JSON without changing its content.Strictly, retain the keys and their values. Make sure you handle markdown, newlines and multiline text"
    
    
    ex_input_text = """{\"Relevancy\":false,
                    \"Bot Response\": \"The information provided in the ABCD Knowedge Library does not contain any relevant information about washing with hope.
                                        But I can tell you that washing with hope is a culture in florida.
                                        It is recommended to improve the culture in florida.\"
                    }"""

    ex_response_text = """{\"Relevancy\":false,
                    \"Bot Response\": \"The information provided in the ABCD Knowedge Library does not contain any relevant information about washing with hope.\n But I can tell you that washing with hope is a culture in florida.\n It is recommended to improve the culture in florida. \"
                    }"""
    
    
    ex_input_text1 = """
                    {\"Refined Query\": \"What is the meaning or significance behind Washing with hope?\"
                    \"knowledge Base\": true
                    }
                    """
    
    ex_response_text1 = """
                    {\"Refined Query\": \"What is the meaning or significance behind Washing with hope?\",
                    \"knowledge Base\": true
                    }
                    """
    
    messages  = [
        {"role": "system", "content": syste_role},
        {"role": "user", "content": ex_input_text},
        {"role": "assistant", "content": ex_response_text},
        {"role": "user", "content": ex_input_text1},
        {"role": "assistant", "content": ex_response_text1},
        {"role": "user", "content": response}
    ]
    
    response = client.chat.completions.create(
        model=possible_models["o3"],
        messages=messages,
        response_format={"type": "json_object"}
    )
    
    gpt_response = response.choices[0].message.content
    
    
    return gpt_response
    
    
def is_valid_json(s):
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


def get_response(model, conversation_string, query, context, source=None):
    
    temperature, prompt = get_gpt_config(source)
    
    prompt+="""\n\nInstructions for Response Generation:\n\n
            Please ensure to generate empathetic responses when addressing greeting messages and provide insightful replies.  
            The context provided is from "ABCD Knowledge Library".
            The CONVERSATION LOG, Query and ABCD Knowledge Library are delimited with triple backticks.
            Always refer "ABCD Knowledge Library" as the context in the response instead of saying "context".
            
            Your response must be structured as a valid JSON object containing two keys: "Bot Response" and "Relevancy". 
            Format the "Relevancy" value as a boolean. The "Relevancy" value should be set to 'true' if the user's query can be answered using the ABCD Knowledge Library, and 'false' if not.
            "Bot Response" will hold answer to the Query. It is a JSON string and can be multiline or multiparagraph.
        
            Please note: If "Bot Response" is multiline or multiparagraph, make sure to include newline character ('\\n') to preserve the structure while maintaining JSON validation. 
            Maintain json validatity by escaping special characters wherever required.
            """

    system_role = prompt

    example_text = f"""\n\nCONVERSATION LOG:\n
                    ```Human: hi
                    Bot Response: hi```\n\n
                    Query:\n ```What is the meaning or significance behind Washing with hope?``` \n\n
                    ABCD Knowledge Library:\n ```I live in florida.``` \n\n
                    """

    response_text = "{\"Relevancy\":false,\"Bot Response\": \"The information provided in the ABCD Knowedge Library does not contain any relevant information about washing with hope.\"}"
    
    example_text_2 = f"""\n\nCONVERSATION LOG: \n
                    ```Human: hi
                    Bot Response: hi```\n\n
                    Query:\n ```What is the meaning or significance behind Washing with hope?```\n\n
                    ABCD Knowledge Library:\n ```I live in florida where washing with hope is a culture.```\n\n
                    """

    response_text_2 = "{\"Relevancy\":true,\"Bot Response\": \"Washing with hope is a culture in florida\"}"
    
    user_input = f"\n\nCONVERSATION LOG:\n```{conversation_string}```\n\nQuery: \n```{query}```\n\n ABCD Knowledge Library:\n ```{context}```\n\n "

    
    response_list = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": example_text},
        {"role": "assistant", "content": response_text},
        {"role": "user", "content": example_text_2},
        {"role": "assistant", "content": response_text_2},
        {"role": "user", "content": user_input},
    ]

    
    response = client.chat.completions.create(
        model=model,
        messages=response_list
    )
    
    gpt_response = response.choices[0].message.content

    # st.write("GPT Response:",gpt_response)

    #print("user Query-Input:")
    #print(user_input)

    #print("bot Output:")
    #print(gpt_response)
    
    if not is_valid_json(gpt_response):
        gpt_response=convert_response_to_valid_json(gpt_response)
    
    # st.write("GPT Response:",gpt_response)
    response_dict = json.loads(gpt_response)  
    response = response_dict["Bot Response"]
    within_knowledge_base = response_dict["Relevancy"]

    return response, prompt, within_knowledge_base


def generate_prompt(prompt_for_customization,pdf_text,llm_type,model,prompt,nature,summary, use_example, prompt_examples):
    if(common_utils.has_placeholder(prompt_for_customization, "Document.Summary")):
            prompt_for_customization = common_utils.replace_placeholder(prompt_for_customization,"Document.Summary","summary of the document")

    if(common_utils.has_placeholder(prompt_for_customization, "Document.Full")):
            prompt_for_customization = common_utils.replace_placeholder(prompt_for_customization,"Document.Full","full document text")

    if "full document text" in prompt_for_customization and "summary of the document" in prompt_for_customization:
        prompt = f""" Nature of Document: {nature}, Full Document text: {pdf_text}, Summary of the document: {summary} Base Prompt:{prompt}"""
    elif "full document text" in prompt_for_customization:
        prompt = f""" Nature of Document: {nature}, Full Document text: {summary}, Base Prompt:{prompt}"""
    else:
        prompt = f""" Nature of Document: {nature}, Summary of the document: {summary}, Base Prompt:{prompt}"""
    system_role = f"""{prompt_for_customization}"""
    
    examples=""
    example_text_3=""
    response_text_3=""
     
    if prompt_examples and prompt_examples["examples"]:
        examples=prompt_examples["examples"][0]
        example_text_3=f"""{examples["prompt_request"]}"""
        response_text_3=f"""{examples["prompt_response"]}"""
    
    is_example_present = bool(example_text_3) and bool(response_text_3)

    if use_example is True and is_example_present:
        response_list = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": example_text_3},
        {"role": "assistant", "content": response_text_3},
        {"role": "user", "content": prompt}
    ]
    else:
        response_list = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": prompt}
    ]
    
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            system_role,
        ),
        ("user", "{prompt}"),
    ])
    llm_type = llm_type.lower()
    model = model.lower()
    if llm_type == "chatgpt":
        if str(model).lower() in ["o3", "gpt-5"]:
            llm = ChatOpenAI(model=model, api_key=api_key, organization=organization, temperature=1)
        else:
            llm = ChatOpenAI(model=model, api_key=api_key, organization=organization)
    elif llm_type == "claude":
        llm = ChatAnthropic(model=model,anthropic_api_key=claude_api_key)
    else:
        if str(model).lower() in ["o3", "gpt-5"]:
            llm = ChatOpenAI(model=model, api_key=api_key, organization=organization, temperature=1)
        else:
            llm = ChatOpenAI(model=model, api_key=api_key, organization=organization)
    
    chain = prompt_template | llm

    with get_openai_callback() as cb:
        response = chain.invoke({"prompt":prompt})

    input_tokens, output_tokens = cb.prompt_tokens, cb.completion_tokens
    
    # gpt_response = response.choices[0].message.content
    # input_tokens = response.usage.prompt_tokens
    # output_tokens = response.usage.completion_tokens

    return response.content, input_tokens, output_tokens


def generate_customised_prompt(selected_model,nature,prompt_label,summary_text,text,prompt_data,use_example):
    prompt_for_customization,prompt_string, wisdom_1, wisdom_2,chunks, prompt_corpus, prompt_examples,which_chunks,wisdom_received,llm_flow,llm,model,label_for_output,show_on_frontend = prompt_data
    generated_prompt, input_tokens, output_tokens = generate_prompt(prompt_for_customization,text, llm,model,prompt_string, nature,summary_text,use_example, prompt_examples)
    return prompt_label, generated_prompt, wisdom_1, wisdom_2,chunks, prompt_corpus,which_chunks,wisdom_received,llm,model,input_tokens, output_tokens


def generate_prompts(selected_model,nature,summary_text,text,use_example,selected_prompts):
    total_input_tokens = {}
    total_output_tokens = {}
    generated_prompts={}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_customised_prompt,selected_model,nature, prompt_label,summary_text, text,prompt_data,use_example) for prompt_label, prompt_data in selected_prompts.items()]
        for future in concurrent.futures.as_completed(futures):
            prompt_label, generated_prompt, wisdom_1, wisdom_2,chunks, prompt_corpus,which_chunks,wisdom_received, llm,model,input_tokens, output_tokens = future.result()
            generated_prompts[prompt_label] = {"generated_prompt":generated_prompt,"chunks":chunks,"prompt_corpus":prompt_corpus, "wisdom_1":wisdom_1, "wisdom_2":wisdom_2,"which_chunks":which_chunks, "wisdom_received":wisdom_received,"llm":llm, "model":model}
    
    return generated_prompts, total_input_tokens, total_output_tokens


@traceable(tags=["analyzer-comments-generator"])
def generate_analyze_comments(analyzer_prompt, context, model, prompt_label=None, tokens_counter=None, wisdom1=None, wisdom2=None, comments=None):
    try:
        response_list, context = get_analyzer_messages(analyzer_prompt, context, wisdom1, wisdom2, comments)
        
        prompt = response_list[1]["content"]

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", response_list[0]["content"]),
                ("user", "{prompt}"),
            ]
        )
        
        chain = prompt_template | llm
        
        response = chain.invoke({"prompt": prompt})
        
        gpt_response = response.content
        if not is_valid_json(gpt_response):
            gpt_response = convert_response_to_valid_json(gpt_response)
            
        return gpt_response, response_list
    except Exception as e:
        print("Error:", e)
        return None, None
    response = client.chat.completions.create(
        model=model,
        messages=response_list,
        response_format={"type": "json_object"}
    )
    
    gpt_response = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    if not is_valid_json(gpt_response):
        gpt_response = convert_response_to_valid_json(gpt_response)
    
    if tokens_counter is not None and prompt_label is not None:
        prompt_label = prompt_label.lower()
        tokens_counter[0]["analyzer_comments"][prompt_label] = input_tokens
        tokens_counter[1]["analyzer_comments"][prompt_label] = output_tokens
    
    return gpt_response, context


def convert_response_to_valid_json(response):
    syste_role = "Convert given text into valid JSON without changing its content.Strictly, retain the keys and their values."
      
    ex_input_text = """
    {"Hypothesis": ["A) Gender norms and patriarchal values can significantly influence the level of access to health facilities and treatments, potentially leading to higher rates of AMR infections in men.", "B) The prescribing behaviors of antibiotics can vary based on the gender of the patient, potentially leading to unnecessary prescriptions and increased AMR."]}
    {"Development Program Ideas": ["A) Implement community-based outreach programs that specifically target marginalized groups such as low-caste individuals and Muslims, aiming to increase awareness and provide resources for mental health support.", "B) Develop interventions that address the unique mental health challenges faced by marginalized groups, incorporating considerations for factors such as gender, caste, and socioeconomic status."]}
    """

    ex_response_text = """{
        "Hypothesis": ["A) Gender norms and patriarchal values can significantly influence the level of access to health facilities and treatments, potentially leading to higher rates of AMR infections in men.", "B) The prescribing behaviors of antibiotics can vary based on the gender of the patient, potentially leading to unnecessary prescriptions and increased AMR."],
        "Development Program Ideas": ["A) Implement community-based outreach programs that specifically target marginalized groups such as low-caste individuals and Muslims, aiming to increase awareness and provide resources for mental health support.", "B) Develop interventions that address the unique mental health challenges faced by marginalized groups, incorporating considerations for factors such as gender, caste, and socioeconomic status."]  
    }
    """
    
    messages  = [
        {"role": "system", "content": syste_role},
        {"role": "user", "content": ex_input_text},
        {"role": "assistant", "content": ex_response_text},
        {"role": "user", "content": response}
    ]
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    gpt_response = response.choices[0].message.content
    
    return gpt_response


def get_analyzer_messages(analyzer_prompt, context, wisdom_1, wisdom_2):
    additional_prompt = """Instructions for Response Generation:
    Strictly adhere to the questions asked and answer each of the questions by referring the CONTEXT. Do not give the same answer multiple times.
    Strictly, ensure the output is in the form of valid JSON with the key as "Comments" and the value as a valid string or valid array of strings.
    AVOID generating a JSON object as the value of "Comments", it should always be either string or array of strings.
    If the value of spans multiple lines, ensure the new lines are represented using the "\\n" escape sequence within the string. Do not use literal line breaks.
    
    Examples of Valid JSON Responses:
    
    Example 1:
        {{
            "Comments":["comment1","comment2"]  
        }}
    Example 2:
        {{
            "Comments":"comment"
        }}
    
    Examples of Invalid JSON Responses:

    Example 1:
        {{
            "Comments": [
                "{{\\"comment\\": \\"comment1\\"}}",
                "{{\\"comment\\": \\"comment2\\"}}",
            ]
        }}
    
    Example 2:
        {{
            "Comments": ["{{\\"comment1\\"}}", "{{\\"comment2\\"}}"]
        }}

    """
    if wisdom_1 and wisdom_2:
        user_input = f"""CONTEXT: \n\n ```{context}``` \n\n Wisdom1: \n\n ```{wisdom_1}``` \n\n Wisdom2: \n\n ```{wisdom_2}```"""
        system_role = analyzer_prompt + "\n" + additional_prompt + "\n\n The Context, Wisdom1 and Wisdom2 are enclosed within three backticks"
    else:
        user_input = f"""CONTEXT: \n\n ```{context}```"""
        system_role = analyzer_prompt + "\n" + additional_prompt + "\n\n The Context is enclosed within three backticks"

    response_list = [
        {"role": "system", "content": system_role},]
    
    response_list.extend([
        {"role": "user", "content": user_input}
    ])
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            system_role,
        ),
        ("user", "{user_input}"),
    ])
    
    return prompt_template, response_list,user_input


def generate_prompt_analyze_comments_langchain(analyzer_prompt,context, wisdom_1, wisdom_2, model, llm_type, llm_model):    
    prompt_template, response_list, user_input = get_analyzer_messages(analyzer_prompt, context, wisdom_1, wisdom_2)
    llm_type = llm_type.lower()
    llm_model = llm_model.lower()
    if llm_type == "chatgpt":
        if str(llm_model).lower().startswith("o"):
            llm = ChatOpenAI(model=llm_model, api_key=api_key, organization=organization, temperature=1)
        else:
            llm = ChatOpenAI(model=llm_model, api_key=api_key, organization=organization)
    elif llm_type == "claude":
        llm = ChatAnthropic(model=llm_model,anthropic_api_key=claude_api_key)
    else:
        if str(llm_model).lower().startswith("o"):
            llm = ChatOpenAI(model=llm_model, api_key=api_key, organization=organization, temperature=1)
        else:
            llm = ChatOpenAI(model=llm_model, api_key=api_key, organization=organization)
        
    chain = prompt_template | llm

    with get_openai_callback() as cb:
        response = chain.invoke({"user_input":user_input})

    gpt_response = response.content
    if not is_valid_json(gpt_response):
        gpt_response = convert_response_to_valid_json(gpt_response)
    
    input_tokens, output_tokens = cb.prompt_tokens, cb.completion_tokens
    
    return gpt_response, response_list, input_tokens, output_tokens


def get_belonggrag_output(prompt_label, prompt_flows, doc_type, identity, pdf_text, pdf_summary, model, filters, dependencies,result_store,sti):
    logger.info(f"Generating BelonggRAG output for {prompt_label}")
    # BelonggRAG Flow
    selected_prompts = {}
    
    f1_base_prompt = prompt_flows[prompt_label]["base_prompt"]
    f1_custom_prompt = prompt_flows[prompt_label]["customization_prompt"]
    f1_wisdom_1 = prompt_flows[prompt_label]["wisdom_1"]
    f1_wisdom_2 = prompt_flows[prompt_label]["wisdom_2"]
    f1_chunks = prompt_flows[prompt_label]["chunks"]
    f1_prompt_corpus = prompt_flows[prompt_label]["prompt_corpus"]
    f1_which_chunks = prompt_flows[prompt_label]["which_chunks"]
    f1_wisdom_received = prompt_flows[prompt_label]["wisdom_received"]
    f1_customise_prompt_based_on = prompt_flows[prompt_label]["customize_prompt_based_on"]
    f1_send_along_customised_prompt = prompt_flows[prompt_label]["send_along_customised_prompt"]
    llm = prompt_flows[prompt_label]["llm"]
    llm_model = prompt_flows[prompt_label]["model"]
    label_for_output = prompt_flows[prompt_label]["label_for_output"]
    show_on_frontend = prompt_flows[prompt_label]["show_on_frontend"]
    for dep in dependencies[prompt_label]:
        if dep in result_store:
            if(common_utils.has_placeholder(f1_wisdom_1, str(dep))):
                    f1_wisdom_1 = common_utils.replace_placeholder(f1_wisdom_1,str(dep), result_store[dep])
            if(common_utils.has_placeholder(f1_wisdom_2, str(dep))):
                    f1_wisdom_2 = common_utils.replace_placeholder(f1_wisdom_2,str(dep), result_store[dep])
            if(common_utils.has_placeholder(f1_base_prompt, str(dep))):
                    f1_base_prompt = common_utils.replace_placeholder(f1_base_prompt,str(dep), result_store[dep])
            if(common_utils.has_placeholder(f1_custom_prompt, str(dep))):
                    f1_custom_prompt = common_utils.replace_placeholder(f1_custom_prompt,str(dep), result_store[dep])

    received_wisdom = "Wisdom_Received" + "\n\n"
    for wisdom in f1_wisdom_received:
        if wisdom in result_store:
            logger.info(f"{prompt_label} using wisdom from {wisdom}")
            received_wisdom += str(result_store[wisdom]) + "\n\n"
            
    logger.info(f1_customise_prompt_based_on)
    summary_text = pdf_summary
    for customizing_string in f1_customise_prompt_based_on:
        if customizing_string =="Document.Summary":
            if(common_utils.has_placeholder(f1_custom_prompt, "Document.Summary")):
                    f1_custom_prompt = common_utils.replace_placeholder(f1_custom_prompt,"Document.Summary","summary of the document")
                    summary_text = pdf_summary
        else:
            if(common_utils.has_placeholder(f1_custom_prompt, "Document.Full")):
                    f1_custom_prompt = common_utils.replace_placeholder(f1_custom_prompt,"Document.Full","full document text")
                    summary_text = pdf_text

    selected_prompts[prompt_label] = (f1_custom_prompt,f1_base_prompt, f1_wisdom_1, f1_wisdom_2,f1_chunks, "", {}, f1_which_chunks, f1_wisdom_received,"BelonggRAG",llm,llm_model,label_for_output,show_on_frontend)
    logger.info(f"Generating BelonggRAG {prompt_label} prompts using {llm} & {llm_model}")
    f1_generated_prompt, f1_generate_prompt_input_token, f1_generate_prompt_output_token = generate_prompts(model,doc_type,summary_text,pdf_text,False,selected_prompts)
    logger.info(f"BelonggRAG {prompt_label} prompts generated using {llm} & {llm_model}")
    
    generated_comments_json = {}
    if(f1_prompt_corpus and prompt_label not in ["P_Custom"]):
        filter = filters[f1_prompt_corpus]
    else:
        filter = None

    f1_generated_prompt = f1_generated_prompt[prompt_label]['generated_prompt']
    logger.info(f1_send_along_customised_prompt)
    chunk_match_text = ""
    for i in f1_send_along_customised_prompt:
        if(i=='Document.Summary'):
            chunk_match_text += f"{pdf_summary}\n"
        else:
            chunk_match_text += f"{pdf_text}\n"
    chunk_match_text += f1_generated_prompt
    
    logger.info(f"Fetching relevant chunks for BelonggRAG {prompt_label}")
    relevant_chunks = pinecone_utils.extract_unique_chunks_langchain(chunk_match_text, top_k=f1_chunks, multiplier=3,filter=filter)  
    logger.info(f"Relevant chunks for BelonggRAG {prompt_label} fetched")
    context = relevant_chunks['all_context'] + "\n\n" + received_wisdom
    logger.info(f"Generating comments for BelonggRAG {prompt_label} using {llm} & {llm_model}")
    f1_generated_comments, prompt_with_example, input_tokens, output_tokens = generate_prompt_analyze_comments_langchain(f1_generated_prompt,context, f1_wisdom_1, f1_wisdom_2, model, llm, llm_model)
    logger.info(f"Comments generated for BelonggRAG {prompt_label} using {llm} & {llm_model}")
    analyze_comments = []
    if('json' in f1_generated_comments):
        l_index = f1_generated_comments.find('{')
        r_index = f1_generated_comments.rfind('}')
        f1_generated_comments = f1_generated_comments[l_index:r_index+1]
    try:
        generated_comments_json = json.loads(f1_generated_comments)

        if isinstance(generated_comments_json['Comments'], list):
            for comment in generated_comments_json['Comments']:
                if isinstance(comment, dict):
                    for key, value in comment.items():
                        analyze_comments.append({"comment": f"{key}: {value}"})
                else:
                    analyze_comments.append({"comment": comment})
        elif isinstance(generated_comments_json['Comments'], str):
            analyze_comments.append({"comment": generated_comments_json['Comments']})
        else:
            raise Exception(f"Invalid value for generated_comments_json['Comments'],{generated_comments_json}")
    except Exception as e:
        logger.info(f"Analyze Comment Exception: {str(e)}")
        traceback.print_exc() 
        analyze_comments.append({"comment":"Invalid comment generated by GPT","sources":[]})
    logger.info(f"{prompt_label} output generated using BelonggRAG, {llm} & {llm_model}")
    prompts = {"base_prompt":f1_base_prompt,"customization_prompt":f1_custom_prompt,"generated_prompt":f1_generated_prompt, "llm_flow":"BelonggRAG","llm":llm, "model":llm_model}
    contexts = {'wisdom_1':f1_wisdom_1,"wisdom_2":f1_wisdom_2,'context':context}
    return prompt_label,analyze_comments, relevant_chunks, prompts, contexts


def generate_flow2_output(prompt_label, generated_prompt, context,model,llm_type, llm_model):
    response_list = [
        {"role": "system", "content": generated_prompt},
        {"role": "user", "content": context}
    ]
    
    prompt_template = ChatPromptTemplate.from_messages([
                        (
                            "system",
                            generated_prompt,
                        ),
                        ("user", "{user_input}"),
                    ])
    
    llm_type = llm_type.lower()
    llm_model = llm_model.lower()
    if llm_type == "chatgpt":
        if str(llm_model).lower() in ["o3", "gpt-5"]:
            llm = ChatOpenAI(model=llm_model, api_key=api_key, organization=organization, temperature=1)
        else:
            llm = ChatOpenAI(model=llm_model, api_key=api_key, organization=organization)
    elif llm_type == "claude":
        llm = ChatAnthropic(model=llm_model,anthropic_api_key=claude_api_key)
    else:
        if str(llm_model).lower() in ["o3", "gpt-5"]:
            llm = ChatOpenAI(model=llm_model, api_key=api_key, organization=organization, temperature=1)
        else:
            llm = ChatOpenAI(model=llm_model, api_key=api_key, organization=organization)
        
    chain = prompt_template | llm

    with get_openai_callback() as cb:
        response = chain.invoke({"user_input":context})

    gpt_response = response.content
    
    input_tokens, output_tokens = cb.prompt_tokens, cb.completion_tokens
    
    return prompt_label, gpt_response, response_list,input_tokens, output_tokens


def generate_flow2_output_multithreaded(generated_prompts, summary_text,model):
    total_input_tokens = {}
    total_output_tokens = {}
    flow2_response={}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_flow2_output,prompt_label,prompt_data, summary_text, model) for prompt_label, prompt_data in generated_prompts.items()]
        for future in concurrent.futures.as_completed(futures):
            prompt_label, generated_response, input_tokens, output_tokens = future.result()
            total_input_tokens[prompt_label] = input_tokens
            total_output_tokens[prompt_label] = output_tokens
            flow2_response[prompt_label] = {"generated_response":generated_response}
    
    return flow2_response, total_input_tokens, total_output_tokens


def get_directllm_output(prompt_label, prompt_flows, doc_type, identity, pdf_text, pdf_summary, model, filters, dependencies,result_store,st):
    logger.info(f"Generating DirectLLM Output for {prompt_label}")
    selected_prompts = {}
    f2_base_prompt = prompt_flows[prompt_label]["base_prompt"]
    f2_custom_prompt = prompt_flows[prompt_label]["customization_prompt"]
    f2_wisdom_1 = prompt_flows[prompt_label]["wisdom_1"]
    f2_wisdom_2 = prompt_flows[prompt_label]["wisdom_2"]
    f2_which_chunks = prompt_flows[prompt_label]["which_chunks"]
    f2_wisdom_received = prompt_flows[prompt_label]["wisdom_received"]
    llm = prompt_flows[prompt_label]["llm"]
    llm_model = prompt_flows[prompt_label]["model"]
    label_for_output = prompt_flows[prompt_label]["label_for_output"]
    show_on_frontend = prompt_flows[prompt_label]["show_on_frontend"]
    for dep in dependencies[prompt_label]:
        if dep in result_store:
            if(common_utils.has_placeholder(f2_wisdom_1, str(dep))):
                    f2_wisdom_1 = common_utils.replace_placeholder(f2_wisdom_1,str(dep), result_store[dep])
            if(common_utils.has_placeholder(f2_wisdom_2, str(dep))):
                    f2_wisdom_2 = common_utils.replace_placeholder(f2_wisdom_2,str(dep), result_store[dep])
            if(common_utils.has_placeholder(f2_base_prompt, str(dep))):
                    f2_base_prompt = common_utils.replace_placeholder(f2_base_prompt,str(dep), result_store[dep])
            if(common_utils.has_placeholder(f2_custom_prompt, str(dep))):
                    f2_custom_prompt = common_utils.replace_placeholder(f2_custom_prompt,str(dep), result_store[dep])

    received_wisdom = "Wisdom_Received" + "\n\n"
    for wisdom in f2_wisdom_received:
        if wisdom in result_store:
            logger.info(f"{prompt_label}.F2 using wisdom from {wisdom}")
            received_wisdom += str(result_store[wisdom]) + "\n\n"
            
    f2_chunks = prompt_flows[prompt_label]["chunks"]
    f2_prompt_corpus = prompt_flows[prompt_label]["prompt_corpus"]
    f2_customise_prompt_based_on = prompt_flows[prompt_label]["customize_prompt_based_on"]
    f2_send_along_customised_prompt = prompt_flows[prompt_label]["send_along_customised_prompt"]
    logger.info(f"DirectLLM {prompt_label} Customize prompt based on: {f2_customise_prompt_based_on}")
    summary_text = pdf_summary
    for customizing_string in f2_customise_prompt_based_on:
        if customizing_string =="Document.Summary":
            if(common_utils.has_placeholder(f2_custom_prompt, "Document.Summary")):
                    f2_custom_prompt = common_utils.replace_placeholder(f2_custom_prompt,"Document.Summary","summary of the document")
                    summary_text = pdf_summary
        else:
            if(common_utils.has_placeholder(f2_custom_prompt, "Document.Full")):
                    f2_custom_prompt = common_utils.replace_placeholder(f2_custom_prompt,"Document.Full","full document text")
                    summary_text = pdf_text
    logger.info(f"Generating prompts for DirectLLM {prompt_label} using {llm} & {llm_model}")
    selected_prompts[prompt_label] = (f2_custom_prompt,f2_base_prompt, f2_wisdom_1, f2_wisdom_2,f2_chunks, "", {}, f2_which_chunks, f2_wisdom_received,"DirectLLM",llm,llm_model, label_for_output, show_on_frontend)
    f2_generated_prompt, f2_generate_prompt_input_token, f2_generate_prompt_output_token = generate_prompts(model,doc_type,summary_text,pdf_text,False,selected_prompts)
    logger.info(f"Prompts generated for DirectLLM {prompt_label} using {llm} & {llm_model}")
    f2_context = ""
    logger.info(f2_send_along_customised_prompt)
    for i in f2_send_along_customised_prompt:
        if(i=='Document.Summary'):
            f2_context += f"{pdf_summary}\n"
        else:
            f2_context += f"{pdf_text}\n"
    
    f2_context += "\n\n" + received_wisdom
    f2_generated_prompt = f2_generated_prompt[prompt_label]['generated_prompt']
    logger.info(f"Generating comments for DirectLLM {prompt_label} using {llm} & {llm_model}")
    prompt_label, f2_output,f2_response_list, f2_input_tokens, f2_output_tokens = generate_flow2_output(prompt_label, f2_generated_prompt, f2_context,model,llm, llm_model)
    logger.info(f"Comments generated for DirectLLM {prompt_label} using {llm} & {llm_model}")
    logger.info(f"{prompt_label} output generated using DirectLLM, {llm} & {llm_model}")
    prompts = {"base_prompt":f2_base_prompt,"customization_prompt":f2_custom_prompt,"generated_prompt":f2_generated_prompt,"llm_flow":"DirectLLM","llm":llm,"model":model}
    contexts = {'wisdom_1':f2_wisdom_1,"wisdom_2":f2_wisdom_2,'context':f2_context}
    return prompt_label, f2_output, {}, prompts, contexts


def get_hyde_output(prompt_label, prompt_flows, doc_type, identity, pdf_text, pdf_summary, model, filters, dependencies,result_store,sti):
    logger.info(f"Generating HYDE output for {prompt_label}")
    # HYDE Flow
    selected_prompts = {}
    
    f3_base_prompt = prompt_flows[prompt_label]["base_prompt"]
    f3_custom_prompt = prompt_flows[prompt_label]["customization_prompt"]
    f3_wisdom_1 = prompt_flows[prompt_label]["wisdom_1"]
    f3_wisdom_2 = prompt_flows[prompt_label]["wisdom_2"]
    f3_chunks = prompt_flows[prompt_label]["chunks"]
    f3_prompt_corpus = prompt_flows[prompt_label]["prompt_corpus"]
    f3_which_chunks = prompt_flows[prompt_label]["which_chunks"]
    f3_wisdom_received = prompt_flows[prompt_label]["wisdom_received"]
    f3_customise_prompt_based_on = prompt_flows[prompt_label]["customize_prompt_based_on"]
    f3_send_along_customised_prompt = prompt_flows[prompt_label]["send_along_customised_prompt"]
    llm = prompt_flows[prompt_label]["llm"]
    llm_model = prompt_flows[prompt_label]["model"]
    label_for_output = prompt_flows[prompt_label]["label_for_output"]
    show_on_frontend = prompt_flows[prompt_label]["show_on_frontend"]
    
    for dep in dependencies[prompt_label]:
        if dep in result_store:
            if(common_utils.has_placeholder(f3_wisdom_1, str(dep))):
                    f3_wisdom_1 = common_utils.replace_placeholder(f3_wisdom_1,str(dep), result_store[dep])
            if(common_utils.has_placeholder(f3_wisdom_2, str(dep))):
                    f3_wisdom_2 = common_utils.replace_placeholder(f3_wisdom_2,str(dep), result_store[dep])
            if(common_utils.has_placeholder(f3_base_prompt, str(dep))):
                    f3_base_prompt = common_utils.replace_placeholder(f3_base_prompt,str(dep), result_store[dep])
            if(common_utils.has_placeholder(f3_custom_prompt, str(dep))):
                    f3_custom_prompt = common_utils.replace_placeholder(f3_custom_prompt,str(dep), result_store[dep])

    received_wisdom = "Wisdom_Received" + "\n\n"
    for wisdom in f3_wisdom_received:
        if wisdom in result_store:
            logger.info(f"{prompt_label} using wisdom from {wisdom}")
            received_wisdom += str(result_store[wisdom]) + "\n\n"
            
    logger.info(f3_customise_prompt_based_on)
    summary_text = pdf_summary
    for customizing_string in f3_customise_prompt_based_on:
        if customizing_string =="Document.Summary":
            if(common_utils.has_placeholder(f3_custom_prompt, "Document.Summary")):
                    f3_custom_prompt = common_utils.replace_placeholder(f3_custom_prompt,"Document.Summary","summary of the document")
                    summary_text = pdf_summary
        else:
            if(common_utils.has_placeholder(f3_custom_prompt, "Document.Full")):
                    f3_custom_prompt = common_utils.replace_placeholder(f3_custom_prompt,"Document.Full","full document text")
                    summary_text = pdf_text

    selected_prompts[prompt_label] = (f3_custom_prompt,f3_base_prompt, f3_wisdom_1, f3_wisdom_2,f3_chunks, "", {}, f3_which_chunks, f3_wisdom_received,"BelonggRAG",llm,llm_model,label_for_output,show_on_frontend)
    logger.info(f"Generating HYDE {prompt_label} prompts using {llm} & {llm_model}")
    f3_generated_prompt, f3_generate_prompt_input_token, f3_generate_prompt_output_token = generate_prompts(model,doc_type,summary_text,pdf_text,False,selected_prompts)
    logger.info(f"HYDE {prompt_label} prompts generated using {llm} & {llm_model}")
    
    generated_comments_json = {}
    if(f3_prompt_corpus and prompt_label not in ["P_Custom"]):
        filter = filters[f3_prompt_corpus]
    else:
        filter = None

    f3_generated_prompt = f3_generated_prompt[prompt_label]['generated_prompt']
    logger.info(f3_send_along_customised_prompt)
    chunk_match_text = ""
    for i in f3_send_along_customised_prompt:
        if(i=='Document.Summary'):
            chunk_match_text += f"{pdf_summary}\n"
        elif (i=='Document.Full'):
            chunk_match_text += f"{pdf_text}\n"
        else:
            if i in result_store.keys():
                chunk_match_text += f"{str(result_store[i])}\n"
    
    chunk_match_text += f3_generated_prompt
    
    logger.info(f"Fetching relevant chunks for HYDE {prompt_label}")
    relevant_chunks = pinecone_utils.extract_unique_chunks_langchain(chunk_match_text, top_k=f3_chunks, multiplier=3,filter=filter)  
    logger.info(f"Relevant chunks for HYDE {prompt_label} fetched")
    context = relevant_chunks['all_context'] + "\n\n" + received_wisdom
    logger.info(f"Generating comments for HYDE {prompt_label} using {llm} & {llm_model}")
    f1_generated_comments, prompt_with_example, input_tokens, output_tokens = generate_prompt_analyze_comments_langchain(f3_generated_prompt,context, f3_wisdom_1, f3_wisdom_2, model, llm, llm_model)
    logger.info(f"Comments generated for HYDE {prompt_label} using {llm} & {llm_model}")
    analyze_comments = []
    if('json' in f1_generated_comments):
        l_index = f1_generated_comments.find('{')
        r_index = f1_generated_comments.rfind('}')
        f1_generated_comments = f1_generated_comments[l_index:r_index+1]
    try:
        generated_comments_json = json.loads(f1_generated_comments)

        if isinstance(generated_comments_json['Comments'], list):
            for comment in generated_comments_json['Comments']:
                if isinstance(comment, dict):
                    for key, value in comment.items():
                        analyze_comments.append({"comment": f"{key}: {value}"})
                else:
                    analyze_comments.append({"comment": comment})
        elif isinstance(generated_comments_json['Comments'], str):
            analyze_comments.append({"comment": generated_comments_json['Comments']})
        else:
            raise Exception(f"Invalid value for generated_comments_json['Comments'],{generated_comments_json}")
    except Exception as e:
        logger.info(f"Analyze Comment Exception: {str(e)}")
        traceback.print_exc() 
        analyze_comments.append({"comment":"Invalid comment generated by GPT","sources":[]})
    logger.info(f"{prompt_label} output generated using HYDE, {llm} & {llm_model}")
    prompts = {"base_prompt":f3_base_prompt,"customization_prompt":f3_custom_prompt,"generated_prompt":f3_generated_prompt, "llm_flow":"BelonggRAG","llm":llm, "model":llm_model}
    contexts = {'wisdom_1':f3_wisdom_1,"wisdom_2":f3_wisdom_2,'context':context}
    return prompt_label,analyze_comments, relevant_chunks, prompts, contexts


def analyze_text_flows(pdf_text, pdf_summary,doc_type, identity,parent_prompt_label, prompt_info,prompt_flows, filters, model, organization_id, dependencies, result_store,completed,st):
    result = {}
    prompt_contexts = {}
    result_prompts = {}
    flow_dependencies = {}
    for i in prompt_flows.keys():
        flow_dependencies[i] = prompt_flows[i]['additional_dependencies']
    
    flows_ordered = common_utils.topological_sort(flow_dependencies)
    flows_ordered = [item for item in flows_ordered if item in flow_dependencies]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}

        while flows_ordered:
            parallel_tasks = []
            for prompt_label in list(flows_ordered):
                if all(dep in completed for dep in flow_dependencies[prompt_label]):
                    if prompt_flows[prompt_label]["llm_flow"] == "ABCDRAG":
                        futures[prompt_label] = executor.submit(get_belonggrag_output,prompt_label,prompt_flows, doc_type, identity, pdf_text, pdf_summary, model, filters, flow_dependencies,result_store,True)
                    elif prompt_flows[prompt_label]["llm_flow"] == "DirectLLM":
                        futures[prompt_label] = executor.submit(get_directllm_output,prompt_label,prompt_flows, doc_type, identity, pdf_text, pdf_summary, model, filters, flow_dependencies,result_store,st)
                    elif prompt_flows[prompt_label]["llm_flow"] == "HYDE":
                        futures[prompt_label] = executor.submit(get_hyde_output,prompt_label,prompt_flows, doc_type, identity, pdf_text, pdf_summary, model, filters, flow_dependencies,result_store,st)
                    parallel_tasks.append(prompt_label)
                    flows_ordered.remove(prompt_label)


            if parallel_tasks:
                logger.info(f"Processing in parallel: {parallel_tasks}")

            for future in concurrent.futures.as_completed(futures.values()):
                input_tokens, output_tokens = 0,0
                prompt_label, generated_comments, relevant_chunks, prompts,contexts = future.result()
                completed.add(prompt_label)
                result[prompt_label] = {"comments":generated_comments, "relevant_chunks":relevant_chunks}
                result_prompts[prompt_label] = prompts
                result_store[prompt_label] = generated_comments
                prompt_contexts[prompt_label] = contexts
    return result, result_prompts, prompt_contexts


def analyze_text_multiflow(pdf_text, pdf_summary, doc_type, identity,prompt_label, prompt_info, prompt_flow, filters, model, organization_id, dependencies, result_store,completed,st):
    
    generated_prompt = prompt_info["generated_prompt"]
    prompt_chunks_count = prompt_info["chunks"]
    prompt_corpus = prompt_info["prompt_corpus"]
    wisdom_1 = prompt_info["wisdom_1"]
    wisdom_2 = prompt_info["wisdom_2"]
    which_chunks = prompt_info["which_chunks"]
    wisdom_received = prompt_info["wisdom_received"]
    llm = prompt_info["llm"]
    llm_model = prompt_info["model"]
    input_tokens, output_tokens = 0,0
    knowledge_objects, knowledge_chunks = {},{}
    generated_comments_json = {}
    prompt_flow_results,prompt_flow_prompts, prompt_flow_contexts = {},{},{}
    relevant_chunks_f1, relevant_chunks_f2,relevant_chunks_f3, relevant_chunks = {},{},{},{}
    print(f"{prompt_label} has the value in result_store as {result_store.keys()}")
    print(f"{prompt_label} has the value in completed set as {completed}")
    
    if(len(prompt_flow) > 0):
        prompt_flow_results, prompt_flow_prompts, prompt_flow_contexts = analyze_text_flows(pdf_text, pdf_summary,doc_type, identity,prompt_label, prompt_info,prompt_flow, filters, model, organization_id, dependencies, result_store,completed,st)
    all_wisdoms = {}
    if(f"{prompt_label}.F1" in prompt_flow_results.keys()):
        f1_comments = prompt_flow_results[f"{prompt_label}.F1"]["comments"]
        relevant_chunks_f1 = prompt_flow_results[f"{prompt_label}.F1"]["relevant_chunks"]
        all_wisdoms[f"{prompt_label}.F1"] = str(f1_comments)
    if(f"{prompt_label}.F2" in prompt_flow_results.keys()):
        f2_comments = prompt_flow_results[f"{prompt_label}.F2"]["comments"]
        relevant_chunks_f2 = prompt_flow_results[f"{prompt_label}.F2"]["relevant_chunks"]
        all_wisdoms[f"{prompt_label}.F2"] = str(f2_comments)
    
    print(wisdom_received)
    received_wisdom = "Wisdom_Received" + "\n\n"
    for wisdom in wisdom_received:
        if wisdom in all_wisdoms.keys():
            logger.info(f"{prompt_label} using wisdom from {wisdom}")
            received_wisdom += str(all_wisdoms[wisdom]) + "\n\n"
        elif wisdom in result_store.keys():
            logger.info(f"{prompt_label} using wisdom from {wisdom}")
            received_wisdom += str(result_store[wisdom]) + "\n\n"
    
    print(which_chunks)
    relevant_chunks = common_utils.merge_unique_chunks(prompt_label, which_chunks, relevant_chunks_f1, relevant_chunks_f2,relevant_chunks_f3)
    
    if prompt_label == "P_Custom":
        filter = {"Organization": {"$eq": organization_id}}
        chunk_match_text = f"{pdf_summary}\n{generated_prompt}"
        relevant_chunks = pinecone_utils.extract_unique_chunks_langchain(chunk_match_text, top_k=prompt_chunks_count, multiplier=3,filter=filter)  
        context = relevant_chunks['all_context']
        
    sources_info = {}
    context = "Prompt_Context\n\n"
    if len(relevant_chunks) > 0:
        for i in range(prompt_chunks_count):
            try:
                name = relevant_chunks[f'meta_{i+1}']
                url = relevant_chunks[f'url_{i+1}']
                chunk_id=str(int(relevant_chunks[f'chunk_id_{i+1}']))
                sources_info[chunk_id] = {"name":name,"url":url}
                context += relevant_chunks[f'context_{i+1}'] + "\n"
            except Exception as e:
                print(f"Exception: {str(e)}")
                traceback.print_exc() 
                
    # inputs = [result_store[dep] for dep in dependencies[prompt_label]]
    deps = []
    inputs = []
    dependency_context = ""
    for dep in dependencies[prompt_label]:
        inputs.append(result_store[dep])
        deps.append(dep)
    print(f"{prompt_label} using output of {deps}")
    if(len(inputs)>0):
        dependency_context = "ContextChain"
        for i in range(len(inputs)):
            dependency_context += f"\n\n{inputs[i]}"

    context += "\n\n" + dependency_context + "\n\n" + received_wisdom

    generated_comments, prompt_with_example, input_tokens, output_tokens = generate_prompt_analyze_comments_langchain(generated_prompt,context, wisdom_1, wisdom_2, model,llm, llm_model)
    if wisdom_1 and wisdom_2:
        context = f"{context} \n\n Wisdom 1: \n {wisdom_1} \n\n Wisdom 2: \n {wisdom_2}"

    analyze_comments = []
    if('json' in generated_comments):
        l_index = generated_comments.find('{')
        r_index = generated_comments.rfind('}')
        generated_comments = generated_comments[l_index:r_index+1]
    try:
        generated_comments_json = json.loads(generated_comments)
        sources = list(sources_info.values())

        directory = "temp_files"
        file_name = f"{prompt_label}_generated_comments.json"
        file_name_2 = f"{prompt_label}_context.json"
        file_path = os.path.join(directory, file_name)
        file_path_2 = os.path.join(directory, file_name_2)

        os.makedirs(directory, exist_ok=True)

        # Write the content to the file
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(generated_comments_json, file, ensure_ascii=False, indent=4)
        
        with open(file_path_2, "w",encoding="utf-8") as file_2:
            file_2.write(str(prompt_with_example))

        if isinstance(generated_comments_json['Comments'], list):
            for comment in generated_comments_json['Comments']:
                if isinstance(comment, dict):
                    for key, value in comment.items():
                        analyze_comments.append({"comment": f"{key}: {value}", "sources": sources})
                else:
                    analyze_comments.append({"comment": comment, "sources": sources})
        elif isinstance(generated_comments_json['Comments'], str):
            analyze_comments.append({"comment": generated_comments_json['Comments'], "sources": sources})
        else:
            raise Exception(f"Invalid value for generated_comments_json['Comments'],{generated_comments_json}")
    except Exception as e:
        print(f"Analyze Comment Exception: {str(e)}")
        traceback.print_exc() 
        analyze_comments.append({"comment":"Invalid comment generated by GPT","sources":[]})
    # result_store[prompt_label] = generated_comments_json
    return prompt_corpus, generated_comments_json,knowledge_objects, knowledge_chunks, analyze_comments, prompt_label, context, relevant_chunks, prompt_flow_results,prompt_flow_prompts,prompt_flow_contexts,input_tokens, output_tokens


def generate_analyze_comments_dependencies(pdf_text, doc_type, identity,pdf_summary,filters,model,generated_prompts,organization_id,section_titles,dependencies,topo_order,prompt_label_flows,st=None):
    total_input_tokens = {}
    total_output_tokens = {}
    analyze_context_used={}
    generated_analyze_comments={}
    relevant_chunks={}
    collected_data = []
    result_store = {}
    prompt_label_flow = {}
    completed = set()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
    
        while topo_order:
            parallel_tasks = []
            for prompt_label in list(topo_order):
                if all(dep in completed for dep in dependencies[prompt_label]):
                    if prompt_label == "P_Custom":
                        futures[prompt_label] = executor.submit(analyze_text_multiflow,pdf_text,pdf_summary,doc_type,identity, prompt_label, generated_prompts[prompt_label],{},filters,model, organization_id, dependencies,result_store,completed,st)
                    else:
                        futures[prompt_label] = executor.submit(analyze_text_multiflow,pdf_text,pdf_summary,doc_type,identity, prompt_label, generated_prompts[prompt_label],prompt_label_flows[prompt_label],filters,model, organization_id, dependencies,result_store,completed,st)
                    parallel_tasks.append(prompt_label)
                    topo_order.remove(prompt_label)


            if parallel_tasks:
                print(f"Processing in parallel: {parallel_tasks}")

            for future in concurrent.futures.as_completed(futures.values()):
                input_tokens, output_tokens = 0,0
                prompt_corpus, generated_comments,knowledge_objects, knowledge_chunks, analyze_comments,prompt_label,context,relevant_chunks, prompt_flow_result, prompt_flow_prompts,prompt_flow_contexts,input_tokens, output_tokens = future.result()
                completed.add(prompt_label)
                section_title = section_titles[prompt_label]
                if(len(knowledge_objects) == 0):
                    prompt_analyze_comments={"section_title":section_title,"prompt_corpus":prompt_corpus,"analyze_comments":analyze_comments}
                else:
                    prompt_analyze_comments={"section_title":section_title,"prompt_corpus":prompt_corpus,"knowledge_objects":knowledge_objects,"analyze_comments":analyze_comments}
                generated_analyze_comments[prompt_label]=prompt_analyze_comments
                analyze_context_used[prompt_label]=context
                total_input_tokens[prompt_label] = input_tokens
                total_output_tokens[prompt_label] = output_tokens
                result_store[prompt_label] = generated_comments
                prompt_label_flow[prompt_label] = {"prompt_flow_result":prompt_flow_result,"prompt_flow_prompts":prompt_flow_prompts, "prompt_flow_contexts":prompt_flow_contexts}
                collected_data.append((prompt_label, prompt_corpus, prompt_analyze_comments))

    print(result_store.keys())
    return generated_analyze_comments,analyze_context_used, prompt_label_flow,relevant_chunks, collected_data, total_input_tokens, total_output_tokens


def generate_evaluator_prompt(prompt_for_customization,selected_model, base_prompt, nature, identity,tor_summary_text):
    
    if(common_utils.has_placeholder(prompt_for_customization, "Document.Summary")):
            prompt_for_customization = common_utils.replace_placeholder(prompt_for_customization,"Document.Summary","summary of the document")

    if(common_utils.has_placeholder(prompt_for_customization, "Document.Full")):
            prompt_for_customization = common_utils.replace_placeholder(prompt_for_customization,"Document.Full","full document text")
    
    prompt = f""" Nature of Document: {nature}, Identity of Document: {identity}, Terms Of Reference Summary : {tor_summary_text}, Base Prompt:{base_prompt}"""
    
    system_role = f"""{prompt_for_customization}"""
    
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            system_role,
        ),
        ("user", "{prompt}"),
    ])
    print(f"Selected model: {selected_model}")
    # Set temperature explicitly for "o*" models (e.g., o3) which only support the default (1)
    if str(selected_model) in ["o3", "gpt-5"]:
        llm = ChatOpenAI(model=selected_model, api_key=api_key, organization=organization, temperature=1)
    else:
        llm = ChatOpenAI(model=selected_model, api_key=api_key, organization=organization)
    
    chain = prompt_template | llm

    with get_openai_callback() as cb:
        response = chain.invoke({"prompt":prompt})

    input_tokens, output_tokens = cb.prompt_tokens, cb.completion_tokens

    return response.content, input_tokens, output_tokens
    

def generate_evaluator_customised_prompt(selected_model,nature,identity,prompt_label,tor_summary_text,prompt_data):
    prompt_for_customization = prompt_data["customization_prompt"]
    base_prompt = prompt_data["base_prompt"]
    wisdom_1 = prompt_data["wisdom_1"]
    wisdom_2 = prompt_data["wisdom_2"]
    chunks = prompt_data["chunks"]
    customize_prompt_based_on = prompt_data["customize_prompt_based_on"]
    send_along_customised_prompt = prompt_data["send_along_customised_prompt"]
    wisdom_received = prompt_data["wisdom_received"]
    llm_flow = prompt_data["llm_flow"]
    llm = prompt_data["llm"]
    model = prompt_data["model"]
    model = model.lower()
    generated_prompt, input_tokens, output_tokens = generate_evaluator_prompt(prompt_for_customization,model, base_prompt, nature, identity,tor_summary_text)
    
    return prompt_label, generated_prompt, wisdom_1, wisdom_2, chunks, customize_prompt_based_on,send_along_customised_prompt,wisdom_received,llm_flow, llm, model,input_tokens, output_tokens


def generate_evaluator_prompts(selected_model,nature,identity,tor_summary_text,evaluator_prompts):
    total_input_tokens = {}
    total_output_tokens = {}
    generated_prompts={}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_evaluator_customised_prompt,selected_model,nature, identity, prompt_label,tor_summary_text, prompt_data) for prompt_label, prompt_data in evaluator_prompts.items()]
        for future in concurrent.futures.as_completed(futures):
            prompt_label, generated_prompt, wisdom_1, wisdom_2, chunks, customize_prompt_based_on,send_along_customised_prompt,wisdom_received,llm_flow, llm, model,input_tokens, output_tokens = future.result()
            total_input_tokens[prompt_label] = input_tokens
            total_output_tokens[prompt_label] = output_tokens
            generated_prompts[prompt_label] = {"generated_prompt":generated_prompt,"wisdom_1":wisdom_1, "wisdom_2":wisdom_2,"customize_prompt_based_on":customize_prompt_based_on,"send_along_customised_prompt":send_along_customised_prompt,"wisdom_received":wisdom_received,"llm_flow":llm_flow, "llm":llm, "model":model,"chunks":chunks}
    
    return generated_prompts, total_input_tokens, total_output_tokens


def get_evaluator_messages(evaluator_prompt,context, wisdom_1, wisdom_2):
    system_output_instruction = """
        Instruction for Response Generation:
        1. Provide the response strictly in markdown format with markdown formatting. 
        2. Do not add ```markdown with the start or end of response. Just make the formatting as markdown. Do not provide the response in simple plain text. 
        3. Strictly adhere to the markdown response format.
        """
    if wisdom_1 and wisdom_2:
        user_input = f"""```{context}``` \n\n Wisdom1: \n\n ```{wisdom_1}``` \n\n Wisdom2: \n\n ```{wisdom_2}```"""
        system_role=evaluator_prompt + "\n\nThe context, wisdom1 and wisdom2 are enclosed in backticks\n\n" + system_output_instruction + "\n\nCONTEXT: \n\n " 
    else:
        user_input = "```"+ context + "````"
        system_role=evaluator_prompt + "\n\nThe context is enclosed in three backticks\n\n" + system_output_instruction + "\n\nCONTEXT: \n\n " 
        
    # Write system role to the text file
    # with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "system_role.txt"), "w") as f:
    #     f.write(system_role)
    response_list = [
        {"role": "system", "content": system_role},]
    
    # user_input = f"""Context:{context}"""
   
    response_list.extend([
        {"role": "user", "content": user_input}
    ])

    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            system_role,
        ),
        ("user", "{user_input}"),
    ])
    
    return prompt_template, response_list,user_input


@traceable(tags=["evaluator-comments-generator"])
def generate_evaluator_output(proposal_text, p_internal_prompt, model, wisdom_1, wisdom_2, prompt_label=None, dependencies=None, tokens_counter=None):
    response_list, context = get_evaluator_messages(p_internal_prompt, proposal_text, wisdom_1, wisdom_2)
    
    prompt = response_list[1]["content"]

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", response_list[0]["content"]),
            ("user", "{prompt}"),
        ]
    )
    
    chain = prompt_template | llm
    
    response = chain.invoke({"prompt": prompt})
    
    return response.content, response_list
    
    response = client.chat.completions.create(
        model=model,
        messages=response_list
    )
    gpt_response = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    
    if tokens_counter is not None and prompt_label is not None:
        prompt_label = prompt_label.lower()
        tokens_counter[0]["evaluator_comments"][prompt_label] = input_tokens
        tokens_counter[1]["evaluator_comments"][prompt_label] = output_tokens
    
    return gpt_response, response_list


def generate_evaluator_comments_old(proposal_text, model, generated_evaluator_prompts, analyze_comments, st=None):
    
    p_internal_output=""
    p_external_output=""
    p_delta_output=""
    p_internal_response_list = ""
    p_external_response_list = ""
    p_delta_response_list = ""
    evaluator_response = {}
    
    if "P_Internal" in generated_evaluator_prompts:
        p_internal_prompt = generated_evaluator_prompts["P_Internal"]
        p_internal_output,p_internal_response_list = generate_evaluator_output(proposal_text,p_internal_prompt,model)
        evaluator_response["P_Internal"]=p_internal_output
          
    if "P_External" in generated_evaluator_prompts:
        p_external_prompt = generated_evaluator_prompts["P_External"]
        p_external_output,p_external_response_list = generate_evaluator_output(analyze_comments, p_external_prompt, model)
        evaluator_response["P_External"]=p_external_output
    
    if "P_Delta" in generated_evaluator_prompts:
        p_delta_prompt = generated_evaluator_prompts["P_Delta"]
        p_delta_context = f"{p_internal_output}\n{p_external_output}"
        p_delta_output,p_delta_response_list = generate_evaluator_output(p_delta_context, p_delta_prompt, model)
        evaluator_response["P_Delta"]=p_delta_output
    
    return evaluator_response, p_internal_response_list, p_external_response_list, p_delta_response_list


def generate_evaluator_comments_concurrently(proposal_text, model, generated_evaluator_prompts, analyze_comments, evaluator_wisdom, tokens_counter=None, st=None):
    print("Generating Evaluator Comments Concurrently...")
    p_internal_output = ""
    p_external_output = ""
    p_delta_output = ""
    p_internal_response_list = ""
    p_external_response_list = ""
    p_delta_response_list = ""
    evaluator_response = {}
    
    dependency_list = {
        "p_internal": [],
        "p_external": ["p_internal"]
    }

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}

        if "P_Internal" in generated_evaluator_prompts:
            p_internal_prompt = generated_evaluator_prompts["P_Internal"]
            wisdom_1, wisdom_2 = evaluator_wisdom["P_Internal"][0], evaluator_wisdom["P_Internal"][1]
            futures["P_Internal"] = executor.submit(
                generate_evaluator_output, proposal_text, p_internal_prompt, model, wisdom_1, wisdom_2, "p_internal", dependency_list, tokens_counter
            )

        if "P_External" in generated_evaluator_prompts:
            p_external_prompt = generated_evaluator_prompts["P_External"]
            wisdom_1, wisdom_2 = evaluator_wisdom["P_External"][0], evaluator_wisdom["P_External"][1]
            futures["P_External"] = executor.submit(
                generate_evaluator_output, analyze_comments, p_external_prompt, model, wisdom_1, wisdom_2, "p_external", dependency_list, tokens_counter
            )

        # Wait for both tasks to complete
        if "P_Internal" in futures:
            p_internal_output, p_internal_response_list = futures["P_Internal"].result()
            evaluator_response["P_Internal"] = p_internal_output

        if "P_External" in futures:
            p_external_output, p_external_response_list = futures["P_External"].result()
            evaluator_response["P_External"] = p_external_output

    # Execute P_Delta after P_Internal and P_External are complete
    if "P_Delta" in generated_evaluator_prompts:
        p_delta_prompt = generated_evaluator_prompts["P_Delta"]
        p_delta_context = f"{p_internal_output}\n{p_external_output}"
        wisdom_1, wisdom_2 = evaluator_wisdom["P_Delta"][0], evaluator_wisdom["P_Delta"][1]
        p_delta_output, p_delta_response_list = generate_evaluator_output(p_delta_context, p_delta_prompt, model, wisdom_1, wisdom_2, "p_delta", dependency_list, tokens_counter)
        evaluator_response["P_Delta"] = p_delta_output

    return (
        evaluator_response,
        p_internal_response_list,
        p_external_response_list,
        p_delta_response_list,
    )


def generate_evaluator_comments_concurrently_test(proposal_text, model, generated_evaluator_prompts, analyze_comments, evaluator_wisdom, dependency_graph, tokens_counter=None, st=None):
    start_time = time.time()
    evaluator_response = {}
    response_lists = {}

    def execute_evaluator(evaluator_name, input_text):
        print(f"Starting execution of {evaluator_name}...")
        evaluator_start_time = time.time()
        prompt = generated_evaluator_prompts[evaluator_name]
        wisdom_1, wisdom_2 = evaluator_wisdom[evaluator_name][0], evaluator_wisdom[evaluator_name][1]
        output, response_list = generate_evaluator_output(input_text, prompt, model, wisdom_1, wisdom_2, evaluator_name.lower(), dependency_graph, tokens_counter)
        evaluator_response[evaluator_name] = output
        response_lists[evaluator_name] = response_list
        evaluator_end_time = time.time()
        print(f"Finished execution of {evaluator_name}. Time taken: {evaluator_end_time - evaluator_start_time:.2f} seconds")
        return output

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures_dict = {}

        if "P_Internal" in generated_evaluator_prompts and "P_External" in generated_evaluator_prompts:
            if "P_External" in dependency_graph["P_Internal"]:
                print("Dependency detected: P_Internal depends on P_External")
                futures_dict["P_External"] = executor.submit(execute_evaluator, "P_External", analyze_comments)
                p_external_output = futures_dict["P_External"].result()
                futures_dict["P_Internal"] = executor.submit(execute_evaluator, "P_Internal", f"{proposal_text}\n\n Context 1 \n {p_external_output}")
            elif "P_Internal" in dependency_graph["P_External"]:
                print("Dependency detected: P_External depends on P_Internal")
                futures_dict["P_Internal"] = executor.submit(execute_evaluator, "P_Internal", proposal_text)
                p_internal_output = futures_dict["P_Internal"].result()
                futures_dict["P_External"] = executor.submit(execute_evaluator, "P_External", f"{analyze_comments}\n\n Context 1 \n {p_internal_output}")
            else:
                print("No dependencies detected, executing P_Internal and P_External in parallel")
                futures_dict["P_Internal"] = executor.submit(execute_evaluator, "P_Internal", proposal_text)
                futures_dict["P_External"] = executor.submit(execute_evaluator, "P_External", analyze_comments)
        else:
            print("Executing available evaluators")
            if "P_Internal" in generated_evaluator_prompts:
                futures_dict["P_Internal"] = executor.submit(execute_evaluator, "P_Internal", proposal_text)
            if "P_External" in generated_evaluator_prompts:
                futures_dict["P_External"] = executor.submit(execute_evaluator, "P_External", analyze_comments)

        print("Waiting for all evaluator tasks to complete...")
        concurrent.futures.wait(futures_dict.values())
        print("All evaluator tasks completed")

    if "P_Delta" in generated_evaluator_prompts:
        p_delta_context = f"{evaluator_response.get('P_Internal', '')}\n{evaluator_response.get('P_External', '')}"
        execute_evaluator("P_Delta", p_delta_context)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Finished: Generating Evaluator Comments Concurrently. Total time taken: {total_time:.2f} seconds")

    return (
        evaluator_response,
        response_lists.get("P_Internal", ""),
        response_lists.get("P_External", ""),
        response_lists.get("P_Delta", ""),
    )


def get_evaluator_flow1_output(prompt_label, prompt_flows, doc_type, identity, proposal_text, proposal_summary, model, dependencies,result_store):
    print(f"Generating {prompt_label}.F1 output")
    # F1 Flow
    selected_prompts = {}
    f1_prompt_label = f"{prompt_label}.F1"
    f1_base_prompt = prompt_flows[f"{prompt_label}.F1"]["base_prompt"]
    f1_custom_prompt = prompt_flows[f"{prompt_label}.F1"]["customization_prompt"]
    f1_wisdom_1 = prompt_flows[f"{prompt_label}.F1"]["wisdom_1"]
    f1_wisdom_2 = prompt_flows[f"{prompt_label}.F1"]["wisdom_2"]
    f1_chunks = prompt_flows[f"{prompt_label}.F1"]["chunks"]
    f1_wisdom_received = prompt_flows[f"{prompt_label}.F1"]["wisdom_received"]
    f1_customise_prompt_based_on = prompt_flows[f"{prompt_label}.F1"]["customize_prompt_based_on"]
    f1_send_along_customised_prompt = prompt_flows[f"{prompt_label}.F1"]["send_along_customised_prompt"]
    label_for_output = prompt_flows[f"{prompt_label}.F1"]["label_for_output"]
    show_on_frontend = prompt_flows[f"{prompt_label}.F1"]["show_on_frontend"]
    llm_flow = prompt_flows[f"{prompt_label}.F1"]["llm_flow"]
    llm = prompt_flows[f"{prompt_label}.F1"]["llm"]
    llm_model = prompt_flows[f"{prompt_label}.F1"]["model"]
    
    for dep in dependencies[f1_prompt_label]:
        if dep in result_store:
            if(common_utils.has_placeholder(f1_wisdom_1, str(dep))):
                    f1_wisdom_1 = common_utils.replace_placeholder(f1_wisdom_1,str(dep), result_store[dep])
            if(common_utils.has_placeholder(f1_wisdom_2, str(dep))):
                    f1_wisdom_2 = common_utils.replace_placeholder(f1_wisdom_2,str(dep), result_store[dep])
            if(common_utils.has_placeholder(f1_base_prompt, str(dep))):
                    f1_base_prompt = common_utils.replace_placeholder(f1_base_prompt,str(dep), result_store[dep])
            if(common_utils.has_placeholder(f1_custom_prompt, str(dep))):
                    f1_custom_prompt = common_utils.replace_placeholder(f1_custom_prompt,str(dep), result_store[dep])

    print(f1_customise_prompt_based_on)
    summary_text = proposal_summary
    for customizing_string in f1_customise_prompt_based_on:
        if customizing_string =="Document.Summary":
            if(common_utils.has_placeholder(f1_custom_prompt, "Document.Summary")):
                    f1_custom_prompt = common_utils.replace_placeholder(f1_custom_prompt,"Document.Summary","summary of the document")
                    summary_text = proposal_summary
        else:
            if(common_utils.has_placeholder(f1_custom_prompt, "Document.Full")):
                    f1_custom_prompt = common_utils.replace_placeholder(f1_custom_prompt,"Document.Full","full document text")
                    summary_text = proposal_text

    selected_prompts[f1_prompt_label] = (f1_custom_prompt,f1_base_prompt, f1_wisdom_1, f1_wisdom_2,f1_chunks, "", {}, [], f1_wisdom_received, llm_flow, llm, llm_model, label_for_output, show_on_frontend)
    print(f"Generating {prompt_label}.F1 prompts")
    print(f"{prompt_label}.F1 prompts generated")
    f1_generated_prompt, f1_generate_prompt_input_token, f1_generate_prompt_output_token = generate_prompts(model,doc_type,summary_text,proposal_text,False,selected_prompts)

    f1_generated_prompt = f1_generated_prompt[f1_prompt_label]['generated_prompt']
    print("Generating comments for F1")
    f1_generated_comments, prompt_with_example, input_tokens, output_tokens = generate_evaluator_Output(proposal_text,f1_generated_prompt,f1_wisdom_1, f1_wisdom_2,model, llm, llm_model)
    print("Comments generated for F1")
    print(f"{prompt_label}.F1 output generated")
    prompts = {"base_prompt":f1_base_prompt,"customization_prompt":f1_custom_prompt,"generated_prompt":f1_generated_prompt, "llm":llm, "model":llm_model}
    context = {'wisdom_1':f1_wisdom_1,'wisdom_2':f1_wisdom_2,'context':prompt_with_example}
    return f1_prompt_label,f1_generated_comments, prompts,context


def get_evaluator_flow2_output(prompt_label, prompt_flows, doc_type, identity, proposal_text, proposal_summary, model, dependencies,result_store):
    print(f"Generating {prompt_label}.F2 output")
    selected_prompts = {}
    f2_prompt_label = f"{prompt_label}.F2"
    f2_base_prompt = prompt_flows[f"{prompt_label}.F2"]["base_prompt"]
    f2_custom_prompt = prompt_flows[f"{prompt_label}.F2"]["customization_prompt"]
    f2_wisdom_1 = prompt_flows[f"{prompt_label}.F2"]["wisdom_1"]
    f2_wisdom_2 = prompt_flows[f"{prompt_label}.F2"]["wisdom_2"]
    f2_wisdom_received = prompt_flows[f"{prompt_label}.F2"]["wisdom_received"]
    label_for_output = prompt_flows[f"{prompt_label}.F2"]["label_for_output"]
    show_on_frontend = prompt_flows[f"{prompt_label}.F2"]["show_on_frontend"]
    llm_flow = prompt_flows[f"{prompt_label}.F2"]["llm_flow"]
    llm = prompt_flows[f"{prompt_label}.F2"]["llm"]
    llm_model = prompt_flows[f"{prompt_label}.F2"]["model"]
    
    for dep in dependencies[f2_prompt_label]:
        if dep in result_store:
            if(common_utils.has_placeholder(f2_wisdom_1, str(dep))):
                    f2_wisdom_1 = common_utils.replace_placeholder(f2_wisdom_1,str(dep), result_store[dep])
            if(common_utils.has_placeholder(f2_wisdom_2, str(dep))):
                    f2_wisdom_2 = common_utils.replace_placeholder(f2_wisdom_2,str(dep), result_store[dep])
            if(common_utils.has_placeholder(f2_base_prompt, str(dep))):
                    f2_base_prompt = common_utils.replace_placeholder(f2_base_prompt,str(dep), result_store[dep])
            if(common_utils.has_placeholder(f2_custom_prompt, str(dep))):
                    f2_custom_prompt = common_utils.replace_placeholder(f2_custom_prompt,str(dep), result_store[dep])

    received_wisdom = "Wisdom_Received" + "\n\n"
    for wisdom in f2_wisdom_received:
        if wisdom in result_store:
            print(f"{prompt_label}.F2 using wisdom from {wisdom}")
            received_wisdom += str(result_store[wisdom]) + "\n\n" 
        
    f2_chunks = prompt_flows[f"{prompt_label}.F2"]["chunks"]
    f2_customise_prompt_based_on = prompt_flows[f"{prompt_label}.F2"]["customize_prompt_based_on"]
    f2_send_along_customised_prompt = prompt_flows[f"{prompt_label}.F2"]["send_along_customised_prompt"]
    print(f"F2 Customize prompt based on: {f2_customise_prompt_based_on}")
    summary_text = proposal_summary
    for customizing_string in f2_customise_prompt_based_on:
        if customizing_string =="Document.Summary":
            if(common_utils.has_placeholder(f2_custom_prompt, "Document.Summary")):
                    f2_custom_prompt = common_utils.replace_placeholder(f2_custom_prompt,"Document.Summary","summary of the document")
                    summary_text = proposal_summary
        else:
            if(common_utils.has_placeholder(f2_custom_prompt, "Document.Full")):
                    f2_custom_prompt = common_utils.replace_placeholder(f2_custom_prompt,"Document.Full","full document text")
                    summary_text = proposal_text
    
    print("Generating prompts for F2")
    selected_prompts[f2_prompt_label] = (f2_custom_prompt,f2_base_prompt, f2_wisdom_1, f2_wisdom_2,f2_chunks, "", {}, [], f2_wisdom_received, llm_flow, llm, llm_model, label_for_output, show_on_frontend)
    f2_generated_prompt, f2_generate_prompt_input_token, f2_generate_prompt_output_token = generate_prompts(model,doc_type,summary_text,proposal_text,False,selected_prompts)
    print("Prompts generated for F2")
    f2_context = ""
    print(f2_send_along_customised_prompt)
    for i in f2_send_along_customised_prompt:
        if(i=='Document.Full'):
            f2_context += f"{proposal_text}\n"
        else:
            f2_context += f"{proposal_summary}\n"

    f2_context += "\n\n" + received_wisdom
    
    f2_generated_prompt = f2_generated_prompt[f2_prompt_label]['generated_prompt']
    print("Generating comments for F2")
    f2_prompt_label, f2_output, f2_response_list, f2_input_tokens, f2_output_tokens = generate_flow2_output(f2_prompt_label, f2_generated_prompt, f2_context,model, llm, llm_model)
    print("Comments generated for F2")
    print(f"{prompt_label}.F2 output generated")
    prompts = {"base_prompt":f2_base_prompt,"customization_prompt":f2_custom_prompt,"generated_prompt":f2_generated_prompt,"llm":llm,"model":llm_model}
    context = {'wisdom_1':f2_wisdom_1,'wisdom_2':f2_wisdom_2,'context':f2_response_list}
    return f2_prompt_label, f2_output, prompts,context


def evaluator_flows(proposal_text, proposal_summary,doc_type, identity,parent_prompt_label, prompt_flows, model, dependencies, result_store):
    result = {}
    result_prompts = {}
    result_context = {}
    flow_dependencies = {}
    for i in prompt_flows.keys():
        flow_dependencies[i] = prompt_flows[i]['additional_dependencies']
    
    print(flow_dependencies)
    flows_ordered = common_utils.topological_sort(flow_dependencies)
    print(flows_ordered)
    result_store = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        completed = set()

        while flows_ordered:
            parallel_tasks = []
            for prompt_label in list(flows_ordered):
                if all(dep in completed for dep in flow_dependencies[prompt_label]):
                    if 'F1' in prompt_label:
                        futures[prompt_label] = executor.submit(get_evaluator_flow1_output,parent_prompt_label, prompt_flows, doc_type, identity, proposal_text, proposal_summary, model, flow_dependencies,result_store)
                    elif 'F2' in prompt_label:
                        futures[prompt_label] = executor.submit(get_evaluator_flow2_output,parent_prompt_label, prompt_flows, doc_type, identity, proposal_text, proposal_summary, model, flow_dependencies,result_store)
                    parallel_tasks.append(prompt_label)
                    flows_ordered.remove(prompt_label)


            if parallel_tasks:
                print(f"Processing in parallel: {parallel_tasks}")

            for future in concurrent.futures.as_completed(futures.values()):
                input_tokens, output_tokens = 0,0
                prompt_label, generated_comments, prompts, contexts = future.result()
                completed.add(prompt_label)
                result[prompt_label] = {"comments":generated_comments}
                result_prompts[prompt_label] = prompts
                result_context[prompt_label] = contexts
                result_store[prompt_label] = generated_comments
    return result, result_prompts, result_context


def generate_evaluator_Output(proposal_text,p_internal_prompt,wisdom_1, wisdom_2,model, llm_type, llm_model):
    prompt_template, response_list, user_input = get_evaluator_messages(p_internal_prompt,proposal_text, wisdom_1, wisdom_2)
    llm_type = llm_type.lower()
    llm_model = llm_model.lower()
    if llm_type == "chatgpt":
        if str(llm_model).lower() in ["o3", "gpt-5"]:
            llm = ChatOpenAI(model=llm_model, api_key=api_key, organization=organization, temperature=1)
        else:
            llm = ChatOpenAI(model=llm_model, api_key=api_key, organization=organization)
    elif llm_type == "claude":
        llm = ChatAnthropic(model=llm_model,anthropic_api_key=claude_api_key,max_tokens=8192)
    else:
        if str(llm_model).lower() in ["o3", "gpt-5"]:
            llm = ChatOpenAI(model=llm_model, api_key=api_key, organization=organization, temperature=1)
        else:
            llm = ChatOpenAI(model=llm_model, api_key=api_key, organization=organization)
    
    chain = prompt_template | llm

    with get_openai_callback() as cb:
        response = chain.invoke({"user_input":user_input})

    gpt_response = response.content

    input_tokens, output_tokens = cb.prompt_tokens, cb.completion_tokens
    # response_list = get_evaluator_messages(p_internal_prompt,proposal_text, wisdom_1, wisdom_2)
    # response = client.chat.completions.create(
    #     model=model,
    #     temperature=0,
    #     messages=response_list
    # )
    # gpt_response = response.choices[0].message.content
    # input_tokens = response.usage.prompt_tokens
    # output_tokens = response.usage.completion_tokens
    # if not is_valid_json(gpt_response):
    #     gpt_response = convert_response_to_valid_json(gpt_response)
    
    # print(gpt_response)
    # try:
    #     generated_comments_json = json.loads(gpt_response)
    #     if("Comment" in generated_comments_json):
    #         if isinstance(generated_comments_json['Comment'], str):
    #             gpt_response = generated_comments_json["Comment"]
    #         else:
    #             raise Exception(f"Invalid value for generated_comments_json['Comments'],{generated_comments_json}")
    #     if("Comments" in generated_comments_json):
    #         if isinstance(generated_comments_json['Comments'], str):
    #             gpt_response = generated_comments_json["Comments"]
    #         else:
    #             raise Exception(f"Invalid value for generated_comments_json['Comments'],{generated_comments_json}")
    # except Exception as e:
    #     print(f"Analyze Comment Exception: {str(e)}")
    #     traceback.print_exc() 
    #     gpt_response = "Invalid comment generated by GPT"

    
    return gpt_response,response_list, input_tokens,output_tokens


def generate_evaluator_comments(proposal_text,model,generated_evaluator_prompts,analyze_comments,st=None):
    p_internal_output=""
    p_external_output=""
    p_delta_output=""
    p_internal_response_list = ""
    p_external_response_list = ""
    p_delta_response_list = ""
    evaluator_response = {}
    
    if "P_Internal" in generated_evaluator_prompts:
        p_internal_prompt = generated_evaluator_prompts["P_Internal"]
        p_internal_output,p_internal_response_list = generate_evaluator_Output(proposal_text,p_internal_prompt,model)
        evaluator_response["P_Internal"]=p_internal_output
          
    if "P_External" in generated_evaluator_prompts:
        p_external_prompt = generated_evaluator_prompts["P_External"]
        p_external_output,p_external_response_list = generate_evaluator_Output(analyze_comments,p_external_prompt,model)
        evaluator_response["P_External"]=p_external_output
    
    if "P_Delta" in generated_evaluator_prompts:
        p_delta_prompt = generated_evaluator_prompts["P_Delta"]
        p_delta_context = f"Context 1: \n {p_internal_output}\n\n Context 2: \n {p_external_output}"
        p_delta_output,p_delta_response_list = generate_evaluator_Output(p_delta_context,p_delta_prompt,model)
        evaluator_response["P_Delta"]=p_delta_output
    
    return evaluator_response, p_internal_response_list, p_external_response_list, p_delta_response_list
    

def generate_evaluate_comments(proposal_text,model,generated_evaluator_prompts,analyze_comments,section_titles,st=None):
    p_internal_output = ""
    p_external_output = ""
    p_delta_output = ""
    p_internal_response_list = ""
    p_external_response_list = ""
    p_delta_response_list = ""
    p_internal_input_tokens, p_internal_output_tokens = 0,0
    p_external_input_tokens, p_external_output_tokens = 0,0
    p_delta_input_tokens, p_delta_output_tokens = 0,0
    total_input_tokens = {}
    total_output_tokens = {}
    evaluator_response = {}

    def generate_p_internal_output():
        nonlocal p_internal_output, p_internal_response_list
        input_tokens, output_tokens = 0,0
        if "P_Internal" in generated_evaluator_prompts:
            p_internal_prompt = generated_evaluator_prompts["P_Internal"]["generated_prompt"]
            p_internal_wisdom_1 = generated_evaluator_prompts["P_Internal"]["wisdom_1"]
            p_internal_wisdom_2 = generated_evaluator_prompts["P_Internal"]["wisdom_2"]
            p_internal_output, p_internal_response_list, input_tokens, output_tokens = generate_evaluator_Output(proposal_text, p_internal_prompt, p_internal_wisdom_1,p_internal_wisdom_2,model)
        return p_internal_output, p_internal_response_list, input_tokens, output_tokens

    def generate_p_external_output():
        nonlocal p_external_output, p_external_response_list
        input_tokens, output_tokens = 0,0
        if "P_External" in generated_evaluator_prompts:
            p_external_prompt = generated_evaluator_prompts["P_External"]["generated_prompt"]
            p_external_wisdom_1 = generated_evaluator_prompts["P_External"]["wisdom_1"]
            p_external_wisdom_2 = generated_evaluator_prompts["P_External"]["wisdom_2"]
            p_external_output, p_external_response_list, input_tokens, output_tokens = generate_evaluator_Output(analyze_comments, p_external_prompt,p_external_wisdom_1,p_external_wisdom_2, model)
        return p_external_output, p_external_response_list, input_tokens, output_tokens


    with concurrent.futures.ThreadPoolExecutor() as executor:
        internal_future = executor.submit(generate_p_internal_output)
        external_future = executor.submit(generate_p_external_output)

        concurrent.futures.wait([internal_future, external_future])


        p_internal_output, p_internal_response_list, p_internal_input_tokens, p_internal_output_tokens = internal_future.result()
        p_external_output, p_external_response_list, p_external_input_tokens, p_external_output_tokens = external_future.result()
    
    evaluator_response = {
        "P_Internal": {},
        "P_External": {},
        "P_Delta": {}
    }
    evaluator_response["P_Internal"]=p_internal_output
    evaluator_response["P_Internal_section_title"] = section_titles["P_Internal"]
    evaluator_response["P_External"]=p_external_output
    evaluator_response["P_External_section_title"] = section_titles["P_External"]

    if "P_Delta" in generated_evaluator_prompts:
        p_delta_prompt = generated_evaluator_prompts["P_Delta"]["generated_prompt"]
        p_delta_context = f"Context 1: \n {p_internal_output}\n\n Context 2: \n {p_external_output}"
        p_delta_output,p_delta_response_list, p_delta_input_tokens, p_delta_output_tokens = generate_evaluator_Output(p_delta_context,p_delta_prompt,"", "",model)
        evaluator_response["P_Delta"]=p_delta_output
        evaluator_response["P_Delta_section_title"] = section_titles["P_Delta"]
    
    total_input_tokens["P_Internal"] = p_internal_input_tokens
    total_output_tokens["P_Internal"] = p_internal_output_tokens
    total_input_tokens["P_External"] = p_external_input_tokens
    total_output_tokens["P_External"] = p_external_output_tokens
    total_input_tokens["P_Delta"] = p_delta_input_tokens
    total_output_tokens["P_Delta"] = p_delta_output_tokens
    return evaluator_response, p_internal_response_list, p_external_response_list, p_delta_response_list, total_input_tokens, total_output_tokens


def generate_evaluate_comments_dependencies(proposal_text,model,generated_evaluator_prompts,analyze_comments,section_titles,dependencies, topo_order,st=None):
    p_internal_output = ""
    p_external_output = ""
    p_delta_output = ""
    p_internal_response_list = ""
    p_external_response_list = ""
    p_delta_response_list = ""
    p_output=""
    p_response_list=""
    p_internal_input_tokens, p_internal_output_tokens = 0,0
    p_external_input_tokens, p_external_output_tokens = 0,0
    p_delta_input_tokens, p_delta_output_tokens = 0,0
    total_input_tokens = {}
    total_output_tokens = {}
    evaluator_response = {}
    evaluator_response_list = {}


    def generate_p_output(prompt_label, dependencies, result_store):
        nonlocal p_output, p_response_list
        input_tokens, output_tokens = 0,0
        if prompt_label in generated_evaluator_prompts:
            prompt = generated_evaluator_prompts[prompt_label]["generated_prompt"]
            wisdom_1 = generated_evaluator_prompts[prompt_label]["wisdom_1"]
            wisdom_2 = generated_evaluator_prompts[prompt_label]["wisdom_2"]
            deps = []
            inputs = []
            dependency_context = ""
            for dep in dependencies[prompt_label]:
                inputs.append(result_store[dep])
                deps.append(dep)
            if prompt_label != "P_Delta":
                print(f"{prompt_label} using output of {deps}")
                if(len(inputs)>0):
                    dependency_context = f"ContextChain"
                    for i in range(len(inputs)):
                        dependency_context += f"\n\n{inputs[i]}"
                if prompt_label == "P_Internal":
                    context = proposal_text + "\n\n" + dependency_context
                else:
                    context = analyze_comments + "\n\n" + dependency_context
            else:   
                print(f"{prompt_label} using output of {deps}")
                p_internal_output = result_store["P_Internal"]
                p_external_output = result_store["P_External"]
                if "P_Internal" in deps:
                    context = f"Context 1: \n {p_internal_output}\n"
                elif "P_External" in deps:
                    context = f"Context 1: \n {p_external_output}\n"
                else:
                    context = f"Context 1: \n {p_internal_output}\n\n Context 2: \n {p_external_output}"
            p_output, p_response_list, input_tokens, output_tokens = generate_evaluator_Output(context, prompt, wisdom_1,wisdom_2,model)
        return prompt_label, p_output, p_response_list, input_tokens, output_tokens


    result_store = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        completed = set()

        while topo_order:
            parallel_tasks = []
            for prompt_label in list(topo_order):
                if all(dep in completed for dep in dependencies[prompt_label]):
                    futures[prompt_label] = executor.submit(generate_p_output,prompt_label, dependencies,result_store)
                    parallel_tasks.append(prompt_label)
                    topo_order.remove(prompt_label)


            if parallel_tasks:
                print(f"Processing in parallel: {parallel_tasks}")

            for future in concurrent.futures.as_completed(futures.values()):
                input_tokens, output_tokens = 0,0
                prompt_label,output,response_list, input_tokens, output_tokens = future.result()
                completed.add(prompt_label)
                section_title = section_titles[prompt_label]
                evaluator_response[prompt_label] = output
                evaluator_response[f"{prompt_label}_section_title"] = section_title
                evaluator_response_list[prompt_label] = response_list
                result_store[prompt_label] = output
                total_input_tokens[f"{prompt_label}"] = input_tokens
                total_output_tokens[f"{prompt_label}"] = output_tokens

    if "P_Internal" in evaluator_response_list.keys():
        p_internal_response_list = evaluator_response_list["P_Internal"]
    if "P_External" in evaluator_response_list.keys():
        p_external_response_list = evaluator_response_list["P_External"]
    if "P_Delta" in evaluator_response_list.keys():
        p_delta_response_list = evaluator_response_list["P_Delta"]
    return evaluator_response, p_internal_response_list, p_external_response_list, p_delta_response_list,total_input_tokens, total_output_tokens


def generate_p_output_flows(generated_evaluator_prompts,prompt_label,prompt_flow, proposal_text,proposal_summary,doc_type, identity,dependencies, result_store, model):
    p_output, p_response_list = "", ""
    input_tokens, output_tokens = 0,0
    prompt_flow_results,prompt_flow_prompts,prompt_flow_contexts = {},{},{}
    if(len(prompt_flow) > 0):
        print(f"Getting flow wise output for {prompt_label}")
        prompt_flow_results, prompt_flow_prompts, prompt_flow_contexts = evaluator_flows(proposal_text, proposal_summary,doc_type, identity,prompt_label, prompt_flow, model, dependencies, result_store)
        print(f"Flow Wise Output Fetched for {prompt_label}")
    
    if prompt_label in generated_evaluator_prompts:
        prompt = generated_evaluator_prompts[prompt_label]["generated_prompt"]
        wisdom_1 = generated_evaluator_prompts[prompt_label]["wisdom_1"]
        wisdom_2 = generated_evaluator_prompts[prompt_label]["wisdom_2"]
        wisdom_received = generated_evaluator_prompts[prompt_label]["wisdom_received"]
        llm = generated_evaluator_prompts[prompt_label]["llm"]
        llm_model = generated_evaluator_prompts[prompt_label]["model"]
        deps = []
        inputs = []
        dependency_context = ""
        for dep in dependencies[prompt_label]:
            inputs.append(result_store[dep])
            deps.append(dep)

        print(f"{prompt_label} using output of {deps}")
        if(len(inputs)>0):
            dependency_context = f"ContextChain"
            for i in range(len(inputs)):
                dependency_context += f"\n\n{inputs[i]}"
        all_wisdoms = {}
        if(f"{prompt_label}.F1" in prompt_flow_results.keys()):
            f1_comments = prompt_flow_results[f"{prompt_label}.F1"]["comments"]
            all_wisdoms[f"{prompt_label}.F1"] = str(f1_comments)
        if(f"{prompt_label}.F2" in prompt_flow_results.keys()):
            f2_comments = prompt_flow_results[f"{prompt_label}.F2"]["comments"]
            all_wisdoms[f"{prompt_label}.F2"] = str(f2_comments)
        
        print(wisdom_received)
        received_wisdom = "Wisdom_Received" + "\n\n"
        for wisdom in wisdom_received:
            if wisdom in all_wisdoms.keys():
                print(f"{prompt_label} using wisdom from {wisdom}")
                received_wisdom += all_wisdoms[wisdom] + "\n\n" 
        
        if prompt_label == "P_Internal":
            context = proposal_text + "\n\n" + dependency_context + "\n\n" + received_wisdom

        p_output, p_response_list, input_tokens, output_tokens = generate_evaluator_Output(context, prompt, wisdom_1,wisdom_2,model, llm, llm_model)
    return prompt_label, p_output, p_response_list, prompt_flow_results,prompt_flow_prompts, prompt_flow_contexts,input_tokens, output_tokens


@traceable(tags=["Evaluator Comments"])
def generate_evaluate_comments_multiflow(proposal_text,proposal_summary,doc_type, identity,prompt_flow,model,generated_evaluator_prompts,section_titles,dependencies, topo_order):
    p_internal_output = ""
    p_internal_response_list = ""
    p_internal_input_tokens, p_internal_output_tokens = 0,0
    total_input_tokens = {}
    total_output_tokens = {}
    evaluator_response = {}
    evaluator_response_list = {}
    child_flows = {}

    result_store = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        completed = set()

        while topo_order:
            parallel_tasks = []
            for prompt_label in list(topo_order):
                if all(dep in completed for dep in dependencies[prompt_label]):
                    futures[prompt_label] = executor.submit(generate_p_output_flows,generated_evaluator_prompts,prompt_label, prompt_flow[prompt_label],proposal_text, proposal_summary,doc_type, identity,dependencies,result_store,model)
                    parallel_tasks.append(prompt_label)
                    topo_order.remove(prompt_label)


            if parallel_tasks:
                print(f"Processing in parallel: {parallel_tasks}")

            for future in concurrent.futures.as_completed(futures.values()):
                input_tokens, output_tokens = 0,0
                prompt_label,output,response_list, prompt_flow_results,prompt_flow_prompts, prompt_flow_contexts,input_tokens, output_tokens = future.result()
                completed.add(prompt_label)
                section_title = section_titles[prompt_label]
                evaluator_response[prompt_label] = output
                evaluator_response[f"{prompt_label}_section_title"] = section_title
                evaluator_response_list[prompt_label] = response_list
                child_flows[prompt_label] = {"prompt_flow_prompts":prompt_flow_prompts,"prompt_flow_contexts":prompt_flow_contexts,"prompt_flow_results":prompt_flow_results}
                result_store[prompt_label] = output
                total_input_tokens[f"{prompt_label}"] = input_tokens
                total_output_tokens[f"{prompt_label}"] = output_tokens

    if "P_Internal" in evaluator_response_list.keys():
        p_internal_response_list = evaluator_response_list["P_Internal"]

    return evaluator_response, p_internal_response_list, child_flows,total_input_tokens, total_output_tokens

    
@traceable(tags=["analyzer-prompts-generator"])
def generate_prompt_original(prompt_for_customization, selected_model, prompt, nature, summary, use_example, prompt_examples, prompt_label=None, tokens_counter=None):
    prompt = f""" Nature of the Document: {nature} \n\n Summary of the document:\n{summary} \n\n Base Prompt:\n{prompt}"""
    system_role = f"""{prompt_for_customization}"""
    examples=""
    example_text_3=""
    response_text_3=""
    is_example_present = bool(example_text_3) and bool(response_text_3)
    
    if use_example is True and is_example_present:
        response_list = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": example_text_3},
        {"role": "assistant", "content": response_text_3},
        {"role": "user", "content": prompt}
    ]
    else:
        response_list = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": prompt}
    ]
    
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_role),
            ("user", "{prompt}"),
        ]
    )
    
    chain = prompt_template | llm
    
    response = chain.invoke({"prompt": prompt})
    
    return response.content

    # response = client.chat.completions.create(
    #     model=selected_model,
    #     messages=response_list,
    #     temperature=0
    # )
    
    gpt_response = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
        
    if tokens_counter is not None and prompt_label is not None:
        prompt_label = prompt_label.lower()
        
        tokens_counter[0]["analyzer_prompts"][prompt_label] = input_tokens
        tokens_counter[1]["analyzer_prompts"][prompt_label] = output_tokens
        
    return gpt_response


def generate_evaluator_prompt_old(prompt_for_customization, selected_model, base_prompt, nature, tor_summary_text, prompt_label=None, tokens_counter=None):
    prompt = f""" Nature of Document: {nature}\n\nTerms Of Reference Summary: \n{tor_summary_text}\n\nBase Prompt: \n{base_prompt}"""
    system_role = f"""{prompt_for_customization}"""
    
    response_list = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": prompt}
    ]
    
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_role),
            ("user", "{prompt}"),
        ]
    )
    
    chain = prompt_template | llm
    
    response = chain.invoke({"prompt": prompt})
    
    return response.content
    
    response = client.chat.completions.create(
        model=selected_model,
        messages=response_list
    )
    
    gpt_response = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    
    if tokens_counter is not None and prompt_label is not None:
        prompt_label = prompt_label.lower()
        tokens_counter[0]["evaluator_prompts"][prompt_label] = input_tokens
        tokens_counter[1]["evaluator_prompts"][prompt_label] = output_tokens
    
    return gpt_response


@traceable(tags=["proposal-summary-generator"])
def get_summary(text, model, nature_of_document, tokens_counter=None):
    summary_prompt = api_utils.get_analyzer_proposal_summary_prompts(nature_of_document)
    
    system_prompt = f"""
    {summary_prompt}
    
    Instruction for Response Generation:
        1. Provide the response strictly in markdown format with markdown formatting. 
        2. Do not add ```markdown with the start or end of response. Just make the formatting as markdown. Do not provide the response in simple plain text. 
        3. Strictly adhere to the markdown response format.
    """
    
    prompt = f"""Context: \n{text}"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "{prompt}"),
        ]
    )
    
    chain = prompt_template | llm
    
    response = chain.invoke({"prompt": prompt})

    return response.content


@traceable
def get_summary_streamlit(text, model, nature_of_document):
    summary_length = os.getenv("summary_length")
    summary_prompt = api_utils.get_analyzer_proposal_summary_prompts(nature_of_document)
    system_prompt = f"""
    {summary_prompt}
    
    Instruction for Response Generation:
        1. Provide the response strictly in markdown format with markdown formatting. 
        2. Do not add ```markdown with the start or end of response. Just make the formatting as markdown. Do not provide the response in simple plain text. 
        3. Strictly adhere to the markdown response format.
    """
    prompt = f"""Context: {text}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
        
    gpt_response = response.choices[0].message.content

    return gpt_response, summary_prompt


@traceable(tags=["tor-summary-generator"])
def get_tor_summary(text, model, nature_of_document, organization_id=None, tokens_counter=None):
    summary_length = os.getenv("summary_length")
    print("Retrieving TOR Summary Prompt!")
    summary_prompt = api_utils.get_tor_summary_prompts(nature_of_document, organization_id)
    system_prompt = f"""
    {summary_prompt}
    
    Instruction for Response Generation:
        1. Provide the response strictly in markdown format with markdown formatting. 
        2. Do not add ```markdown with the start or end of response. Just make the formatting as markdown. Do not provide the response in simple plain text. 
        3. Strictly adhere to the markdown response format.
    """
    prompt = f"""Context: {text}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "{prompt}"),
        ]
    )
    
    chain = prompt_template | llm
    
    response = chain.invoke({"prompt": prompt})
    
    return response.content

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
        
    gpt_response = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    
    if tokens_counter is not None:
        if tokens_counter[0].has_key("tor_summary"):
            tokens_counter[0]["tor_summary"] = input_tokens
            tokens_counter[1]["tor_summary"] = output_tokens

    return gpt_response


@traceable
def get_tor_summary_streamlit(text, model, nature_of_document, organization_id=None):
    summary_length = os.getenv("summary_length")
    print("Retrieving TOR Summary Prompt!")
    summary_prompt = api_utils.get_tor_summary_prompts(nature_of_document, organization_id)
    system_prompt = f"""
    {summary_prompt}
    
    Instruction for Response Generation:
        1. Provide the response strictly in markdown format with markdown formatting. 
        2. Do not add ```markdown with the start or end of response. Just make the formatting as markdown. Do not provide the response in simple plain text. 
        3. Strictly adhere to the markdown response format.
    """
    prompt = f"""Context: {text}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
        
    gpt_response = response.choices[0].message.content

    return gpt_response, summary_prompt


@traceable(tags=["p0-summary-generator"])
def get_P0_summary(text, model, nature_of_document, tokens_counter=None):
    
    summary_length = os.getenv("summary_length")
    print("Retrieving P0 Comments Summary Prompt!")
    summary_prompt = api_utils.get_analyzer_comments_summary_prompts(nature_of_document)
    system_prompt = f"""
    {summary_prompt}
    
    Instruction for Response Generation:
        1. Summary should be within {summary_length} characters.
        2. Provide the response strictly in markdown format with markdown formatting. 
        3. Do not add ```markdown with the start or end of response. Just make the formatting as markdown. Do not provide the response in simple plain text. 
        4. Strictly adhere to the markdown response format.
    """
    prompt = f"""
    Context: {text}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "{prompt}"),
        ]
    )
    
    chain = prompt_template | llm
    
    response = chain.invoke({"prompt": prompt})
    
    return response.content


@traceable
def get_section_query_response(query,context,conversation_string):
    
    prompt="""
           Answer the given Query based on the given 'Context' and 'Conversation Log'.Dont make any assumptions, use only the given context to answer the query.
        """

    user_input=f"Context:{context}\n\nConversation Log:{conversation_string}\n\nQuery:{query}\n\n"
    
    response_list = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input}          
    ]
    
    
    response = client.chat.completions.create(
        model=possible_models["o3"],
        messages=response_list
    )
    gpt_response = response.choices[0].message.content

    return gpt_response


@traceable
def get_analyzer_section_query_response(query,context,conversation_string):
    prompt="""
        Answer the given Query based on the given 'Context' and 'Conversation Log'.Dont make any assumptions, use only the given context to answer the query.
    """
    user_input=f"Context:{context}\n\nConversation Log:{conversation_string}\n\nQuery:{query}\n\n"
    response_list = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input}          
    ]
    response = client.chat.completions.create(
        model=possible_models["o3"],
        messages=response_list
    )
    gpt_response = response.choices[0].message.content
    return gpt_response


@traceable
def get_summary_analyzer_streamlit(text,model,doc_type, user_role):
    summary_length=os.getenv("summary_length")
    
    summary_prompt=endpoint_utils.get_current_summary_prompt_from_api(doc_type)
    
    if(user_role is not None):
        if(common_utils.has_placeholder(summary_prompt, "user_role")):
            summary_prompt = common_utils.replace_placeholder(summary_prompt,"user_role", user_role)
            
    system_prompt = f"""{summary_prompt}\nSummary should be within {summary_length} words."""
    prompt = f"""Context: {text}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    
    gpt_response = response.choices[0].message.content

    return gpt_response, summary_prompt



if __name__ == "__main__":
    evaluator_prompt = "Please evaluate the following content:"
    proposal_text = "This is the main context."
    wisdom_1 = "This is wisdom 1."
    wisdom_2 = "This is wisdom 2."
    model = "gpt-4"
    tokens_counter = [{"evaluator_comments": {}}, {"evaluator_comments": {}}]
    prompt_label = "example"

    gpt_response, response_list = generate_evaluator_output(proposal_text, evaluator_prompt, model, wisdom_1, wisdom_2, prompt_label, tokens_counter)
    print(gpt_response)
    print(response_list)
