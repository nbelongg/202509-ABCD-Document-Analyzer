import requests
from fastapi import HTTPException
import os
from dotenv import load_dotenv
import traceback
from typing import Optional

load_dotenv(override=True)

api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")


def get_prompt_corpus(prompt_label: str) -> str:
    prompt_corpus_mapping = {
        "P1": "C1(Universal corpus)",
        "P2": "C2(MBS and GPP)",
        "P3": "C3(LC and IID)",
        "P4": "C4(SDSC)",
        "P5": "C5(CSS)"
    }

    return prompt_corpus_mapping.get(prompt_label, "Invalid Prompt Label")


def get_prompts(prompt_label, doc_type):
    url = "http://13.232.173.41:8006/get_prompts"
    headers = {
        'api-key': api_key,
        'api-secret': api_secret
    }
    params = {
        "prompt_label": prompt_label,
        "doc_type": doc_type
    }
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        return data["prompts"]
        
        base_prompt = data['prompts'][0]['base_prompt']
        customization_prompt = data['prompts'][0]['customization_prompt']
        wisdom_1 = data['prompts'][0]['wisdom_1']
        wisdom_2 = data['prompts'][0]['wisdom_2']
        chunks = data['prompts'][0]['chunks']
        prompt_corpus = data['prompts'][0]['corpus_id']
        prompt_examples = None
        section_title = data['prompts'][0]['section_title']
        dependencies = data['prompts'][0]['dependencies']
        return base_prompt, customization_prompt, chunks, prompt_corpus, prompt_examples, wisdom_1, wisdom_2, section_title, dependencies
    except requests.exceptions.RequestException as e:
        print(f"Error fetching prompts: {e}")
        return []
    except Exception as e:
        traceback.print_exc()
        return []


def get_analyzer_comments_summary_prompts(doc_type):
    url = "http://13.232.173.41:8006/get_analyzer_comments_summary_prompts/"
    params = {'doc_type': doc_type}
    headers = {
        'api-key': api_key,
        'api-secret': api_secret
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data['prompts'][0]['summary_prompt']
        else:
            print("Failed to retrieve analyzer summary prompts. Status code:", response.status_code)
            print("Response:", response.text)
    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)


def get_analyzer_proposal_summary_prompts(doc_type):
    url = "http://13.232.173.41:8006/get_analyzer_proposal_summary_prompts/"
    params = {'doc_type': doc_type}
    headers = {
        'api-key': api_key,
        'api-secret': api_secret
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data['prompts'][0]['proposal_prompt']
            prompts = data.get("prompts", [])
            if prompts:
                for prompt in prompts:
                    print("Prompt Label:", prompt.get("prompt_label"))
                    print("Doc Type:", prompt.get("doc_type"))
                    print("Summary Prompt:", prompt.get("summary_prompt"))
            else:
                print("No analyzer summary prompts found.")
        else:
            print("Failed to retrieve analyzer summary prompts. Status code:", response.status_code)
            print("Response:", response.text)
    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)


def get_custom_prompts(nature_of_document, organization_id):
    url = "http://13.232.173.41:8006/get_custom_prompts"
    data = {
        'doc_type': nature_of_document,
        'organization_id': organization_id
    }
    headers = {
        'api-key': api_key,
        'api-secret': api_secret,
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    try:
        response = requests.post(url, data=data, headers=headers)
        if response.status_code == 200:
            data = response.json()
            base_prompt = data["prompts"][0]["base_prompt"]
            customization_prompt = data["prompts"][0]["customization_prompt"]
            chunks = data["prompts"][0]["chunks"]
            section_title = data["prompts"][0]["section_title"]
            return base_prompt, customization_prompt, chunks, section_title
        else:
            raise HTTPException(status_code=404, detail=f"No prompts found for organization: {organization_id}")
    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)
        raise HTTPException(status_code=500, detail="Error retrieving custom prompts from API.")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No prompts found for organization: {organization_id}")


def get_tor_summary_prompts(nature_of_document, organization_id):
    url = "http://13.232.173.41:8006/get_tor_summary_prompts/"
    data = {
        "doc_type": nature_of_document,
        "organization_id": organization_id
    }
    headers = {
        'api-key': api_key,
        'api-secret': api_secret
    }
    try:
        response = requests.post(url, data=data, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data["prompts"][0]["tor_summary_prompt"]
        else:
            print("Failed to retrieve analyzer summary prompts. Status code:", response.status_code)
            print("Response:", response.text)
    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)


def get_current_evaluator_prompts(prompt_labels, doc_type = None, organization_id = None, org_guideline_id = None):
    api_url = "http://13.232.173.41:8006/get_evaluator_prompts"
    headers = {
        'accept': 'application/json',
        'api-key': api_key,
        'api-secret': api_secret
    }
    
    if prompt_labels:
        prompts_configurations = {}
        prompt_flows= {}
        for prompt_label in prompt_labels:
            data = {
                
                'doc_type': doc_type ,
                'organization_id': organization_id,
                'org_guideline_id': org_guideline_id
            }
            print(data)
            try:
                response = requests.post(api_url, headers=headers, data=data)
                
                print(response.status_code)
                if response.status_code == 200:
                    
                    prompts = response.json().get("prompts", [])  
                    prompt_flows= {}
                    for prompt in prompts:
                        prompt_label = prompt.get("prompt_label")
                        base_prompt = prompt.get("base_prompt")
                        customization_prompt = prompt.get("customization_prompt")
                        wisdom_1 = prompt.get("wisdom_1")
                        wisdom_2 = prompt.get("wisdom_2")
                        chunks = prompt.get("chunks")
                        sec_title =  prompt.get("section_title", "")
                        dependencies = prompt.get("additional_dependencies","")
                        customize_prompt_based_on = prompt.get("customize_prompt_based_on","")
                        send_along_customised_prompt = prompt.get("send_along_customised_prompt","")
                        wisdom_received = prompt.get("wisdom_received","")
                        llm_flow = prompt.get("LLM_Flow","")
                        llm = prompt.get("LLM","")
                        model = prompt.get("Model","")
                        show_on_frontend = prompt.get("show_on_frontend","")
                        label_for_output = prompt.get("label_for_output","")
                        
                        if wisdom_1 is None:
                            wisdom_1 = ""
                            
                        if wisdom_2 is None:
                            wisdom_2 = ""
                        
                        if sec_title is None:
                            sec_title = ""
                            
                        if not dependencies or dependencies == "":
                            dependencies = []
                        else:
                            dependencies = dependencies.split(",")
                            
                        if not wisdom_received or wisdom_received == "":
                            wisdom_received = []
                        else:
                            wisdom_received = wisdom_received.split(",")
                            
                        if not customize_prompt_based_on or customize_prompt_based_on == "":
                            customize_prompt_based_on = []
                        else:
                            customize_prompt_based_on = customize_prompt_based_on.split(",")
                            
                        if not send_along_customised_prompt or send_along_customised_prompt == "":
                            send_along_customised_prompt = []
                        else:
                            send_along_customised_prompt = send_along_customised_prompt.split(",")
                        
                        if not llm or llm == "":
                            llm = "ChatGPT"
                        
                        if not model or model == "":
                            model = "gpt-4o-2024-08-06"
                            
                        if('.F1' in prompt_label or '.F2' in prompt_label):
                            flow = {}
                            flow[prompt_label]={"base_prompt":base_prompt,"customization_prompt":customization_prompt,"wisdom_1":wisdom_1, "wisdom_2": wisdom_2,"chunks":chunks,"wisdom_received":wisdom_received,"additional_dependencies":dependencies,"customize_prompt_based_on":customize_prompt_based_on,"send_along_customised_prompt":send_along_customised_prompt, "show_on_frontend":show_on_frontend,"label_for_output":label_for_output,"llm_flow":llm_flow,"llm":llm,"model":model}
                            if 'P_Internal' in prompt_flows.keys():
                                prompt_flows['P_Internal'].update(flow)
                            else:
                                prompt_flows['P_Internal'] = flow
                        else:
                            if prompt_label == 'P_Internal':
                                prompts_configurations[prompt_label]={"base_prompt":base_prompt,"customization_prompt":customization_prompt,"wisdom_1":wisdom_1, "wisdom_2": wisdom_2,"chunks":chunks,"customize_prompt_based_on":customize_prompt_based_on,"send_along_customised_prompt":send_along_customised_prompt,"additional_dependencies":dependencies,"wisdom_received":wisdom_received, "llm_flow":llm_flow,"llm":llm,"model":model, "show_on_frontend":show_on_frontend,"label_for_output":label_for_output}
                        
                elif response.status_code == 404:
                    continue
                    raise HTTPException(status_code=404, detail="No prompts found.")
                else:
                    raise HTTPException(status_code=response.status_code, detail=f"Failed to fetch prompts from the API: {response.text}")
            
            except Exception as e:
                print(f"An error occurred during the query execution for \n Document Type:{doc_type}\nOrganisation ID:{organization_id}: {str(e)}")
                traceback.print_exc()
                raise e
    else:
        data = {
                
                'doc_type': doc_type ,
                'organization_id': organization_id,
                'org_guideline_id': org_guideline_id
        }

        print(data)
            
        prompts_configurations = {}
        prompt_flows= {}
        try:
            response = requests.post(api_url, headers=headers, data=data)
                
            if response.status_code == 200:
                    
                prompts = response.json().get("prompts", [])  
            
                for prompt in prompts:
                    prompt_label = prompt.get("prompt_label")
                    base_prompt = prompt.get("base_prompt")
                    customization_prompt = prompt.get("customization_prompt")
                    wisdom_1 = prompt.get("wisdom_1")
                    wisdom_2 = prompt.get("wisdom_2")
                    chunks = prompt.get("chunks")
                    sec_title =  prompt.get("section_title", "")
                    dependencies = prompt.get("additional_dependencies","")
                    customize_prompt_based_on = prompt.get("customize_prompt_based_on","")
                    send_along_customised_prompt = prompt.get("send_along_customised_prompt","")
                    wisdom_received = prompt.get("wisdom_received","")
                    llm_flow = prompt.get("LLM_Flow","")
                    llm = prompt.get("LLM","")
                    model = prompt.get("Model","")
                    show_on_frontend = prompt.get("show_on_frontend","")
                    label_for_output = prompt.get("label_for_output","")
                    
                    if wisdom_1 is None:
                        wisdom_1 = ""
                        
                    if wisdom_2 is None:
                        wisdom_2 = ""
                    
                    if sec_title is None:
                        sec_title = ""
                    if not dependencies or dependencies == "":
                        dependencies = []
                    else:
                        dependencies = dependencies.split(",")
                    if not wisdom_received or wisdom_received == "":
                        wisdom_received = []
                    else:
                        wisdom_received = wisdom_received.split(",")
                        
                    if not customize_prompt_based_on or customize_prompt_based_on == "":
                        customize_prompt_based_on = []
                    else:
                        customize_prompt_based_on = customize_prompt_based_on.split(",")
                        
                    if not send_along_customised_prompt or send_along_customised_prompt == "":
                        send_along_customised_prompt = []
                    else:
                        send_along_customised_prompt = send_along_customised_prompt.split(",")
                    
                    if not llm or llm == "":
                        llm = "ChatGPT"
                        
                    if not model or model == "":
                        model = "gpt-4o-2024-08-06"
                            
                    if('.F1' in prompt_label or '.F2' in prompt_label):
                        flow = {}
                        flow[prompt_label]={"base_prompt":base_prompt,"customization_prompt":customization_prompt,"wisdom_1":wisdom_1, "wisdom_2": wisdom_2,"chunks":chunks,"wisdom_received":wisdom_received,"additional_dependencies":dependencies,"customize_prompt_based_on":customize_prompt_based_on,"send_along_customised_prompt":send_along_customised_prompt, "show_on_frontend":show_on_frontend,"label_for_output":label_for_output,"llm_flow":llm_flow,"llm":llm,"model":model}
                        if 'P_Internal' in prompt_flows.keys():
                            prompt_flows['P_Internal'].update(flow)
                        else:
                            prompt_flows['P_Internal'] = flow
                    else:
                        if prompt_label == 'P_Internal':
                            prompts_configurations[prompt_label]={"base_prompt":base_prompt,"customization_prompt":customization_prompt,"wisdom_1":wisdom_1, "wisdom_2": wisdom_2,"chunks":chunks,"customize_prompt_based_on":customize_prompt_based_on,"send_along_customised_prompt":send_along_customised_prompt,"additional_dependencies":dependencies,"wisdom_received":wisdom_received, "llm_flow":llm_flow,"llm":llm,"model":model, "show_on_frontend":show_on_frontend,"label_for_output":label_for_output}
                        
            elif response.status_code == 404:
                return {}
                raise HTTPException(status_code=404, detail="No prompts found.")
            else:
                raise HTTPException(status_code=response.status_code, detail=f"Failed to fetch prompts from the API: {response.text}")
            
        except Exception as e:
            print(f"An error occurred during the query execution for \n Document Type:{doc_type}\nOrganisation ID:{organization_id}: {str(e)}")
            traceback.print_exc()
            raise e
        
    return prompts_configurations, prompt_flows


if __name__ == "__main__":
    tor_summary_prompt = get_tor_summary_prompts("Policy Document", "UNICEF")
    data1 = get_prompts("P1", "Policy Document")
    print(data1)
    