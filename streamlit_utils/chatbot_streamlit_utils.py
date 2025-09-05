import pinecone_utils
import gpt_utils
from streamlit_chat import message
import db_utils

def get_conversation_string(st):
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

   
def chatbot_current_config_panel(st):
    current_prompt,current_temperature = db_utils.get_current_gpt_config()
    st.sidebar.markdown(f"**Current Prompt:**\n\n{str(current_prompt)}")
    st.sidebar.markdown(f"**Current Temperature:**\n\n{str(current_temperature)}")
         
def chat_panel(st):
    
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

def generate_query_response(st,query,model,pdf_mappings):
    
    conBuffWindow=3
    chunks_count=3
    
    if query:
        with st.spinner("typing..."):
            
            conversation_string=""
            if len(st.session_state['requests'])>0:
                conversation_string = get_conversation_string(st)
            
            requires_retrieval,refined_query = gpt_utils.query_refiner(conversation_string, query) 
    
            conversation_string=""
            if len(st.session_state['requests']) >= conBuffWindow:
                start_index = len(st.session_state['requests']) - conBuffWindow
            else:
                start_index = 0

            for i in range(start_index, len(st.session_state['requests'])):   
                conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
                conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n" 
            
            st.subheader("Refined Query:")
            st.write(refined_query)
            
            context=[]
            if requires_retrieval:
                result_dict=pinecone_utils.extract_unique_chunks(refined_query,top_k=chunks_count,multiplier=2)
                print("Result Dict:",result_dict)
                context = result_dict['all_context']
                
            response,prompt,within_knowledge_base=gpt_utils.get_response(model,conversation_string,query,context)
            
            print("Context:",context)

            with st.expander("Prompt Used:"):
                st.write(prompt)

            with st.expander("Context used by GPT to answer question:"):
                
                sources_info=[]
                if context:
                    for i in range(chunks_count):
                        pdf_name=result_dict[f'meta_{i+1}']
                        
                        pdf_info={}
        
                        if pdf_name in pdf_mappings: 
                            pdf_info=pdf_mappings[pdf_name]
                            pdf_info['context']=result_dict[f'context_{i+1}']
                        
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
                            pdf_info['context']=result_dict[f'context_{i+1}']
                        
                        sources_info.append(pdf_info)
                    
                st.json(sources_info)
            
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 
            

def chatbot_panel(st,pdf_mappings):
    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["How can I assist you?"]

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    filters_container = st.container()
    chat_container = st.container()
    query_container = st.container()

    with filters_container:
        model = st.selectbox('Model',('gpt-3.5-turbo-1106','gpt-4o'),index=1)
    
    with query_container:
        query = st.text_input("Query: ", key="input")
        generate_query_response(st,query,model,pdf_mappings)
        
    with chat_container:
        chat_panel(st)


