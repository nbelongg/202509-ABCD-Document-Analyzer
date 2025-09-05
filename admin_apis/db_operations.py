import traceback
from dotenv import load_dotenv
import os
import mysql.connector
from fastapi import HTTPException
from models import DocType
import traceback
import time

load_dotenv(override=True)

host=os.getenv("mysql_host")
database=os.getenv("mysql_database")
user=os.getenv("mysql_user")
password=os.getenv("mysql_password")

corpus_mapping = {
    "P1": "C1(Universal corpus)",
    "P2": "C2(MBS and GPP)",
    "P3": "C3(LC and IID)",
    "P4": "C4(SDSC)",
    "P5": "C5(CSS)"
}


def connect_to_mysql():
    try:
        connection = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        return connection
    except mysql.connector.Error as err:
        traceback.print_exc()
        return None
   
        
def get_prompts_from_db(prompt_label, doc_type):
    conn = connect_to_mysql()
    if conn:
        prompts = []
        try:
            cursor = conn.cursor(dictionary=True)
            query = f"""SELECT * FROM analyzer_prompts"""
            params = []

            if prompt_label:
                query += f" WHERE prompt_label like '{prompt_label}%'"
                if doc_type:
                    query += " AND doc_type = %s"
                    params.append(doc_type.value)
            elif doc_type:
                query += " WHERE doc_type = %s"
                params.append(doc_type.value)
            cursor.execute(query, params)
            prompts = cursor.fetchall()
        finally:
            if conn:
                conn.close()
        return {"prompts": prompts}   


def update_prompts_in_db(prompt_label, prompts_update):
    conn = connect_to_mysql()
    print("Prompt Label: ", prompt_label)
    try:
        cursor = conn.cursor()
        updated_prompts = []

        for prompt in prompts_update:
            # Validate the provided data
            if not prompt.base_prompt and not prompt.customization_prompt:
                return {"prompts": f"No data provided for update for doc_type: {prompt.doc_type}"}

            # Set default values if necessary
            prompt.doc_type = prompt.doc_type or "Program design Document"

            # Fetch the existing record for the combination of prompt label and doc type
            cursor.execute("""
                SELECT *
                FROM analyzer_prompts
                WHERE prompt_label = %s AND doc_type = %s
            """, (prompt_label, prompt.doc_type))
            existing_prompt = cursor.fetchone()
            print("Existing Prompt: ", existing_prompt)

            if existing_prompt:
                # Update the existing record
                update_values = {}
                if prompt.base_prompt:
                    update_values["base_prompt"] = prompt.base_prompt
                if prompt.customization_prompt:
                    update_values["customization_prompt"] = prompt.customization_prompt
                if prompt.wisdom_1:
                    update_values["wisdom_1"] = prompt.wisdom_1
                if prompt.wisdom_2:
                    update_values["wisdom_2"] = prompt.wisdom_2
                if prompt.corpus_id:
                    update_values["corpus_id"] = prompt.corpus_id
                if prompt.section_title:
                    update_values["section_title"] = prompt.section_title
                if prompt.number_of_chunks:
                    update_values["chunks"] = prompt.number_of_chunks
                if prompt.customize_prompt_based_on:
                    update_values["customize_prompt_based_on"] = prompt.customize_prompt_based_on
                if prompt.send_along_customised_prompt:
                    update_values["send_along_customised_prompt"] = prompt.send_along_customised_prompt
                if prompt.which_chunks:
                    update_values["which_chunks"] = prompt.which_chunks
                if prompt.wisdom_received:
                    update_values["wisdom_received"] = prompt.wisdom_received
                if prompt.llm_flow:
                    update_values["llm_flow"] = prompt.llm_flow
                if prompt.llm:
                    update_values["llm"] = prompt.llm
                if prompt.model:
                    update_values["model"] = prompt.model
                if prompt.show_on_frontend:
                    update_values["show_on_frontend"] = prompt.show_on_frontend
                if prompt.label_for_output:
                    update_values["label_for_output"] = prompt.label_for_output
                
                update_values["dependencies"] = prompt.dependencies

                if update_values:
                    # Construct the update query dynamically
                    query = "UPDATE analyzer_prompts SET "
                    query += ", ".join([f"{key} = %s" for key in update_values.keys()])
                    query += " WHERE prompt_label = %s AND doc_type = %s"
                    cursor.execute(query, tuple(update_values.values()) + (prompt_label, prompt.doc_type))
            else:
                print("Inserting new record")
                # Insert a new record
                cursor.execute("""
                    INSERT INTO analyzer_prompts (prompt_label, doc_type, base_prompt, customization_prompt, wisdom_1, wisdom_2, corpus_id, section_title, chunks, dependencies, 
                    customize_prompt_based_on, send_along_customised_prompt, which_chunks, wisdom_received, llm_flow, llm, model, show_on_frontend, label_for_output)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (prompt_label, prompt.doc_type, prompt.base_prompt, prompt.customization_prompt, prompt.wisdom_1, prompt.wisdom_2, prompt.corpus_id, prompt.section_title,
                      prompt.number_of_chunks, prompt.dependencies, prompt.customize_prompt_based_on, prompt.send_along_customised_prompt, prompt.which_chunks, prompt.wisdom_received,
                      prompt.llm_flow, prompt.llm, prompt.model, prompt.show_on_frontend, prompt.label_for_output))
                print("Inserted new record")

            # Fetch the updated or inserted record
            cursor.execute("""
                SELECT prompt_id, prompt_label, section_title, doc_type, corpus_id, base_prompt, customization_prompt, wisdom_1, wisdom_2, chunks, dependencies,
                customize_prompt_based_on, send_along_customised_prompt, which_chunks, wisdom_received, llm_flow, llm, model, show_on_frontend, label_for_output
                FROM analyzer_prompts
                WHERE doc_type = %s AND prompt_label = %s
            """, (prompt.doc_type, prompt_label))
            updated_prompt = cursor.fetchone()

            if updated_prompt:
                updated_prompt_dict = {
                    "prompt_id": updated_prompt[0],
                    "prompt_label": updated_prompt[1],
                    "section_title": updated_prompt[2],
                    "doc_type": updated_prompt[3],
                    "corpus_id": updated_prompt[4],
                    "base_prompt": updated_prompt[5],
                    "customization_prompt": updated_prompt[6],
                    "wisdom_1": updated_prompt[7],
                    "wisdom_2": updated_prompt[8],
                    "chunks": updated_prompt[9],
                    "dependencies": updated_prompt[10],
                    "customize_prompt_based_on": updated_prompt[11],
                    "send_along_customised_prompt": updated_prompt[12],
                    "which_chunks": updated_prompt[13],
                    "wisdom_received": updated_prompt[14],
                    "llm_flow": updated_prompt[15],
                    "llm": updated_prompt[16],
                    "model": updated_prompt[17],
                    "show_on_frontend": updated_prompt[18],
                    "label_for_output": updated_prompt[19]
                }
                updated_prompts.append(updated_prompt_dict)

        conn.commit()
        return {"prompts": updated_prompts}
    except Exception as e:
        traceback.print_exc()
        return {"Error": str(e)}
    finally:
        if conn:
            conn.close()        


def delete_prompts_from_db(prompt_label, doc_type):
    try:
        conn = connect_to_mysql()
        cursor = conn.cursor(dictionary=True)

        # Check if the prompt exists
        query = """
            SELECT * FROM analyzer_prompts_test
            WHERE prompt_label = %s AND doc_type = %s
        """
        cursor.execute(query, (prompt_label, doc_type))
        prompt = cursor.fetchone()

        if prompt:
            # Delete the prompt
            delete_query = """
                DELETE FROM analyzer_prompts_test
                WHERE prompt_label = %s AND doc_type = %s
            """
            cursor.execute(delete_query, (prompt_label, doc_type))
            conn.commit()

            # Return the deleted prompt
            return prompt
        else:
            raise HTTPException(status_code=404, detail="Prompt not found")
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None
    finally:
        cursor.close()
        conn.close()
    

def fetch_corpus_id_analyzer_prompts():
    conn = connect_to_mysql()
    corpus =[]
    try:
        cursor = conn.cursor(dictionary=True)
        query = """SELECT DISTINCT corpus_id FROM analyzer_prompts"""
        cursor.execute(query)
        rows = cursor.fetchall()
        corpus =[]
        for row in rows:
            if row["corpus_id"]:
                corpus.append(row["corpus_id"])
    finally:
        if conn:
            conn.close()
    return {"corpus_id": corpus}


def get_all_summary_prompts_from_db(doc_type=None):
    connection = connect_to_mysql()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            # Fetch the most recently updated prompt
            if doc_type:
                query = '''SELECT prompt_label, doc_type, summary_prompt 
                    FROM analyzer_comments_summary_prompts 
                    WHERE doc_type = %s
                    ORDER BY created_at DESC
                    LIMIT 1'''
                cursor.execute(query, (doc_type.value,))
            else:
                query = '''SELECT prompt_label, doc_type, summary_prompt 
                    FROM analyzer_comments_summary_prompts 
                    WHERE (doc_type, created_at) 
                    IN (SELECT doc_type, MAX(created_at) 
                        FROM analyzer_comments_summary_prompts 
                        GROUP BY doc_type)'''
                cursor.execute(query)

            prompts = cursor.fetchall()

            if not prompts:
                raise HTTPException(status_code=404, detail="No analyzer summary prompts found")
            
            response = {"prompts": []}
            for prompt in prompts:
                response["prompts"].append(prompt)

            return response

        except mysql.connector.Error as err:
            print(f"Error: {err}")
            raise HTTPException(status_code=500, detail="Internal server error")

        finally:
            cursor.close()
            connection.close()
    else:
        raise HTTPException(status_code=500, detail="Unable to establish connection with database")


def update_summary_prompts_in_db(prompts):
    try:
        connection = connect_to_mysql()
        
        if not connection:
            raise HTTPException(status_code=500, detail="Unable to establish connection with database")

        try:
            cursor = connection.cursor(dictionary=True)
            updated_prompts = []

            # Start a transaction
            connection.start_transaction()

            for prompt in prompts:
                doc_type = prompt.doc_type.value
                summary_prompt = prompt.summary_prompt

                # Insert or update the summary prompt for the given document type
                query = """
                    INSERT INTO analyzer_comments_summary_prompts (doc_type, summary_prompt) 
                    VALUES (%s, %s) 
                    ON DUPLICATE KEY UPDATE summary_prompt = VALUES(summary_prompt)
                """
                cursor.execute(query, (doc_type, summary_prompt))
                
                # Commit the changes to ensure data persistence
                connection.commit()

                # Fetch the most recently inserted prompt based on doc_type
                query = "SELECT doc_type, summary_prompt FROM analyzer_comments_summary_prompts WHERE doc_type = %s ORDER BY created_at DESC LIMIT 1"
                cursor.execute(query, (doc_type,))

                updated_prompt = cursor.fetchone()
                
                updated_prompts.append(updated_prompt)
                if not updated_prompts:
                    raise HTTPException(status_code=404, detail="No analyzer proposal summary prompts found for the given document types")

            if not updated_prompts:
                raise HTTPException(status_code=404, detail="No analyzer summary prompts found for the given document types")

            return updated_prompts

        except mysql.connector.Error as err:
            # Handle MySQL-related errors
            connection.rollback()  # Roll back the transaction to maintain data integrity
            raise HTTPException(status_code=500, detail=f"Database error: {err}")

        finally:
            cursor.close()
            connection.close()

    except HTTPException as http_err:
        # Re-raise the HTTP error if it was already raised
        raise http_err
    
    except Exception as ex:
        # Handle any other exceptions
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {ex}")


def get_proposal_summary_prompts_from_db(doc_type):
    connection = connect_to_mysql()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)

            if doc_type:
                query = """
                    SELECT doc_type, proposal_prompt
                    FROM analyzer_proposal_summary_prompts
                    WHERE doc_type = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                cursor.execute(query, (doc_type.value,))
            else:
                query = """
                    SELECT doc_type, proposal_prompt
                    FROM analyzer_proposal_summary_prompts
                    WHERE (doc_type, created_at) IN (
                        SELECT doc_type, MAX(created_at)
                        FROM analyzer_proposal_summary_prompts
                        GROUP BY doc_type
                    )
                """
                cursor.execute(query)

            prompts = cursor.fetchall()

            if not prompts:
                raise HTTPException(status_code=404, detail="No analyzer summary prompts found")
            
            response = {"prompts": []}
            for prompt in prompts:
                response["prompts"].append(prompt)

            return {"prompts": prompts}

        except mysql.connector.Error as err:
            print(f"Error: {err}")
            raise HTTPException(status_code=500, detail="Internal server error")

        finally:
            cursor.close()
            connection.close()
    else:
        raise HTTPException(status_code=500, detail="Unable to establish connection with database")


def update_proposal_prompts_in_db(prompts):
    try:
        connection = connect_to_mysql()
        
        if not connection:
            raise HTTPException(status_code=500, detail="Unable to establish connection with database")

        try:
            cursor = connection.cursor(dictionary=True)
            updated_prompts = []

            # Start a transaction
            connection.start_transaction()

            for prompt in prompts:
                doc_type = prompt.doc_type.value
                proposal_prompt = prompt.proposal_prompt

                # Insert or update the proposal summary prompt for the given document type
                query = """
                    INSERT INTO analyzer_proposal_summary_prompts (doc_type, proposal_prompt) 
                    VALUES (%s, %s) 
                    ON DUPLICATE KEY UPDATE proposal_prompt = VALUES(proposal_prompt)
                """
                cursor.execute(query, (doc_type, proposal_prompt))

                # Commit the changes to ensure data persistence
                connection.commit()

                # Fetch the most recently inserted prompt based on doc_type
                query = "SELECT doc_type, proposal_prompt FROM analyzer_proposal_summary_prompts WHERE doc_type = %s ORDER BY created_at DESC LIMIT 1"
                cursor.execute(query, (doc_type,))

                updated_prompt = cursor.fetchone()
                
                updated_prompts.append(updated_prompt)
                if not updated_prompts:
                    raise HTTPException(status_code=404, detail="No analyzer proposal summary prompts found for the given document types")
            
            return updated_prompts

        except mysql.connector.Error as err:
            # If there's a MySQL error, rollback and raise a custom error message
            connection.rollback()  # Ensure the transaction is not half-finished
            raise HTTPException(status_code=500, detail=f"Database error: {err}")

        finally:
            cursor.close()
            connection.close()  # Close the connection safely

    except HTTPException as http_err:
        # Handle expected HTTP exceptions and re-raise them
        raise http_err
    
    except Exception as ex:
        # Handle unexpected exceptions with a generic error message
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {ex}")

  
def get_all_custom_prompts_from_db(doc_type=None, organization_id=None):
    conn = connect_to_mysql()
    prompts = []
    try:
        cursor = conn.cursor(dictionary=True)
        query = """SELECT * FROM analyzer_custom_partition_prompts"""
        params = []

        if doc_type:
            query += " WHERE doc_type = %s"
            params.append(doc_type)
    
        if organization_id:
            if not doc_type:
                query += " WHERE"
            else:
                query += " AND"
            query += " organization_id = %s"
            params.append(organization_id)
            
        cursor.execute(query, params)
        prompts = cursor.fetchall()
    finally:
        if conn:
            conn.close()
    return prompts


def update_custom_prompts_in_db(prompts_update,  prompt_label="P_Custom"):
    conn = connect_to_mysql()
    try:
        cursor = conn.cursor()
        updated_prompts = []
        for prompt in prompts_update:
            
            corpus_id = prompt.corpus_id
            section_title = prompt.section_title
            if prompt.number_of_chunks is None:
                print(f"Chunks is None")
                prompt.number_of_chunks = 6
            
            if not prompt.doc_type or not prompt.organization_id:
                return {"prompts": f"doc_type or organization_id missing"}
            
            if not prompt.base_prompt and not prompt.customization_prompt:
                return {"prompts": f"No data provided for update for doc_type: {prompt.doc_type.value}"}
            
            cursor.execute("""SELECT COUNT(*) FROM analyzer_custom_partition_prompts
                                WHERE prompt_label = %s AND doc_type = %s AND organization_id = %s""",
                            (prompt_label, prompt.doc_type, prompt.organization_id))
            row_count = cursor.fetchone()[0]

            if row_count > 0:  # If present, perform update
                if prompt.section_title is not None:
                    query = """UPDATE analyzer_custom_partition_prompts SET base_prompt = %s, customization_prompt = %s, corpus_id = %s, section_title=%s, chunks=%s
                            WHERE prompt_label = %s AND doc_type = %s AND organization_id = %s"""
                    cursor.execute(query,(
                        prompt.base_prompt,
                        prompt.customization_prompt,
                        prompt.corpus_id,
                        prompt.section_title,
                        prompt.number_of_chunks,
                        prompt_label,
                        prompt.doc_type,
                        prompt.organization_id,
                    ))
                else:
                    query = """UPDATE analyzer_custom_partition_prompts SET base_prompt = %s, customization_prompt = %s, corpus_id = %s, chunks=%s
                        WHERE prompt_label = %s AND doc_type = %s AND organization_id = %s"""
                    cursor.execute(query,(
                        prompt.base_prompt,
                        prompt.customization_prompt,
                        prompt.corpus_id,
                        prompt.number_of_chunks,
                        prompt_label,
                        prompt.doc_type,
                        prompt.organization_id
                    ))
            else:  # Otherwise, perform insert
                if prompt.section_title is not None:
                    query = """INSERT INTO analyzer_custom_partition_prompts (prompt_label, base_prompt, customization_prompt, corpus_id, section_title, doc_type, organization_id, chunks)
                                VALUES (%s,%s, %s, %s, %s, %s, %s, %s)"""
                    cursor.execute(query, (
                        prompt_label,
                        prompt.base_prompt,
                        prompt.customization_prompt,
                        prompt.corpus_id,
                        prompt.section_title,
                        prompt.doc_type,
                        prompt.organization_id,
                        prompt.number_of_chunks
                    ))
                else:
                    query = """INSERT INTO analyzer_custom_partition_prompts (prompt_label, base_prompt, customization_prompt, corpus_id, doc_type, organization_id, chunks)
                                VALUES (%s,%s, %s, %s, %s, %s, %s)"""
                    cursor.execute(query, (
                        prompt_label,
                        prompt.base_prompt,
                        prompt.customization_prompt,
                        prompt.corpus_id,
                        prompt.doc_type,
                        prompt.organization_id,
                        prompt.number_of_chunks
                    ))
            
            # Fetcg updated or inserted value
            query = "SELECT prompt_label, doc_type, organization_id, base_prompt, customization_prompt, corpus_id, section_title, chunks FROM analyzer_custom_partition_prompts WHERE doc_type = %s AND organization_id=%s"
            cursor.execute(query, (prompt.doc_type, prompt.organization_id))
            updated_prompt = cursor.fetchone()
            if updated_prompt:
                updated_prompt = {
                    "prompt_label": prompt_label,
                    "doc_type": prompt.doc_type,
                    "organization_id": prompt.organization_id,
                    "base_prompt": prompt.base_prompt,
                    "customization_prompt": prompt.customization_prompt,
                    "corpus_id": prompt.corpus_id,
                    "section_title": prompt.section_title,
                    "chunks": prompt.number_of_chunks
                }
                updated_prompts.append(updated_prompt)

        conn.commit()
        return updated_prompts
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        if conn:
            conn.close()


def delete_custom_prompts_from_db(doc_type, organization_id):
    try:
        conn = connect_to_mysql()
        cursor = conn.cursor(dictionary=True)

        # Check if the prompt exists
        query = """
            SELECT * FROM analyzer_custom_partition_prompts_test
            WHERE doc_type = %s AND organization_id = %s
        """
        cursor.execute(query, (doc_type, organization_id))
        prompt = cursor.fetchone()

        if prompt:
            # Delete the prompt
            delete_query = """
                DELETE FROM analyzer_custom_partition_prompts_test
                WHERE doc_type = %s AND organization_id = %s
            """
            cursor.execute(delete_query, (doc_type, organization_id))
            conn.commit()

            # Return the deleted prompt
            return prompt
        else:
            raise HTTPException(status_code=404, detail="Prompt not found")
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None
    finally:
        cursor.close()
        conn.close()
    

def fetch_corpus_id_organization_prompts():
    conn = connect_to_mysql()
    corpus =[]
    try:
        cursor = conn.cursor(dictionary=True)
        query = """SELECT DISTINCT corpus_id FROM analyzer_custom_partition_prompts"""
        cursor.execute(query)
        rows = cursor.fetchall()
        corpus =[]
        for row in rows:
            if row["corpus_id"]:
                corpus.append(row["corpus_id"])
    finally:
        if conn:
            conn.close()
    return {"corpus_id": corpus}


def get_all_evaluator_prompts_from_db(paritition_type=None, doc_type=None, organization_id=None, org_guideline_id=None):
    conn = connect_to_mysql()
    prompts = []
    try:
        cursor = conn.cursor(dictionary=True)
        query = """SELECT * FROM analyzer_custom_evaluator_prompts"""
        params = []

        if doc_type:
            query += " WHERE doc_type = %s"
            params.append(doc_type)
        if organization_id:
            if not doc_type:
                query += " WHERE"
            else:
                query += " AND"
            query += " organization_id = %s"
            params.append(organization_id)
        if paritition_type:
            if not doc_type and not organization_id:
                query += " WHERE"
            else:
                query += " AND"
            query += " prompt_label = %s"
            params.append(paritition_type)
        if org_guideline_id is not None:
            if not doc_type and not organization_id and not paritition_type:
                query += " WHERE"
            else:
                query += " AND"
            query += " org_guideline_id = %s"
            params.append(org_guideline_id)
        cursor.execute(query, params)
        prompts = cursor.fetchall()
    finally:
        if conn:
            conn.close()
    return prompts

    
def update_evaluator_prompts_in_db(prompts_update):
    conn = connect_to_mysql()
    try:
        cursor = conn.cursor()
        updated_prompts = []
        for prompt in prompts_update:
            if not prompt.prompt_label:
                return {"prompts": f"Prompt Label Missing"}
            
            if not prompt.doc_type or not prompt.organization_id or not prompt.org_guideline_id:
                return {"prompts": "doc_type, organization_id or org_guideline_id is missing"}
            
            if not prompt.base_prompt and not prompt.customization_prompt:
                return {"prompts": f"No data provided for update for doc_type: {prompt.doc_type.value}"}
            
            cursor.execute("""SELECT section_title,chunks,additional_dependencies,customize_prompt_based_on, send_along_customised_prompt, prompt_context, wisdom_received,COUNT(*) FROM analyzer_custom_evaluator_prompts
                                WHERE prompt_label = %s AND doc_type = %s AND organization_id = %s AND org_guideline_id = %s""",
                            (prompt.prompt_label, prompt.doc_type, prompt.organization_id, prompt.org_guideline_id))
            result = cursor.fetchall()
            row_count = result[0][7]
            section_title = result[0][0]
            chunks = result[0][1]
            dependency = result[0][2]
            customize_prompt_based_on = result[0][3]
            send_along_customised_prompt = result[0][4]
            prompt_context = result[0][5]
            wisdom_received = result[0][6]

            if not prompt.section_title:
                if section_title is not None:
                    prompt.section_title = section_title
                else:
                    prompt.section_title = ""
            
            if not prompt.number_of_chunks:
                if chunks is not None:
                    prompt.number_of_chunks = chunks
                else:
                    prompt.number_of_chunks= 12
            
            if not prompt.additional_dependencies:
                if dependency is not None:
                    prompt.additional_dependencies = dependency
                else:
                    prompt.additional_dependencies = ""
            
            if not prompt.customize_prompt_based_on:
                if customize_prompt_based_on is not None:
                    prompt.customize_prompt_based_on = customize_prompt_based_on
                else:
                    prompt.customize_prompt_based_on = ""
            
            if not prompt.send_along_customised_prompt:
                if send_along_customised_prompt is not None:
                    prompt.send_along_customised_prompt = send_along_customised_prompt
                else:
                    prompt.send_along_customised_prompt = ""
            
            if not prompt.prompt_context:
                if prompt_context is not None:
                    prompt.prompt_context = prompt_context
                else:
                    prompt.prompt_context = ""

            if not prompt.wisdom_received:
                if wisdom_received is not None:
                    prompt.wisdom_received = wisdom_received
                else:
                    prompt.wisdom_received = ""

            if prompt.wisdom_1 is not None and prompt.wisdom_2 is not None:
                if row_count > 0:  # If present, perform update
                    query = """UPDATE analyzer_custom_evaluator_prompts SET base_prompt = %s, customization_prompt = %s, wisdom_1=%s, wisdom_2=%s, chunks=%s,section_title=%s, additional_dependencies=%s, customize_prompt_based_on=%s, send_along_customised_prompt=%s, prompt_context=%s, wisdom_received=%s,LLM_Flow=%s,LLM=%s,Model=%s,show_on_frontend=%s,label_for_output=%s
                            WHERE prompt_label = %s AND doc_type = %s AND organization_id = %s AND org_guideline_id = %s"""
                    cursor.execute(query,(
                        prompt.base_prompt,
                        prompt.customization_prompt,
                        prompt.wisdom_1,
                        prompt.wisdom_2,
                        prompt.number_of_chunks,
                        prompt.section_title,
                        prompt.additional_dependencies,
                        prompt.customize_prompt_based_on,
                        prompt.send_along_customised_prompt,
                        prompt.prompt_context,
                        prompt.wisdom_received,
                        prompt.llm_flow,
                        prompt.llm,
                        prompt.model,
                        prompt.show_on_frontend,
                        prompt.label_for_output,
                        prompt.prompt_label,
                        prompt.doc_type,
                        prompt.organization_id,
                        prompt.org_guideline_id
                    ))
                else:  # Otherwise, perform insert
                    query = """INSERT INTO analyzer_custom_evaluator_prompts (prompt_label, base_prompt, customization_prompt, wisdom_1, wisdom_2, chunks, section_title,additional_dependencies,customize_prompt_based_on, send_along_customised_prompt, prompt_context, wisdom_received,LLM_Flow,LLM,Model,show_on_frontend,label_for_output,doc_type, organization_id, org_guideline_id)
                                VALUES (%s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,  %s, %s, %s, %s, %s, %s, %s, %s)"""
                    cursor.execute(query, (
                        prompt.prompt_label,
                        prompt.base_prompt,
                        prompt.customization_prompt,
                        prompt.wisdom_1,
                        prompt.wisdom_2,
                        prompt.number_of_chunks,
                        prompt.section_title,
                        prompt.additional_dependencies,
                        prompt.customize_prompt_based_on,
                        prompt.send_along_customised_prompt,
                        prompt.prompt_context,
                        prompt.wisdom_received,
                        prompt.llm_flow,
                        prompt.llm,
                        prompt.model,
                        prompt.show_on_frontend,
                        prompt.label_for_output,
                        prompt.doc_type,
                        prompt.organization_id,
                        prompt.org_guideline_id
                    ))
                # Fetcg updated or inserted value
                query = "SELECT prompt_label, doc_type, organization_id, org_guideline_id, base_prompt, customization_prompt, wisdom_1, wisdom_2, section_title, chunks, additional_dependencies,customize_prompt_based_on, send_along_customised_prompt, prompt_context, wisdom_received,LLM_Flow,LLM,Model,show_on_frontend,label_for_output FROM analyzer_custom_evaluator_prompts WHERE prompt_label = %s AND doc_type = %s AND organization_id=%s AND org_guideline_id = %s"
                cursor.execute(query, (prompt.prompt_label, prompt.doc_type, prompt.organization_id, prompt.org_guideline_id))
                updated_prompt = cursor.fetchall()

                if updated_prompt:
                    #updated_prompt = {"prompt_label": prompt.prompt_label, "additional_dependencies":updated_prompt[0][11],"doc_type": prompt.doc_type, "identity": prompt.identity, "organization_id": prompt.organization_id, "org_guideline_id": prompt.org_guideline_id,"section_title":updated_prompt[0][9],"chunks":updated_prompt[0][10], "base_prompt": prompt.base_prompt, "customization_prompt": prompt.customization_prompt, "wisdom_1": prompt.wisdom_1, "wisdom_2": prompt.wisdom_2,"customize_prompt_based_on":updated_prompt[0][12], "send_along_customised_prompt":updated_prompt[0][13], "prompt_context":updated_prompt[0][14], "wisdom_received":updated_prompt[0][15],"LLM_Flow":updated_prompt[0][16],"LLM":updated_prompt[0][17],"Model":updated_prompt[0][18],"show_on_frontend":updated_prompt[0][19],"label_for_output":updated_prompt[0][20]}
                    updated_prompts.append(updated_prompt)
            else:
                if row_count > 0:  # If present, perform update
                    query = """UPDATE analyzer_custom_evaluator_prompts SET base_prompt = %s, customization_prompt = %s, chunks = %s, section_title = %s, additional_dependencies=%s, customize_prompt_based_on=%s, send_along_customised_prompt=%s, prompt_context=%s, wisdom_received=%s,LLM_Flow=%s,LLM=%s,Model=%s,show_on_frontend=%s,label_for_output=%s
                            WHERE prompt_label = %s AND doc_type = %s AND organization_id = %s AND org_guideline_id = %s"""
                    cursor.execute(query,(
                        prompt.base_prompt,
                        prompt.customization_prompt,
                        prompt.number_of_chunks,
                        prompt.section_title,
                        prompt.additional_dependencies,
                        prompt.customize_prompt_based_on,
                        prompt.send_along_customised_prompt,
                        prompt.prompt_context,
                        prompt.wisdom_received,
                        prompt.llm_flow,
                        prompt.llm,
                        prompt.model,
                        prompt.show_on_frontend,
                        prompt.label_for_output,
                        prompt.prompt_label,
                        prompt.doc_type,
                        prompt.organization_id,
                        prompt.org_guideline_id
                    ))
                else:  # Otherwise, perform insert
                    query = """INSERT INTO analyzer_custom_evaluator_prompts (prompt_label, base_prompt, customization_prompt, chunks, section_title, additional_dependencies, customize_prompt_based_on, send_along_customised_prompt, prompt_context, wisdom_received,LLM_Flow,LLM,Model,show_on_frontend,label_for_output,doc_type, organization_id, org_guideline_id)
                                VALUES (%s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s)"""
                    cursor.execute(query, (
                        prompt.prompt_label,
                        prompt.base_prompt,
                        prompt.customization_prompt,
                        prompt.number_of_chunks,
                        prompt.section_title,
                        prompt.additional_dependencies,
                        prompt.customize_prompt_based_on,
                        prompt.send_along_customised_prompt,
                        prompt.prompt_context,
                        prompt.wisdom_received,
                        prompt.llm_flow,
                        prompt.llm,
                        prompt.model,
                        prompt.show_on_frontend,
                        prompt.label_for_output,
                        prompt.doc_type,
                        prompt.organization_id,
                        prompt.org_guideline_id
                    ))
                
                # Fetch updated or inserted value
                query = "SELECT prompt_label, doc_type, organization_id, org_guideline_id, base_prompt, customization_prompt, section_title, chunks, additional_dependencies, customize_prompt_based_on, send_along_customised_prompt, prompt_context, wisdom_received,LLM_Flow,LLM,Model,show_on_frontend,label_for_output FROM analyzer_custom_evaluator_prompts WHERE prompt_label = %s AND doc_type = %s AND idntity = %s AND organization_id=%s AND org_guideline_id = %s"
                cursor.execute(query, (prompt.prompt_label, prompt.doc_type, prompt.organization_id, prompt.org_guideline_id))
                updated_prompt = cursor.fetchall()

                if updated_prompt:
                    updated_prompt = {"prompt_label": prompt.prompt_label,"additional_dependencies":updated_prompt[0][9], "doc_type": prompt.doc_type, "organization_id": prompt.organization_id, "org_guideline_id": prompt.org_guideline_id, "section_title":updated_prompt[0][7],"chunks":updated_prompt[0][8],"base_prompt": prompt.base_prompt, "customization_prompt": prompt.customization_prompt, "customize_prompt_based_on":updated_prompt[0][10], "send_along_customised_prompt":updated_prompt[0][11], "prompt_context":updated_prompt[0][12], "wisdom_received":updated_prompt[0][13],"LLM_Flow":updated_prompt[0][14],"LLM":updated_prompt[0][15],"Model":updated_prompt[0][16],"show_on_frontend":updated_prompt[0][17],"label_for_output":updated_prompt[0][18] }
                    updated_prompts.append(updated_prompt)

        conn.commit()
        return {"prompts": updated_prompts}
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        if conn:
            conn.close()


def update_evaluator_prompts_in_db_test(prompts_update):
    conn = connect_to_mysql()
    try:
        cursor = conn.cursor()
        updated_prompts = []
        print(f"Starting update for {len(prompts_update)} prompts")
        
        for prompt in prompts_update:
            print(f"Processing prompt: {prompt.prompt_label}")
            print(f"Unique identifier: prompt_label={prompt.prompt_label}, doc_type={prompt.doc_type} organization_id={prompt.organization_id}, org_guideline_id={prompt.org_guideline_id}")
            
            if not prompt.prompt_label:
                print("Error: Prompt Label Missing")
                return {"prompts": "Prompt Label Missing"}

            if not prompt.doc_type or not prompt.organization_id:
                print(f"Error: doc_type or organization_id missing for prompt {prompt.prompt_label}")
                return {"prompts": "doc_type or organization_id missing"}

            if not prompt.base_prompt and not prompt.customization_prompt:
                print(f"Error: No data provided for update for doc_type: {prompt.doc_type.value}")
                return {"prompts": f"No data provided for update for doc_type: {prompt.doc_type.value}"}

            # Fetch existing data
            if prompt.org_guideline_id is not None:
                cursor.execute("""SELECT COUNT(*) FROM analyzer_custom_evaluator_prompts
                                  WHERE prompt_label = %s AND doc_type = %s AND organization_id = %s AND org_guideline_id = %s""",
                               (prompt.prompt_label, prompt.doc_type, prompt.organization_id, prompt.org_guideline_id))
            else:
                cursor.execute("""SELECT COUNT(*) FROM analyzer_custom_evaluator_prompts
                                  WHERE prompt_label = %s AND doc_type = %s AND organization_id = %s AND org_guideline_id IS NULL""",
                               (prompt.prompt_label, prompt.doc_type, prompt.organization_id))
            row_count = cursor.fetchone()[0]
            print(f"Found {row_count} existing rows for prompt {prompt.prompt_label}")

            update_fields = []
            update_values = []
            if prompt.base_prompt:
                update_fields.append("base_prompt = %s")
                update_values.append(prompt.base_prompt)
            if prompt.customization_prompt:
                update_fields.append("customization_prompt = %s")
                update_values.append(prompt.customization_prompt)
            if prompt.wisdom_1 is not None:
                update_fields.append("wisdom_1 = %s")
                update_values.append(prompt.wisdom_1)
            if prompt.wisdom_2 is not None:
                update_fields.append("wisdom_2 = %s")
                update_values.append(prompt.wisdom_2)
            if prompt.section_title is not None:
                update_fields.append("section_title = %s")
                update_values.append(prompt.section_title)
            if prompt.number_of_chunks is not None:
                update_fields.append("chunks = %s")
                update_values.append(prompt.number_of_chunks)
            if prompt.dependencies is not None:
                update_fields.append("dependencies = %s")
                update_values.append(prompt.dependencies)
            
            if row_count == 1:
                print(f"Updating existing prompt: {prompt.prompt_label}")
                if update_fields:
                    query = f"""UPDATE analyzer_custom_evaluator_prompts 
                                SET {', '.join(update_fields)}
                                WHERE prompt_label = %s AND doc_type = %s AND organization_id = %s"""
                    params = update_values + [prompt.prompt_label, prompt.doc_type, prompt.organization_id]
                    if prompt.org_guideline_id is not None:
                        query += " AND org_guideline_id = %s"
                        params.append(prompt.org_guideline_id)
                    else:
                        query += " AND org_guideline_id IS NULL"
                    cursor.execute(query, params)
                    print(f"Updated {cursor.rowcount} rows for prompt {prompt.prompt_label}")
            else:
                if row_count > 1:
                    print(f"Found multiple rows for prompt {prompt.prompt_label}. Deleting and reinserting.")
                    # Delete all existing records
                    if prompt.org_guideline_id is not None:
                        cursor.execute("""DELETE FROM analyzer_custom_evaluator_prompts
                                          WHERE prompt_label = %s AND doc_type = %s AND organization_id = %s AND org_guideline_id = %s""",
                                       (prompt.prompt_label, prompt.doc_type, prompt.organization_id, prompt.org_guideline_id))
                    else:
                        cursor.execute("""DELETE FROM analyzer_custom_evaluator_prompts
                                          WHERE prompt_label = %s AND doc_type = %s AND organization_id = %s AND org_guideline_id IS NULL""",
                                       (prompt.prompt_label, prompt.doc_type, prompt.organization_id))
                    print(f"Deleted {cursor.rowcount} rows for prompt {prompt.prompt_label}")
                
                print(f"Inserting new record for prompt {prompt.prompt_label}")
                # Insert new record
                query = """INSERT INTO analyzer_custom_evaluator_prompts 
                           (prompt_label, base_prompt, customization_prompt, wisdom_1, wisdom_2, 
                            doc_type, organization_id, org_guideline_id, section_title, chunks, dependencies)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                cursor.execute(query, (
                    prompt.prompt_label,
                    prompt.base_prompt,
                    prompt.customization_prompt,
                    prompt.wisdom_1,
                    prompt.wisdom_2,
                    prompt.doc_type,
                    prompt.organization_id,
                    prompt.org_guideline_id,
                    prompt.section_title,
                    prompt.number_of_chunks,
                    prompt.dependencies
                ))
                print(f"Inserted new row for prompt {prompt.prompt_label}")

            # Fetch the updated or inserted value
            if prompt.org_guideline_id is not None:
                cursor.execute("""SELECT prompt_label, doc_type, organization_id, org_guideline_id, base_prompt, 
                                         customization_prompt, wisdom_1, wisdom_2, section_title, chunks, dependencies
                                  FROM analyzer_custom_evaluator_prompts 
                                  WHERE prompt_label = %s AND doc_type = %s AND organization_id = %s AND org_guideline_id = %s
                                  ORDER BY created_at DESC LIMIT 1""",
                               (prompt.prompt_label, prompt.doc_type, prompt.organization_id, prompt.org_guideline_id))
            else:
                cursor.execute("""SELECT prompt_label, doc_type, organization_id, org_guideline_id, base_prompt, 
                                         customization_prompt, wisdom_1, wisdom_2, section_title, chunks, dependencies
                                  FROM analyzer_custom_evaluator_prompts 
                                  WHERE prompt_label = %s AND doc_type = %s AND organization_id = %s AND org_guideline_id IS NULL
                                  ORDER BY created_at DESC LIMIT 1""",
                               (prompt.prompt_label, prompt.doc_type, prompt.organization_id))
            updated_prompt = cursor.fetchone()

            if updated_prompt:
                updated_prompt = {
                    "prompt_label": updated_prompt[0],
                    "doc_type": updated_prompt[1],
                    "organization_id": updated_prompt[2],
                    "org_guideline_id": updated_prompt[3],
                    "base_prompt": updated_prompt[4],
                    "customization_prompt": updated_prompt[5],
                    "wisdom_1": updated_prompt[6],
                    "wisdom_2": updated_prompt[7],
                    "section_title": updated_prompt[8],
                    "chunks": updated_prompt[9],
                    "dependencies": updated_prompt[10]
                }
                updated_prompts.append(updated_prompt)
                print(f"Successfully processed prompt: {prompt.prompt_label}")
            else:
                print(f"Warning: Could not fetch updated data for prompt {prompt.prompt_label}")

        conn.commit()
        print(f"Successfully updated {len(updated_prompts)} prompts")
        return updated_prompts
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        if conn:
            conn.close()
        print("Database connection closed")


def delete_evaluator_prompt(evaluator_prompt_label, doc_type, organization_id, org_guideline_id):
    try:
        conn = connect_to_mysql()
        cursor = conn.cursor(dictionary=True)

        # Check if the prompt exists
        query = """
            SELECT * FROM analyzer_custom_evaluator_prompts
            WHERE prompt_label = %s AND doc_type = %s AND organization_id = %s AND org_guideline_id = %s
        """
        cursor.execute(query, (evaluator_prompt_label, doc_type, organization_id, org_guideline_id))
        prompt = cursor.fetchone()

        if prompt:
            # Delete the prompt
            delete_query = """
                DELETE FROM analyzer_custom_evaluator_prompts
                WHERE prompt_label = %s AND doc_type = %s AND organization_id = %s AND org_guideline_id = %s
            """
            cursor.execute(delete_query, (evaluator_prompt_label, doc_type, organization_id, org_guideline_id))
            conn.commit()

            # Return the deleted prompt
            return prompt
        else:
            raise HTTPException(status_code=404, detail="Prompt not found")
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None
    finally:
        cursor.close()
        conn.close()

            
def get_tor_summary_prompts_from_db(doc_type=None, organization_id=None):
    conn = connect_to_mysql()
    prompts = []
    try:
        cursor = conn.cursor(dictionary=True)
        query = """SELECT * FROM tor_summary_prompts"""
        params = []

        if doc_type:
            query += " WHERE doc_type = %s"
            params.append(doc_type)
        if organization_id:
            if not doc_type:
                query += " WHERE"
            else:
                query += " AND"
            query += " organization_id = %s"
            params.append(organization_id)
        
        cursor.execute(query, params)
        prompts = cursor.fetchall()
    finally:
        if conn:
            conn.close()
    return prompts


def update_tor_prompts_in_db(prompts_update):
    conn = connect_to_mysql()
    try:
        cursor = conn.cursor()
        updated_prompts = []
        for prompt in prompts_update:
                
            cursor.execute("""SELECT COUNT(*) FROM tor_summary_prompts WHERE doc_type = %s AND organization_id= %s""", (prompt.doc_type,prompt.organization_id))
            row_count = cursor.fetchone()[0]

            if row_count > 0:  # If present, perform update
                query = """UPDATE tor_summary_prompts SET tor_summary_prompt = %s WHERE doc_type = %s AND organization_id= %s"""
                cursor.execute(query,(
                    prompt.tor_summary_prompt,
                    prompt.doc_type,
                    prompt.organization_id
                ))
            else:  # Otherwise, perform insert
                query = """INSERT INTO tor_summary_prompts (tor_summary_prompt, doc_type, organization_id) VALUES (%s,%s, %s)"""
                cursor.execute(query, (
                    prompt.tor_summary_prompt,
                    prompt.doc_type,
                    prompt.organization_id
                ))
            
            # Fetch updated or inserted value
            query = "SELECT doc_type, organization_id, tor_summary_prompt FROM tor_summary_prompts WHERE doc_type = %s AND organization_id= %s"
            cursor.execute(query, (prompt.doc_type, prompt.organization_id))
            updated_prompt = cursor.fetchone()

            if updated_prompt:
                updated_prompt = {"doc_type": prompt.doc_type, "organization_id": prompt.organization_id, "tor_summary_prompt": prompt.tor_summary_prompt}
                updated_prompts.append(updated_prompt)

        conn.commit()
        return updated_prompts
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        if conn:
            conn.close()
            
            
def measure_time(func, num_connections=10):
    total_time = 0
    successful_connections = 0
    for _ in range(num_connections):
        start_time = time.time()
        try:
            conn = func()
            if conn.is_connected():
                successful_connections += 1
                conn.close()
            else:
                print("Connection not successful")
        except Exception as e:
            print(f"Exception: {e}")
        end_time = time.time()
        connection_time = end_time - start_time
        total_time += connection_time
        print(f"Time taken for this connection: {connection_time:.2f} seconds")
    return total_time, successful_connections


if __name__ == "__main__":
    # Measure time taken for direct connections
    time_direct, successful_direct = measure_time(connect_to_mysql)
    print(f"Time taken for direct connections: {time_direct:.2f} seconds")
    print(f"Successful direct connections: {successful_direct}/10")