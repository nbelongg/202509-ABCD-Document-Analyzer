import os
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
import traceback
from logger import streamlit_logger as logger

# Load environment variables from .env file
load_dotenv()


def connect_to_database():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("mysql_host"),
            database=os.getenv("mysql_database"),
            user=os.getenv("mysql_user"),
            password=os.getenv("mysql_password"),
        )
        if connection.is_connected():
            logger.info("Successfully connected to the database")
            return connection
    except Error as e:
        logger.error(f"Database connection error: {e}")
        return None


def get_prompts():
    connection = connect_to_database()
    if not connection:
        return []

    try:
        cursor = connection.cursor()
        query = "SELECT prompt, prompt_title, prompt_content FROM pdf_analyzer_prompts where image_prompt=0 ORDER BY created_at DESC"
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    except Error as e:
        logger.error(f"Error retrieving prompts and titles: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def get_image_prompts():
    connection = connect_to_database()
    if not connection:
        return []

    try:
        cursor = connection.cursor()
        query = "SELECT prompt, prompt_title, prompt_content FROM pdf_analyzer_prompts where image_prompt=1 ORDER BY created_at DESC"
        cursor.execute(query)
        results = cursor.fetchall()

        prompts_list = []
        for result in results:
            prompt_dict = {"prompt": result[2], "prompt_title": result[1]}
            prompts_list.append(prompt_dict)

        return prompts_list
    except Error as e:
        logger.error(f"Error retrieving prompts and titles: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def get_prompt_data():
    connection = connect_to_database()
    if not connection:
        return []

    try:
        cursor = connection.cursor(dictionary=True)
        query = "SELECT prompt_title, prompt_content FROM pdf_analyzer_prompts ORDER BY created_at DESC"
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    except Error as e:
        logger.error(f"Error retrieving prompt data: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def bulk_update_prompts(prompt_data):
    connection = connect_to_database()
    if not connection:
        return False

    try:
        cursor = connection.cursor()

        # Get all existing prompt titles
        cursor.execute("SELECT prompt_title FROM pdf_analyzer_prompts")
        existing_titles = set(row[0] for row in cursor.fetchall())

        # Separate new and existing prompts
        new_prompts = [
            (title, content, image_prompt)
            for title, content, image_prompt in prompt_data
            if title not in existing_titles
        ]
        existing_prompts = [
            (title, content, image_prompt)
            for title, content, image_prompt in prompt_data
            if title and content and title in existing_titles
        ]

        logger.info(f"Number of new prompts to insert: {len(new_prompts)}")

        # Insert new prompts
        if new_prompts:
            insert_query = """
                INSERT INTO pdf_analyzer_prompts 
                (prompt_title, prompt_content, image_prompt) 
                VALUES (%s, %s, %s)
            """
            for title, content, image_prompt in new_prompts:
                logger.info(f"Inserting new prompt: {title}")
                cursor.execute(insert_query, (title, content, image_prompt))
                connection.commit()
                logger.info(f"Successfully inserted prompt: {title}")
            logger.info(f"Total new prompts inserted: {len(new_prompts)}")

        # Update existing prompts
        if existing_prompts:
            update_query = """
                UPDATE pdf_analyzer_prompts 
                SET prompt_content = %s, image_prompt = %s 
                WHERE prompt_title = %s
            """
            cursor.executemany(
                update_query, [(content, image_prompt, title) for title, content, image_prompt in existing_prompts]
            )
            logger.info(f"Updated {len(existing_prompts)} existing prompts")

        # Delete prompts not in the new data
        new_titles = set(title for title, _, _ in prompt_data)
        titles_to_delete = existing_titles - new_titles
        if titles_to_delete:
            delete_query = "DELETE FROM pdf_analyzer_prompts WHERE prompt_title IN ({})".format(
                ",".join(["%s"] * len(titles_to_delete))
            )
            cursor.execute(delete_query, tuple(titles_to_delete))
            logger.info(f"Deleted {len(titles_to_delete)} prompts not in the new data")

        connection.commit()
        logger.info(f"Prompts update completed. Total rows affected: {cursor.rowcount}")
        return True
    except Error as e:
        logger.error(f"Error updating prompt data: {e}")
        traceback.print_exc()
        connection.rollback()
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def delete_prompts(prompt_titles):
    connection = connect_to_database()
    if not connection:
        return False

    try:
        cursor = connection.cursor()

        # Delete prompts with the given titles
        delete_query = "DELETE FROM pdf_analyzer_prompts WHERE prompt_title IN (%s)"
        format_strings = ",".join(["%s"] * len(prompt_titles))
        cursor.execute(delete_query % format_strings, tuple(prompt_titles))

        connection.commit()
        logger.info(f"Successfully deleted {cursor.rowcount} prompts with specified titles")
        return True
    except Error as e:
        logger.error(f"Error deleting prompts: {e}")
        connection.rollback()
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def log_analysis_request(
    user_email, client_email, batch_id, pdf_file_names, pdf_file_content, selected_prompts, model_name, use_llama_parse
):
    try:
        connection = connect_to_database()
        cursor = connection.cursor()

        insert_query = """
        INSERT INTO pdf_analyzer_logs 
        (user_email, client_email, batch_id, pdf_name, pdf_content, prompt_title, model_used, use_llama_parse) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        for pdf_name, pdf_content in zip(pdf_file_names, pdf_file_content):
            for prompt_title in selected_prompts:
                data = (
                    user_email,
                    client_email,
                    batch_id,
                    pdf_name,
                    pdf_content,
                    prompt_title,
                    model_name,
                    use_llama_parse,
                )
                cursor.execute(insert_query, data)

        connection.commit()
        logger.info(f"Successfully logged analysis request for batch ID: {batch_id}")
        return

    except Error as e:
        logger.error(f"Error logging analysis request for batch ID {batch_id}: {e}")
        connection.rollback()
        return None

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def main():
    prompts = get_prompts()
    prompt_options = []
    prompt_dict = {}
    if prompts:
        for prompt, prompt_title, prompt_content in prompts:
            logger.info(f"Prompt: {prompt_title}")
            prompt_options.append(prompt_title)
            logger.info(f"Content: {prompt}")
            logger.info(f"Content: {prompt_content[:15]}")
            prompt_dict[prompt] = prompt_content[:15]
    else:
        logger.info("No prompts and titles found or an error occurred.")
    logger.info(f"Prompt options: {prompt_options}")
    logger.info(f"Prompt dictionary: {prompt_dict}")


if __name__ == "__main__":
    main()
