import os

import mysql.connector
from mysql.connector import Error

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(override=True)


def connect_to_database():
    try:
        connection = mysql.connector.connect(
            host=os.getenv('mysql_host'),
            database=os.getenv('mysql_database'),
            user=os.getenv('mysql_user'),
            password=os.getenv('mysql_password')
        )
        if connection.is_connected():
            print("Successfully connected to the database")
            return connection
    except Error as e:
        print(f"Error connecting to the database: {e}")
        return None


def log_analysis_request(log_data):
    try:
        connection = connect_to_database()
        cursor = connection.cursor()
        
        update_query = """
        UPDATE pdf_analyzer_logs 
        SET input_tokens = %s, output_tokens = %s, total_tokens = %s
        WHERE batch_id = %s AND pdf_name = %s AND prompt_title = %s
        """
        
        data = (
            log_data['input_tokens'],
            log_data['output_tokens'],
            log_data['total_tokens'],
            log_data['batch_id'],
            log_data['filename'],
            log_data['prefix']
        )
        
        cursor.execute(update_query, data)
        connection.commit()
        print(f"Successfully updated log for batch ID: {log_data['batch_id']}, filename: {log_data['filename']}, prefix: {log_data['prefix']}")
        return

    except Error as e:
        print(f"Error updating log: {e}")
        connection.rollback()
        return None

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            


