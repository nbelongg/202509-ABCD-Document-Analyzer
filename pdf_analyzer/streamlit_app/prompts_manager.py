# import streamlit_app.db_utils as db_utils
import streamlit_app.db_utils as db_utils
import pandas as pd
import json
from logger import streamlit_logger as logger
import traceback


def prompt_update_ui(st):
    st.header("Update Prompts")

    uploaded_file = st.file_uploader("Upload CSV to update prompts", type="csv")

    if st.button("Update Prompts", key="update_prompts_button"):
        if uploaded_file is None:
            st.error("Please select a CSV file to update prompts.")
        else:
            try:
                logger.info("CSV file uploaded, starting update process.")
                success = update_prompts_from_csv(uploaded_file, st)
                if "prompts" in st.session_state:
                    logger.info("Clearing prompts and prompts_dict from session state.")
                    del st.session_state["prompts"]
                    del st.session_state["prompts_dict"]
                if success:
                    st.success("Prompts updated successfully!")
                    logger.info("Prompts updated successfully.")
                    display_all_prompts(st)
                else:
                    logger.error("Failed to update prompts due to a database issue.")
            except Exception as e:
                logger.error(f"Error processing CSV: {str(e)}")
                st.error(f"Error processing CSV: {str(e)}")


def update_prompts_from_csv(csv_file_path, st):
    try:
        logger.info("Attempting to read the CSV file.")
        try:
            df = pd.read_csv(csv_file_path)
            logger.info("CSV file read successfully.")
        except pd.errors.EmptyDataError:
            logger.error("CSV file is empty.")
            st.error("CSV file is empty.")
            return False
        except pd.errors.ParserError:
            logger.error("Error parsing CSV file.")
            st.error("Error parsing CSV file.")
            return False

        # Check if required columns exist
        logger.info("Checking for required columns in the CSV file.")
        required_columns = ["prompt_label", "prompt", "image_prompt"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            error_message = f"CSV file is missing required columns: {', '.join(missing_columns)}"
            logger.error(error_message)
            st.error(error_message)
            return False

        logger.info("Filtering out rows with null values in required columns.")
        df = df.dropna(subset=["prompt_label", "prompt"])

        logger.info("Checking for image prompt column.")
        df["image_prompt"] = df["image_prompt"].fillna("").astype(str)
        df["image_prompt"] = df["image_prompt"].str.lower().map({"true": 1, "": 0, "false": 0})

        logger.info("Creating a list of tuples from the dataframe.")
        prompt_data = list(df[["prompt_label", "prompt", "image_prompt"]].itertuples(index=False, name=None))

        logger.info("Attempting to update/insert prompts in the database.")
        try:
            success = db_utils.bulk_update_prompts(prompt_data)
            if success:
                logger.info("Prompts updated/inserted successfully in the database.")
            else:
                logger.error("Failed to update/insert prompts in the database.")
        except Exception as e:
            logger.error(f"Database update error: {e}")
            traceback.print_exc()
            st.error("Failed to update prompts due to a database issue.")
            return False

        if "delete" in df.columns:
            logger.info("Processing 'delete' column for prompt deletions.")
            df["delete"] = df["delete"].fillna("").astype(str)
            prompts_to_delete = df[df["delete"].str.lower() == "true"]["prompt_label"].tolist()
            if prompts_to_delete:
                try:
                    delete_success = db_utils.delete_prompts(prompts_to_delete)
                    if delete_success:
                        logger.info("Prompts deleted successfully.")
                    else:
                        logger.warning("Failed to delete some prompts.")
                        st.warning("Some prompts could not be deleted.")
                except Exception as e:
                    logger.error(f"Database delete error: {e}")
                    st.error("Failed to delete prompts due to a database issue.")

        return success

    except Exception as e:
        logger.error(f"Unexpected error processing CSV file: {e}")
        st.error(f"Unexpected error processing CSV file: {e}")
        return False


def display_all_prompts(st):
    st.subheader("All Prompts")
    prompt_data = db_utils.get_prompt_data()

    if prompt_data:
        # Create a dictionary to hold all prompts
        all_prompts = {}

        for i, row in enumerate(prompt_data):
            prompt = {"Prompt": row["prompt_content"]}
            all_prompts[row["prompt_title"]] = prompt

        # Display the JSON string in a code block
        with st.expander("All prompts"):
            st.json(all_prompts)
    else:
        st.warning("No prompts found in the database.")


def export_prompts_to_csv(output_file_path="prompts_export.csv"):
    try:
        # Fetch prompt data from the database
        prompt_data = db_utils.get_prompt_data()

        if not prompt_data:
            logger.info("No prompts found in the database.")
            return False

        # Convert the prompt data to a pandas DataFrame
        df = pd.DataFrame(prompt_data)

        # Rename columns to match the expected format for importing
        df = df.rename(columns={"prompt_title": "prompt_label", "prompt_content": "prompt"})

        # Select only the required columns
        df = df[["prompt_label", "prompt"]]

        # Export the DataFrame to a CSV file
        df.to_csv(output_file_path, index=False)

        logger.info(f"Prompts exported successfully to {output_file_path}")
        return True

    except Exception as e:
        logger.error(f"Error exporting prompts to CSV: {str(e)}")
        return False


def main():
    # Fetch prompt data
    # prompt_data = db_utils.get_prompt_data()
    # #prompt_data=None
    # if prompt_data:
    #     for row in prompt_data:
    #         print(f"Prompt Title: {row['prompt_title']}")
    #         print(f"Prompt Content: {row['prompt_content'][:50]}...")  # Print first 50 characters
    # else:
    #     print("No prompt data found or an error occurred.")
    # Example of updating/inserting prompt data
    # csv_file_path = 'prompts_export.csv'
    csv_file_path = "pdf_analyser_prompts.csv"
    success = update_prompts_from_csv(csv_file_path)
    if success:
        print("Bulk update/insert from CSV completed successfully")
    else:
        print("Bulk update/insert from CSV failed")


if __name__ == "__main__":
    main()
