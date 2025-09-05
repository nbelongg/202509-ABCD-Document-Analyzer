import pinecone
import pandas as pd
import re

# Configuration settings
pinecone_api_key = '947ea359-754d-4075-9cbf-cf37e7a9521e'
pinecone_env = 'us-east-1-aws'
index_name = 'open-source-meaningful'
csv_filename = 'ABCD_Analyzer_Corpus_27_11.csv'

def replace_values(val):
    if "1" in str(val):
        return True
    elif pd.isna(val):
        return False
    else:
        return val

def is_filename_part(df, filename):
    
    filename = re.escape(filename)
    matching_rows = df[(df['Title'].str.contains(filename, regex=True)) | (df['PDF Title'].str.contains(filename, regex=True))]
    
    
    print(f"Matching rows: {matching_rows}")
    return matching_rows.head(1)

def main():
    
    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    index = pinecone.Index(index_name)

    # Get total vector count
    totvect = index.describe_index_stats()['total_vector_count']
    
    df = pd.read_csv(csv_filename)
    df['Title'] = df['Title'].str.replace('.pdf', '', regex=False)
    df['PDF Title'] = df['PDF Title'].str.replace('.pdf', '', regex=False)

    # Rename columns
    columns_to_rename = {
        "MBS (Marketing and Behavior Science 101)": "MBS",
        "GPP (Governance & Public Policy 101)": "GPP",
        "SDSC (Systems Delivery and Supply Chains)": "SDSC",
        "LC (Local Context of Behavioural Determinants â€” Culture, Norms, Attitudes, Perceptions)": "LC",
        "IID (Intersectional Identities)": "IID",
        "CSS (Case studies and solutions)": "CSS",
        "DS (Datasets 101)": "DS"
    }

    for col in df.columns:
        for key, value in columns_to_rename.items():
            df.columns = [col.split(' ')[0] if col not in ['S No.', 'PDF Title'] else col for col in df.columns]

    for i in range(50000, 70000):
        a = index.fetch([str(i)])
        if a['vectors']:
            
            filename = a['vectors'][str(i)]['metadata']['filename'].replace('.pdf', '')
            matching_rows = is_filename_part(df, filename)
            print(f"Matching rows: {matching_rows}")
            if not matching_rows.empty:
                selected_columns = ['MBS', 'GPP', 'SDSC', 'LC', 'IID', 'CSS', 'DS']
                extracted_data = matching_rows[selected_columns]
                extracted_data = extracted_data.copy()
                for col in selected_columns:
                    extracted_data[col] = extracted_data[col].apply(replace_values)
                    
                
                new_dict = extracted_data.to_dict(orient='list')

                new_dict = {key: value[0] for key, value in new_dict.items()}
                
                print(new_dict)
                
                old_dict = a['vectors'][str(i)]['metadata']
                
                
                old_dict.update(new_dict)
                
                index.update(id=str(i), set_metadata=old_dict)
                
                print(f"""Updated metadata for id,{str(i)}""")
    
    # for i in range(0, 10):
    #     a = index.fetch([str(i)])
    #     if a['vectors']:
    #         print(a['vectors'][str(i)]['metadata'])

if __name__ == '__main__':
    main()