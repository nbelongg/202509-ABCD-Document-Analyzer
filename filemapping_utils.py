import csv

csv_file_path = 'ML_Model_Masterlist.csv'  


def get_pdf_mappings():
    pdf_mappings = {}    
    key_mapping = {
        'S No.': 'sno',
        'Title': 'title',
        'Author/Organization': 'author_organization',
        'Publication Year': 'publication_year',
        'Link': 'link',
        'PDF Title': 'pdf_title'
    }
    
    with open(csv_file_path, 'r', newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)

        for row in reader:
            pdf_details = {key_mapping[key]: value for key, value in row.items() if key in key_mapping}
            pdf_title = pdf_details['pdf_title']

            if not pdf_title.endswith('.pdf'):
                pdf_title += '.pdf'

        
            pdf_mappings[pdf_title] = pdf_details
            
    return pdf_mappings


            
