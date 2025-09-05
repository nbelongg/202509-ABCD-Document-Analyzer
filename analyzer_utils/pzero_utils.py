import json
import db_utils
import gpt_utils
import re
import common_utils

def get_pzero_summary(generated_analyze_comments, doc_type, model, user_role):
    combined_comments = combine_comment(generated_analyze_comments)
    summary_prompt = db_utils.get_proposal_summary_prompt(doc_type)
    if(common_utils.has_placeholder(summary_prompt, "user_role")):
        summary_prompt = common_utils.replace_placeholder(summary_prompt,"user_role", user_role)
        
    pzero_summary = gpt_utils.get_P0_summary(combined_comments, model, doc_type)
    return pzero_summary, summary_prompt


def combine_comment(generated_analyze_comments):
    data_dict = generated_analyze_comments
    combined_comments = ""
    for key, value in data_dict.items():
        if "analyze_comments" in value:
            for item in value["analyze_comments"]:
                combined_comments += item["comment"] + "\n"
    
    return combined_comments


def extract_sources_from_analyze_comments(generated_analyze_comments):
    all_sources = []
    for key, value in generated_analyze_comments.items():
        analyze_comments = value.get('analyze_comments', [])
        for comment_data in analyze_comments:
            sources = comment_data.get('sources', [])
            for source in sources:
                all_sources.append(source)

    unique_entries = []
    seen_combinations = set()

    for entry in all_sources:
        name = entry["name"]
        url = entry["url"]
        
        # Check if the combination of name and url has been seen before
        if (name, url) not in seen_combinations:
            # If not seen, add the entry to the unique_entries list
            unique_entries.append(entry)
            # Add the combination to the set of seen combinations
            seen_combinations.add((name, url))

    return unique_entries
