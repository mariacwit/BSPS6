import json

def clean_result_explanation_from_json(file_path):
    # Load data from JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Clean the "result" field
    for entry in data:
        if "result" in entry:
            entry["result"] = entry["result"].split("\n")[0].strip()
    
    # Save cleaned data back to JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


file_path = "data/harmful_output/gpt-3.5-turbo/AttackEval/evaluation_results_OG.json"
clean_result_explanation_from_json(file_path)