import os
import sys
import json
import logging
import openai
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import time

sys.path.append("/home/carolina/01-Luxembourg/Semester_6/BSPS6/AutoDefenceCode")
from defense.explicit_detector.agency.explicit_1_agent import VanillaJailbreakDetector

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  

# Hardcoded config_list for OpenAI setup
config_list = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 1000
}

# Initialize the VanillaJailbreakDetector with the hardcoded config_list
detector = VanillaJailbreakDetector(config_list=config_list)
detector.strategy_name = "explicit_1_agent"  # Use a valid strategy name from defense_prompts.json

def process_file(input_file, output_dir):
    """
    Processes a single harmful output JSON file using the VanillaJailbreakDetector and saves the result.
    """
    logging.info(f"Processing file: {input_file}")

    # Load harmful output
    with open(input_file, 'r') as f:
        harmful_output = json.load(f)

    # Run the defense strategy
    results = []
    for item in harmful_output:
        harmful_content = item["raw_response"]  # Extract the harmful response
        #logging.info(f"Processing response: {harmful_content}")
        try:
            valid, success, response = detector(harmful_content)
            logging.info(f"LLM Evaluation Result: valid={valid}, success={success}, response={response}")
        except Exception as e:
            logging.error(f"Error processing response: {harmful_content}, Error: {e}")
            valid, success, response = False, False, "ERROR"

        results.append({
            "name": item["name"],  # Keep the original name for reference
            "raw_response": harmful_content,
            "valid": valid,
            "success": success,
            "defense_response": response
        })
    # Save the result
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(input_file))
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    #logging.info(f"Processed {input_file} -> {output_file}")

class OpenAIWrapper:
    def __init__(self, config_list):
        self.config_list = config_list

    def create(self, messages):
        # Retry logic for handling connection errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(model=self.config_list["model"],
                messages=messages,
                temperature=self.config_list["temperature"],
                max_tokens=self.config_list["max_tokens"])
                return response
            except openai.OpenAIError as e:
                logging.error(f"OpenAI API error: {e}. Attempt {attempt + 1} of {max_retries}.")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
            except Exception as e:
                logging.error(f"Unexpected error: {e}. Attempt {attempt + 1} of {max_retries}.")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise


if __name__ == "__main__":
    # Input and output paths
    input_file = "data/harmful_output/gpt-3.5-turbo/attack-OG_0.json"  # Replace with the actual file path
    output_dir = "defense_output/gpt3.5-turbo/1Agent"

    # Process the file
    process_file(input_file, output_dir)
