import os
import sys
import json
import logging
import openai
import time
from openai import OpenAI

# Add the project directory to the Python path
sys.path.append("/home/carolina/01-Luxembourg/Semester_6/BSPS6/AutoDefenceCode")

# Import necessary classes
from defense.explicit_detector.agency.explicit_3_agents import AutoGenDetectorThreeAgency
from defense.explicit_detector.explicit_defense_arch import DefenseGroupChat  # Import DefenseGroupChat

# Disable Docker usage by setting the environment variable
os.environ["AUTOGEN_USE_DOCKER"] = "False"

# Hardcoded config_list for OpenAI setup
config_list = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "model": "gpt-3.5-turbo",
    "max_tokens": 1000  # Removed invalid 'temperature' key
}

# Initialize the AutoGenDetectorV1 with the hardcoded config_list
detector = AutoGenDetectorThreeAgency(config_list=config_list)
detector.strategy_name = "explicit_3_agent"  # Use a valid strategy name from defense_prompts.json

# Ensure the DefenseGroupChat fallback logic is used
detector.groupchat_class = DefenseGroupChat

def process_file(input_file, output_dir, limit=1):
    """
    Processes a limited number of harmful output JSON items using the AutoGenDetectorV1 and saves the result.
    """
    logging.info(f"Processing file: {input_file}")
    logging.debug(f"Limit parameter passed to process_file: {limit}")

    # Load harmful output
    with open(input_file, 'r') as f:
        harmful_output = json.load(f)

    # Limit the number of items to process
    harmful_output = harmful_output[:limit]
    logging.debug(f"Number of items to process after applying limit: {len(harmful_output)}")

    # Run the defense strategy
    results = []
    for item in harmful_output:
        harmful_content = item["raw_response"]  # Extract the harmful response
        try:
            logging.debug(f"Processing harmful content: {harmful_content}")
            valid, success, response = detector(harmful_content)

            # Debugging: Log the response from the detector
            logging.debug(f"Detector response: valid={valid}, success={success}, response={response}")

            # Ensure response is not empty or malformed
            if not response or not isinstance(response, str):
                raise ValueError(f"Malformed response: {response}")

            logging.info(f"LLM Evaluation Result: valid={valid}, success={success}, response={response}")
        except Exception as e:
            # Log the exact error details for debugging
            logging.error(f"Error processing response: {harmful_content}, Error: {e}", exc_info=True)
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

    logging.info(f"Processed {input_file} -> {output_file}")


class OpenAIWrapper:
    def __init__(self, config_list):
        self.config_list = config_list

    def create(self, messages):
        # Retry logic for handling connection errors
        max_retries = 1
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.config_list["model"],
                    messages=messages,
                    temperature=self.config_list["temperature"],
                    max_tokens=self.config_list["max_tokens"]
                )
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
    input_file = "data/harmful_output/gpt-3.5-turbo/attack-BetterDan_0.json"  # Replace with the actual file path
    output_dir = "defense_output/gpt3.5-turbo/3Agent"

    # Enable debug logging
    logging.basicConfig(level=logging.DEBUG)

    # Suppress debug logs from the OpenAI library
    logging.getLogger("openai").setLevel(logging.WARNING)

    # Process only the first 1 item
    process_file(input_file, output_dir, limit=390)
