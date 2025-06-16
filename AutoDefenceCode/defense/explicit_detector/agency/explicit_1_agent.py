from openai import OpenAI

client = OpenAI()
from autogen import OpenAIWrapper
from defense.explicit_detector.explicit_defense_arch import ExplicitMultiAgentDefense
from defense.utility import load_defense_prompt
from evaluator.evaluate_helper import evaluate_defense_with_response
import json

def load_defense_prompt():
    prompt_file = "/home/carolina/01-Luxembourg/Semester_6/BSPS6/AutoDefenceCode/data/prompt/defense_prompts.json"
    with open(prompt_file, "r") as f:
        return json.load(f)

class OpenAIWrapper:
    def __init__(self, config_list):
        self.api_key = config_list["api_key"]
        self.model = config_list["model"]
        self.temperature = config_list["temperature"]
        self.max_tokens = config_list["max_tokens"]

    def create(self, messages):
        # Use the new OpenAI API method
        response = client.chat.completions.create(model=self.model,
        messages=messages,
        temperature=self.temperature,
        max_tokens=self.max_tokens)
        return response

class VanillaJailbreakDetector:
    def __init__(self, config_list):
        self.config_list = config_list
        self.defense_prompt = load_defense_prompt()
        self.strategy_name = "explicit_1_agent"
        self.llm = OpenAIWrapper(config_list=config_list)  # Initialize the LLM wrapper with the hardcoded config_list

    def __call__(self, prompt):
        # Ensure the strategy exists
        if self.strategy_name not in self.defense_prompt:
            raise ValueError(f"Strategy '{self.strategy_name}' not found in defense_prompt.")

        # Construct the LLM input
        strategy_data = self.defense_prompt[self.strategy_name]
        messages = [
            {"role": "system", "content": strategy_data["1_system"]},
            {"role": "user", "content": strategy_data["1_user"].replace("[INSERT INPUT HERE]", prompt)}
        ]

        # Send the input to the LLM
        response = self.llm.create(messages=messages).choices[0].message.content

        # Analyze the response
        valid = "VALID" in response.upper()
        success = "INVALID" not in response.upper()
        return valid, success, response

    def agency_execute(self, llm_output):
        valid, success, llm_output = self(llm_output)
        return llm_output


class CoT(VanillaJailbreakDetector):
    def __init__(self, config_list):
        self.defense_prompt = load_defense_prompt()
        self.strategy_name = "explicit_1_agent_CoT"
        self.llm = OpenAIWrapper(config_list=config_list, cache_seed=123)
        self.messages = [
            {
                "role": "system",
                "content": self.defense_prompt[self.strategy_name]["1_system"],
            }
        ]


class CoTV2(VanillaJailbreakDetector):
    def __init__(self, config_list):
        self.defense_prompt = load_defense_prompt()
        self.strategy_name = "explicit_1_agent_CoT_v2"
        self.llm = OpenAIWrapper(config_list=config_list, cache_seed=123)
        self.messages = [
            {
                "role": "system",
                "content": self.defense_prompt[self.strategy_name]["1_system"],
            }
        ]

class CoTV3(VanillaJailbreakDetector):
    def __init__(self, config_list):
        self.defense_prompt = load_defense_prompt()
        self.strategy_name = "explicit_1_agent_CoT_v3"
        self.llm = OpenAIWrapper(config_list=config_list, cache_seed=123)
        self.messages = [
            {
                "role": "system",
                "content": self.defense_prompt[self.strategy_name]["1_system"],
            }
        ]

class VanillaJailbreakDetectorV0125(VanillaJailbreakDetector):
    def __init__(self, config_list):
        self.defense_prompt = load_defense_prompt()
        self.strategy_name = "explicit_1_agent_v0125"
        self.llm = OpenAIWrapper(config_list=config_list, cache_seed=123)
        self.messages = [
            {
                "role": "system",
                "content": self.defense_prompt[self.strategy_name]["1_system"],
            }
        ]

if __name__ == '__main__':
    # args = argparse.ArgumentParser()
    # args.add_argument("--log_file", type=str, default="data/defense_output/detection_summary_vanilla.json")
    # args = args.parse_args()
    #
    # evaluate_explicit_detector(VanillaJailbreakDetector(), log_file=args.log_file)

    evaluate_defense_with_response(task_agency=CoTV3,
                                   defense_agency=ExplicitMultiAgentDefense,
                                   defense_output_name="/tmp/tmp.json",
                                   model_name="gpt-3.5-turbo-1106")
