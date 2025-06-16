import os
from os.path import join, exists
from joblib import Parallel, delayed
import pandas as pd
from glob import glob
from tqdm import tqdm
import sys
sys.path.append("/home/carolina/01-Luxembourg/Semester_6/BSPS6/AutoDefenceCode")
from defense.explicit_detector.agency.explicit_1_agent import (VanillaJailbreakDetector, CoT, CoTV2, CoTV3,
                                                               VanillaJailbreakDetectorV0125)
from defense.explicit_detector.agency.explicit_2_agents import AutoGenDetectorV1, AutoGenDetectorV0125
from defense.explicit_detector.agency.explicit_3_agents import AutoGenDetectorThreeAgency, AutoGenDetectorThreeAgencyV2
#from defense.explicit_detector.agency.explicit_5_agents import DetectorFiveAgency
from defense.explicit_detector.explicit_defense_arch import ExplicitMultiAgentDefense
# from defense.implicit_detector.agency.implicit_1_agent import MoralAdvisor
# from defense.implicit_detector.agency.implicit_2_agents import MoralAdvisor2Agent
# from defense.implicit_detector.agency.implicit_3_agents import MoralAdvisor3Agent
# from defense.implicit_detector.implicit_defense_arch import ImplicitMultiAgentDefense
from evaluator.evaluate_helper import evaluate_defense_with_output_list, evaluate_defense_with_response
from defense.utility import load_harmful_prompt, load_defense_prompt
import argparse
import openai
import json

defense_strategies = [
    # {"name": "im-1", "defense_agency": ImplicitMultiAgentDefense, "task_agency": MoralAdvisor},
    # {"name": "im-2", "defense_agency": ImplicitMultiAgentDefense, "task_agency": MoralAdvisor2Agent},
    # {"name": "im-3", "defense_agency": ImplicitMultiAgentDefense, "task_agency": MoralAdvisor3Agent},
    # {"name": "ex-1", "defense_agency": ExplicitMultiAgentDefense, "task_agency": VanillaJailbreakDetector},
    {"name": "ex-2", "defense_agency": ExplicitMultiAgentDefense, "task_agency": AutoGenDetectorV1},
    {"name": "ex-3", "defense_agency": ExplicitMultiAgentDefense, "task_agency": AutoGenDetectorThreeAgency},
    {"name": "ex-cot", "defense_agency": ExplicitMultiAgentDefense, "task_agency": CoT},
    # {"name": "ex-1-0125", "defense_agency": ExplicitMultiAgentDefense, "task_agency": VanillaJailbreakDetectorV0125},
    # {"name": "ex-2-0125", "defense_agency": ExplicitMultiAgentDefense, "task_agency": AutoGenDetectorV0125},
    # {"name": "ex-cot-5", "defense_agency": ExplicitMultiAgentDefense, "task_agency": CoTV3},
    # {"name": "ex-5", "defense_agency": ExplicitMultiAgentDefense, "task_agency": DetectorFiveAgency},
    # {"name": "ex-3-v2", "defense_agency": ExplicitMultiAgentDefense, "task_agency": AutoGenDetectorThreeAgencyV2},
    # {"name": "ex-cot-v2", "defense_agency": ExplicitMultiAgentDefense, "task_agency": CoTV2},
]


def eval_csv_from_yuan():
    attack_csv_list = glob("data/harmful_output/multiple_attack_output/*.csv")
    attack_csv_list.sort()
    for attack_csv in tqdm(attack_csv_list):
        df = pd.read_csv(attack_csv)
        for defense_strategy in defense_strategies:
            df[defense_strategy["name"]] = evaluate_defense_with_output_list(
                task_agency=defense_strategy["task_agency"],
                defense_agency=defense_strategy["defense_agency"],
                output_list=df["target"].tolist())

        df.to_csv(attack_csv, index=False)


def eval_defense_strategies(llm_name, output_suffix, ignore_existing, chat_file,
                            presence_penalty=0.0, frequency_penalty=0.0, temperature=0.7):
    """
    Evaluate defense strategies for a given LLM and configuration.
    """
    defense_output_prefix = join(f"data/defense_output/open-llm-defense{output_suffix}", llm_name)
    os.makedirs(defense_output_prefix, exist_ok=True)
    for defense_strategy in defense_strategies:
        output_file = join(defense_output_prefix, defense_strategy["name"] + ".json")
        if exists(output_file) and ignore_existing:
            print(f"Defense output exists, skip {output_file}")
            continue
        print(f"Evaluating {llm_name} with strategy {defense_strategy['name']}")
        print(f"Chat file: {chat_file}")
        print(f"Output file: {output_file}")
        try:
            result = evaluate_defense_with_response(
                task_agency=defense_strategy["task_agency"],
                defense_agency=defense_strategy["defense_agency"],
                chat_file=chat_file,
                defense_output_name=output_file,
                model_name=llm_name,
                parallel=True,
                frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
                temperature=temperature)
            print(f"Result for {defense_strategy['name']}: {result}")
        except Exception as e:
            print(f"Error during evaluation of {defense_strategy['name']}: {e}")


def eval_with_open_llms(model_list, chat_file, ignore_existing=True,
                        host_name="127.0.0.1", output_suffix="", frequency_penalty=1.3,
                        temperature=0.7, eval_safe=True, eval_harm=True, presence_penalty=0.0):
    # "llama-2-13b", "llama-2-7b", "llama-pro-8b", "llama-2-70b", "tinyllama-1.1b", "vicuna-13b-v1.5", "vicuna-33b", "vicuna-7b-v1.5", "vicuna-13b-v1.3.0"
    for llm_name in model_list:
        print("Evaluating", llm_name)
        if eval_harm:
            eval_defense_strategies(llm_name, output_suffix, ignore_existing=ignore_existing,
                                    chat_file=chat_file,
                                    presence_penalty=presence_penalty,
                                    frequency_penalty=frequency_penalty, temperature=temperature)
        if eval_safe:
            eval_defense_strategies(llm_name, "-safe" + output_suffix, ignore_existing=ignore_existing,
                                    chat_file=chat_file.replace("attack", "safe"),
                                    presence_penalty=presence_penalty,
                                    frequency_penalty=frequency_penalty, temperature=temperature)


def eval_with_openai(model_list, chat_file, ignore_existing=True, output_suffix="",
                     temperature=0.7, eval_safe=True, eval_harm=True, presence_penalty=0.0):
    for llm_name in model_list:
        print("Evaluating", llm_name)
        if eval_harm:
            eval_defense_strategies(llm_name, output_suffix, ignore_existing=ignore_existing,
                                    chat_file=chat_file, presence_penalty=presence_penalty,
                                    temperature=temperature)
        if eval_safe:
            eval_defense_strategies(llm_name, "-safe" + output_suffix, ignore_existing=ignore_existing,
                                    chat_file=chat_file.replace("attack_gpt3.5", "safe_gpt3.5"),
                                    presence_penalty=presence_penalty,
                                    temperature=temperature)


def eval_harmful_prompts_with_multiagent(llm_name, harmful_prompt_pattern, output_suffix, ignore_existing=False,
                                         presence_penalty=0.0, frequency_penalty=0.0, temperature=0.7):
    """
    Evaluate harmful prompts using the multi-agent system.
    """
    harmful_files = glob(harmful_prompt_pattern)
    if not harmful_files:
        raise FileNotFoundError(f"No files found matching the pattern: {harmful_prompt_pattern}")

    defense_output_prefix = join(f"data/defense_output/open-llm-defense{output_suffix}", llm_name)
    os.makedirs(defense_output_prefix, exist_ok=True)

    for harmful_file in harmful_files:
        print(f"Processing harmful prompts from file: {harmful_file}")
        harmful_prompts = load_harmful_prompt(json_path=harmful_file)

        for defense_strategy in defense_strategies:
            output_file = join(defense_output_prefix, f"{defense_strategy['name']}_{os.path.basename(harmful_file)}")
            if exists(output_file) and ignore_existing:
                print(f"Defense output exists, skip {output_file}")
                continue

            print(f"Evaluating {llm_name} with strategy {defense_strategy['name']}")
            try:
                results = []
                for prompt_name, prompt_content in tqdm(harmful_prompts.items()):
                    result = evaluate_defense_with_response(
                        task_agency=defense_strategy["task_agency"],
                        defense_agency=defense_strategy["defense_agency"],
                        chat_file=None,  # Not using a chat file, passing prompts directly
                        defense_output_name=None,  # Results will be collected in memory
                        model_name=llm_name,
                        parallel=False,
                        temperature=temperature,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty
                    )
                    results.append({"name": prompt_name, "prompt": prompt_content, "response": result})

                # Save results to output file
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
                print(f"Results saved to {output_file}")
            except Exception as e:
                print(f"Error during evaluation of {defense_strategy['name']} for file {harmful_file}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat_file_pattern", type=str, default="data/harmful_output/gpt-3.5-turbo/attack-*.json",
                        help="Glob pattern to match harmful prompt files.")
    parser.add_argument("--output_suffix", type=str, default="")
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--ignore_existing", action="store_true")
    args = parser.parse_args()

    llm_name = "gpt-3.5-turbo"  # Fixed model name
    eval_harmful_prompts_with_multiagent(
        llm_name=llm_name,
        harmful_prompt_pattern=args.chat_file_pattern,
        output_suffix=args.output_suffix,
        ignore_existing=args.ignore_existing,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        temperature=args.temperature
    )
