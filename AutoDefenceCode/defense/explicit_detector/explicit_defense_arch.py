import re
import logging
from typing import *

import autogen
import openai
from autogen import Agent, ConversableAgent, UserProxyAgent, AssistantAgent, OpenAIWrapper

from defense.utility import load_defense_prompt

defense_prompts = load_defense_prompt()
prompt = defense_prompts["explicit_2_agent"]

class DefenseGroupChat(autogen.GroupChat):
    def __init__(self, agents, messages=None, max_round=4, speaker_selection_method="round_robin"):
        # Ensure the fallback agent is added to the group chat
        fallback_agent = next((agent for agent in agents if agent.name == "Task_Agency_Agent"), None)
        if fallback_agent is None:
            fallback_agent = ConversableAgent(name="Task_Agency_Agent")
            agents.append(fallback_agent)

        super().__init__(agents=agents, messages=messages, max_round=max_round, speaker_selection_method=speaker_selection_method)

    def select_speaker(self, last_speaker: Agent, selector: ConversableAgent):
        if last_speaker.name != "Coordinator":
            return self.agent_by_name("Coordinator")
        else:
            # Extract mentions from the Coordinator's last message
            coordinator_message = self.agent_by_name("Coordinator").last_message()['content']
            mentions = self._mentioned_agents(coordinator_message, self.agents)

            # Debugging: Log the Coordinator's message and mentions
            logging.debug(f"Coordinator's message: {coordinator_message}")
            logging.debug(f"Mentions extracted: {mentions}")

            # Handle cases where no agent or multiple agents are mentioned
            if len(mentions) != 1:
                logging.warning(f"Coordinator did not mention exactly one agent. Defaulting to the next agent in round-robin.")
                next_agent = self._get_next_agent_round_robin(last_speaker)
                logging.debug(f"Selected fallback agent: {next_agent.name}")
                return next_agent

            # Return the mentioned agent
            name = next(iter(mentions))
            agent = self.agent_by_name(name)
            if agent is None:
                raise ValueError(f"Mentioned agent '{name}' does not exist in the group chat.")
            return agent

    def _get_next_agent_round_robin(self, last_speaker: Agent) -> Agent:
        """
        Implements round-robin logic to select the next agent.
        """
        agent_names = [agent.name for agent in self.agents if agent.name != "Coordinator"]
        if last_speaker.name not in agent_names:
            return self.agents[0]  # Default to the first agent if last_speaker is not in the list

        current_index = agent_names.index(last_speaker.name)
        next_index = (current_index + 1) % len(agent_names)
        return self.agent_by_name(agent_names[next_index])

    def _mentioned_agents(self, message: str, agents: List[Agent]) -> Dict[str, int]:
        """
        Extracts the agent explicitly mentioned after the last occurrence of 'Next:' in the message.
        """
        mentions = {}
        # Log the Coordinator's response for debugging
        logging.debug(f"Coordinator's full response: {message}")

        # Use regex to find all occurrences of 'Next:' and extract the last one
        matches = re.findall(r"Next:\s*([a-zA-Z_]+)\b", message)
        if matches:
            mentioned_agent = matches[-1]  # Get the last match
            logging.debug(f"Extracted agent after the last 'Next:': {mentioned_agent}")
            # Check if the mentioned agent exists in the list of agents
            if any(agent.name == mentioned_agent for agent in agents):
                mentions[mentioned_agent] = 1
            else:
                logging.warning(f"Extracted agent '{mentioned_agent}' is not in the list of valid agents.")
        else:
            logging.warning("No agent explicitly mentioned after 'Next:' in the message.")

        logging.debug(f"Explicit mentions extracted: {mentions}")
        return mentions


class DefenseAssistantAgent(AssistantAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_reply([Agent, None], DefenseAssistantAgent.generate_oai_reply)

    def generate_oai_reply(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[OpenAIWrapper] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        final, response = super().generate_oai_reply(messages, sender, config)
        if response is None:
            return final, "Due to Azure OpenAI's content management policy, the response is too sensitive to be shown."
        return final, response


class TaskAgencyAgent(ConversableAgent):
    def __init__(self, agency, **kwargs):
        super().__init__(**kwargs)
        self.register_reply([Agent, None], TaskAgencyAgent.execute_agency)
        self.agency = agency

    def execute_agency(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, None]]:
        client = self.client if config is None else config
        if client is None:
            raise ValueError("client is None")
        if messages is None:
            raise ValueError("messages is None")

        response = self.agency.agency_execute(messages[-1]["content"])

        return True, response


class OutputAgent(ConversableAgent):
    def __init__(self, strict_sorry_for_invalid, **kwargs):
        super().__init__(**kwargs)
        self.strict_sorry_for_invalid = strict_sorry_for_invalid
        self.register_reply([Agent, None], OutputAgent.refine_invalid_output)
        self.register_reply([Agent, None], OutputAgent.exam_valid_output)

    def exam_valid_output(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, None]]:
        if messages is None:
            raise ValueError("messages is None")

        if "I apologize" in messages[-1]["content"]:
            return True, "I'm sorry, but I'm not able to provide the information you're looking for."

        valid_in = "VALID" in messages[-1]["content"].upper()
        invalid_in = "INVALID" in messages[-1]["content"].upper()

        if valid_in and not invalid_in:
            if len([m for m in messages if m["name"] == 'LLM_Victim_Agent']) != 0:
                return True, [m for m in messages if m["name"] == 'LLM_Victim_Agent'][0]["content"]
            else:
                first_input = [m for m in messages if m["name"] == 'User_Prompt_Agent'][0]["content"]
                first_input = re.search(r'--SYSTEM INPUT START--\n((.|\n)*)\n--SYSTEM INPUT END--', first_input).group(
                    1)
                return True, first_input
        elif invalid_in:
            if self.strict_sorry_for_invalid:
                return True, "I'm sorry, but I'm not able to provide the information you're looking for."
            else:
                return False, "The output needs refinement."
        else:
            if "Azure OpenAI's content management policy" in messages[-1]["content"]:
                return True, "I'm sorry, but I'm not able to provide the information you're looking for."
            else:
                return True, f"DEFENSE ERROR: Judgement not exist. Output: {messages[-1]['content']}"

    def refine_invalid_output(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, None]]:
        client = self.client if config is None else config
        if client is None:
            raise ValueError("client is None")
        if messages is None:
            raise ValueError("messages is None")

        final, response = self.generate_oai_reply(messages, sender, config)
        return final, response


class LLMVitimAgent(ConversableAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_reply([Agent, None], LLMVitimAgent.generate_oai_azure_reply)

    def generate_oai_azure_reply(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, None]]:

        try:
            response = super().generate_oai_reply(messages, sender, config)
        except openai.BadRequestError as e:
            if e.status_code == 400:
                return True, "I'm sorry, but I'm not able to provide the information you're looking for. (From Azure Content Management)"
        return response


class CoordinatorAgent(DefenseAssistantAgent):
    def generate_coordinate_reply(
            self,
            messages: Optional[List[Dict]] = None,
            sender: Optional[Agent] = None,
            config: Optional[OpenAIWrapper] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        try:
            # Attempt to extract the system input using the expected markers
            content = self._oai_messages[sender][0]['content']
            logging.debug(f"Coordinator received content: {content}")

            match = re.findall(r"--SYSTEM INPUT START--\n((.|\n)*)--SYSTEM INPUT END--", content)
            if not match:
                logging.error("SYSTEM INPUT markers not found in the message content.")
                return False, "ERROR: SYSTEM INPUT markers not found."

            system_input = match[0][0]  # Extract the first match
            logging.debug(f"Extracted system input: {system_input}")

            last_agent_name = self._oai_messages[sender][-1]['name']
            logging.debug(f"Last agent name: {last_agent_name}")

            # Directly call the Judge after the IntentionAnalyzer
            if last_agent_name == "IntentionAnalyzer":
                final = True
                response = f"{self.defense_prompt['explicit_2_agent']['2_user'].replace('[INSERT INPUT HERE]', system_input)} Next: Judge"
            elif last_agent_name == "Judge":
                final = True
                response = "Next: TaskInputAgent"
            else:
                raise ValueError(f"Unknown agent name: {last_agent_name}")

            logging.debug(f"Coordinator response: {response}")
            return final, response

        except Exception as e:
            logging.error(f"Error in CoordinatorAgent: {e}")
            return False, f"ERROR: {e}"


class ExplicitMultiAgentDefense:
    def __init__(self, task_agency, config_list, cache_seed=123):
        self.config_list = config_list
        self.cache_seed = cache_seed
        self.defense_prompt = load_defense_prompt()
        logging.debug(f"Loaded defense prompts: {self.defense_prompt}")  # Add this for debugging
        self.user_input_agent = UserProxyAgent(name="User_Prompt_Agent", is_termination_msg=lambda x: True,
                                               human_input_mode="NEVER",
                                               llm_config={"config_list": config_list, "cache_seed": cache_seed})
        self.llm_agent = LLMVitimAgent(name="LLM_Victim_Agent",
                                       system_message="You are a helpful AI assistant.",
                                       llm_config={"config_list": config_list, "cache_seed": cache_seed})

        self.taskagency_agent = TaskAgencyAgent(task_agency, name="Task_Agency_Agent",
                                                llm_config={"config_list": config_list, "cache_seed": cache_seed})
        self.output_agent = OutputAgent(True, name="Output_Agent",
                                        system_message="Your task is to refuse to answer user's question. Do not mention the invalid content in your response.",
                                        llm_config={"config_list": config_list, "cache_seed": cache_seed})

    def clear_history(self):
        self.user_input_agent.clear_history()
        self.llm_agent.clear_history()
        self.taskagency_agent.clear_history()
        self.output_agent.clear_history()

    def defense_with_response(self, response):
        self.clear_history()
        groupchat = autogen.GroupChat(
            agents=[self.user_input_agent, self.taskagency_agent, self.output_agent], messages=[],
            max_round=4, speaker_selection_method="round_robin")
        manager = autogen.GroupChatManager(groupchat=groupchat,
                                           llm_config={"config_list": self.config_list, "cache_seed": self.cache_seed})
        prompt = self.defense_prompt["explicit_2_agent"]["1_user"].replace("[INSERT INPUT HERE]", response)
        logging.debug(f"Using prompt: {prompt}")  # Add this for debugging
        self.user_input_agent.initiate_chat(manager, message=prompt, silent=True)
        result = self.output_agent.last_message()

        find_reg_output = re.search(r'--SYSTEM OUTPUT START--\n((.|\n)*)\n--SYSTEM OUTPUT END--', result["content"])
        if find_reg_output is not None:
            result["content"] = find_reg_output.group(1)

        return result

    def defense_with_prompt(self, prompt):
        self.clear_history()
        groupchat = autogen.GroupChat(
            agents=[self.user_input_agent, self.llm_agent, self.taskagency_agent, self.output_agent], messages=[],
            max_round=4, speaker_selection_method="round_robin")
        manager = autogen.GroupChatManager(groupchat=groupchat,
                                           llm_config={"config_list": self.config_list, "cache_seed": self.cache_seed})
        self.user_input_agent.initiate_chat(manager, message=prompt)
        result = self.output_agent.last_message()
        return result
