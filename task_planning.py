# YAGTA - Yet another autonomous task agent - experiments in autonomous GPT agents that learn over time
# jack@latrobe.group

## main.py - main program loop for YAGTA

# Base Imports
import os
import logging
import json
from typing import Dict, List, Any

# Langchain Imports
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.vectorstores import FAISS

# YAGTA Imports
from yagta_task import AgentTask

# OpenAI LLM - The API key for your Azure OpenAI resource.  You can find this in the Azure portal under your Azure OpenAI resource.
if "OPENAI_API_KEY" not in os.environ:
    logging.critical("main: Env OPENAI_API_KEY not set")
    raise ValueError("Env OPENAI_API_KEY not set")


# BabyAGI - Program main loop
def task_planner(LLM: ChatOpenAI, VECTORSTORE: FAISS, OBJECTIVE: str, TOOLS: list) -> List[Dict[str, Any]]:

    # Check to see if we've done similar tasks before
    task_results = VECTORSTORE.similarity_search(OBJECTIVE, k=3)
    context_list = "\n - ".join(task.page_content for task in task_results)
    
    PLANNED_TASKS = []

    # Attempt to generate a list of tasks
    # This iterates to avoid transient hallucinations that impact valid JSON generation
    iteration = 0
    while iteration < 3:
        iteration += 1
        logging.debug(f"task_planner: Planning - Iteration {iteration}")
        try:
            writing_prompt = PromptTemplate.from_template(
                "You are a expert task planner given the following objective: {OBJECTIVE}\n"
                "You've previously completed these relevant tasks, among others:\n{CONTEXT}\n\n"
                "Return a JSON object with a list of tasks that a researcher could do to gather background on this objective.\n"
                "The researcherhas access to the following tools to achieve their tasks:\n{TOOLS}"
                "Respond only in valid JSON in the following format:\n"
                "[{{task_description: 'A description of the task'}},\n"
                "{{task_description: 'A description of the next task'}}]\n"
            )
            writing_chain = LLMChain(llm=LLM, prompt=writing_prompt)

            # Run planning chain
            plan_response = writing_chain.run(
                CONTEXT=context_list,
                OBJECTIVE=OBJECTIVE,
                TOOLS=str(TOOLS),
            )

            # Validate JSON from LLM
            logging.debug(f"task_planner: Planning - Plan response: {plan_response}")
            json_plan = json.loads(plan_response)

            # Explore the plan JSON
            temp_tasks = []
            for task in json_plan:
                temp_tasks.append(
                    AgentTask(
                        task_description = task["task_description"],
                        task_objective = OBJECTIVE,
                        )
                )
            PLANNED_TASKS.extend(temp_tasks)
            return PLANNED_TASKS
        except ValueError as ex:
            logging.error(f"task_planner: Error: {ex}")
            # If the LLM returns invalid JSON, we loop and try again until max iterations is reached
            continue
    assert False, "task_planner: Unable to plan tasks"

