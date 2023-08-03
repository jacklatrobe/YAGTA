# YAGTA - Yet another autonomous task agent - experiments in autonomous GPT agents that learn over time
# jack@latrobe.group

## main.py - main program loop for YAGTA

# Base Imports
import os
import sys
import logging
import json
from collections import deque
from typing import Dict, List, Optional, Any

# Langchain Imports
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.vectorstores import FAISS

# OpenAI LLM - The API key for your Azure OpenAI resource.  You can find this in the Azure portal under your Azure OpenAI resource.
if "OPENAI_API_KEY" not in os.environ:
    logging.critical("task_planner: Env OPENAI_API_KEY not set")
    raise ValueError("Env OPENAI_API_KEY not set")


# BabyAGI - Program main loop
def task_planner(vectorstore: FAISS, OBJECTIVE: str, DESIRED_TASKS: int = 3) -> List[Dict[str, Any]]:
    # OpenAI LLM - Initialise
    llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo", max_tokens=2000)
    # Pull most relevant task results from Vectorstore
    MAX_TASK_RESULTS = 10

    task_results = vectorstore.similarity_search(OBJECTIVE, k=MAX_TASK_RESULTS)
    context_list = "\n - ".join(task.page_content for task in task_results)
    
    PLANNED_TASKS = []

    max_iterations = 10
    loop = True
    iteration = 0
    while loop:
        iteration += 1
        logging.debug(f"task_planner: Planning - Iteration {iteration}")
        if iteration > max_iterations:
            logging.info(f"task_planner: Planning - Max iterations reached, exiting")
            raise RecursionError("task_planner: Max iterations reached")
        try:
            writing_prompt = PromptTemplate.from_template(
                "You are a expert task planner given the following objective: {OBJECTIVE}\n"
                "You've previously completed a number of tasks and have the following context:\n{CONTEXT}\n\n"
                "Return a JSON object with a list of {DESIRED_TASKS} tasks that a researcher could do to gather background on this objective.\n"
                "The researcher can use the internet to achieve their task.\n"
                "Respond only in valid JSON in the following format:\n"
                "[{{task_id: 1, task_description: 'A description of the task'}},\n"
                "{{task_id: 2, task_description: 'A description of the task'}}]\n"
            )
            writing_chain = LLMChain(llm=llm, prompt=writing_prompt)

            # Run planning chain
            plan_response = writing_chain.run(
                CONTEXT=context_list,
                OBJECTIVE=OBJECTIVE, 
                DESIRED_TASKS=DESIRED_TASKS
            )

            # Validate JSON from LLM
            logging.debug(f"task_planner: Planning - Plan response: {plan_response}")
            json_plan = json.loads(plan_response)

            # Explore the plan JSON
            temp_tasks = []
            for task in json_plan:
                temp_tasks.append(
                    {
                        "task_id": task["task_id"],
                        "task_description": task["task_description"],
                    }
                )
            PLANNED_TASKS.extend(temp_tasks)
            loop = False
            return PLANNED_TASKS
        except ValueError as ex:
            logging.error(f"task_planner: Error: {ex}")
            # If the LLM returns invalid JSON, we loop and try again until max iterations is reached
            continue
    raise NotImplementedError(
        "task_planner: Task planner failed to return a valid plan"
    )

