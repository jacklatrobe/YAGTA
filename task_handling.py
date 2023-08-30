# YAGTA - Yet another autonomous task agent - experiments in autonomous GPT agents that learn over time
# jack@latrobe.group

##  task_handling.py - Task execution functions for YAGTA

# Base Imports
import logging

# Langchain Imports
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI

# YAGTA Imports
from yagta_task import AgentTask

# Task Execution Function
def execute_task(llm: ChatOpenAI, task: AgentTask, tools: list) -> str:
    logging.info(f"execute_task: Executing Task: {task.task_description}")

    agent_chain = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
        handle_parsing_errors="Check your response formatting and try again.")
    result = agent_chain.run(input=f"You are an autonomous task agent. Use your tools to achieve this task: {task.task_description}\n#####\n" + 
                             f"Do not make anything up. Do your best to preserve any URLs, citations or sources in your final response.\n#####\n" +
                             f"This task is being done as part of a broader objective, so try to shape your response so it progresses this objective: {task.task_objective}\n#####\n")

    logging.debug(f"execute_task: Task {task.task_description} result: {result}")
    
    return AgentTask(
        task_description = task.task_description,
        task_objective= task.task_objective,
        task_result = result
    )