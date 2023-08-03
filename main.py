# YAGTA - Yet another autonomous task agent - experiments in autonomous GPT agents that learn over time
# jack@latrobe.group

## main.py - main program loop for YAGTA

# Base Imports
import json
import sys
import logging
import threading
from collections import deque
from typing import Dict, List, Optional, Any

# Logging - Initialise
logging.basicConfig(encoding='utf-8', level=logging.INFO)

# Langchain Imports
from langchain.agents import AgentType
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.agents import Tool
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import DuckDuckGoSearchRun

# Vectorstore - Imports
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
import faiss

# Vectorstore - Configuration
embeddings_model = OpenAIEmbeddings()
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
try:
    previous_vectors = FAISS.load_local("yagta_index", embeddings_model)
    vectorstore.merge_from(previous_vectors)
    logging.info("main: Loaded previous vectors")
except RuntimeError as rEx:
    logging.warning(f"main: Unable to load previous vectors: {rEx}")

# Set Global Variables
INITIAL_OBJECTIVE = "Conduct research into LangChain, LLMs, the Reason and Act Framework, and secondbrain theory. Each task you do is saved as a unique research result, so generate lots of tasks and suggest any follow up actions in the descriptions."
DESIRED_TASKS = 4
MAX_TASK_RESULTS = 7

# Global Task List
TASKS = []

# YAGTA Imports
from task_planner import task_planner

# Task Execution Function
def execute_task(agent_chain, vectorstore, TASK):
    logging.debug(f"main: Executing Task: {TASK}")
    result = agent_chain.run(input=f"Use your tools to achieve this task - you may need to try to use multiple tools, different search terms, or add new tasks for later investigation or follow-up to complete this task: {TASK['task_description']}")

    vectorstore.add_texts(
        texts=[result],
        metadatas=[{"task_description": TASK["task_description"],
                    "task_status": TASK["task_status"],}],
        ids=[TASK["task_id"]],
    )

# Function - Adds new pending tasks to TASKS array
def add_new_pending_task(OBJECTIVE, task_description: str):
    logging.info(f"main: Adding new pending task: {task_description}")
    try:
        index = len(TASKS)
        results_with_scores = vectorstore.similarity_search_with_score(task_description)
        for doc, score in results_with_scores:
            if score > 0.95:
                logging.debug(f"main: Similar task already exists in TASKS list, skipping: {task_description}")
                return f"Task already exists in TASKS list, skipping: {task_description}"
            else:
                TASKS.insert(index, {"task_id": index, "task_description": task_description, "task_objective": OBJECTIVE, "task_status": "pending"})
                return f"New task added successfully added to the queue: {task_description}"
    except Exception as ex:
        return f"Error adding new task: {str(ex)}"

# Define toolset
TOOLS = [
    Tool(
        name="Search Wikipedia",
        func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
        description="useful for searching wikipedia for facts about companies, places and famous people. Input should be two or three keywords to search for.",
    ),
    Tool(
        name="Search DuckDuckGo",
        func=DuckDuckGoSearchRun(),
        description="useful for searching the internet for news, websites, answers and general knowledge. Input should be a search term or query to search for.",
    ),
    Tool(
        name="Dispatch New Task",
        func=lambda desc: add_new_pending_task(OBJECTIVE, desc),
        description="useful for adding a new task for later when another tool doesn't work or you can't find what you need . Input should be a description of the next task for you to complete to progress the objective.",
    )
]

def main(OBJECTIVE: str):
    # Establish LLM
    agent_llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo-16k", max_tokens=4000)
    memory_llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo", max_tokens=1000)

    # Create initial plan
    #try:
    for TASK in task_planner(OBJECTIVE, DESIRED_TASKS):
        add_new_pending_task(OBJECTIVE, TASK['task_description'])
    # except RecursionError as rEx:
    #     logging.critical(f"main: Planner reached max iterations attempting to generate valid JSON plan for objective: {rEx}")
    #     sys.exit(1)
    # except Exception as ex:
    #     logging.critical(f"main: Planner encountered an unknown error while trying to generate initial plan for objective: {ex}")
    #     sys.exit(1)

    # Create Agent Chain
    memory = ConversationSummaryBufferMemory(
        llm=memory_llm, max_token_limit=250, return_messages=True, memory_key = "chat_history"
    )
    agent_chain = initialize_agent(
        TOOLS, 
        agent_llm, 
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
        memory=memory,
        handle_parsing_errors="Error from OpenAI - check your response formatting and try again.")
    # Main program loop - executes pending TASKS in a thread
    while len([task for task in TASKS if task["task_status"] == "pending"]) > 0:
        logging.info(f"main: Pending tasks detected, starting new task execution iteration")
        ## Start task threads - WHY ARE MY INDEXES OUT OF SYNC
        task_threads = {}
        for TASK in [task for task in TASKS if task["task_status"] == "pending"]:
            index = int(TASK["task_id"])
            logging.info(f"main: Starting new task thread for task ID: {index}")
            x = threading.Thread(target=execute_task, args=(agent_chain, vectorstore, TASK,))
            task_threads[index] = x
            x.start()
            TASKS[index]["task_status"] = "running"
        
        ## Wait for task threads to finish
        for TASK in [task for task in TASKS if task["task_status"] == "running"]:
            index = int(TASK["task_id"])
            task_threads[index].join()
            TASKS[index]["task_status"] = "complete"
            logging.info(f"main: Thread {index} has finished executing task")

    # Pull most relevant task results from Vectorstore
    task_results = vectorstore.similarity_search(OBJECTIVE, k=MAX_TASK_RESULTS)
    context_list = "\n - ".join(task.page_content for task in task_results)
     
    # Generate summary of task results
    summary_prompt = PromptTemplate.from_template(
                    "You are an AI agent who has been given the following objective: {objective}.\n"
                    "You have performed the following tasks to achieve this objective: {context_list}.\n"
                    "Write a detailed response that answers the objective, based only on the information you found while doing the research tasks above\n"
                    "Be honest about when no information was found or where we ran into errors, and be sure to include helpful suggestions or follow up actions for the user"
                )
    summary_chain = LLMChain(llm=agent_llm, prompt=summary_prompt)
    summary = summary_chain.run(objective=OBJECTIVE, context_list=context_list)

    # Save vectorstore to disk for next YAGTA run
    vectorstore.save_local("yagta_index")

    # Check the summary meets the objective, or loop again
    objective_prompt = PromptTemplate.from_template(
                    "An AI agent was given the following objective: {OBJECTIVE}\n\n"
                    "It generated this response: {response}.\n\n"
                    "Assess the response against the objective and respond with only one of the following JSON responses:\n"
                    "{{assessment: 'complete'}} or {{assessment: 'incomplete', new_objective: 'new objective to generate better response'}}\n"
                )
    objective_chain = LLMChain(llm=agent_llm, prompt=objective_prompt)
    check_objective = objective_chain.run(OBJECTIVE=OBJECTIVE, response=summary)
    try:
        check_result = json.loads(check_objective)
        if check_result["assessment"] == "complete":
            return summary
        else:
            OBJECTIVE = check_result["new_objective"]
            return False
    except Exception as ex:
        logging.warning(f"main: Error parsing objective check response: {ex}")
        return summary


if __name__ == "__main__":
    loop = True
    OBJECTIVE = INITIAL_OBJECTIVE
    while loop:
        result = main(OBJECTIVE)
        if result:
            print(result)
            break
        else:
            OBJECTIVE = result