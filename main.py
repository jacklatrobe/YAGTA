# YAGTA - Yet another autonomous task agent - experiments in autonomous GPT agents that learn over time
# jack@latrobe.group

## main.py - main program loop for YAGTA

# Base Imports
import json
import sys
import logging
import threading
import validators
from collections import deque
from typing import Dict, List, Optional, Any

import html2text
from langchain.document_transformers import Html2TextTransformer

# Logging - Initialise
logging.basicConfig(encoding='utf-8', level=logging.INFO, filename="yagta.log")

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
from langchain.tools import DuckDuckGoSearchResults
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer

# Vectorstore - Imports
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
import faiss

# YAGTA Imports
from task_planner import task_planner

# Set Global Variables
INITIAL_OBJECTIVE: str = "Learn everything you can about writing LangChain, using OpenAIs APIs and leveraging Large Language Models to build autonomous agents."
DESIRED_TASKS: int = 4
MAX_TASK_RESULTS: int = 7
SIMILAR_TASK_SCORE: float = 0.75

# Global Task List
TASKS: list = []

# Vectorstore - Init
def init_vectorstore() -> FAISS:
    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    try:
        previous_vectors = FAISS.load_local("yagta_index", embeddings_model)
        vectorstore.merge_from(previous_vectors)
        logging.info("init_vectorstore: Loaded previous vectors")
    except RuntimeError as rEx:
        logging.warning(f"init_vectorstore: Unable to load previous vectors: {rEx}")
    return vectorstore

# URL Loader Function
def get_url_content(url: str) -> str:
    if validators.url(url) == True:
        try:
            logging.info("get_url_content: Loading page content from {url}")
            loader = AsyncHtmlLoader(url)
            docs = loader.load()
            html2text = Html2TextTransformer()
            doc_transformed = html2text.transform_documents(docs)
            page_content = doc_transformed[0].page_content
            return f"Content from {url}:\n{page_content}"
        except Exception as ex:
            logging.warning(f"get_url_content: Error loading page content from {url}: {str(ex)}")
            return f"Error loading page content from {url}: {str(ex)}"
    else:
        logging.warning (f"{url} is not a valid URL - unable to load page content")
        return f"{url} is not a valid URL - unable to load page content"

# Task Execution Function
def execute_task(agent_llm: ChatOpenAI, memory_llm: ChatOpenAI, vectorstore: FAISS, TASK: dict) -> None:
    logging.info(f"execute_task: Executing Task {TASK['task_id']} - {TASK['task_description']}")

    agent_chain = initialize_agent(
        TOOLS, 
        agent_llm, 
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
        handle_parsing_errors="Check your response formatting and try again.")
    result = agent_chain.run(input=f"Use your tools to achieve this task: {TASK['task_description']}\nRespond only with information from your tools - do not make anything up. Do your best to preserve any URLs, citations or sources in your final response.")

    logging.debug(f"execute_task: Task {TASK['task_id']} result: {result}")
    vectorstore.add_texts(
        texts=[result],
        metadatas=[{
            "task_description": TASK["task_description"],
            "task_objective": TASK["task_objective"],
            }],
        ids=[TASK["task_id"]],
    )

# Function - Adds new pending tasks to TASKS array
def add_new_pending_task(vectorstore: FAISS, OBJECTIVE: str, task_description: str) -> str:
    try:
        index = len(TASKS)
        results_with_scores = vectorstore.similarity_search_with_score(task_description)
        logging.info(f"add_new_pending_task: {len(results_with_scores)} similarity search results returned for {task_description}.")
        logging.debug(f"add_new_pending_task: Similarity search results: {results_with_scores}")
        for doc, score in results_with_scores:
            if score > SIMILAR_TASK_SCORE:
                logging.info(f"add_new_pending_task: Similar task already exists in TASKS list, skipping: {task_description}")
                return f"Task already exists in TASKS list, skipping: {task_description}"
            else:
                TASKS.insert(index, {"task_id": index, "task_description": task_description, "task_objective": OBJECTIVE, "task_status": "pending"})
                logging.info(f"add_new_pending_task: New task added successfully added to the queue: {task_description}")
                return f"New task added successfully added to the queue: {task_description}"
    except Exception as ex:
        logging.warning(f"add_new_pending_task: Error adding new task: {str(ex)}")
        return f"Error adding new task: {str(ex)}"

# Define toolset
websearch = DuckDuckGoSearchResults(backend="text")
newssearch = DuckDuckGoSearchResults(backend="news")
TOOLS = [
    Tool(
        name="Search Wikipedia",
        func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
        description="useful for searching wikipedia for facts about companies, places and famous people. Input should be two or three keywords to search for.",
    ),
    Tool(
        name="Search DuckDuckGo",
        func=websearch,
        description="useful for searching the internet for information, web pages or documentation. Input should be a search term or key words to search for.",
    ),
    Tool(
        name="Search News Articles",
        func=newssearch,
        description="useful for searching the internet for news articles on a topic. Input should be a search term or key words to search for.",
    ),
    Tool(
        name="Load Web Page",
        func=get_url_content,
        description="useful for loading content from a web page, such as those found from DuckDuckGo. Input should be a valid URL and the output will be text content from the page.",
    ),
    Tool(
        name="Dispatch New Task",
        func=lambda desc: add_new_pending_task(OBJECTIVE, desc),
        description="useful for adding a new task for later when another tool doesn't work or you can't find what you need. Input should be a description of the next task for you to complete to progress the objective.",
    )
]

def execute_objective(OBJECTIVE: str) -> list:
    # Establish LLM
    agent_llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo-16k-0613", max_tokens=4000)
    memory_llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo-0613", max_tokens=1000)

    # (Re)Load Vectorstore
    vectorstore = init_vectorstore()

    # Create initial plan
    try:
        for PLANNED_TASK in task_planner(vectorstore, OBJECTIVE, DESIRED_TASKS):
            add_new_pending_task(vectorstore, OBJECTIVE, PLANNED_TASK['task_description'])
    except RecursionError as rEx:
        logging.critical(f"execute_objective: Planner reached max iterations attempting to generate valid JSON plan for objective: {rEx}")
        sys.exit(1)
    except Exception as ex:
        logging.critical(f"execute_objective: Planner encountered an unknown error while trying to generate initial plan for objective: {ex}")
        sys.exit(1)

    # Main program loop - executes pending TASKS in a thread
    while len([task for task in TASKS if task["task_status"] == "pending"]) > 0:
        logging.info(f"execute_objective: Pending tasks detected, starting new task execution iteration")
        ## Start task threads - WHY ARE MY INDEXES OUT OF SYNC
        task_threads = {}
        for TASK in [task for task in TASKS if task["task_status"] == "pending"]:
            index = int(TASK["task_id"])
            logging.info(f"execute_objective: Starting new task thread for task ID: {index}")
            x = threading.Thread(target=execute_task, args=(agent_llm, memory_llm, vectorstore, TASK,))
            task_threads[index] = x
            x.start()
            TASKS[index]["task_status"] = "running"
        
        ## Wait for task threads to finish
        for TASK in [task for task in TASKS if task["task_status"] == "running"]:
            index = int(TASK["task_id"])
            task_threads[index].join()
            TASKS[index]["task_status"] = "complete"
            logging.info(f"execute_objective: Thread {index} has finished executing task")

    # Pull most relevant task results from Vectorstore
    task_results = []
    for TASK in [task for task in TASKS if task["task_status"] == "complete" and task["task_objective"] == OBJECTIVE]:
        task_description = TASK["task_description"]
        task_context = vectorstore.similarity_search(task_description, k=MAX_TASK_RESULTS)
        logging.info(f"execute_objective: {len(task_context)} similarity search results returned for {task_description}.")
        logging.debug(f"execute_objective: Similarity search results: {task_context}")
        task_results.append(task_context)
    context_list = "\n - ".join(task_results)
     
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
                    "You are trying to find a new objective. Assess the response against the objective and respond with only one of the following JSON responses:\n"
                    ' - If you can find a new objective to continue with: {{"assessment": "incomplete", "new_objective": "any suggested actions, objectives or suggestions to proceed with"}}\n'
                    ' - If you cannot find a new task or objective to continue with: {{"assessment": "complete"}}\n'
                    "Remember to respond with ONLY JSON in your response."
                )
    objective_chain = LLMChain(llm=agent_llm, prompt=objective_prompt)
    check_objective = objective_chain.run(OBJECTIVE=OBJECTIVE, response=summary)
    try:
        check_result = json.loads(check_objective)
        if check_result["assessment"] == "complete":
            logging.info(f"execute_objective: ending loop with final summary")
            return [True, summary]
        else:
            OBJECTIVE = check_result["new_objective"]
            logging.info(f"execute_objective: restarting loop with new objective: {OBJECTIVE}")
            return [False, OBJECTIVE]
    except Exception as ex:
        logging.warning(f"execute_objective: Error parsing objective check response: {ex}")
        return [True, summary]


if __name__ == "__main__":
    loop = True
    OBJECTIVE = INITIAL_OBJECTIVE
    while loop:
        result = execute_objective(OBJECTIVE)
        if result[0] == True:
            print(result[1])
            break
        else:
            OBJECTIVE = result[1]
            continue