# YAGTA - Yet another autonomous task agent - experiments in autonomous GPT agents that learn over time
# jack@latrobe.group

## main.py - main program loop for YAGTA

# Base Imports
import os
import logging
import concurrent.futures
from collections import deque

# Logging - Initialise
logging.basicConfig(encoding='utf-8', level=logging.INFO, filename="yagta.log")

# OpenAI LLM - The API key for your Azure OpenAI resource.  You can find this in the Azure portal under your Azure OpenAI resource.
if "OPENAI_API_KEY" not in os.environ:
    logging.critical("main: Env OPENAI_API_KEY not set")
    raise ValueError("Env OPENAI_API_KEY not set")

# Langchain Imports
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# Vectorstore - Imports
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
import faiss

# YAGTA Imports
from yagta_task import AgentTask
from task_planning import task_planner
from task_handling import execute_task
from yagta_tools.agent_tools import AgentTools

# Global Task List
PENDING_TASKS = deque()
COMPLETED_TASKS = deque()

# Worker Pool - Init
THREADPOOL = concurrent.futures.ThreadPoolExecutor(max_workers=3)
THREADFUTURES: list = []

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

# Main Program Loop - Take an objective from the user, plan tasks, execute tasks, save results, repeat
if __name__ == "__main__":
    OBJECTIVE = input("Welcome to YAGTA - Please enter your initial objective: ")
    VECTORSTORE = init_vectorstore()
    LLM = ChatOpenAI(model="gpt-3.5-turbo-0613", max_tokens=1500)
    AGENTTOOLS = AgentTools(VECTORSTORE)
    # Plan initial tasks
    initial_tasks = task_planner(LLM=LLM, VECTORSTORE=VECTORSTORE, OBJECTIVE=OBJECTIVE, TOOLS=AGENTTOOLS)
    logging.info(f"main: Planning complete - {len(initial_tasks)} tasks planned")

    # Add initial tasks to pending queue
    for task in initial_tasks:
        logging.info(f"main: Adding task to pending queue")
        PENDING_TASKS.append(task)

    # Loop through task dispatch
    final_result = None
    loop = True
    while loop:
        # Dispatch pending tasks
        while len(PENDING_TASKS) > 0:
            task = PENDING_TASKS.pop()
            THREADFUTURES.append(THREADPOOL.submit(execute_task, llm=LLM, task=task, tools=AGENTTOOLS.TOOLS))
            logging.info(f"main: Dispatched task to worker pool - {task.task_description}")
        
        # Check for completed tasks
        for task in THREADFUTURES:
            if task.done():
                result = task.result()
                task_obj = AgentTask(
                    task_description=result.task_description,
                    task_objective=result.task_objective,
                    task_result=result.task_result
                )
                COMPLETED_TASKS.append(task_obj)
                logging.info(f"main: Task completed - {task_obj.task_description}")

        # Save task results to Vectorstore
        result_summary = "Task Results:"
        for task in COMPLETED_TASKS:
            VECTORSTORE.add_texts(task.task_result, [{"task_description": task.task_description, "task_objective": task.task_objective}])
            result_summary = f"{result_summary}\n - {task.task_description}: {task.task_result}"

        # Summarise task results
        final_result = LLM.predict(f"Summarise the following tasks and their results for this objective: {OBJECTIVE}\n#####\n{result_summary}")
        if final_result:
            break

    # Persist your previous tasks to disk to learn for next time.
    VECTORSTORE.save_local("yagta_index")
    print(final_result)