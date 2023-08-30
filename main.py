# YAGTA - Yet another autonomous task agent - experiments in autonomous GPT agents that learn over time
# jack@latrobe.group

## main.py - main program loop for YAGTA

# Base Imports
import os
import logging
from multiprocessing import Process, Queue, Lock
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
def YAGTA(OBJECTIVE: str = None):
    if OBJECTIVE is None:
        OBJECTIVE = input("Welcome to YAGTA - Please enter your initial objective: ")

    VECTORSTORE = init_vectorstore()
    LLM = ChatOpenAI(model="gpt-3.5-turbo-0613", max_tokens=1500)
    AGENTTOOLS = AgentTools(VECTORSTORE)

    # Plan initial tasks
    initial_tasks = task_planner(LLM=LLM, VECTORSTORE=VECTORSTORE, OBJECTIVE=OBJECTIVE, TOOLS=AGENTTOOLS)
    PENDING_TASKS.extend(initial_tasks)
    logging.info(f"main: Planning complete - {len(initial_tasks)} tasks planned")

    # Add initial tasks to pending queue
    logging.info("main: Starting parallel task execution")
    q = Queue()
    FUTURES = []
    for task in PENDING_TASKS:
        process = Process(target=execute_task, args=(LLM, task, AGENTTOOLS, q))
        process.start()
        FUTURES.append(process)
    while len(FUTURES) > 0:
        for thread in FUTURES:
            if not thread.is_alive():
                thread.join()
                FUTURES.remove(thread)

    # Save task results to Vectorstore and generate summary string
    logging.info("main: Saving task results to Vectorstore")
    result_summary = "Task Results:"
    for task in q.get():
        VECTORSTORE.add_texts(task.task_result, [{"task_description": task.task_description, "task_objective": task.task_objective}])
        result_summary = f"{result_summary}\n - {task.task_description}: {task.task_result}"

    # Summarise task results
    logging.info("main: Generating summary of task results")
    final_result = LLM.predict(f"Summarise the following tasks and their results for this objective: {OBJECTIVE}\n#####\n{result_summary}")

    # Persist your previous tasks to disk to learn for next time.
    VECTORSTORE.save_local("yagta_index")
    print(final_result)

if __name__ == "__main__":
    lock = Lock()
    YAGTA()