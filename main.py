# YAGTA - Yet another autonomous task agent - experiments in autonomous GPT agents that learn over time
# jack@latrobe.group

## main.py - main program loop for YAGTA

# Base Imports
import os
import logging
import concurrent.futures
from queue import Queue

# Logging - Initialise
logging.basicConfig(encoding='utf-8', level=logging.INFO, filename="yagta.log")

# OpenAI LLM - The API key for your Azure OpenAI resource.  You can find this in the Azure portal under your Azure OpenAI resource.
if "OPENAI_API_KEY" not in os.environ:
    logging.critical("main: Env OPENAI_API_KEY not set")
    raise ValueError("Env OPENAI_API_KEY not set")

# Langchain Imports
from langchain.embeddings import OpenAIEmbeddings

# Vectorstore - Imports
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
import faiss

# YAGTA Imports
from task_planning import task_planner
from url_loader import get_url_content

# Set Global Variables
INITIAL_OBJECTIVE: str = "Learn everything you can about writing LangChain, using OpenAIs APIs and leveraging Large Language Models to build autonomous agents."
DESIRED_TASKS: int = 4
MAX_TASK_RESULTS: int = 7
SIMILAR_TASK_SCORE: float = 0.75

# Global Task List
TASKS = Queue()

# Worker Pool - Init
THREADPOOL = concurrent.futures.ThreadPoolExecutor(max_workers=3)
THREADFUTURES: list = []

# AgentTask - define class
class AgentTask:
    def __init__(self, task_id: int, task_description: str, task_objective: str, task_status: str) -> None:
        self.task_id = task_id
        self.task_description = task_description
        self.task_objective = task_objective
        self.task_status = task_status

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