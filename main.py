# YAGTA - Yet another autonomous task agent - experiments in autonomous GPT agents that learn over time
# jack@latrobe.group

## main.py - main program loop for YAGTA

# Base Imports
import os
import sys
import logging
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Any

# Logging - Initialise
logging.basicConfig(encoding='utf-8', level=logging.INFO)

# Langchain Imports
from langchain.agents import AgentType
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate, OpenAI
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import DuckDuckGoSearchRun

# Vectorstore - Imports
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
import faiss

# Vectorstore - Configuration
embeddings_model = OpenAIEmbeddings().embed_query
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})

# YAGTA Imports
from task_planner import task_planner

def execute_task(agent_chain, vectorstore, TASK):
    logging.info(f"main: Executing Task: {TASK}")
    result = agent_chain.run(input=f"Use your tools to achieve this task - you may need to use a tool multiple times with different search terms or key words to get enough information: {TASK['task_description']}")

    vectorstore.add_texts(
        texts=[result],
        metadatas=[{"task": TASK["task_description"]}],
        ids=[TASK["task_id"]],
    )

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
]

def main():
    llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo-16k", max_tokens=5000)

    OBJECTIVE = "Compile a detailed report on the intersection of artificial general intelligences, large language models and frameworks such as Reason and Act (ReAct) and LangChain"
    DESIRED_TASKS = 10
    MAX_TASK_RESULTS = 5

    try:
        TASKS = task_planner(OBJECTIVE, DESIRED_TASKS)

    except RecursionError as rEx:
        logging.critical(f"main: Planner reached max iterations attempting to generate valid JSON plan for objective: {rEx}")
        sys.exit(1)
    except Exception as ex:
        logging.critical(f"main: Planner encountered an unknown error while trying to generate initial plan for objective: {ex}")
        sys.exit(1)
    memory = ConversationSummaryBufferMemory(
        llm=llm, max_token_limit=250, return_messages=True, memory_key = "chat_history"
    )
    agent_chain = initialize_agent(TOOLS, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, memory=memory)

    task_threads = []
    for TASK in TASKS:
        x = threading.Thread(target=execute_task, args=(agent_chain, vectorstore, TASK,))
        task_threads.append(x)
        x.start()
    
    for index, thread in enumerate(task_threads):
        thread.join()
        logging.info(f"main: Thread {index} has finished executing task")

    task_results = vectorstore.similarity_search(OBJECTIVE, k=MAX_TASK_RESULTS)
    context_list = "\n - ".join(task.page_content for task in task_results)
    summary_prompt = PromptTemplate.from_template(
                    "You are an AI agent who has been given the following objective: {objective}.\n"
                    "You have performed the following tasks to achieve this objective: {context_list}.\n"
                    "Write a detailed report that answers the objective, or at least includes an explaination of the tasks and actions taken and a detailed list of actions required to yet achieve the objective.\n"
                )
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    summary = summary_chain.run(objective=OBJECTIVE, context_list=context_list)
    print(summary)


if __name__ == "__main__":
    main()