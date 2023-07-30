# YAGTA - Yet another autonomous task agent - experiments in autonomous GPT agents that learn over time
# jack@latrobe.group

## main.py - main program loop for YAGTA

# Base Imports
import os
import sys
import logging
from collections import deque
from typing import Dict, List, Optional, Any

# Logging - Initialise
logging.basicConfig(encoding='utf-8', level=logging.INFO)

# Langchain Imports
from langchain import LLMChain, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM, OpenAI
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from langchain.chains.base import Chain

# YAGTA Imports
from babyagi import *

# Vectorstore - Imports
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore

# OpenAI LLM - The API key for your Azure OpenAI resource.  You can find this in the Azure portal under your Azure OpenAI resource.
if "OPENAI_API_KEY" not in os.environ:
    logging.CRITICAL("Env OPENAI_API_KEY not set - exiting")
    sys.exit(1)


# Program main loop
def main():
    # OpenAI LLM - Establish connection to the GPT LLM
    llm = OpenAI(
        deployment_name="chat",
        model_name="gpt-35-turbo",
        temperature=0.1,
    )

    # BabyAGI - Define an objective for the AI
    OBJECTIVE = "Write a weather report for Melbourne, VIC today"

    # Vectorstore - Define your embedding model
    embeddings_model = OpenAIEmbeddings()

    # Vectorstore - Connect to the vector store
    embedding_size = 1536
    index = FAISS.IndexFlatL2(embedding_size)

    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    # OpenAI LLM - Logging of LLMChains
    verbose = False

    # If None, will keep on going forever
    max_iterations: Optional[int] = 5
    baby_agi = BabyAGI.from_llm(
        llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
    )

    baby_agi({"objective": OBJECTIVE})
    

if __name__ == "__main__":
    main()
