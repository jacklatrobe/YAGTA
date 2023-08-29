# YAGTA - Yet another autonomous task agent - experiments in autonomous GPT agents that learn over time
# jack@latrobe.group

## yagta_tools.py - Toolset for YAGTA

# Base Imports
import logging

# Langchain Imports
from langchain.agents import Tool
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import DuckDuckGoSearchResults
from langchain.vectorstores import FAISS

# YAGTA Imports
from yagta_tools.url_loader import get_url_content

class AgentTools:
    def __init__(self, VECTORSTORE: FAISS):
        self.vectorstore = VECTORSTORE
        websearch = DuckDuckGoSearchResults(backend="text")
        newssearch = DuckDuckGoSearchResults(backend="news")
        vectorsearch = VECTORSTORE.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5})

        self.TOOLS = [
            Tool(
                name="Search Vectorstore",
                func=vectorsearch.get_relevant_documents,
                description="Use this tool first every time to check if you've completed a similar previous task and what your results were. Input is a search query or keywords",
            ),
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
                description="useful for loading content from a web page, such as those found from DuckDuckGo. Input must be a valid URL and the output will text from the web page.",
            ),
        ]