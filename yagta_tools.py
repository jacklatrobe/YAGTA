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

# YAGTA Imports
from url_loader import get_url_content

# Define Tools
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
    )
]