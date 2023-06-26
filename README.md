# YAGTA - Yet Another GPT Task Agent

YAGTA is an open-source project that aims to create an autonomous task agent by combining the innovative approaches of BabyAGI and AutoGPT. It leverages the power of GPT-4 and the modularity of LangChain to build a versatile and powerful language model application.

## Introduction

YAGTA draws inspiration from BabyAGI's approach to creating an autonomous agent with internet access, memory management, and extensibility with plugins. It adopts the concept of an agent that can perform tasks, gather information, and learn from its interactions. This allows YAGTA to interact with the web, manage its short-term and long-term memory, and extend its capabilities with various plugins.

From AutoGPT, YAGTA inherits the idea of using GPT-4 instances for text generation and information summarization. This enables YAGTA to generate human-like text, summarize complex information, and interact with users in a natural and intuitive manner.

Furthermore, YAGTA utilizes LangChain's modular approach to building language model applications. This allows YAGTA to be flexible and adaptable, capable of handling a wide range of tasks and applications. With LangChain's modules, YAGTA can perform tasks such as web searches, file storage, and access to popular websites and platforms.

One of the roadmap features of YAGTA is its ability to hopefully use LangChain's libraries to write Python files. This feature is aimed at enabling YAGTA to write and troubleshoot its own LangChain tools. The ultimate goal is for YAGTA to learn to do new things and interact with new systems autonomously, expanding its capabilities over time.

## File structure
YAGTA/
├── agents/  # custom Agent classes go here
│   ├── agent1.py
│   ├── agent2.py
│   └── __init__.py
├── tools/  # custom tools for agents go here
│   ├── tool1.py
│   ├── tool2.py
│   └── __init__.py
├── main.py  # main file to run the project and program loop
├── DOCKERFILE # builds a container for the YAGTA app
└── requirements.txt  # project dependencies

## Quickstart

1. Get an OpenAI [API Key](https://platform.openai.com/account/api-keys)
2. Download the [latest release](https://github.com/jacklatrobe/YAGTA/releases/latest)
3. Follow the installation instructions in the `docs` folder of the repository
4. Configure any additional features you want, or install some plugins as described in the `docs` folder
5. Run the app as per the instructions in the `docs` folder
