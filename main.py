from agents.execution_agent import ExecutionAgent
from agents.context_agent import ContextAgent
from agents.task_creation_agent import TaskCreationAgent
from agents.priority_agent import PriorityAgent
from agents.tool_creation_agent import ToolCreationAgent
from langchain.agents import AgentExecutor
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.llms import AzureOpenAI
from re import sub
import os
import sys
import logging

# Set this to `azure`
os.getenv("OPENAI_API_TYPE", default="azure")

# The API version you want to use: set this to `2023-03-15-preview` for the released version.
os.getenv("OPENAI_API_VERSION", default="2023-03-15-preview")

# The base URL for your Azure OpenAI resource.  You can find this in the Azure portal under your Azure OpenAI resource.
os.getenv("OPENAI_API_BASE", default="https://your-resource-name.openai.azure.com")

# The API key for your Azure OpenAI resource.  You can find this in the Azure portal under your Azure OpenAI resource.
if "OPENAI_API_KEY" not in os.environ:
    logging.CRITICAL("Env OPENAI_API_KEY not set - exiting")
    sys.exit(1)

# Program main loop
def main():
    # Establish connection to LLM
    llm = AzureOpenAI(
        deployment_name="davinci",
        model_name="text-davinci-003",
        temperature=0.1,
    )

    # Initialize your agents
    execution_agent = ExecutionAgent(llm=llm)
    context_agent = ContextAgent(llm=llm)
    task_creation_agent = TaskCreationAgent(llm=llm)
    priority_agent = PriorityAgent(llm=llm)
    tool_creation_agent = ToolCreationAgent(llm=llm)

    # List of your agents
    agents = [execution_agent, context_agent, task_creation_agent, priority_agent, tool_creation_agent]

    # Loop through your agent flow
    keep_working = True
    while keep_working:
        current_objective = "This is a test"
        available_tools = get_tools(current_objective)
        query = "What should we do to achieve the current objective: {current_objective}".format(current_objective=current_objective)
        for agent in agents:
            agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=available_tools, verbose=True)
            response = agent_executor.run(query)
            query = response

# Function for dynamically loading any new tools written between each iteration
def discover_tools():
    tools = []
    for directories, subdirectories, files in os.walk("./tools/"):
        for file in files:
            base_name = str(file).removesuffix(".py")
            function_name = camel_case(base_name)
            try:
                eval("exec('from tools.{base_name} import {function_name}')".format(base_name=base_name, function_name=function_name))
                tools.append(eval("{object_name} = {function_name}()".format(object_name=function_name, function_name=function_name)))
            except Exception as ex:
                # TODO - Failure to import should raise a task to have ToolCreationAgent re-examine this tools src code
                pass
    return tools

def get_tools(query):
    tools = discover_tools()
    docs = [
            Document(page_content=t.description, metadata={"index": i})
            for i, t in enumerate(tools)
        ]
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    retriever = vector_store.as_retriever()
    docs = retriever.get_relevant_documents(query)
    return [tools[d.metadata["index"]] for d in docs]

# String handler for case conversion
def camel_case(s):
    s = sub(r"(_|-)+", " ", s).title().replace(" ", "")
    return ''.join([s[0].lower(), s[1:]])

if __name__ == "__main__":
    main()
