# YAGTA - Yet another autonomous task agent

YAGTA (Yet another autonomous task agent) is an experimental project that focuses on creating autonomous GPT (Generative Pre-trained Transformer) agents that can learn and improve over time. The goal of YAGTA is to develop an agent that can generate and execute tasks based on given objectives.

## Getting Started

To get started with YAGTA, follow these steps:

### Prerequisites

- Python 3.7 or higher
- OpenAI API key

### Installation

1. Clone the YAGTA repository from GitHub:

```
git clone https://github.com/your-username/yagta.git
```

2. Navigate to the cloned repository:

```
cd yagta
```

3. Install the required dependencies:

```
pip install -r requirements.txt
```

### Usage

1. Set up your OpenAI API key:

   - Go to the Azure portal and find your Azure OpenAI resource.
   - Retrieve your API key.

2. Run the main program loop:

   ```
   python main.py
   ```

   This will start the YAGTA agent and execute the tasks based on the given objectives.

## Configuration

YAGTA uses the following components and configurations:

- LangChain: LangChain is a framework that simplifies the creation of applications using large language models (LLMs). It provides tools, components, and interfaces to manage interactions with language models.

- Vectorstore Memory: YAGTA uses VectorStoreRetrieverMemory to store and retrieve memories. It stores previous conversation snippets and uses an in-memory vectorstore for efficient retrieval.

- OpenAI LLM: YAGTA utilizes the OpenAI language model for generating responses and executing tasks. Make sure to set up your OpenAI API key before running YAGTA.

## Contributing

Contributions to YAGTA are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.

## License

YAGTA is released under the [MIT License](https://opensource.org/licenses/MIT).