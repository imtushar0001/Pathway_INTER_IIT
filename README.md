# Agentic RAG with LangGraph and Pathway

This project demonstrates the integration of **Agentic Retrieval-Augmented Generation (RAG)** workflows using **LangGraph** and **Pathway**. Agentic RAG enables advanced information retrieval and synthesis, making it a powerful tool for building intelligent agents. The project leverages key technologies, libraries, and frameworks, including LangGraph, OpenAI, and other ecosystem tools, to create a modular and scalable solution.

## Key Components

- **LangGraph**: A graph-based framework designed for creating AI workflows with reusable components and integrations.
- **Pathway**: Enables orchestrating data workflows, optimizing retrieval processes, and facilitating scalability as well as Real Time indexing
- **OpenAI API**: Provides GPT-based capabilities for natural language understanding and generation.
- **SerpAPI and bs4**: Used for search and web scraping functionalities.
- **SQLite Checkpoints**: Facilitates lightweight, checkpoint-based storage during development and testing.

The code for the same with proper description is there in `Pathway_LangGraph.ipynb`
## Setup Instructions

Before running the project, ensure all dependencies are installed. The commands below will install the required libraries, including specific versions for compatibility. These dependencies include **LangChain**, **LangGraph**, **OpenAI**, and supporting libraries for data manipulation and API calls.

### Initial Metrics : 
Performance metrics were evaluated using the Opik library, with an LLM-based judge assessing hallucination and answer relevance.
The opik library leverages Large Language Models (LLMs) as judge evaluation metrics to assess the quality of generated text. It specifically focuses on evaluating two key aspects: hallucination, ensuring the generated answers are factually accurate, and answer relevance, ensuring the responses are contextually appropriate. By using LLMs to automate and scale the evaluation of these factors, opik enables consistent and effective monitoring of LLM performance in tasks like question answering and dialogue systems.
An accompanying Jupyter notebook (`LLM_Judge.ipynb`) has been added for further reference.
#### Performance of the model:
| Metric      | Average Relevance Score | Average Hallucination Score |
|-------------|-------------------------|-----------------------------|
| LLM_Judge   | 0.83                    |0                            |

The hallucination score and relevance score of each query is given in `Performance_Metrics.xlsx`. The context used for judging these queries is given in `data` directory.






