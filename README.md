### Workflow Overview
![2](https://github.com/Duvvuri-Srinath/Tech-Meet-Pathway/blob/pathway/Workflow.png?raw=true)


---

# Query Processing System: Detailed Workflow

This README outlines the complete workflow for our Query Processing System, which leverages advanced AI techniques to handle and process user queries using multiple layers of context retrieval, including document indexing, embedding, vector stores, query classification, leader-analyst agents, and external web search APIs. 

## 1. Document Indexing and Vector Store Initialization

The system starts by indexing available documents to make them searchable based on relevance to incoming queries. Here’s how it works:

- **Document Parsing**: Documents are parsed using a pre-trained model such as Gemini Embedder. This model processes the documents and converts them into a format that’s easier to analyze.
  
- **Text Embedding**: The text from each document is embedded using a machine learning model. The embeddings convert the textual data into a numerical form that captures the meaning and context of the text.

- **Vector Store**: These embedded documents are stored in a vector store (Pathway’s VectorStoreServer). The vector store enables fast similarity-based retrieval, which allows the system to quickly retrieve relevant documents in response to user queries.

## 2. Server Setup and Query Handling

Once the vector store is initialized, the server is started, ready to process user queries. Here’s the process for handling incoming queries:

- **Query Reception**: When a user submits a query, the system first uses an **input guardrail** to scan the query for inappropriate language, actions, or irrelevant content.

- **Document Retrieval**: The system retrieves the top 3 most relevant documents from the vector store based on their similarity to the incoming query. These documents form the initial context for the query.

- **Query Grading**: The system then classifies the query as either:
  - **Correct**: If the query can be answered using the retrieved documents.
  - **Incorrect**: If additional context is needed (e.g., external web search).

## 3. Leader-Analyst Agent Process (Correct Queries)

If the query is classified as **Correct**, meaning it can be answered using the retrieved documents, the following steps occur:

- **Leader Analyst Chain**: The **Leader agent** divides the query into two subtasks based on the retrieved context. These subtasks are fed into **Analyst agents**, which are responsible for answering the query.

- **Unifying Responses**: After receiving answers from both Analysts, the Leader agent combines their responses to form a final answer. The Leader ensures that the answer is coherent and accurate.

- **Follow-up Queries**: If the Leader agent determines that follow-up queries are necessary for further clarification, it divides the query into two new subtasks, and the process repeats with the Analysts.

- **Human-in-the-Loop**: After generating the final response, a **human-in-the-loop** approach is used. The user is asked for feedback on the response. If the user is satisfied, the process ends; otherwise, the user can refine the query and provide additional context.

## 4. Leader-Analyst Agent Process (Incorrect Queries)

If the query is classified as **Incorrect**, meaning it requires external context, the system follows a similar process but with the addition of web search:

- **Leader-Analyst Subtask Division**: Similar to the **Correct** query process, the **Leader agent** divides the query into two subtasks. These are passed to the Analysts for processing.

- **External Web Search**:
  - The system first attempts to use the **SERPER API** to gather additional context from the web.
  - If the **SERPER API** fails, the system falls back to using the **SERP API**, which scrapes the top URLs and collects additional context (such as AI overviews and stock information).

- **Unifying Responses**: After retrieving the additional context from the web, the **Leader agent** unifies the information from the Analysts and the external search sources to generate a final response.

- **Follow-up Queries**: If the Leader agent determines that further refinement is needed, the query is divided into new subtasks, and the external search process is repeated.

- **Human-in-the-Loop**: As with correct queries, after generating the final response, user feedback is gathered. If the user is satisfied, the process is complete; if not, the user is asked to refine the query and provide additional context.

## 5. Final Steps and User Interaction (Human in the Loop)

- **Feedback Loop**: At each stage, users can provide feedback on the system’s response. If the response is deemed unsatisfactory, users can refine their queries and add more context to guide the system toward a better answer.

- **Query Resolution**: Once the system provides an answer that satisfies the user, the query is considered resolved, and the system is ready to handle the next query.

## 6. Adaptive RAG Question Answering System:
Adaptive RAG dynamically adjusts the number of supporting documents in prompts based on question difficulty and model feedback. Starting with a minimal context, it expands geometrically (e.g., doubling documents) if the model cannot confidently answer, ensuring cost-efficiency and low latency. Overlapping prompts retain previously used relevant documents, maintaining accuracy while progressively refining context for harder questions.

---

This workflow ensures that queries are processed efficiently, with a combination of advanced AI techniques and fallback mechanisms to ensure accuracy and user satisfaction. Each component, from document retrieval to external web searches and human feedback, plays a vital role in delivering the best possible answers.





---

### Key Processes Summary

1. **Document Indexing & Embedding**: Documents are parsed, embedded using a pre-trained model, and stored in a vector store for fast similarity-based retrieval.

2. **Query Reception & Guardrail**: The system scans incoming queries for inappropriate content, then retrieves the top 3 relevant documents from the vector store.

3. **Query Classification**: The query is classified as correct (can be answered with existing documents) or incorrect (requires additional context).

4. **Leader-Analyst Process (Correct Queries)**: 
   - For correct queries, the Leader divides the query into subtasks for two Analysts, who process and return responses. The Leader unifies their answers.
   - If follow-up is needed, the process repeats. User feedback ensures the response's accuracy.

5. **Leader-Analyst Process (Incorrect Queries)**: 
   - For incorrect queries, the Leader divides the query and gathers additional context via web search (using SERPER or SERP API). Responses are unified by the Leader.
   - Follow-ups may be generated, with user feedback guiding the system.

6. **Human-in-the-Loop**: User feedback is gathered to ensure accuracy. If not satisfied, users refine the query, and the process restarts. 

This process ensures efficient, accurate, and adaptable query handling.

### Overview

This system processes user queries by retrieving relevant documents, classifying them as correct or incorrect, and utilizing a Leader-Analyst approach to generate answers. If needed, external web searches are conducted to gather more context, with user feedback ensuring accurate and refined responses.

### Installation

1. Install Required Packages

    - ```pip install -r requirements.txt```


2. Set Up Environment Variables: 

    - Define your API keys in ```main.py``` during ```key``` class initialisation and set environment variables for SERP_API_KEY, SERPER_API_KEY,GEMINI_API_KEY, OPENAI_API_KEY, and LICENSE_KEY.

3. SERP_API Setup

    - If there are some dependency issues while using ```GoogleSearch``` from ```serpapi```, unzip the google_search_results-2.4.2.zip file and work in that directory

### Usage

1. Configure API Keys in main.py

2. Data Directory

   - Add the required pdf file in the data directory (Context PDF)

3. Run the Application:

   - If you are using docker then run the following :

        - ```docker build -t my-pathway-app .```
        - ```docker run -it --rm -p 9000:9000 -v //c/Users/ADMIN/Downloads/ps_3_pathway_3/ps_3_pathway/data:/app/data --name my-pathway-app my-pathway-app```
     
4. Query Input:

   - Type questions in the command-line interface.
   - Type exit to stop the server.
   - Type Yes if you are satisfied with the response, else type No and enter a refined query with some additional context

## Components

### 1. Scraper.py

Here is a detailed description of the key components in the provided Python script:

---

#### **1. `ContentScraper` Class**
This class is designed to scrape web content, perform Google searches using the SERP API, and retrieve stock price data.

- **Attributes**:
  - `serp_api_key`: API key used to authenticate requests to the SERP API.

- **Methods**:
  - **`__init__(self, serp_api_key)`**: Initializes the `ContentScraper` class with a given SERP API key.
  - **`scrape_content(self, url)`**: Scrapes content from the specified URL, extracts textual information, and returns the first 800 characters of content.
  - **`search_google(self, query)`**: Searches Google using the SERP API for the given query and extracts relevant sources and AI overview context from the search results.
  - **`get_content_from_urls(self, source_description_list)`**: Retrieves and compiles content from a list of source URLs, calling `scrape_content` to get content from each URL.
  - **`get_stock_price(self, query)`**: Retrieves stock price information if available through the SERP API's answer box, and formats the result into a statement.

#### **2. `GoogleSerperAPI` Class**
This class provides an interface to interact with the Serper.dev API to perform Google searches and retrieve search results.

- **Attributes**:
  - `api_key`: The API key to authenticate requests to the Serper.dev API.
  - `k`: The number of search results to retrieve.
  - `gl`: Geolocation of the search results.
  - `hl`: Language of the search results.
  - `search_type`: Type of search (e.g., "search", "images").
  - `initialised`: A flag indicating if the class is initialized with an API key.

- **Methods**:
  - **`__init__(self, api_key, k, gl, hl, search_type)`**: Initializes the `GoogleSerperAPI` class with the provided configuration. Ensures the API key is provided.
  - **`_make_request(self, search_term, **kwargs)`**: Makes a synchronous HTTP request to the Serper API to fetch search results for a given query.
  - **`_make_async_request(self, search_term, **kwargs)`**: Makes an asynchronous HTTP request to the Serper API for search results.
  - **`get_results(self, query, **kwargs)`**: Retrieves search results synchronously for the given query.
  - **`get_async_results(self, query, **kwargs)`**: Retrieves search results asynchronously for the given query.
  - **`parse_snippets(self, results)`**: Extracts and returns search result snippets from the JSON response.
  - **`search(self, query, **kwargs)`**: Performs a synchronous search and returns concatenated search result snippets.
  - **`async_search(self, query, **kwargs)`**: Performs an asynchronous search and returns concatenated search result snippets.

  
#### **Main Use Cases**:
1. **Web Scraping**: `ContentScraper` can extract and return the text content from a given URL.
2. **Google Search**: `GoogleSerperAPI` helps perform Google searches via the SERPER API and parse relevant search results.
3. **Stock Price Retrieval**: `ContentScraper` also fetches stock price information from the Google search results.

#### **Summary of Process**:
- The script first uses the `ContentScraper` class to scrape content from web pages and search Google using the Serper API.
- It organizes and formats the retrieved content and stock price data.
- Through the `GoogleSerperAPI`, it allows both synchronous and asynchronous search operations to fetch and parse relevant search snippets, which can be used to enrich responses or provide context to users.

### 2. llm.py
Here is a detailed description of the `llm.py` script:

#### **1. `OpenAIClient` Class**
This class provides an interface to interact with OpenAI's API, specifically to get completions using the GPT model.

- **Attributes**:
  - `api_key`: API key used to authenticate requests to OpenAI's API.
  - `model`: The model used for generating completions (defaults to `"gpt-4o"`).
  - `client`: An instance of OpenAI client initialized with the provided API key.

- **Methods**:
  - **`__init__(self, api_key, model="gpt-4o")`**: Initializes the `OpenAIClient` class with the API key and optionally accepts a model name (default is `"gpt-4o"`). The `OpenAI` client instance is created with the provided API key.
  - **`get_completion(self, prompt)`**: Sends a prompt to the OpenAI API to generate a completion using the specified model (`gpt-4o`). The response is parsed to return the content of the completion from the first choice provided by the model.

#### **Main Libraries Used**:
- **`openai`**: This is the official Python client for OpenAI's API, which enables interaction with OpenAI models like GPT for text generation tasks.

#### **Key Features**:
- **Initialization**: The `OpenAIClient` is initialized with an API key and model name (default is GPT-4).
- **Completion Request**: The `get_completion` method sends a prompt to the GPT model and returns the content of the generated completion.
  
#### **Summary of Process**:
- The script defines a client class (`OpenAIClient`) that interfaces with OpenAI's API to generate completions based on a given prompt. The class sends a chat completion request to the API, using the specified model (default is GPT-4), and returns the model's response as text.

### 3. Guardrail.py

Here’s a detailed breakdown of the `guardrail.py` script:

#### **1. `GuardrailChecker` Class**
The `GuardrailChecker` class is designed to ensure that user queries comply with company communication policies. It uses OpenAI's GPT models to evaluate and check whether a given message adheres to predefined guidelines.

- **Attributes**:
  - `openai.api_key`: The OpenAI API key used to authenticate requests to OpenAI’s services.
  - `api_key`: Stores the OpenAI API key.
  - `client`: An instance of OpenAI’s API client, initialized with the API key.
  - `model`: The OpenAI model (default is `"gpt-4"`).
  - `guardrail_system_message`: A system message that specifies the company's communication policies, which include rules against harmful content, impersonation, extraction of system prompts, sharing sensitive information, and nonsensical language.

- **Methods**:
  - **`__init__(self, openai_api_key: str, model="gpt-4")`**: Initializes the `GuardrailChecker` with an OpenAI API key and the model name (defaults to `"gpt-4"`). The guardrail system message is also set, outlining the company’s communication policies.
  - **`check_compliance(self, question: str) -> str`**: Sends the user’s query to the OpenAI API, requesting an evaluation against the company's communication policies. Returns `'yes'` if the query complies, and `'no'` if it violates any policy.
  - **`generate_response(self, question: str) -> str`**: If a query violates company policies, this method generates a response to the query using the OpenAI model, indicating how to handle the violation.
  - **`guardrail_check(self, question: str) -> Dict[str, Union[str, None]]`**: Combines the compliance check and the response generation steps. If the query violates policies, it generates a response; otherwise, it confirms the query is fine.
  - **`decide_guardrail(self, state: Dict[str, Any]) -> str`**: Makes a decision based on the guardrail check result. If the query is inappropriate, it returns `'Inappropriate query'`; otherwise, it returns `'fine'`.

#### **Key Libraries Used**:
- **`openai`**: This is the official Python client for OpenAI’s API, which is used to interact with the GPT models for generating completions and evaluating compliance.
- **`typing`**: Used to specify the expected types for function arguments and return values (e.g., `Dict`, `Union`).

#### **Main Processes**:
1. **Compliance Check**: The `check_compliance` method evaluates the user's query against a predefined set of communication policies. It uses the OpenAI API to check if the query violates any rules.
2. **Response Generation**: If a query violates the policies, the `generate_response` method generates a response that explains why the query is inappropriate.
3. **Guardrail Check**: The `guardrail_check` method first checks if the query complies with the policies. If not, it generates a response. If the query complies, it simply prints a message saying the query is fine.
4. **Decision Making**: The `decide_guardrail` method uses the result of the guardrail check to decide if the query is acceptable or inappropriate.

#### **Summary of Process**:
- This script ensures that user queries are evaluated for compliance with company policies. If a query violates any policies, the system generates an appropriate response explaining the violation. If the query is fine, the system confirms it. The main evaluation and response generation are powered by OpenAI's GPT model.

### 4. grade.py
Here’s a detailed breakdown of the `grade.py` script:

#### **1. `grade_doc` Class**
The `grade_doc` class is responsible for grading the relevance of a retrieved document to a user query using OpenAI's API. It uses the GPT model to assess whether the content of the document is related to the user's question.

- **Attributes**:
  - `openai_api_key`: The API key used to authenticate the OpenAI API requests.
  - `model`: The name of the OpenAI model used for grading (defaults to `"gpt-4"`).
  - `grade_msg`: A fixed instruction message that provides the model with the criteria for grading the relevance of a document to a query. It instructs the model to give a binary score (`"yes"` or `"no"`) based on whether the document is relevant to the user’s question.

- **Methods**:
  - **`__init__(self, openai_api_key: str, model="gpt-4")`**: Initializes the `grade_doc` class with the OpenAI API key and the model name (defaults to `"gpt-4"`). It sets the API key for OpenAI, initializes the client, stores the model name, and defines the grading instruction message.
  - **`grade_document(self, query: str, context: str) -> str`**: This method takes the user’s query and the retrieved document (context) and sends them to the OpenAI API to grade the document's relevance. The response is a binary score (`"yes"` or `"no"`) indicating whether the document is relevant to the query.

#### **Key Libraries Used**:
- **`openai`**: This is the official Python client for OpenAI’s API, used to interact with GPT models for generating completions and assessing document relevance.

#### **Main Process**:
1. **Grading the Document**:
   - The `grade_document` method combines the user’s query and the retrieved document into a prompt and sends it to the OpenAI API.
   - The system message (`grade_msg`) provides instructions on how to grade the relevance of the document. The GPT model evaluates whether the document contains keywords or semantic meaning related to the query.
   - Based on this evaluation, the model returns a binary score: `"yes"` if the document is relevant and `"no"` if it is not.

2. **API Call**:
   - The method constructs a prompt consisting of the user’s query and the document, then sends it to the OpenAI API with the grading instructions.
   - The response is parsed, and the binary score is extracted to indicate whether the document is relevant to the question.

#### **Summary of Process**:
- This script helps in filtering out irrelevant documents in a retrieval process by using OpenAI’s GPT model to evaluate the relevance of a document in relation to a user’s query. It provides a simple binary score (`"yes"` or `"no"`) to indicate the relevance, which can be used for improving the quality of search results or document retrieval in a system.

### 5. conversational_agent.py
This script defines a `ConversationalPipeline` class that orchestrates a multi-step query processing pipeline using OpenAI's GPT models. 
It integrates multiple subtasks, including query analysis, task division, response unification, and follow-up handling, to generate accurate and informative responses.
- **Attributes**:
  - `openai_api_key`: The API key used to authenticate the OpenAI API requests.
  - `model`: The name of the OpenAI model used for grading (defaults to `"gpt-4"`).


- **Methods**:
    - __init__(self, openai_api_key, model="gpt-4o"): Initializes the ConversationalPipeline with the API key and model.
    - call_openai(prompt, model="gpt-4"): Sends a prompt to the OpenAI API and returns the response.
    - analyst_task(query, context): Analyzes a query using the provided document context.
    - leader_task(response_1, response_2, query, context): Unifies responses from two analysts into a single final response.
    - check_follow_up(query, context, final_response): Determines if a follow-up is needed based on the final response.
    - divide_correct_task_into_subtasks(query, context): Divides a correct query into two distinct subtasks.
    - divide_incorrect_task_into_subtasks(query): Divides an incorrect query into subtasks for further analysis.
    - generate_new_subtasks(query, subtask_1, subtask_2, context): Generates two new subtasks based on the existing ones.
    - run_pipeline(query, context_a, context_b, subtask_1, subtask_2): Orchestrates responses from two Analysts for distinct subtasks and unifies them using a Leader task.
    - final_unification_task(combined_response, response_3, response_4, query, context): Unifies responses from multiple analysts into one coherent response.
    - run_pipeline_if_needed(query, context_c, context_d, subtask_3, subtask_4, final_response, context): Executes additional pipeline tasks if needed, handling Subtask 3 and Subtask 4 and unifying their responses.
 
  - **Usage**
    1. Instantiate the `ConversationalPipeline` class with the API key.
    2. Use the methods to process user queries by analyzing the query, dividing tasks, unifying responses, and handling follow-ups as needed.


### 6. config.py
   - Sets up and configures API keys for SERP API, Gemini API, OpenAI, and Pathway licenses.

#### 7. main.py
   - Launches the server, manages user interactions, performs searches, and generates responses based on query relevance.

### Initial Metrics : 
Performance metrics were evaluated using the Opik library, with an LLM-based judge assessing hallucination and answer relevance.
The opik library leverages Large Language Models (LLMs) as judge evaluation metrics to assess the quality of generated text. It specifically focuses on evaluating two key aspects: hallucination, ensuring the generated answers are factually accurate, and answer relevance, ensuring the responses are contextually appropriate. By using LLMs to automate and scale the evaluation of these factors, opik enables consistent and effective monitoring of LLM performance in tasks like question answering and dialogue systems.
An accompanying Jupyter notebook (`LLM_Judge.ipynb`) has been added for further reference.
#### Performance of the model:
| Metric      | Average Relevance Score | Average Hallucination Score |
|-------------|-------------------------|-----------------------------|
| LLM_Judge   | 0.839                   |0                            |

The hallucination score and relevance score of each query is given in `Performance_Metrics.xlsx`. The context used for judging these queries is given in `data` directory.

