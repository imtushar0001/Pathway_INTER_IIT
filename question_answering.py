# question_answering.py
import os
import pathway as pw
from pathway.xpacks.llm import llms, embedders, prompts, parsers, splitters
from pathway.xpacks.llm.vector_store import VectorStoreServer
from pathway.xpacks.llm.question_answering import RAGClient, AdaptiveRAGQuestionAnswerer
from pathway.udfs import ExponentialBackoffRetryStrategy, DiskCache
from guardrail import GuardrailChecker
from grade import grade_doc
from conversational_agent import ConversationalPipeline
from scraper import ContentScraper, GoogleSerperAPI
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import threading
import time
import google.generativeai as genai  # Add this import

router = APIRouter()

# Configuration
SERP_API_KEY = "Enter your Serp API Key"
GEMINI_API_KEY = "Enter your Gemini API Key"
OPENAI_API_KEY = "Enter your OpenAI API Key"
SERPER_API_KEY = "Enter your Serper API Key"

os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract/tessdata/"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

pw.set_license_key("Enter your Pathway License Key")
genai.configure(api_key=GEMINI_API_KEY)

# Setup Pathway components
folder = pw.io.fs.read(path="./data/", format="binary", with_metadata=True)
text_splitter = splitters.TokenCountSplitter(max_tokens=400)
embedder = embedders.OpenAIEmbedder(cache_strategy=DiskCache())
sources = [folder]

chat = llms.OpenAIChat(
    model="gpt-4o",
    retry_strategy=ExponentialBackoffRetryStrategy(max_retries=6),
    cache_strategy=DiskCache(),
    temperature=0.05,
)

table_args = {
    "parsing_algorithm": "llm",
    "llm": chat,
    "prompt": prompts.DEFAULT_MD_TABLE_PARSE_PROMPT,
}
parser = parsers.OpenParse(table_args=table_args)

doc_store = VectorStoreServer(
    *sources,
    embedder=embedder,
    splitter=text_splitter,
    parser=parser
)

app_rag = AdaptiveRAGQuestionAnswerer(
    llm=chat,
    indexer=doc_store,
)

app_host = "0.0.0.0"
app_port = 8000

def start_server():
    app_rag.build_server(host=app_host, port=app_port)
    app_rag.run_server()

server_thread = threading.Thread(target=start_server, name="AdaptiveRAGQuestionAnswerer")
server_thread.daemon = True
server_thread.start()
time.sleep(2)

client = RAGClient(host=app_host, port=app_port)

class QueryRequest(BaseModel):
    question: str

@router.post("/api/v1/users")
async def ask_questions(request: QueryRequest):
    question = request.question
    guard = GuardrailChecker(OPENAI_API_KEY)
    grader = grade_doc(OPENAI_API_KEY)
    if question.lower() == "exit":
        return JSONResponse(content={"message": "Exiting the app."}, status_code=200)
    
    if guard.check_compliance(question) == "no":
        return JSONResponse(content={"message": "Inappropriate query " + guard.generate_response(question)}, status_code=200)
    
    docs = client.retrieve(question)
    texts = [item['text'] for item in docs]
    
    status = grader.grade_document(question, texts)
    leader_analyst = ConversationalPipeline(OPENAI_API_KEY)

    if status.lower() == "yes":
        subtask_1, subtask_2 = leader_analyst.divide_correct_task_into_subtasks(question, texts)
        context_a = client.retrieve(subtask_1)
        context_a = [item['text'] for item in context_a]
        context_b = []
        if subtask_2:
            context_b = client.retrieve(subtask_2)
            context_b = [item['text'] for item in context_b]
        final_response = leader_analyst.run_pipeline(question, context_a, context_b, subtask_1, subtask_2)
        follow_up_status = leader_analyst.check_follow_up(question, context_a + context_b, final_response)
        if follow_up_status == "Yes":
            subtask_3, subtask_4 = leader_analyst.generate_new_subtasks(question, subtask_1, subtask_2, texts)
            context_c = client.retrieve(subtask_3)
            context_c = [item['text'] for item in context_c]
            context_d = client.retrieve(subtask_4)
            context_d = [item['text'] for item in context_d]
            context = context_a + context_b
            final_response = leader_analyst.run_pipeline_if_needed(question, context_c, context_d, subtask_3, subtask_4, final_response, context)
            return JSONResponse(content={"message": final_response}, status_code=200)
        return JSONResponse(content={"message": final_response}, status_code=200)
    else:
        web_scraper = GoogleSerperAPI(SERPER_API_KEY)
        flag = 0
        if web_scraper.initialised & flag:
            subtask_1, subtask_2 = leader_analyst.divide_incorrect_task_into_subtasks(question)
            context_a = web_scraper.search(subtask_1)
            context_b = ""
            if subtask_2:
                context_b = web_scraper.search(subtask_2)
            final_response = leader_analyst.run_pipeline(question, context_a, context_b, subtask_1, subtask_2)
            context = context_a + context_b
            follow_up_status = leader_analyst.check_follow_up(question, context, final_response)
            if follow_up_status == "Yes" and subtask_2:
                subtask_3, subtask_4 = leader_analyst.generate_new_subtasks(question, subtask_1, subtask_2, texts)
                context_c = web_scraper.search(subtask_3)
                context_d = ""
                if subtask_4:
                    context_d = web_scraper.search(subtask_4)
                final_response = leader_analyst.run_pipeline_if_needed(question, context_c, context_d, subtask_3, subtask_4, final_response, context)
                return JSONResponse(content={"message": final_response}, status_code=200)
            return JSONResponse(content={"message": final_response}, status_code=200)
        else:
            web_scraper = ContentScraper(SERP_API_KEY)
            subtask_1, subtask_2 = leader_analyst.divide_incorrect_task_into_subtasks(question)
            source_description_list_a, ai_overview_context_a = web_scraper.search_google(subtask_1)
            all_content_a, context_a = web_scraper.get_content_from_urls(source_description_list_a)
            stock_info_a = web_scraper.get_stock_price(subtask_1)
            source_description_list_b, ai_overview_context_b = web_scraper.search_google(subtask_2)
            all_content_b, context_b = web_scraper.get_content_from_urls(source_description_list_b)
            stock_info_b = web_scraper.get_stock_price(subtask_2)
            context_a.extend(ai_overview_context_a)
            context_a.extend(stock_info_a)
            context_b.extend(ai_overview_context_b)
            context_b.extend(stock_info_b)
            final_response = leader_analyst.run_pipeline(question, context_a, context_b, subtask_1, subtask_2)
            context = context_a + context_b
            follow_up_status = leader_analyst.check_follow_up(question, context, final_response)
            if follow_up_status == "Yes" and subtask_2:
                subtask_3, subtask_4 = leader_analyst.generate_new_subtasks(question, subtask_1, subtask_2, texts)
                source_description_list_c, ai_overview_context_c = web_scraper.search_google(subtask_3)
                all_content_c, context_c = web_scraper.get_content_from_urls(source_description_list_c)
                stock_info_c = web_scraper.get_stock_price(subtask_3)
                source_description_list_d, ai_overview_context_d = web_scraper.search_google(subtask_4)
                all_content_d, context_d = web_scraper.get_content_from_urls(source_description_list_d)
                stock_info_d = web_scraper.get_stock_price(subtask_4)
                context_c.extend(ai_overview_context_c)
                context_c.extend(stock_info_c)
                context_d.extend(ai_overview_context_d)
                context_d.extend(stock_info_d)
                final_response = leader_analyst.run_pipeline_if_needed(question, context_c, context_d, subtask_3, subtask_4, final_response, context)
                return JSONResponse(content={"message": final_response}, status_code=200)
            return JSONResponse(content={"message": final_response}, status_code=200)
