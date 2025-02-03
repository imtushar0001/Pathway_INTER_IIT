import requests
from bs4 import BeautifulSoup
import os
from serpapi.google_search import GoogleSearch as search
import os
import aiohttp
from typing import Any, Dict, List, Optional

class ContentScraper:
    def __init__(self, serp_api_key):
        self.serp_api_key = serp_api_key

    def scrape_content(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            content = ' '.join(paragraph.text for paragraph in paragraphs)
            return content[:800]
        except requests.RequestException:
            return None

    def search_google(self, query):
        params = {
            "engine": "google_finance",
            "q": query,
            "api_key": self.serp_api_key
        }

        store = search(params).get_dict()

        source_description_list = []

        if 'knowledge_graph' in store and store['knowledge_graph'] is not None:
            new_dict = {
                "source": store["knowledge_graph"].get("source", ""),
                "description": store["knowledge_graph"].get("description", "")
            }
            source_description_list.append(new_dict)

        if 'related_questions' in store and store['related_questions']:
            for question in store['related_questions']:
                source_description_list.append({
                    'source': question.get('link', ''),
                    'description': question.get('snippet', '')
                })

        ai_overview_context = []
        if 'ai_overview' in store and 'text_blocks' in store['ai_overview']:
          for block in store["ai_overview"]["text_blocks"]:
              # Check if 'snippet' is in the block
              if block.get("snippet"):
                  ai_overview_context.append(block["snippet"])

              # If 'list' is in the block, iterate through its items
              if block.get("list"):
                  for item in block["list"]:
                      if item.get("snippet"):
                          ai_overview_context.append(item["snippet"])

        return source_description_list, ai_overview_context

    def get_content_from_urls(self, source_description_list):
        urls = [item["source"] for item in source_description_list]
        all_content = []
        context = []

        for url in urls:
            content = self.scrape_content(url)
            if content:
                all_content.append({"url": url, "content": content})
                context.append(content)

        return all_content, context
    
    def get_stock_price(self, query):
        """
        Gets stock price information if present in the store dictionary.
        Returns a list containing a formatted statement with stock price information.
        """
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serp_api_key
        }

        store = search(params).get_dict()
        stock_info = []
        if "answer_box" in store and store["answer_box"]:
            answer_box = store["answer_box"]
            if "list" in answer_box:
                stock_info.extend(answer_box["list"])
                

            # Ensure 'price' exists, then retrieve other optional fields if available
            if "price" in answer_box:
                price = answer_box["price"]
                stock = answer_box.get("stock", "Stock")
                currency = answer_box.get("currency", "")
                exchange = answer_box.get("exchange", "an Exchange")

                stock_info_more = [
                    f"According to {exchange}, the stock price is {currency} {price} for {stock}."
                ]
                stock_info.extend(stock_info_more)

        # Return empty list if stock price information is not found
        return stock_info




class GoogleSerperAPI:
    def __init__(self, api_key: Optional[str] = None, k: int = 10, gl: str = "us", hl: str = "en", search_type: str = "search"):
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise ValueError("API key for Serper.dev is required.")
        self.k = k
        self.gl = gl
        self.hl = hl
        self.search_type = search_type
        self.initialised = True

    def _make_request(self, search_term: str, **kwargs: Any) -> Dict:
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        url = f"https://google.serper.dev/{self.search_type}"
        params = {
            "q": search_term,
            "gl": self.gl,
            "hl": self.hl,
            "num": self.k,
            **kwargs,
        }
        response = requests.post(url, headers=headers, json=params)
        response.raise_for_status()
        return response.json()

    async def _make_async_request(self, search_term: str, **kwargs: Any) -> Dict:
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        url = f"https://google.serper.dev/{self.search_type}"
        params = {
            "q": search_term,
            "gl": self.gl,
            "hl": self.hl,
            "num": self.k,
            **kwargs,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=params, headers=headers) as response:
                response.raise_for_status()
                return await response.json()

    def get_results(self, query: str, **kwargs: Any) -> Dict:
        return self._make_request(query, **kwargs)

    async def get_async_results(self, query: str, **kwargs: Any) -> Dict:
        return await self._make_async_request(query, **kwargs)

    def parse_snippets(self, results: Dict) -> List[str]:
        snippets = []
        if "organic" in results:
            for item in results["organic"][:self.k]:
                if "snippet" in item:
                    snippets.append(item["snippet"])
        return snippets or ["No good results found."]

    def search(self, query: str, **kwargs: Any) -> str:
        results = self.get_results(query, **kwargs)
        return " ".join(self.parse_snippets(results))

    async def async_search(self, query: str, **kwargs: Any) -> str:
        results = await self.get_async_results(query, **kwargs)
        return " ".join(self.parse_snippets(results))
