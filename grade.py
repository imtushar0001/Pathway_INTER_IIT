import openai
from openai import OpenAI

class grade_doc:
  def __init__(self, openai_api_key: str, model="gpt-4"):
        openai.api_key = openai_api_key
        self.api_key = openai_api_key
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.grade_msg = f"""You are a grader assessing relevance of a retrieved document to a user question. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

  def grade_document(self, query, context):
        prompt = f"""
        User Question : {query}
        Document : {context}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.grade_msg},
                {"role": "user", "content": prompt}
            ]
        )
        score = response.choices[0].message.content.strip().lower()
        return score

