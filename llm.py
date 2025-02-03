import openai
from openai import OpenAI

class OpenAIClient:
    def __init__(self, api_key, model="gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def get_completion(self, prompt):
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model,
        )
        return response.choices[0].message.content