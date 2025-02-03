import os
import pathway as pw
import google.generativeai as genai

class key:
    def __init__(self, serp_api_key, gemini_api_key, openai_api_key, license_key):
        self.serp_api_key = serp_api_key
        self.gemini_api_key = gemini_api_key
        self.openai_api_key = openai_api_key
        pw.set_license_key(license_key)
        os.environ['GEMINI_API_KEY'] = gemini_api_key
        os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract/tessdata/"
        os.environ["OPENAI_API_KEY"] = openai_api_key
        genai.configure(api_key = gemini_api_key)