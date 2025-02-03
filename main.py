import os
import pathway as pw
from werkzeug.utils import secure_filename
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from file_upload import router as file_upload_router
from question_answering import router as question_answering_router

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(file_upload_router)
app.include_router(question_answering_router)

pw.set_license_key("Enter your Pathway License Key")
SERP_API_KEY = "Enter your Serp API Key"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
