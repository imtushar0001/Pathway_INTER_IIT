# file_upload.py
from fastapi import APIRouter, File, UploadFile, HTTPException
import shutil
import os

router = APIRouter()
UPLOAD_DIR = "/app/data"

os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/api/v1/users/uploadDocument")
def upload_document(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"filename": file.filename, "status": "uploaded"}
    except Exception:
        raise HTTPException(status_code=500, detail="Something went wrong")
    finally:
        file.file.close()