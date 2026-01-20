from fastapi import FastAPI, UploadFile, File
import os
import openai

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.get("/")
def root():
    return {"status": "yard2garden-backend is alive 🌱"}

@app.post("/analyze-yard")
async def analyze_yard(file: UploadFile = File(...)):
    # Placeholder response for now
    return {
        "message": "AI yard analysis endpoint ready",
        "filename": file.filename
    }
