from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import base64
import os
import base64

app = FastAPI(title="Yard2Garden AI Planner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()

@app.get("/")
def root():
    return {"status": "yard2garden-backend is alive 🌱"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/analyze-yard")
async def analyze_yard(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
image_b64 = base64.b64encode(image_bytes).decode("utf-8")

response = client.responses.create(
    model="gpt-5.2",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Analyze this yard and generate a planting plan."},
            {"type": "input_image", "image_base64": image_b64}
        ]
    }]
)

result = response.output_text
return {"result": result}
 {
            "filename": file.filename,
            "garden_plan": response.output_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
