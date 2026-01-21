from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
import base64

app = FastAPI(title="Yard2Garden AI Planner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        response = client.responses.create(
            model="gpt-5.2",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Analyze this yard and create a complete food garden plan with sun zones, planting layout, and step-by-step instructions."},
                    {"type": "input_image", "image_base64": image_base64}
                ]
            }]
        )

        return {
            "filename": file.filename,
            "garden_plan": response.output_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
