from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import base64
import os
import json

app = FastAPI(title="Yard2Garden AI Planner")

# CORS (safe for testing; later restrict origins to your Unity app domain/app scheme)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client (expects OPENAI_API_KEY to be set in Cloud Run env vars / secret)
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
        # Read uploaded image
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Prompt forcing strict JSON output
        system_prompt = (
            "You are a garden planning AI. "
            "You MUST return ONLY valid JSON. "
            "No markdown. No code fences. No commentary. No extra text."
        )

        user_prompt = """
Analyze the yard photo and return ONLY valid JSON using this exact schema:

{
  "summary": string,
  "sun_exposure": string,
  "recommended_zones": [
    {
      "zone_name": string,
      "plants": [string, ...],
      "notes": string
    }
  ],
  "step_by_step_plan": [string, ...],
  "shopping_list": [
    { "item": string, "quantity": string }
  ]
}

Rules:
- Output MUST be valid JSON.
- Do not wrap in ``` fences.
- Do not include any text before or after the JSON.
"""

        # OpenAI Responses API call (vision)
        response = client.responses.create(
            model="gpt-5.2",
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                        {"type": "input_image", "image_base64": image_b64},
                    ],
                },
            ],
        )

        raw_text = response.output_text or ""

        # Attempt to parse JSON strictly
        try:
            plan = json.loads(raw_text)
        except json.JSONDecodeError:
            # Try a small cleanup in case of accidental leading/trailing whitespace
            cleaned = raw_text.strip()
            try:
                plan = json.loads(cleaned)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "error": "Model returned non-JSON output.",
                        "raw_output_preview": cleaned[:500],
                    },
                )

        return {
            "filename": file.filename,
            "plan": plan
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
