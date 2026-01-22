from __future__ import annotations
import os, json, base64
from typing import Any, Dict, Optional, Literal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI(title="Yard2Garden AI Planner", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

@app.get("/")
def root():
    return {"status": "yard2garden-backend is alive 🌱"}

@app.get("/health")
def health():
    return {"ok": True}

def clean_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return json.loads(text)

@app.post("/analyze-yard")
async def analyze_yard(
    file: UploadFile = File(...),
    zip_code: str = Form(...),
    budget: int = Form(...),
    upkeep: Literal["low","medium","high"] = Form(...)
):
    if not client:
        raise HTTPException(500, "OpenAI key missing")

    image = await file.read()
    if not image:
        raise HTTPException(422, "Empty image")

    if len(image) > 8*1024*1024:
        raise HTTPException(413, "Image too large")

    image_b64 = base64.b64encode(image).decode()

    prompt = f"""
You are Yard2Garden AI Planner.

zip_code={zip_code}
budget_usd={budget}
upkeep={upkeep}

Return ONLY valid JSON in this schema:

{{
  "summary": string,
  "assumptions": {{
    "zip_code": string,
    "budget_usd": number,
    "upkeep_level": "low"|"medium"|"high"
  }},
  "site_observations": {{
    "sun_exposure": string,
    "space_notes": string,
    "soil_notes": string,
    "constraints": [string]
  }},
  "zones": [
    {{
      "name": string,
      "purpose": string,
      "plants": [string],
      "layout_notes": string
    }}
  ],
  "steps": [string],
  "shopping_list": [
    {{
      "item": string,
      "quantity": string,
      "cost_usd": number
    }}
  ],
  "total_cost_usd": number,
  "maintenance": {{
    "weekly_minutes": number,
    "watering": string,
    "weeding": string,
    "seasonal": [string]
  }}
}}

Total cost must not exceed budget_usd.
"""

    response = client.responses.create(
        model="gpt-5.2",
        input=[{
            "role": "user",
            "content": [
                {"type":"input_text","text":prompt},
                {"type":"input_image","image_base64":image_b64}
            ]
        }]
    )

    data = clean_json(response.output_text)

    data["assumptions"]["zip_code"] = zip_code
    data["assumptions"]["budget_usd"] = budget
    data["assumptions"]["upkeep_level"] = upkeep

    return {
        "filename": file.filename,
        "inputs": {
            "zip_code": zip_code,
            "budget": budget,
            "upkeep": upkeep
        },
        "garden_plan": data
    }
