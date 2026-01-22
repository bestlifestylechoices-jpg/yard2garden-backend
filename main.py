from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import base64
import json
import os
import re

app = FastAPI(title="Yard2Garden AI Planner")

# CORS (safe default for public API + Swagger UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client (expects OPENAI_API_KEY in Cloud Run env)
# In Cloud Run: Variables & Secrets -> add env var OPENAI_API_KEY from Secret Manager
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/")
def root():
    return {"status": "yard2garden-backend is alive 🌱"}


@app.get("/health")
def health():
    return {"ok": True}


def _extract_json(text: str) -> dict:
    """
    Tries hard to pull a JSON object from a model response.
    Accepts raw JSON, or JSON inside ``` fences, or text that contains a JSON object.
    """
    if not text:
        return {}

    # If it's already pure JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # If wrapped in ```json ... ```
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except Exception:
            pass

    # Last resort: find the first {...} block
    brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except Exception:
            pass

    # Could not parse -> return the raw text in a predictable shape
    return {"raw_text": text}


@app.post("/analyze-yard")
async def analyze_yard(
    file: UploadFile = File(...),
    zip_code: str = Form(..., description="US ZIP code (e.g., 97401)"),
    budget: float = Form(..., description="Budget in USD (e.g., 150)"),
    upkeep_level: str = Form(..., description="low / medium / high"),
):
    try:
        # Basic validation
        zip_code = zip_code.strip()
        upkeep_level = upkeep_level.strip().lower()
        if len(zip_code) < 5:
            raise HTTPException(status_code=422, detail="zip_code must be at least 5 characters.")
        if budget <= 0:
            raise HTTPException(status_code=422, detail="budget must be > 0.")
        if upkeep_level not in {"low", "medium", "high"}:
            raise HTTPException(status_code=422, detail="upkeep_level must be: low, medium, or high.")

        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=422, detail="Empty file upload.")

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        prompt = f"""
You are Yard2Garden AI Planner.
Analyze the yard photo and produce a detailed garden plan customized to:
- zip_code: {zip_code}
- budget_usd: {budget}
- upkeep_level: {upkeep_level}

Return ONLY valid JSON (no markdown, no backticks) with this schema:

{{
  "summary": string,
  "sun_exposure": string,
  "assumptions": {{
    "zip_code": string,
    "budget_usd": number,
    "upkeep_level": "low" | "medium" | "high"
  }},
  "recommended_zones": [
    {{
      "zone_name": string,
      "why_here": string,
      "plants": [string],
      "materials": [string],
      "notes": [string]
    }}
  ],
  "step_by_step_plan": [
    {{
      "step": number,
      "title": string,
      "details": [string]
    }}
  ],
  "shopping_list": [
    {{
      "item": string,
      "quantity": string,
      "estimated_cost_usd": number
    }}
  ],
  "budget_breakdown": {{
    "must_have_total_usd": number,
    "nice_to_have_total_usd": number,
    "stays_within_budget": boolean
  }},
  "upkeep_schedule": {{
    "weekly_minutes": number,
    "weekly_tasks": [string],
    "monthly_tasks": [string]
  }}
}}
""".strip()

        response = client.responses.create(
            model="gpt-5.2",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_base64": image_b64},
                    ],
                }
            ],
        )

        parsed = _extract_json(response.output_text)

        return {
            "filename": file.filename,
            "inputs": {
                "zip_code": zip_code,
                "budget_usd": budget,
                "upkeep_level": upkeep_level,
            },
            "garden_plan": parsed,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
