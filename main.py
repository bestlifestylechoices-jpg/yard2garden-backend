from __future__ import annotations

import os
import json
import base64
from typing import Any, Dict, Optional, Literal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI


# ---------- App ----------
app = FastAPI(title="Yard2Garden AI Planner", version="0.1.0")

# CORS: Swagger UI is served from same origin, but this also helps future web clients.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later, replace with your real frontend domain(s)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- OpenAI Client ----------
# Cloud Run should provide OPENAI_API_KEY via env var (recommended).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # Don't crash the whole app; just error when endpoint is used.
    # This keeps /health alive for debugging deployments.
    client: Optional[OpenAI] = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)


# ---------- Helpers ----------
def _safe_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _clean_model_json(text: str) -> Dict[str, Any]:
    """
    The model sometimes returns:
      ```json
      {...}
      ```
    or plain JSON.
    This extracts and parses JSON robustly.
    """
    if not text:
        raise ValueError("Empty model output")

    cleaned = text.strip()

    # Remove code fences if present
    if cleaned.startswith("```"):
        # Strip leading ```json or ``` and trailing ```
        cleaned = cleaned.strip("`")
        # After stripping backticks, it might still start with 'json'
        cleaned = cleaned.lstrip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()

    # Now parse JSON
    return json.loads(cleaned)


# ---------- Routes ----------
@app.get("/")
def root():
    return {"status": "yard2garden-backend is alive 🌱"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze-yard")
async def analyze_yard(
    file: UploadFile = File(...),
    zip_code: str = Form(..., description="US ZIP code, e.g. 97401"),
    budget: int = Form(..., description="Budget in USD (integer), e.g. 200"),
    upkeep: Literal["low", "medium", "high"] = Form(..., description="Ongoing maintenance level"),
):
    # Basic validation
    if not zip_code or len(zip_code.strip()) < 5:
        raise HTTPException(status_code=422, detail="zip_code must be at least 5 characters (e.g., 97401).")

    if budget <= 0:
        raise HTTPException(status_code=422, detail="budget must be a positive integer.")

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured on the server (Cloud Run env var missing).",
        )

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=422, detail="Uploaded file is empty.")

        # Keep payloads reasonable (helps avoid timeouts/cost spikes)
        # 8MB is a safe first cap; adjust later if needed.
        if len(image_bytes) > 8 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image too large. Please upload an image under 8MB.")

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Ask for strict JSON output (no markdown, no code fences)
        prompt = f"""
You are Yard2Garden AI Planner.

Analyze the yard image and produce a planting plan that fits:
- zip_code: {zip_code}
- budget_usd: {budget}
- upkeep_level: {upkeep}

Return ONLY valid JSON (no markdown, no backticks, no extra commentary).
The JSON MUST follow this schema:

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
  "recommended_zones": [
    {{
      "zone_name": string,
      "purpose": string,
      "plants": [string],
      "layout_notes": string
    }}
  ],
  "step_by_step_plan": [string],
  "shopping_list": [
    {{
      "item": string,
      "quantity": string,
      "estimated_cost_usd": number
    }}
  ],
  "estimated_total_cost_usd": number,
  "maintenance_plan": {{
    "weekly_minutes": number,
    "watering_guidance": string,
    "weeding_guidance": string,
    "seasonal_tasks": [string]
  }}
}}

Budget rule:
- Keep estimated_total_cost_usd <= budget_usd.
- Use realistic low-cost choices when upkeep_level is "low".

If something is uncertain from the image, state it under constraints/assumptions.
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

        raw_text = response.output_text or ""
        data = _clean_model_json(raw_text)

        # Final safety: enforce assumptions reflect inputs
        data.setdefault("assumptions", {})
        data["assumptions"]["zip_code"] = zip_code
        data["assumptions"]["budget_usd"] = budget
        data["assumptions"]["upkeep_level"] = upkeep

        return {
            "filename": file.filename,
            "zip_code": zip_code,
            "budget_usd": budget,
            "upkeep": upkeep,
            "garden_plan": data,
        }

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        # Return the raw model output for debugging, but keep it short.
        raise HTTPException(
            status_code=500,
            detail=f"Model did not return valid JSON. JSONDecodeError: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
