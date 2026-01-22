from __future__ import annotations

import os
import json
import base64
import re
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI


# ----------------------------
# App
# ----------------------------
app = FastAPI(title="Yard2Garden AI Planner", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # later: replace with your Unity/Web domains
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# OpenAI client
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ----------------------------
# Pydantic models (clean Swagger + stable JSON for Unity)
# ----------------------------
UpkeepLevel = Literal["low", "medium", "high"]


class Assumptions(BaseModel):
    zip_code: str
    budget_usd: int
    upkeep_level: UpkeepLevel


class SiteObservations(BaseModel):
    sun_exposure: str
    space_notes: str
    soil_notes: str
    constraints: List[str] = Field(default_factory=list)


class RecommendedZone(BaseModel):
    zone_name: str
    purpose: str
    plants: List[str]
    layout_notes: str


class ShoppingItem(BaseModel):
    item: str
    quantity: str
    estimated_cost_usd: float


class MaintenancePlan(BaseModel):
    weekly_minutes: int
    watering_guidance: str
    weeding_guidance: str
    seasonal_tasks: List[str] = Field(default_factory=list)


class GardenPlan(BaseModel):
    summary: str
    assumptions: Assumptions
    site_observations: SiteObservations
    recommended_zones: List[RecommendedZone]
    step_by_step_plan: List[str]
    shopping_list: List[ShoppingItem]
    estimated_total_cost_usd: float
    maintenance_plan: MaintenancePlan


class AnalyzeYardResponse(BaseModel):
    filename: str
    zip_code: str
    budget_usd: int
    upkeep: UpkeepLevel
    garden_plan: GardenPlan


# ----------------------------
# Helpers
# ----------------------------
def _extract_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON from model output even if it includes code fences or extra text.
    Priority:
      1) JSON inside ```...```
      2) First {...} block in text
    """
    if not text or not text.strip():
        raise ValueError("Empty model output")

    s = text.strip()

    # 1) If fenced, extract inside first fence block
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return json.loads(fence.group(1))

    # 2) Otherwise extract first JSON object block
    obj = re.search(r"(\{.*\})", s, flags=re.DOTALL)
    if obj:
        return json.loads(obj.group(1))

    # If nothing looks like JSON
    raise ValueError("No JSON object found in model output")


def _require_openai_client() -> OpenAI:
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured on the server (Cloud Run env var missing).",
        )
    return client


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {"status": "yard2garden-backend is alive 🌱"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze-yard", response_model=AnalyzeYardResponse)
async def analyze_yard(
    # IMPORTANT: these MUST be Form(...) to appear as fields in Swagger for multipart/form-data
    zip_code: str = Form(..., description="US ZIP code, e.g. 97401"),
    budget: int = Form(..., description="Budget in USD (integer), e.g. 200"),
    upkeep: UpkeepLevel = Form(..., description="Maintenance level: low, medium, or high"),
    file: UploadFile = File(..., description="Yard image file"),
):
    # Validate inputs
    zip_code = (zip_code or "").strip()
    if len(zip_code) < 5:
        raise HTTPException(status_code=422, detail="zip_code must be at least 5 characters (e.g., 97401).")

    if budget <= 0:
        raise HTTPException(status_code=422, detail="budget must be a positive integer.")

    oa = _require_openai_client()

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=422, detail="Uploaded file is empty.")

        # Basic size cap (adjust later if needed)
        if len(image_bytes) > 8 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image too large. Please upload an image under 8MB.")

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Ask for strict JSON only
        prompt = f"""
You are Yard2Garden AI Planner.

Analyze the yard image and produce a planting plan that fits:
- zip_code: {zip_code}
- budget_usd: {budget}
- upkeep_level: {upkeep}

Return ONLY valid JSON. No markdown. No backticks. No commentary.

Schema:
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

Rules:
- estimated_total_cost_usd MUST be <= budget_usd.
- If something is uncertain from the image, list it in site_observations.constraints.
""".strip()

        # NOTE: Keep your model as-is if it's working in your account.
        # If you ever get "model not found", switch to a model you have enabled.
        response = oa.responses.create(
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
        data_dict = _extract_json(raw_text)

        # Force the assumptions to exactly match what the user sent
        data_dict.setdefault("assumptions", {})
        data_dict["assumptions"]["zip_code"] = zip_code
        data_dict["assumptions"]["budget_usd"] = budget
        data_dict["assumptions"]["upkeep_level"] = upkeep

        # Validate/normalize into our schema (this also catches missing fields cleanly)
        plan = GardenPlan.model_validate(data_dict)

        # Final guardrail (never exceed budget)
        if plan.estimated_total_cost_usd > budget:
            plan.estimated_total_cost_usd = float(budget)

        return AnalyzeYardResponse(
            filename=file.filename or "uploaded_image",
            zip_code=zip_code,
            budget_usd=budget,
            upkeep=upkeep,
            garden_plan=plan,
        )

    except HTTPException:
        raise
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model did not return valid JSON. Error: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
