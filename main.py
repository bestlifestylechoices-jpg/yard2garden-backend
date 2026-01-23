from __future__ import annotations

import os
import json
import base64
from typing import Any, Dict, Optional, Literal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI


# ---------- App ----------
app = FastAPI(title="Yard2Garden AI Planner", version="0.2.0")

# CORS (works for Swagger + future web clients). In production, replace "*" with your real domain(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- OpenAI Client ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ---------- Helpers ----------
def _strip_code_fences(text: str) -> str:
    """
    Removes ```json ... ``` fences if the model includes them.
    """
    t = (text or "").strip()
    if t.startswith("```"):
        # remove starting fence line
        t = t.split("\n", 1)[-1] if "\n" in t else ""
        # remove trailing fence
        if t.endswith("```"):
            t = t[:-3].strip()
        # sometimes it starts with "json"
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return t.strip()


def _parse_model_json(text: str) -> Dict[str, Any]:
    cleaned = _strip_code_fences(text)
    if not cleaned:
        raise ValueError("Empty model output")
    return json.loads(cleaned)


def _validate_lat_lon(lat: float, lon: float) -> None:
    if not (-90.0 <= lat <= 90.0):
        raise HTTPException(status_code=422, detail="latitude must be between -90 and 90")
    if not (-180.0 <= lon <= 180.0):
        raise HTTPException(status_code=422, detail="longitude must be between -180 and 180")


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

    # Location (Unity sends these after permission is granted)
    latitude: float = Form(..., description="Device latitude, e.g. 44.0521"),
    longitude: float = Form(..., description="Device longitude, e.g. -123.0868"),
    accuracy_m: Optional[float] = Form(None, description="Optional: GPS accuracy in meters"),

    # Sliders (send as strings from Unity; keep simple)
    budget: Literal["low", "medium", "high"] = Form(..., description="Budget tier slider"),
    upkeep: Literal["low", "medium", "high"] = Form(..., description="Upkeep tier slider"),
):
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured on the server (Cloud Run env var missing).",
        )

    _validate_lat_lon(latitude, longitude)

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=422, detail="Uploaded file is empty.")

        # Cap to avoid huge payloads/timeouts/cost spikes
        if len(image_bytes) > 8 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image too large. Please upload an image under 8MB.")

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Budget mapping (for the model to target realistic totals)
        budget_caps = {"low": 150, "medium": 350, "high": 800}
        budget_cap_usd = budget_caps[budget]

        # Prompt: strict JSON only
        prompt = f"""
You are Yard2Garden AI Planner.

Goal:
- Analyze the yard image and propose a realistic food-garden plan.
- Tailor the plan to the user's approximate location (lat/lon) for climate/season assumptions.
- Respect budget and upkeep tiers.

Inputs (DO NOT store these; use only for this response):
- latitude: {latitude}
- longitude: {longitude}
- accuracy_m: {accuracy_m if accuracy_m is not None else "unknown"}
- budget_tier: {budget} (max estimated total: ${budget_cap_usd})
- upkeep_tier: {upkeep}

Return ONLY valid JSON (no markdown, no backticks, no extra text).
JSON schema (must match exactly):

{{
  "summary": string,
  "inputs": {{
    "latitude": number,
    "longitude": number,
    "accuracy_m": number | null,
    "budget_tier": "low"|"medium"|"high",
    "upkeep_tier": "low"|"medium"|"high",
    "budget_cap_usd": number
  }},
  "location_inference": {{
    "assumed_region": string,
    "assumed_climate_notes": string,
    "assumed_seasonality": string
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
  "yard_transformation_prompt": {{
    "style": string,
    "camera_view": string,
    "must_include": [string],
    "avoid": [string]
  }},
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
- estimated_total_cost_usd MUST be <= budget_cap_usd
- If upkeep_tier is "low", prioritize perennials, mulch, drip/soaker, and fewer high-maintenance crops.
- If something can’t be determined from the photo, put it in site_observations.constraints.
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
        plan = _parse_model_json(raw_text)

        # Enforce echoing inputs (no surprises)
        plan.setdefault("inputs", {})
        plan["inputs"]["latitude"] = latitude
        plan["inputs"]["longitude"] = longitude
        plan["inputs"]["accuracy_m"] = accuracy_m
        plan["inputs"]["budget_tier"] = budget
        plan["inputs"]["upkeep_tier"] = upkeep
        plan["inputs"]["budget_cap_usd"] = budget_cap_usd

        return {
            "filename": file.filename,
            "garden_plan": plan,
        }

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model did not return valid JSON. JSONDecodeError: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
