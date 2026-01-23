from __future__ import annotations

import os
import io
import json
import base64
import tempfile
from typing import Any, Dict, Optional, Literal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI


# ----------------------------
# App
# ----------------------------
app = FastAPI(title="Yard2Garden AI Planner", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later to your Unity/web domains if needed
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# OpenAI Client
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ----------------------------
# Helpers
# ----------------------------
def _extract_json(text: str) -> Dict[str, Any]:
    """
    Accepts either raw JSON or fenced ```json ... ``` and returns parsed dict.
    """
    if not text or not text.strip():
        raise ValueError("Empty model output")

    s = text.strip()

    # Strip fenced blocks if present
    if s.startswith("```"):
        lines = s.splitlines()
        # drop first line (``` or ```json)
        lines = lines[1:]
        # drop last line if it's ```
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    return json.loads(s)


def _validate_zip(zip_code: Optional[str]) -> Optional[str]:
    if not zip_code:
        return None
    z = zip_code.strip()
    if not z:
        return None
    # allow global formats, but keep US ZIP friendly (5 or 5-4)
    # You can tighten later if desired.
    return z


def _location_ok(lat: Optional[float], lon: Optional[float]) -> bool:
    if lat is None or lon is None:
        return False
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return False
    return True


def _budget_to_usd(budget: Literal["low", "medium", "high"]) -> Dict[str, Any]:
    """
    Keep these conservative; you can tune later.
    The plan model uses these as hard targets.
    """
    if budget == "low":
        return {"tier": "low", "target_usd": 150, "max_usd": 250}
    if budget == "medium":
        return {"tier": "medium", "target_usd": 350, "max_usd": 600}
    return {"tier": "high", "target_usd": 800, "max_usd": 1400}


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {"status": "yard2garden-backend is alive 🌱"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze-yard")
async def analyze_yard(
    file: UploadFile = File(...),

    # Optional location OR optional zip/postal code.
    # Unity can send location when allowed; otherwise user can type zip/postal.
    latitude: Optional[float] = Form(None, description="Latitude (optional)"),
    longitude: Optional[float] = Form(None, description="Longitude (optional)"),
    accuracy_m: Optional[float] = Form(None, description="Location accuracy in meters (optional)"),
    zip_code: Optional[str] = Form(None, description="ZIP/postal code (optional fallback)"),

    # Sliders (use these tiers from Unity)
    budget: Literal["low", "medium", "high"] = Form(..., description="Budget tier"),
    upkeep: Literal["low", "medium", "high"] = Form(..., description="Maintenance tier"),
):
    """
    Returns:
      - structured garden plan JSON (climate-aware)
      - a photorealistic transformed yard image (base64 PNG)
    No server-side storage: everything is in-memory; the transformed image is returned as base64.
    """

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured on the server (Cloud Run env var missing).",
        )

    # Validate image
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    # Keep payloads reasonable for mobile + Cloud Run
    if len(image_bytes) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large. Please upload an image under 8MB.")

    # Validate location/zip fallback
    zip_norm = _validate_zip(zip_code)
    has_location = _location_ok(latitude, longitude)

    if not has_location and not zip_norm:
        raise HTTPException(
            status_code=422,
            detail="Provide either (latitude & longitude) OR zip_code/postal code.",
        )

    # Budget mapping
    budget_info = _budget_to_usd(budget)

    # Base64 for vision/planning call
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # ----------------------------
    # 1) Structured plan (vision + climate-aware)
    # ----------------------------
    # IMPORTANT: We instruct the model to:
    # - be climate-aware using location if present, else zip/postal
    # - avoid hallucinated/exotic plants unless justified
    # - return strict JSON only
    location_block = ""
    if has_location:
        location_block = (
            f'  "location": {{ "latitude": {latitude}, "longitude": {longitude}, "accuracy_m": {accuracy_m if accuracy_m is not None else "null"} }},\n'
        )
    else:
        location_block = '  "location": null,\n'

    zip_block = f'  "zip_code": "{zip_norm}"\n' if zip_norm else '  "zip_code": null\n'

    plan_prompt = f"""
You are Yard2Garden AI Planner.

Goal:
- Analyze the user's yard image.
- Produce a realistic food-garden plan AND instructions that match the user's climate and constraints.
- Use location if provided; otherwise use zip/postal code.

User inputs (for your assumptions object):
{{
{location_block}{zip_block},
  "budget_tier": "{budget}",
  "budget_target_usd": {budget_info["target_usd"]},
  "budget_max_usd": {budget_info["max_usd"]},
  "upkeep_level": "{upkeep}"
}}

Hard rules:
1) Return ONLY valid JSON. No markdown, no backticks, no commentary.
2) Be climate-aware:
   - If location is present, infer a likely USDA hardiness zone / Köppen climate *as an estimate* (label as estimate).
   - If only zip/postal is present, infer climate similarly as an estimate.
3) Avoid "wild plants":
   - Prefer widely available, proven, region-appropriate food plants.
   - If uncertain, include fewer plant species and add uncertainty notes.
   - Do NOT propose tropical/citrus unless the climate clearly supports it.
4) Keep costs realistic:
   - estimated_total_cost_usd MUST be <= budget_max_usd
   - Use budget_target_usd as the aim.
5) Keep the plan buildable by regular people:
   - Clear steps, spacing notes, basic materials.
   - For upkeep="low": fewer beds, more perennials, mulch, drip/soaker, fewer fussy crops.
6) Include comprehensive plant lists but organized:
   - Provide categories with (common_name, scientific_name).
   - Provide “recommended” vs “optional” lists.
   - Provide a small “avoid_in_this_climate” list if relevant.

Schema (must match keys and types):
{{
  "summary": string,
  "assumptions": {{
    "location": {{ "latitude": number, "longitude": number, "accuracy_m": number|null }} | null,
    "zip_code": string|null,
    "climate_estimates": {{
      "usda_zone_estimate": string|null,
      "koppen_estimate": string|null,
      "frost_risk_notes": string
    }},
    "budget_tier": "low"|"medium"|"high",
    "budget_target_usd": number,
    "budget_max_usd": number,
    "upkeep_level": "low"|"medium"|"high"
  }},
  "site_observations": {{
    "sun_exposure": string,
    "space_notes": string,
    "soil_notes": string,
    "drainage_notes": string,
    "constraints": [string]
  }},
  "layout_plan": {{
    "zones": [
      {{
        "zone_name": string,
        "purpose": string,
        "size_estimate": string,
        "plants_recommended": [string],
        "layout_notes": string
      }}
    ],
    "pathing_and_access": string,
    "irrigation_recommendation": string
  }},
  "planting_plan": {{
    "recommended_species": {{
      "annual_vegetables": [{{"common_name": string, "scientific_name": string}}],
      "perennial_vegetables": [{{"common_name": string, "scientific_name": string}}],
      "herbs": [{{"common_name": string, "scientific_name": string}}],
      "berries": [{{"common_name": string, "scientific_name": string}}],
      "fruit_trees": [{{"common_name": string, "scientific_name": string}}],
      "pollinator_support": [{{"common_name": string, "scientific_name": string}}]
    }},
    "optional_species": {{
      "annual_vegetables": [{{"common_name": string, "scientific_name": string}}],
      "perennial_vegetables": [{{"common_name": string, "scientific_name": string}}],
      "herbs": [{{"common_name": string, "scientific_name": string}}],
      "berries": [{{"common_name": string, "scientific_name": string}}],
      "fruit_trees": [{{"common_name": string, "scientific_name": string}}]
    }},
    "avoid_in_this_climate": [string]
  }},
  "step_by_step_plan": {{
    "week_1_setup": [string],
    "week_2_build": [string],
    "week_3_plant": [string],
    "ongoing_weekly": [string],
    "seasonal": {{
      "spring": [string],
      "summer": [string],
      "fall": [string],
      "winter": [string]
    }}
  }},
  "shopping_list": [
    {{"item": string, "quantity": string, "estimated_cost_usd": number}}
  ],
  "estimated_total_cost_usd": number,
  "maintenance_plan": {{
    "weekly_minutes": number,
    "watering_guidance": string,
    "weeding_guidance": string,
    "mulching_guidance": string,
    "pest_management_guidance": string
  }}
}}
""".strip()

    try:
        plan_resp = client.responses.create(
            model="gpt-5.2",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": plan_prompt},
                        {"type": "input_image", "image_base64": image_b64},
                    ],
                }
            ],
        )
        plan_raw = plan_resp.output_text or ""
        plan_json = _extract_json(plan_raw)

        # Force assumptions to reflect actual inputs
        plan_json.setdefault("assumptions", {})
        plan_json["assumptions"]["location"] = (
            {"latitude": latitude, "longitude": longitude, "accuracy_m": accuracy_m} if has_location else None
        )
        plan_json["assumptions"]["zip_code"] = zip_norm
        plan_json["assumptions"]["budget_tier"] = budget
        plan_json["assumptions"]["budget_target_usd"] = budget_info["target_usd"]
        plan_json["assumptions"]["budget_max_usd"] = budget_info["max_usd"]
        plan_json["assumptions"]["upkeep_level"] = upkeep

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Model did not return valid JSON for plan: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plan generation failed: {str(e)}")

    # ----------------------------
    # 2) Photorealistic transformed image (edit the user's photo)
    # ----------------------------
    # We use the Images Edit endpoint with a GPT Image model.
    # This gives the “same camera angle / same yard” effect better than text-only generation.
    try:
        # Write to a temp file ONLY to satisfy the SDK's file upload expectations.
        # This is ephemeral and not persistent storage.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            tmp.write(image_bytes)
            tmp.flush()

            # Build a tight prompt that anchors realism and avoids weirdness.
            # Keep it short enough to be reliable.
            climate_hint = ""
            if has_location:
                climate_hint = f"Location: lat {latitude}, lon {longitude}."
            elif zip_norm:
                climate_hint = f"ZIP/Postal: {zip_norm}."

            after_prompt = f"""
Transform this exact yard photo into a realistic food-producing garden that could actually be built.
Maintain the same camera angle, perspective, and property boundaries.

Constraints:
- Budget tier: {budget} (aim around ${budget_info["target_usd"]}, never imply luxury hardscapes)
- Upkeep level: {upkeep} (low = simpler, more perennials, heavy mulch; high = more intensive beds)
- {climate_hint}

Design notes:
- Add raised beds or in-ground beds, simple paths, mulch, and sensible spacing.
- Use region-appropriate plants (no tropical trees unless climate supports it).
- Photorealistic, natural lighting, no text overlays, no labels, no watermarks.
""".strip()

            img = client.images.edit(
                model="gpt-image-1.5",
                image=[open(tmp.name, "rb")],
                prompt=after_prompt,
                size="1024x1024",
                output_format="png",
            )

            after_b64 = img.data[0].b64_json  # GPT Image models return base64
            # (Optional) validate decodes
            _ = base64.b64decode(after_b64)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image transformation failed: {str(e)}")

    # ----------------------------
    # Response (no server-side storage)
    # ----------------------------
    return {
        "filename": file.filename,
        "inputs": {
            "location": {"latitude": latitude, "longitude": longitude, "accuracy_m": accuracy_m} if has_location else None,
            "zip_code": zip_norm,
            "budget": budget,
            "upkeep": upkeep,
        },
        "garden_plan": plan_json,
        "after_image": {
            "mime": "image/png",
            "base64": after_b64,
        },
        "notes": [
            "No server-side storage: image and plan are generated and returned immediately.",
            "Client should decode after_image.base64 to display and allow user to save locally.",
        ],
    }
