from __future__ import annotations

import os
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
app = FastAPI(title="Yard2Garden AI Planner", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you want
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
    if "```" in s:
        start = s.find("```")
        end = s.rfind("```")
        if start != -1 and end != -1 and end > start:
            inner = s[start + 3 : end].strip()
            if inner.lower().startswith("json"):
                inner = inner[4:].strip()
            s = inner

    # Trim anything before first { and after last }
    lb = s.find("{")
    rb = s.rfind("}")
    if lb == -1 or rb == -1 or rb <= lb:
        raise ValueError("No JSON object found in model output")

    return json.loads(s[lb : rb + 1])


def _validate_zip(zip_code: Optional[str]) -> Optional[str]:
    if not zip_code:
        return None
    z = zip_code.strip()
    return z if z else None


def _location_ok(lat: Optional[float], lon: Optional[float]) -> bool:
    if lat is None or lon is None:
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def _budget_to_usd(budget: Literal["low", "medium", "high"]) -> Dict[str, int]:
    # These are "feel" numbers for the plan; your UI can interpret them however you want.
    if budget == "low":
        return {"target_usd": 150, "max_usd": 350}
    if budget == "medium":
        return {"target_usd": 500, "max_usd": 1200}
    return {"target_usd": 1500, "max_usd": 3500}


async def _process_request(
    upload: UploadFile,
    budget: Literal["low", "medium", "high"],
    upkeep: Literal["low", "medium", "high"],
    latitude: Optional[float],
    longitude: Optional[float],
    accuracy_m: Optional[float],
    zip_code: Optional[str],
) -> Dict[str, Any]:
    """
    Shared handler for /v1/yard2garden and /analyze-yard.
    Returns top-level:
      - image_b64_png: base64 PNG (no data URI prefix)
      - plan: JSON string (for Unity display)
      - plan_json: object (nice for web clients)
    """
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured on the server (Cloud Run env var missing).",
        )

    image_bytes = await upload.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    if len(image_bytes) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large. Please upload an image under 8MB.")

    zip_norm = _validate_zip(zip_code)
    has_location = _location_ok(latitude, longitude)
    budget_info = _budget_to_usd(budget)

    # Base64 for vision call
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # ----------------------------
    # 1) Structured plan
    # ----------------------------
    climate_hint = ""
    if has_location:
        climate_hint = f"Approx location: lat {latitude}, lon {longitude}."
    elif zip_norm:
        climate_hint = f"ZIP/postal: {zip_norm}."
    else:
        climate_hint = "Location unknown (use safe, general assumptions)."

    plan_prompt = f"""
You are Yard2Garden AI Planner.

Goal:
Turn the user's yard photo into a realistic, buildable, food-producing garden plan.

Inputs:
- Budget tier: {budget} (aim around ${budget_info["target_usd"]}, do not exceed ${budget_info["max_usd"]})
- Upkeep tier: {upkeep} (low=low maintenance; high=more intensive)
- {climate_hint}

Output MUST be valid JSON (no markdown, no commentary) matching this schema:

{{
  "summary": string,
  "assumptions": {{
    "location": object|null,
    "zip_code": string|null,
    "budget_tier": string,
    "budget_target_usd": number,
    "budget_max_usd": number,
    "upkeep_level": string
  }},
  "zones": [
    {{
      "name": string,
      "where_in_yard": string,
      "sun_estimate": string,
      "bed_type": string,
      "plants": [string],
      "why_these": string
    }}
  ],
  "step_by_step": [
    {{ "step": number, "title": string, "detail": string, "time_estimate_minutes": number }}
  ],
  "shopping_list": [
    {{ "item": string, "qty": string, "estimated_cost_usd": number }}
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plan generation failed: {str(e)}")

    # ----------------------------
    # 2) Photorealistic transformed image
    # ----------------------------
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            tmp.write(image_bytes)
            tmp.flush()

            after_prompt = f"""
Transform this exact yard photo into a realistic food-producing garden that could actually be built.
Maintain the same camera angle, perspective, and property boundaries.

Constraints:
- Budget tier: {budget} (aim around ${budget_info["target_usd"]}, never imply luxury hardscapes)
- Upkeep level: {upkeep}
- {climate_hint}

Design notes:
- Add beds, simple paths, mulch, and sensible spacing.
- Use region-appropriate plants (avoid tropical unless climate supports it).
- Photorealistic, natural lighting, no text overlays, no labels, no watermarks.
""".strip()

            img = client.images.edit(
                model="gpt-image-1.5",
                image=[open(tmp.name, "rb")],
                prompt=after_prompt,
                size="1024x1024",
                output_format="png",
            )

            after_b64 = img.data[0].b64_json
            _ = base64.b64decode(after_b64)  # validate decodes

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image transformation failed: {str(e)}")

    plan_text = json.dumps(plan_json, ensure_ascii=False, indent=2)

    return {
        "image_b64_png": after_b64,
        "plan": plan_text,
        "plan_json": plan_json,  # optional but useful
    }


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "yard2garden-backend"}


@app.post("/v1/yard2garden")
async def yard2garden(
    file: UploadFile = File(None),
    image: UploadFile = File(None),

    # Location (accept both naming styles from Unity)
    has_location: Optional[bool] = Form(None),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),

    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    accuracy_m: Optional[float] = Form(None),

    zip_code: Optional[str] = Form(None),

    budget: Literal["low", "medium", "high"] = Form(...),
    upkeep: Literal["low", "medium", "high"] = Form(...),
):
    upload = file or image
    if upload is None:
        raise HTTPException(status_code=422, detail="Missing file upload. Use multipart field 'file' (or 'image').")

    # Decide which lat/lon to use
    lat_use = latitude if latitude is not None else lat
    lon_use = longitude if longitude is not None else lon

    # If client sends has_location=false, ignore any lat/lon
    if has_location is False:
        lat_use = None
        lon_use = None

    return await _process_request(
        upload=upload,
        budget=budget,
        upkeep=upkeep,
        latitude=lat_use,
        longitude=lon_use,
        accuracy_m=accuracy_m,
        zip_code=zip_code,
    )


# Backward-compatible alias (older Unity configs might hit this)
@app.post("/analyze-yard")
async def analyze_yard(
    file: UploadFile = File(None),
    image: UploadFile = File(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    accuracy_m: Optional[float] = Form(None),
    zip_code: Optional[str] = Form(None),
    budget: Literal["low", "medium", "high"] = Form(...),
    upkeep: Literal["low", "medium", "high"] = Form(...),
):
    upload = file or image
    if upload is None:
        raise HTTPException(status_code=422, detail="Missing file upload. Use multipart field 'file' (or 'image').")

    return await _process_request(
        upload=upload,
        budget=budget,
        upkeep=upkeep,
        latitude=latitude,
        longitude=longitude,
        accuracy_m=accuracy_m,
        zip_code=zip_code,
    )
