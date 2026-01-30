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
# Config
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# You can override these in Cloud Run env vars without changing code
PLAN_MODEL = os.getenv("PLAN_MODEL", "gpt-4o-mini")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1")

# Hard safety limits
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))  # 8MB default
OUTPUT_IMAGE_SIZE = os.getenv("OUTPUT_IMAGE_SIZE", "1024x1024")            # e.g., 1024x1024


# ----------------------------
# App
# ----------------------------
app = FastAPI(title="Yard2Garden AI Planner Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you want
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


def _location_ok(lat: Optional[float], lon: Optional[float]) -> bool:
    if lat is None or lon is None:
        return False
    return (-90.0 <= lat <= 90.0) and (-180.0 <= lon <= 180.0)


def _budget_to_usd(budget: Literal["low", "medium", "high"]) -> Dict[str, int]:
    # Simple tiers; tune anytime
    if budget == "low":
        return {"target_usd": 150, "max_usd": 350}
    if budget == "medium":
        return {"target_usd": 500, "max_usd": 1200}
    return {"target_usd": 1500, "max_usd": 3500}


async def _handle_yard2garden(
    upload: UploadFile,
    budget: Literal["low", "medium", "high"],
    upkeep: Literal["low", "medium", "high"],
    latitude: Optional[float],
    longitude: Optional[float],
    accuracy_m: Optional[float],
    zip_code: Optional[str],
) -> Dict[str, Any]:
    """
    Generates:
    - photorealistic updated yard image (base64 PNG)
    - step-by-step plan (JSON object + pretty string)

    Returns BOTH modern + backward-compat fields so Unity can always display something.
    """
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured on the server (Cloud Run env var missing).",
        )

    image_bytes = await upload.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"Image too large. Max is {MAX_IMAGE_BYTES} bytes.")

    has_location = _location_ok(latitude, longitude)
    budget_info = _budget_to_usd(budget)
    zip_norm = (zip_code.strip() if zip_code and zip_code.strip() else None)

    # Base64 for vision plan request
    image_b64_in = base64.b64encode(image_bytes).decode("utf-8")

    # ----------------------------
    # 1) Generate plan JSON
    # ----------------------------
    climate_hint = ""
    if has_location:
        climate_hint = f"Approx location: lat {latitude:.6f}, lon {longitude:.6f}."
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
        response = {
    "plan": plan_text,
    "garden_plan": garden_plan,
}

if image_b64_png:
    response["image_b64_png"] = image_b64_png
    response["after_image"] = {"mime": "image/png", "base64": image_b64_png}

if image_url:
    response["image_url"] = image_url

return response

        plan_resp = client.responses.create(
            model=PLAN_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": plan_prompt},
                        {"type": "input_image", "image_base64": image_b64_in},
                    ],
                }
            ],
        )
        plan_raw = plan_resp.output_text or ""
        garden_plan = _extract_json(plan_raw)

        # Ensure assumptions reflect actual inputs
        garden_plan.setdefault("assumptions", {})
        garden_plan["assumptions"]["location"] = (
            {"latitude": latitude, "longitude": longitude, "accuracy_m": accuracy_m} if has_location else None
        )
        garden_plan["assumptions"]["zip_code"] = zip_norm
        garden_plan["assumptions"]["budget_tier"] = budget
        garden_plan["assumptions"]["budget_target_usd"] = budget_info["target_usd"]
        garden_plan["assumptions"]["budget_max_usd"] = budget_info["max_usd"]
        garden_plan["assumptions"]["upkeep_level"] = upkeep

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plan generation failed: {str(e)}")

    plan_text = json.dumps(garden_plan, ensure_ascii=False, indent=2)

    # ----------------------------
    # 2) Generate photorealistic updated image (PNG base64)
    # ----------------------------
    after_prompt = f"""
Transform this exact yard photo into a realistic food-producing garden.
Maintain the same camera angle and property boundaries.
Photorealistic, natural lighting, no text, no labels, no watermark.

Constraints:
- Budget tier: {budget} (aim around ${budget_info["target_usd"]}; avoid luxury hardscape features)
- Upkeep level: {upkeep}
- {climate_hint}

Design notes:
- Add sensible beds, mulch, simple paths, and realistic spacing.
- Region-appropriate plants (avoid tropical unless climate supports it).
- Photorealistic, natural lighting.
- No text overlays, no labels, no watermarks.
""".strip()

    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            tmp.write(image_bytes)
            tmp.flush()

            # OpenAI image edit
            img = client.images.edit(
    model=IMAGE_MODEL,
    image=[open(tmp.name, "rb")],
    prompt=after_prompt,
    size=OUTPUT_IMAGE_SIZE,
)

image_b64_png = None
image_url = None

if hasattr(img.data[0], "b64_json") and img.data[0].b64_json:
    image_b64_png = img.data[0].b64_json
elif hasattr(img.data[0], "url") and img.data[0].url:
    image_url = img.data[0].url
else:
    raise Exception("Image generated but no usable image field returned")

    # Return modern + backward-compat keys
    return {
        # Modern / Unity-friendly
        "image_b64_png": image_b64_png,
        "plan": plan_text,

        # Backward compatibility (older Unity parsing)
        "after_image": {"mime": "image/png", "base64": image_b64_png},
        "garden_plan": garden_plan,
    }


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "yard2garden-backend", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/v1/yard2garden")
async def yard2garden(
    file: UploadFile = File(None),
    image: UploadFile = File(None),

    # Accept both naming styles for location
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

    # If the client explicitly says has_location=false, ignore any lat/lon
    if has_location is False:
        lat_use = None
        lon_use = None

    return await _handle_yard2garden(
        upload=upload,
        budget=budget,
        upkeep=upkeep,
        latitude=lat_use,
        longitude=lon_use,
        accuracy_m=accuracy_m,
        zip_code=zip_code,
    )


# Keep your old route for backward compatibility
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

    return await _handle_yard2garden(
        upload=upload,
        budget=budget,
        upkeep=upkeep,
        latitude=latitude,
        longitude=longitude,
        accuracy_m=accuracy_m,
        zip_code=zip_code,
    )
