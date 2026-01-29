import base64
import json
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google.cloud import secretmanager

from openai import OpenAI


# -----------------------------
# Config
# -----------------------------
APP_NAME = "yard2garden-backend"
OPENAI_SECRET_NAME = os.getenv("OPENAI_SECRET_NAME", "OPENAI_API_KEY")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")  # set to your domain(s) later

# OpenAI model choices (safe defaults; adjust if you prefer)
# - Vision analysis: use a multimodal model via Responses API (works with image input)
# - Image generation: use Images API (gpt-image-* models)
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-5")  # or another vision-capable model
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1.5")

# Output image size
OUTPUT_IMAGE_SIZE = os.getenv("OUTPUT_IMAGE_SIZE", "1024x1024")


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title=APP_NAME)

# CORS
origins = [o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Secret Manager + OpenAI client (cached)
# -----------------------------
_secret_cache: Dict[str, str] = {}
_openai_client: Optional[OpenAI] = None


def _get_secret_from_gsm(secret_id: str) -> str:
    """
    Reads latest version of a secret from Google Secret Manager.
    Caches in-memory for performance.
    """
    if secret_id in _secret_cache:
        return _secret_cache[secret_id]

    if not GCP_PROJECT_ID:
        raise RuntimeError("GCP_PROJECT_ID env var is required to read secrets.")

    sm = secretmanager.SecretManagerServiceClient()
    name = f"projects/{GCP_PROJECT_ID}/secrets/{secret_id}/versions/latest"
    resp = sm.access_secret_version(request={"name": name})
    value = resp.payload.data.decode("utf-8").strip()

    if not value:
        raise RuntimeError(f"Secret {secret_id} was empty.")

    _secret_cache[secret_id] = value
    return value


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    api_key = _get_secret_from_gsm(OPENAI_SECRET_NAME)
    _openai_client = OpenAI(api_key=api_key)
    return _openai_client


# -----------------------------
# Response schema
# -----------------------------
class GardenPlanResponse(BaseModel):
    status: str
    location: Dict[str, Any]
    budget_level: str
    upkeep_level: str
    plan: Dict[str, Any]
    image_b64_png: str  # base64 png (no data: prefix)


# -----------------------------
# Health endpoints
# -----------------------------
@app.get("/")
def root():
    return {"status": f"{APP_NAME} is alive 🌱"}


@app.get("/health")
def health():
    return {"ok": True}


# -----------------------------
# Core endpoint
# -----------------------------
@app.post("/v1/yard2garden", response_model=GardenPlanResponse)
async def yard2garden(
    image: UploadFile = File(...),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    budget_level: str = Form("medium"),  # low | medium | high
    upkeep_level: str = Form("medium"),  # low | medium | high
):
    # Validate sliders
    budget_level = (budget_level or "medium").lower().strip()
    upkeep_level = (upkeep_level or "medium").lower().strip()

    if budget_level not in {"low", "medium", "high"}:
        raise HTTPException(status_code=400, detail="budget_level must be low|medium|high")
    if upkeep_level not in {"low", "medium", "high"}:
        raise HTTPException(status_code=400, detail="upkeep_level must be low|medium|high")

    # Read image bytes
    img_bytes = await image.read()
    if not img_bytes or len(img_bytes) < 1000:
        raise HTTPException(status_code=400, detail="Invalid image upload.")

    # Convert to base64 for OpenAI vision input
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # Simple location object (keep privacy: approximate only)
    location_obj = {
        "lat": lat,
        "lon": lon,
        "note": "Approximate device location used for climate assumptions. No login. No identity.",
    }

    client = _get_openai()

    # -----------------------------
    # 1) Vision + plan (structured JSON)
    # -----------------------------
    plan_system = (
        "You are Yard2Garden AI Planner. "
        "Given a yard photo and constraints, produce a practical, step-by-step edible garden plan. "
        "Assume typical seasonal planting for the user’s region based on approximate lat/lon if provided. "
        "If location is missing, be conservative and provide a flexible plan."
    )

    plan_instructions = {
        "output_format": {
            "title": "short string",
            "summary": "short paragraph",
            "assumptions": ["bullet strings"],
            "layout": {
                "zones": [
                    {
                        "name": "e.g., Front-Left Bed",
                        "sun": "full|partial|shade|unknown",
                        "size_estimate": "small|medium|large|unknown",
                        "plants": [
                            {
                                "name": "plant name",
                                "why": "1 sentence",
                                "spacing": "e.g., 12 inches",
                                "start_method": "seed|starter|either",
                                "notes": "optional"
                            }
                        ]
                    }
                ]
            },
            "shopping_list": [
                {"item": "string", "qty": "string", "priority": "must|should|nice"}
            ],
            "timeline": [
                {"step": 1, "when": "now|week 1|week 2|month 1|seasonal", "action": "string"}
            ],
            "maintenance": [
                {"frequency": "daily|2-3x/week|weekly|monthly", "task": "string"}
            ],
            "budget_notes": "how the plan fits the budget level",
            "upkeep_notes": "how the plan fits the upkeep level",
            "image_prompt": "A photorealistic 'after' prompt describing the same yard transformed into a food garden"
        }
    }

    # Use Responses API with image input (see OpenAI docs for images/vision & responses). :contentReference[oaicite:0]{index=0}
    try:
        resp = client.responses.create(
            model=OPENAI_VISION_MODEL,
            input=[
                {
                    "role": "system",
                    "content": plan_system
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"""
Create an edible garden plan from this yard photo.

Constraints:
- budget_level: {budget_level}
- upkeep_level: {upkeep_level}
- location (approx): lat={lat}, lon={lon}

Requirements:
- Plan must be realistic, achievable, and climate-aware.
- Assume the user wants a family food garden (vegetables + herbs, optional berries).
- DO NOT recommend anything illegal or unsafe.
- Output MUST be STRICT JSON only.

Most important:
- The image_prompt MUST describe the SAME YARD with the SAME camera angle/perspective.
- Preserve existing house/fence/trees and hardscape; only add garden elements.
- Photorealistic, natural lighting, no text overlays, no watermarks, no people.

Return STRICT JSON only, matching this schema exactly:
{...schema...}
,
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_b64}"}
                    ]
                }
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI plan generation failed: {str(e)}")

    # Extract text from response
    plan_text = ""
    try:
        # The SDK returns output items; safest is to grab combined output_text
        plan_text = resp.output_text  # convenience property in newer SDKs
    except Exception:
        # Fallback: try to walk the structure
        plan_text = str(resp)

    # Parse JSON
    try:
        plan_json = json.loads(plan_text)
    except Exception:
        # If model drifted, try to salvage by finding first/last braces
        start = plan_text.find("{")
        end = plan_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise HTTPException(status_code=500, detail="Model did not return valid JSON plan.")
        try:
            plan_json = json.loads(plan_text[start:end + 1])
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to parse JSON plan.")

    # Get the image prompt the model generated for the “after”
    image_prompt = plan_json.get("image_prompt")
    if not image_prompt or not isinstance(image_prompt, str):
        image_prompt = (
            "Photorealistic transformation of the uploaded yard into a thriving food-producing garden, "
            "with raised beds, mulched paths, and visible vegetables and herbs, matching the original yard layout."
        )

    # -----------------------------
    # 2) Generate the photorealistic “after” image
    # -----------------------------
    # Use Images API (see OpenAI Images reference). :contentReference[oaicite:1]{index=1}
    try:
        img_resp = client.images.generate(
            model=OPENAI_IMAGE_MODEL,
            prompt=image_prompt,
            n=1,
            size=OUTPUT_IMAGE_SIZE
        )
        out_b64 = img_resp.data[0].b64_json
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI image generation failed: {str(e)}")

    return GardenPlanResponse(
        status="ok",
        location=location_obj,
        budget_level=budget_level,
        upkeep_level=upkeep_level,
        plan=plan_json,
        image_b64_png=out_b64
    )
