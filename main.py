import base64
import json
import os
import re
import time
import uuid
from typing import Optional, List

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from google.cloud import secretmanager
from openai import OpenAI

# ----------------------------
# Config
# ----------------------------

OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-5.2")
DEFAULT_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1.5")  # per OpenAI image docs :contentReference[oaicite:2]{index=2}

# CORS: for Unity builds; tighten later if you want
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

# Secret Manager name containing OpenAI API key
OPENAI_API_KEY_SECRET_NAME = os.getenv("OPENAI_API_KEY_SECRET_NAME", "openai_api_key")

# Cache secrets in-memory
_cached_openai_key: Optional[str] = None

# ----------------------------
# FastAPI
# ----------------------------

app = FastAPI(title="Yard2Garden Backend", version="1.0.0")


# ----------------------------
# Models
# ----------------------------

class Yard2GardenRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded image bytes (NO data URL prefix).")
    image_mime: str = Field("image/jpeg", description="image/jpeg or image/png recommended.")
    latitude: Optional[float] = Field(None, description="Approximate latitude (optional).")
    longitude: Optional[float] = Field(None, description="Approximate longitude (optional).")
    budget_usd: Optional[float] = Field(None, description="Optional budget guidance.")
    upkeep_level: Optional[str] = Field(None, description="low | medium | high (optional).")


class Yard2GardenResponse(BaseModel):
    image_b64_png: str
    plan_markdown: str
    request_id: str


# ----------------------------
# Utilities
# ----------------------------

def _get_openai_api_key() -> str:
    """
    Resolve OpenAI API key securely.
    Order:
      1) OPENAI_API_KEY env var (local dev)
      2) Google Secret Manager (Cloud Run)
    """
    global _cached_openai_key

    if _cached_openai_key:
        return _cached_openai_key

    env_key = os.getenv("OPENAI_API_KEY")
    if env_key and env_key.strip():
        _cached_openai_key = env_key.strip()
        return _cached_openai_key

    # Secret Manager
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
    if not project_id:
        raise RuntimeError("Missing GOOGLE_CLOUD_PROJECT (or GCP_PROJECT). Cloud Run normally sets this.")

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{OPENAI_API_KEY_SECRET_NAME}/versions/latest"
    resp = client.access_secret_version(request={"name": name})
    key = resp.payload.data.decode("utf-8").strip()

    if not key:
        raise RuntimeError("OpenAI API key retrieved from Secret Manager is empty.")

    _cached_openai_key = key
    return _cached_openai_key


def _ensure_base64_clean(b64_str: str) -> str:
    """
    Accepts raw base64 or a data URL; returns raw base64.
    """
    b64_str = b64_str.strip()
    if b64_str.startswith("data:"):
        # data:image/jpeg;base64,XXXX
        parts = b64_str.split(",", 1)
        if len(parts) == 2:
            return parts[1].strip()
    return b64_str


def _build_data_url(image_mime: str, image_b64: str) -> str:
    image_b64 = _ensure_base64_clean(image_b64)
    return f"data:{image_mime};base64,{image_b64}"


def _safe_upkeep(upkeep: Optional[str]) -> Optional[str]:
    if upkeep is None:
        return None
    u = upkeep.strip().lower()
    if u in ("low", "medium", "high"):
        return u
    return None


def _safe_budget(budget: Optional[float]) -> Optional[float]:
    if budget is None:
        return None
    try:
        b = float(budget)
        if b < 0:
            return None
        return b
    except Exception:
        return None


def _extract_output_text(resp) -> str:
    """
    Works with OpenAI SDK objects or dicts.
    """
    # New SDK often provides resp.output_text
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str):
        return resp.output_text.strip()

    # Dict fallback
    if isinstance(resp, dict):
        if "output_text" in resp and isinstance(resp["output_text"], str):
            return resp["output_text"].strip()

        # Parse "output" items for message content
        out = resp.get("output", [])
        texts: List[str] = []
        for item in out:
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") in ("output_text", "text"):
                        t = c.get("text", "")
                        if t:
                            texts.append(t)
        return "\n".join(texts).strip()

    return ""


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    # remove triple backtick fences if any
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
        s = re.sub(r"\n```$", "", s)
    return s.strip()


# ----------------------------
# Middleware: Simple CORS
# ----------------------------

@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)

    origin = request.headers.get("origin")
    allow_origin = "*"
    if ALLOWED_ORIGINS and ALLOWED_ORIGINS != ["*"] and origin:
        if origin in ALLOWED_ORIGINS:
            allow_origin = origin
        else:
            allow_origin = ALLOWED_ORIGINS[0]  # fallback (tighten later)

    response.headers["Access-Control-Allow-Origin"] = allow_origin
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Max-Age"] = "3600"
    return response


@app.options("/{path:path}")
async def options_handler(path: str):
    return JSONResponse(content={"ok": True})


# ----------------------------
# Health
# ----------------------------

@app.get("/health")
def health():
    return {"ok": True}


# ----------------------------
# Core Endpoint
# ----------------------------

@app.post("/v1/yard2garden", response_model=Yard2GardenResponse)
def yard2garden(payload: Yard2GardenRequest):
    request_id = str(uuid.uuid4())

    # Validate / normalize
    image_b64 = _ensure_base64_clean(payload.image_b64)
    image_mime = (payload.image_mime or "image/jpeg").strip().lower()
    if image_mime not in ("image/jpeg", "image/png", "image/webp"):
        # We still accept it, but enforce a safe default for data URL
        image_mime = "image/jpeg"

    lat = payload.latitude
    lng = payload.longitude
    budget = _safe_budget(payload.budget_usd)
    upkeep = _safe_upkeep(payload.upkeep_level)

    # Build prompt context
    location_line = ""
    if lat is not None and lng is not None:
        location_line = f"Approximate location (lat,lng): {lat:.6f}, {lng:.6f}."

    budget_line = ""
    if budget is not None:
        budget_line = f"Budget constraint: ${budget:,.0f} USD (prioritize high-impact essentials)."

    upkeep_line = ""
    if upkeep is not None:
        upkeep_line = f"Upkeep preference: {upkeep} (design accordingly)."

    # ----------------------------
    # 1) Generate the step-by-step plan (vision-capable text model)
    # ----------------------------

    try:
        api_key = _get_openai_api_key()
        client = OpenAI(api_key=api_key)

        input_image_data_url = _build_data_url(image_mime, image_b64)

        plan_prompt = f"""
You are a professional garden planner.

Task:
1) Analyze the provided yard photo.
2) Produce a professional, practical, step-by-step plan to transform this yard into a thriving edible garden.
3) Use the user's approximate location ONLY to choose climate-appropriate crops and timing (do not mention saving data).
4) Keep it realistic for typical home gardeners.

Context:
{location_line}
{budget_line}
{upkeep_line}

Output format:
Return Markdown with these sections:
- ## Overview
- ## Quick wins (first weekend)
- ## Layout plan (zones)
- ## Crop recommendations (what grows well here)
- ## Soil + compost plan
- ## Irrigation plan
- ## Planting calendar (next 90 days)
- ## Materials list (prioritized)
- ## Maintenance checklist
- ## Safety notes

Be concise, clear, and actionable.
""".strip()

        # Responses API supports image input as base64 data URLs :contentReference[oaicite:3]{index=3}
        resp = client.responses.create(
            model=DEFAULT_TEXT_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": plan_prompt},
                        {"type": "input_image", "image_url": input_image_data_url},
                    ],
                }
            ],
            # Keep it deterministic-ish
            temperature=0.6,
        )

        plan_markdown = _strip_code_fences(_extract_output_text(resp))
        if not plan_markdown:
            raise RuntimeError("Plan generation returned empty text.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[{request_id}] Plan generation failed: {str(e)}")

    # ----------------------------
    # 2) Generate the photorealistic "after" image (Image API)
    # ----------------------------

    try:
        # We use the Image API generations endpoint for GPT Image models :contentReference[oaicite:4]{index=4}
        api_key = _get_openai_api_key()

        # Prompt tuned for photorealistic yard-to-garden transformation
        image_prompt = f"""
Photorealistic "after" transformation of the SAME yard shown in the input photo, converted into a thriving edible garden.
- Preserve the original camera angle and yard boundaries.
- Replace grass/unused areas with raised beds, mulched paths, and healthy vegetables/herbs.
- Natural lighting, realistic materials, realistic scale, no text.
- Looks like a real photo, not a drawing.
{location_line}
""".strip()

        # Call OpenAI Images API directly for maximum compatibility
        # Docs: /v1/images/generations, GPT image models return base64 encoded images :contentReference[oaicite:5]{index=5}
        r = requests.post(
            f"{OPENAI_API_BASE}/images/generations",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEFAULT_IMAGE_MODEL,
                "prompt": image_prompt,
                "n": 1,
                "size": "1536x1024",   # landscape, good for yard photos
                "output_format": "png"  # force png output
            },
            timeout=120,
        )

        if r.status_code != 200:
            raise RuntimeError(f"Images API error {r.status_code}: {r.text}")

        data = r.json()
        # GPT image models: expect base64 in data[0].b64_json (or similar)
        img_b64 = None
        if isinstance(data, dict):
            arr = data.get("data") or []
            if arr and isinstance(arr, list) and isinstance(arr[0], dict):
                img_b64 = arr[0].get("b64_json") or arr[0].get("image_b64") or arr[0].get("base64")

        if not img_b64:
            raise RuntimeError("Image generation returned no base64 payload.")

        image_b64_png = _ensure_base64_clean(img_b64)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[{request_id}] Image generation failed: {str(e)}")

    return Yard2GardenResponse(
        image_b64_png=image_b64_png,
        plan_markdown=plan_markdown,
        request_id=request_id,
    )
