from __future__ import annotations

import os
import json
import base64
import tempfile
from typing import Any, Dict, Optional, Literal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# OpenAI SDK (python package: openai)
# Cloud Run: set OPENAI_API_KEY as an environment variable
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# ----------------------------
# App
# ----------------------------
app = FastAPI(title="Yard2Garden AI Planner", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# OpenAI client
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI is not None and OPENAI_API_KEY) else None

# Models (override via env if you want)
PLAN_MODEL = os.getenv("Y2G_PLAN_MODEL", "gpt-4.1-mini").strip()
IMAGE_MODEL = os.getenv("Y2G_IMAGE_MODEL", "gpt-image-1").strip()  # <-- important


# ----------------------------
# Helpers
# ----------------------------
def _location_ok(lat: Optional[float], lon: Optional[float]) -> bool:
    return lat is not None and lon is not None and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def _validate_zip(zip_code: Optional[str]) -> Optional[str]:
    if not zip_code:
        return None
    z = zip_code.strip()
    if len(z) < 3:
        return None
    return z


def _budget_to_usd(budget: str) -> Dict[str, Any]:
    # Unity sends tiers; we map to friendly guidance.
    return {
        "tier": budget,
        "low": {"min_usd": 50, "max_usd": 350},
        "medium": {"min_usd": 350, "max_usd": 1500},
        "high": {"min_usd": 1500, "max_usd": 5000},
    }


def _safe_json_load(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return None


def _plan_to_text(plan_json: Dict[str, Any]) -> str:
    """
    Human-readable plan text for the app UI.
    Keeps Unity simple: show PlanText immediately.
    """
    lines = []

    summary = plan_json.get("summary") if isinstance(plan_json, dict) else None
    if summary:
        lines.append(str(summary).strip())

    step_list = plan_json.get("step_by_step_plan")
    if isinstance(step_list, list) and step_list:
        lines.append("")
        lines.append("Step-by-step plan:")
        for i, step in enumerate(step_list, start=1):
            lines.append(f"{i}. {str(step).strip()}")

    shopping = plan_json.get("shopping_list")
    if isinstance(shopping, list) and shopping:
        lines.append("")
        lines.append("Shopping list:")
        for item in shopping:
            if isinstance(item, dict):
                name = item.get("item") or item.get("name") or "Item"
                qty = item.get("quantity") or ""
                lines.append(f"- {name} {qty}".strip())
            else:
                lines.append(f"- {str(item).strip()}")

    zones = plan_json.get("recommended_zones")
    if isinstance(zones, list) and zones:
        lines.append("")
        lines.append("Garden zones:")
        for z in zones:
            if not isinstance(z, dict):
                continue
            zn = z.get("zone_name", "Zone")
            plants = z.get("plants", [])
            notes = z.get("notes", "")
            plant_str = ", ".join(plants) if isinstance(plants, list) else str(plants)
            lines.append(f"- {zn}: {plant_str}".strip())
            if notes:
                lines.append(f"  Notes: {str(notes).strip()}")

    return "\n".join(lines).strip()


def _generate_plan(
    image_bytes: bytes,
    latitude: Optional[float],
    longitude: Optional[float],
    accuracy_m: Optional[float],
    zip_code: Optional[str],
    budget: Literal["low", "medium", "high"],
    upkeep: Literal["low", "medium", "high"],
) -> Dict[str, Any]:
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured on the server.")

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    zip_norm = _validate_zip(zip_code)
    has_location = _location_ok(latitude, longitude)

    budget_info = _budget_to_usd(budget)

    location_payload = None
    if has_location:
        location_payload = {"latitude": latitude, "longitude": longitude, "accuracy_m": accuracy_m}

    # Strict JSON schema prompt (concise but rigid)
    schema_hint = {
        "summary": "string",
        "sun_exposure": "string",
        "recommended_zones": [{"zone_name": "string", "plants": ["string"], "notes": "string"}],
        "step_by_step_plan": ["string"],
        "shopping_list": [{"item": "string", "quantity": "string"}],
        "maintenance": {
            "watering_guidance": "string",
            "weeding_guidance": "string",
            "mulching_guidance": "string",
            "pest_management_guidance": "string",
        },
    }

    user_context = {
        "location": location_payload,
        "zip_code": zip_norm,
        "budget_tier": budget,
        "budget_usd_guidance": budget_info,
        "upkeep_tier": upkeep,
    }

    try:
        resp = client.responses.create(
            model=PLAN_MODEL,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a landscape designer and food garden planner. "
                        "Return ONLY valid JSON. No markdown. No extra keys."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Analyze this yard photo and generate a practical, climate-aware food garden plan."},
                        {"type": "input_text", "text": f"User context (use it): {json.dumps(user_context)}"},
                        {"type": "input_text", "text": f"JSON schema (return exactly these top-level keys): {json.dumps(schema_hint)}"},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                    ],
                },
            ],
        )

        plan_text = getattr(resp, "output_text", "") or ""
        plan_json = _safe_json_load(plan_text)
        if not isinstance(plan_json, dict):
            raise HTTPException(status_code=500, detail="Plan model did not return valid JSON.")

        return plan_json

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plan generation failed: {str(e)}")


def _generate_after_image(
    image_bytes: bytes,
    budget: Literal["low", "medium", "high"],
    upkeep: Literal["low", "medium", "high"],
    latitude: Optional[float],
    longitude: Optional[float],
    zip_code: Optional[str],
) -> str:
    """
    Returns base64 PNG (no data: prefix).
    """
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured on the server.")

    zip_norm = _validate_zip(zip_code)
    has_location = _location_ok(latitude, longitude)

    location_phrase = ""
    if has_location:
        location_phrase = f"Climate: use latitude {latitude:.5f}, longitude {longitude:.5f}."
    elif zip_norm:
        location_phrase = f"Climate: use postal/ZIP code {zip_norm}."

    after_prompt = f"""
Transform the uploaded backyard yard photo into a photorealistic, professionally landscaped, food-producing garden on roughly a quarter-acre.
Keep the same camera angle and yard layout, but redesign it with:
- Raised beds, pathways, mulch, drip irrigation, and defined zones
- A balanced mix of vegetables, herbs, berries, and a few small fruit trees suited to the local climate
- Clean, realistic lighting and shadows consistent with the original photo
- No text overlays, no labels, no watermarks, no diagrams

User preferences:
- Budget tier: {budget}
- Upkeep tier: {upkeep}
{location_phrase}
""".strip()

    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp:
            tmp.write(image_bytes)
            tmp.flush()

            img = client.images.edit(
                model=IMAGE_MODEL,  # "gpt-image-1"
                image=[open(tmp.name, "rb")],
                prompt=after_prompt,
                size="1024x1024",
                output_format="png",
            )

            after_b64 = img.data[0].b64_json
            base64.b64decode(after_b64)  # validate
            return after_b64

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image transformation failed: {str(e)}")


def _generate_all(
    image_bytes: bytes,
    latitude: Optional[float],
    longitude: Optional[float],
    accuracy_m: Optional[float],
    zip_code: Optional[str],
    budget: Literal["low", "medium", "high"],
    upkeep: Literal["low", "medium", "high"],
) -> Dict[str, Any]:
    plan_json = _generate_plan(
        image_bytes=image_bytes,
        latitude=latitude,
        longitude=longitude,
        accuracy_m=accuracy_m,
        zip_code=zip_code,
        budget=budget,
        upkeep=upkeep,
    )

    after_b64 = _generate_after_image(
        image_bytes=image_bytes,
        budget=budget,
        upkeep=upkeep,
        latitude=latitude,
        longitude=longitude,
        zip_code=zip_code,
    )

    plan_text = _plan_to_text(plan_json)

    return {
        "plan_json": plan_json,
        "plan_text": plan_text,
        "image_b64_png": after_b64,   # Unity expects this
        "image_base64": after_b64,    # fallback (some clients)
        "after_image": {"mime": "image/png", "base64": after_b64},  # nested format
    }


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "yard2garden-backend is alive 🌱"}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "has_openai_key": bool(OPENAI_API_KEY),
        "plan_model": PLAN_MODEL,
        "image_model": IMAGE_MODEL,
    }


# Swagger / manual testing route
@app.post("/analyze-yard")
async def analyze_yard(
    file: UploadFile = File(...),

    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    accuracy_m: Optional[float] = Form(None),
    zip_code: Optional[str] = Form(None),

    budget: Literal["low", "medium", "high"] = Form(...),
    upkeep: Literal["low", "medium", "high"] = Form(...),
) -> Dict[str, Any]:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(image_bytes) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large. Please upload an image under 8MB.")

    if not _location_ok(latitude, longitude) and not _validate_zip(zip_code):
        raise HTTPException(status_code=422, detail="Provide either (latitude & longitude) OR zip_code/postal code.")

    payload = _generate_all(
        image_bytes=image_bytes,
        latitude=latitude,
        longitude=longitude,
        accuracy_m=accuracy_m,
        zip_code=zip_code,
        budget=budget,
        upkeep=upkeep,
    )

    return {
        "filename": file.filename or "yard.jpg",
        "request": {
            "location": {"latitude": latitude, "longitude": longitude, "accuracy_m": accuracy_m} if _location_ok(latitude, longitude) else None,
            "zip_code": _validate_zip(zip_code),
            "budget": budget,
            "upkeep": upkeep,
        },
        "garden_plan": payload["plan_json"],
        "plan_text": payload["plan_text"],
        "after_image": payload["after_image"],
        # Also include flat keys so Unity can parse this too if it hits /analyze-yard:
        "image_b64_png": payload["image_b64_png"],
        "image_base64": payload["image_base64"],
    }


# Unity route (the one your app should hit)
@app.post("/v1/yard2garden")
async def yard2garden_v1(
    file: UploadFile = File(...),

    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    accuracy_m: Optional[float] = Form(None),
    zip_code: Optional[str] = Form(None),

    budget: Literal["low", "medium", "high"] = Form(...),
    upkeep: Literal["low", "medium", "high"] = Form(...),
) -> Dict[str, Any]:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(image_bytes) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large. Please upload an image under 8MB.")

    if not _location_ok(latitude, longitude) and not _validate_zip(zip_code):
        raise HTTPException(status_code=422, detail="Provide either (latitude & longitude) OR zip_code/postal code.")

    payload = _generate_all(
        image_bytes=image_bytes,
        latitude=latitude,
        longitude=longitude,
        accuracy_m=accuracy_m,
        zip_code=zip_code,
        budget=budget,
        upkeep=upkeep,
    )

    return {
        "ok": True,
        "image_b64_png": payload["image_b64_png"],
        "plan_text": payload["plan_text"],
        "plan_json": payload["plan_json"],
    }
