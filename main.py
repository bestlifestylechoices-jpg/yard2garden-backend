from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import hashlib

app = FastAPI(
    title="Yard2Garden AI Planner Backend",
    version="1.0.0",
    description="Backend API for Yard2Garden AI Planner (Cloud Run)."
)

@app.get("/")
def root():
    return {"status": "yard2garden-backend is alive 🌱"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/analyze-yard")
async def analyze_yard(
    image: UploadFile = File(...),
    zip_code: str | None = Form(default=None),
    notes: str | None = Form(default=None),
):
    # Read bytes (limit size later if you want)
    data = await image.read()

    # Simple fingerprint so you can confirm uploads are arriving
    sha = hashlib.sha256(data).hexdigest()[:12]

    # TODO (next step): call OpenAI Vision + planning prompt using zip_code
    # For now, return a clean stub response Unity can consume immediately.
    result = {
        "request": {
            "filename": image.filename,
            "content_type": image.content_type,
            "bytes": len(data),
            "sha12": sha,
            "zip_code": zip_code,
            "notes": notes,
        },
        "plan": {
            "summary": "Stub plan — next step will generate a full garden layout + step-by-step instructions.",
            "zones": [],
            "steps": [
                "Upload received successfully.",
                "Next: AI will analyze the yard image and propose a layout.",
                "Next: Generate step-by-step instructions tailored to your zip code.",
            ],
        },
    }

    return JSONResponse(result)
