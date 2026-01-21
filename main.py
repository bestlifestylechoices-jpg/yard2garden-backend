from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import base64
import os

app = FastAPI(title="Yard2Garden AI Planner")

# CORS (good for Unity + browser testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # must be False if allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client (reads OPENAI_API_KEY from env automatically)
# You can also do: OpenAI(api_key=os.environ["OPENAI_API_KEY"])
client = OpenAI()

@app.get("/")
def root():
    return {"status": "yard2garden-backend is alive 🌱"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/analyze-yard")
async def analyze_yard(
    file: UploadFile = File(...),
    zip_code: str | None = Form(None),
    budget: str | None = Form(None),
    upkeep: str | None = Form(None),
):
    try:
        # Basic validation
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded.")

        # Optional: restrict to images
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Please upload an image.",
            )

        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file was empty.")

        # Encode image as a data URL (widely supported pattern)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        content_type = file.content_type or "image/png"
        image_data_url = f"data:{content_type};base64,{image_b64}"

        # Build prompt with optional user constraints
        constraints = []
        if zip_code:
            constraints.append(f"ZIP code: {zip_code}")
        if budget:
            constraints.append(f"Budget: {budget}")
        if upkeep:
            constraints.append(f"Upkeep level: {upkeep}")

        constraints_text = ""
        if constraints:
            constraints_text = "\n\nUser constraints:\n- " + "\n- ".join(constraints)

        prompt = (
            "Analyze this yard photo and generate a beginner-friendly food garden plan.\n"
            "Return:\n"
            "1) A short summary of the yard (sun/shade, space, constraints)\n"
            "2) Recommended garden zones (e.g., veg beds, herbs, fruit)\n"
            "3) A step-by-step plan (numbered)\n"
            "4) A simple shopping list with quantities\n"
            "Keep it practical and realistic."
            f"{constraints_text}"
        )

        response = client.responses.create(
            model="gpt-5.2",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }],
        )

        # Return clean JSON
        return {
            "filename": file.filename,
            "zip_code": zip_code,
            "budget": budget,
            "upkeep": upkeep,
            "garden_plan": response.output_text,
        }

    except HTTPException:
        raise
    except Exception as e:
        # Helpful debugging message in Cloud Run logs + client response
        raise HTTPException(status_code=500, detail=f"Analyze failed: {str(e)}")
