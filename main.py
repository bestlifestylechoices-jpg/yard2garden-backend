from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import base64
import os
import json

app = FastAPI(title="Yard2Garden AI Planner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
def root():
    return {"status": "yard2garden-backend is alive 🌱"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/analyze-yard")
async def analyze_yard(
    file: UploadFile = File(...),
    zip_code: str = Form(...),
    budget: str = Form(...),
    maintenance_level: str = Form(...)
):
    try:
        image_bytes = await file.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        prompt = f"""
You are a professional food-forest and garden planner.

The user lives in ZIP CODE: {zip_code}
Their budget is: {budget}
Their maintenance preference is: {maintenance_level}

Analyze the uploaded yard photo and produce a JSON garden plan using this exact schema:

{{
  "summary": "",
  "climate_assumptions": "",
  "sun_exposure": "",
  "recommended_zones": [
    {{
      "zone_name": "",
      "plants": [],
      "notes": ""
    }}
  ],
  "step_by_step_plan": [],
  "shopping_list": [
    {{
      "item": "",
      "quantity": "",
      "estimated_cost": ""
    }}
  ]
}}

Return only valid JSON. No markdown.
"""

        response = client.responses.create(
            model="gpt-5.2",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_base64": image_b64}
                    ]
                }
            ]
        )

        raw_text = response.output_text

        # Parse JSON safely
        garden_json = json.loads(raw_text)

        return {
            "filename": file.filename,
            "zip_code": zip_code,
            "budget": budget,
            "maintenance_level": maintenance_level,
            "garden_plan": garden_json
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
