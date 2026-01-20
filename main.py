import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import openai

# Initialize FastAPI
app = FastAPI(title="Yard2Garden AI Planner Backend")

# Load OpenAI API Key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

openai.api_key = OPENAI_API_KEY


# ---------- Models ----------
class YardRequest(BaseModel):
    zip_code: str
    description: Optional[str] = None


class YardResponse(BaseModel):
    plan: str


# ---------- Routes ----------
@app.get("/")
def root():
    return {"status": "yard2garden-backend is alive 🌱"}


@app.post("/analyze-yard", response_model=YardResponse)
def analyze_yard(request: YardRequest):
    try:
        prompt = f"""
You are an expert permaculture designer.

Create a practical, beginner-friendly food garden plan for a home located in ZIP code {request.zip_code}.

User notes: {request.description or "No additional notes provided."}

Include:
- Layout suggestions
- Beginner crops
- Spacing tips
- Seasonal guidance
- Sustainability focus
"""

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You design sustainable food gardens."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=700,
        )

        plan_text = response.choices[0].message.content.strip()
        return {"plan": plan_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
