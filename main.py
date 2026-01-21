from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
import base64

app = FastAPI(title="Yard2Garden AI Planner")
@app.get("/health")
def health():
    return {"ok": True}
# CORS (safe for mobile apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client (API key comes from env var)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
def root():
    return {"status": "yard2garden-backend is alive 🌱"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/analyze-yard")
async def analyze_yard(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = """
You are a master permaculture designer and food forest planner.

Analyze the uploaded yard photo and return a practical, beginner-friendly
food garden plan in STRICT JSON with this structure:

{
  "summary": "...",
  "sun_exposure": "...",
  "recommended_zones": [
    {
      "zone_name": "...",
      "plants": ["..."],
      "notes": "..."
    }
  ],
  "step_by_step_plan": [
    "Step 1 ...",
    "Step 2 ..."
  ],
  "shopping_list": [
    {"item": "...", "quantity": "..."}
  ]
}

Be realistic, affordable, and focused on food production.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ],
            }
        ],
        temperature=0.4,
    )

    ai_output = response.choices[0].message.content

    return {
        "filename": file.filename,
        "analysis": ai_output
    }
