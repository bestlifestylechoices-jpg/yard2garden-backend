from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
import base64

app = FastAPI()

@app.get("/")
def root():
    return {"status": "yard2garden-backend is alive 🌱"}

@app.post("/v1/yard2garden")
async def yard2garden(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    budget: str = Form("medium"),
    upkeep: str = Form("medium"),
    has_location: str = Form("false"),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
):
    # Accept either "file" or "image"
    upload = file or image
    if upload is None:
        return {"error": "No file uploaded. Send multipart field 'file' or 'image'."}

    img_bytes = await upload.read()

    # TODO: Replace these two calls with YOUR real logic:
    # 1) plan_text = analyze_yard_to_plan(img_bytes, budget, upkeep, lat, lon, has_location)
    # 2) png_bytes = generate_transformed_garden_image(img_bytes, plan_text, budget, upkeep)

    plan_text = f"Step 1: Assess sun exposure.\nStep 2: Mark zones.\nBudget: {budget}, Upkeep: {upkeep}"
    png_bytes = b""  # <-- MUST be real PNG bytes

    if not png_bytes:
        return {"error": "Image generation returned empty bytes. Check OpenAI call + API key + model."}

    image_b64_png = base64.b64encode(png_bytes).decode("utf-8")

    return {
        "image_b64_png": image_b64_png,
        "plan_text": plan_text
    }
