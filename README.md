# Yard2Garden Backend (Cloud Run)

This service:
- Accepts a user photo (base64) + optional approximate lat/lng
- Generates a photorealistic "after" garden image (base64 png)
- Generates a professional step-by-step garden plan (markdown)
- Stores nothing

## Endpoint
POST /v1/yard2garden

## Environment / Secrets
This service loads the OpenAI API key in this order:
1) OPENAI_API_KEY environment variable (local dev)
2) Google Secret Manager secret name from OPENAI_API_KEY_SECRET_NAME (Cloud Run)

Recommended:
- Secret name: openai_api_key
- Env var: OPENAI_API_KEY_SECRET_NAME=openai_api_key

## Run locally
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
uvicorn main:app --reload --port 8080

## Deploy (high level)
Build + deploy to Cloud Run, set:
- OPENAI_API_KEY_SECRET_NAME=openai_api_key
- (optional) ALLOWED_ORIGINS="*"
