import logging
import requests
import time
from fastapi import APIRouter, HTTPException

from config import OPENROUTER_API_KEY, OPENROUTER_URL, LOG_FILE, MODELS
from schemas import GenerateRequest
from utils import post_with_retry



# Логирование
logger = logging.getLogger("server")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

router = APIRouter()

@router.get("/models")
async def get_models():
    return MODELS


@router.post("/generate")
async def generate(request: GenerateRequest):
    if request.model not in MODELS:
        raise HTTPException(status_code=400, detail="Unknown model")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": request.model,
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": request.max_tokens,
    }

    start = time.time()
    resp = post_with_retry(OPENROUTER_URL, headers, payload)
    latency = time.time() - start

    if resp.status_code != 200:
        logger.error(f"OpenRouter error: {resp.status_code} {resp.text}")
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    tokens_used = data.get("usage", {}).get("total_tokens")

    return {
        "response": text,
        "tokens_used": tokens_used,
        "latency_seconds": latency,
    }
