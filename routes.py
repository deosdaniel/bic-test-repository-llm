import logging
import requests
from fastapi import APIRouter, HTTPException
from config import OPENROUTER_API_KEY, OPENROUTER_URL, LOG_FILE, MODELS
from schemas import GenerateRequest


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
    }

    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            logger.error(f"OpenRouter error: {resp.status_code} {resp.text}")
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        data = resp.json()
        # OpenAI-совместимый формат: choices[0].message.content
        text = data["choices"][0]["message"]["content"]
        return {"response": text}

    except Exception as e:
        logger.exception("Error in /generate")
        raise HTTPException(status_code=500, detail=str(e))