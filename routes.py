import logging
import time
import csv
import statistics
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from config import OPENROUTER_API_KEY, OPENROUTER_URL, LOG_FILE, MODELS
from schemas import GenerateRequest
from utils import post_with_retry



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

@router.post("/benchmark")
async def benchmark(
    prompt_file: UploadFile = File(...),
    model: str = Form(...),
    runs: int = Form(5),
):
    if model not in MODELS:
        raise HTTPException(status_code=400, detail="Unknown model")

    content = await prompt_file.read()
    prompts = [line.strip() for line in content.decode("utf-8").splitlines() if line.strip()]
    if not prompts:
        raise HTTPException(status_code=400, detail="No prompts found in file")

    results = []
    for prompt in prompts:
        for i in range(runs):
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 128,
            }
            start = time.time()
            resp = post_with_retry(OPENROUTER_URL, headers, payload)
            elapsed = time.time() - start

            if resp.status_code == 200:
                results.append((prompt, i + 1, elapsed))
            else:
                results.append((prompt, i + 1, None))

    with open("benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "run", "latency"])
        for row in results:
            writer.writerow(row)

    stats = {}
    for prompt in prompts:
        latencies = [lat for p, r, lat in results if p == prompt and lat is not None]
        if latencies:
            stats[prompt] = {
                "avg": statistics.mean(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "std_dev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            }
        else:
            stats[prompt] = {"avg": None, "min": None, "max": None, "std_dev": None}

    return {"stats": stats, "csv_file": "benchmark_results.csv"}