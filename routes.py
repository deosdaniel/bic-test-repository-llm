import logging
import requests
import time
import csv
import statistics
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse, HTMLResponse

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
    logger.info("GET /models called")
    return MODELS


@router.post("/generate")
async def generate(request: GenerateRequest, stream: bool = Query(False)):
    logger.info(f"POST /generate called with model={request.model}, stream={stream}, max_tokens={request.max_tokens}")

    if request.model not in MODELS:
        logger.error(f"Unknown model: {request.model}")
        raise HTTPException(status_code=400, detail="Unknown model")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": request.model,
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": request.max_tokens,
        "stream": stream,
    }

    if not stream:
        start = time.time()
        resp = post_with_retry(OPENROUTER_URL, headers, payload)
        latency = time.time() - start

        if resp.status_code != 200:
            logger.error(f"OpenRouter error: {resp.status_code} {resp.text}")
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        tokens_used = data.get("usage", {}).get("total_tokens")

        logger.info(f"Generate success: model={request.model}, latency={latency:.2f}s, tokens={tokens_used}")

        return {
            "response": text,
            "tokens_used": tokens_used,
            "latency_seconds": latency,
        }

    else:
        logger.info(f"Streaming response started for model={request.model}")

        def event_stream():
            with requests.post(OPENROUTER_URL, headers=headers, json=payload, stream=True) as r:
                for line in r.iter_lines():
                    if line:
                        yield f"data: {line.decode('utf-8')}\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")



@router.post("/benchmark")
async def benchmark(
    prompt_file: UploadFile = File(...),
    model: str = Form(...),
    runs: int = Form(5),
    visualize: bool = Query(False)
):
    logger.info(f"POST /benchmark called with model={model}, runs={runs}, visualize={visualize}")

    if model not in MODELS:
        logger.error(f"Unknown model: {model}")
        raise HTTPException(status_code=400, detail="Unknown model")

    content = await prompt_file.read()
    prompts = [line.strip() for line in content.decode("utf-8").splitlines() if line.strip()]
    if not prompts:
        logger.error("No prompts found in file")
        raise HTTPException(status_code=400, detail="No prompts found in file")

    results = []  # список (prompt, run, latency)
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
                logger.error(f"Benchmark request failed: {resp.status_code} {resp.text}")
                results.append((prompt, i + 1, None))

    # CSV
    with open("benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "run", "latency"])
        for row in results:
            writer.writerow(row)

    # Статистика
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

    logger.info(f"Benchmark finished: prompts={len(prompts)}, runs={runs}, csv=benchmark_results.csv")

    if visualize:
        html = "<html><body><h2>Benchmark Results</h2><table border='1'><tr><th>Prompt</th><th>Avg</th><th>Min</th><th>Max</th><th>StdDev</th></tr>"
        for prompt, data in stats.items():
            html += f"<tr><td>{prompt}</td><td>{data['avg']}</td><td>{data['min']}</td><td>{data['max']}</td><td>{data['std_dev']}</td></tr>"
        html += "</table></body></html>"
        return HTMLResponse(content=html)

    return {"stats": stats, "csv_file": "benchmark_results.csv"}