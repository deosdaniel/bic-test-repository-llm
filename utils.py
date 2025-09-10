import time
import requests
from fastapi import HTTPException
import logging

logger = logging.getLogger("server")

def post_with_retry(url: str, headers: dict, json_payload: dict,
                    max_retries: int = 5, base_delay: float = 0.5):
    for attempt in range(max_retries):
        resp = requests.post(url, headers=headers, json=json_payload, timeout=60)
        if resp.status_code == 429:
            sleep_for = base_delay * (2 ** attempt)
            logger.warning(f"429 Too Many Requests, retry in {sleep_for:.1f}s (attempt {attempt+1})")
            time.sleep(sleep_for)
            continue
        return resp
    logger.error("Rate limit exceeded after all retries")
    raise HTTPException(status_code=429, detail="Rate limit exceeded after retries")