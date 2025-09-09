import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = os.getenv(
    "OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions"
)
LOG_FILE = os.getenv("LOG_FILE", "server_logs.txt")

MODELS = [
    "deepseek/deepseek-chat-v3.1:free",
    "z-ai/glm-4.5-air:free",
    "moonshotai/kimi-k2:free",
]