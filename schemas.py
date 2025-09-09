from pydantic import BaseModel

class GenerateRequest(BaseModel):
    prompt: str
    model: str