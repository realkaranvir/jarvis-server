from pydantic import BaseModel, Field


class LLMRequest(BaseModel):
    prompt: str = Field(min_length=1)
    backend: str | None = None


class LLMResponse(BaseModel):
    answer: str
    backend: str
    model: str


class STTResponse(BaseModel):
    text: str
    model: str
