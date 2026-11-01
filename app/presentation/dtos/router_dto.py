from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ChatMessageDto(BaseModel):
    """OpenAI standard message format."""

    role: str
    content: str


class ChatCompletionRequestDto(BaseModel):
    """OpenAI compatible chat completions payload with dynamic routing strategy."""

    model: str = "default-instruct"
    messages: List[ChatMessageDto]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    routing_strategy: Optional[str] = "COMPLEXITY_ADAPTIVE"
    stream: Optional[bool] = False


class UsageDto(BaseModel):
    """Token usage."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatChoiceDto(BaseModel):
    """Choice output."""

    index: int
    message: ChatMessageDto
    finish_reason: str = "stop"


class ChatCompletionResponseDto(BaseModel):
    """OpenAI compatible response enriched with routing metadata."""

    id: str
    object: str = "chat.completion"
    model: str
    choices: List[ChatChoiceDto]
    usage: UsageDto
    # Enriched routing transparency
    routed_node: str
    runtime_type: str
    routing_reason: str
    latency_ms: float
    is_cached: bool = False
    user_message: str = Field(..., description="Mensagem contextual em pt-BR.")
