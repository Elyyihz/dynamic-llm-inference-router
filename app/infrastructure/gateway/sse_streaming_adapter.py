import json
import asyncio
import time
from typing import AsyncGenerator
from app.core.domain.models.route import InferenceResponse


async def generate_openai_sse_stream(response: InferenceResponse) -> AsyncGenerator[str, None]:
    """Generates an OpenAI-compatible Server-Sent Events (SSE) token stream.

    Deconstructs the complete generated text into word/sub-word chunks, streaming them
    as JSON SSE payloads followed by the standard '[DONE]' token.

    Args:
        response: Inferred response entity.

    Yields:
        SSE formatted event lines.
    """
    words = response.content.split(" ")
    created_ts = int(time.time())

    for idx, word in enumerate(words):
        chunk_text = word if idx == len(words) - 1 else f"{word} "
        chunk_payload = {
            "id": response.request_id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": response.model_used,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk_text},
                    "finish_reason": None if idx < len(words) - 1 else "stop",
                }
            ],
            "routed_node": response.node_used,
        }
        yield f"data: {json.dumps(chunk_payload)}\n\n"
        # Simulate realistic token generation rate (~25ms per token)
        await asyncio.sleep(0.02)

    yield "data: [DONE]\n\n"
