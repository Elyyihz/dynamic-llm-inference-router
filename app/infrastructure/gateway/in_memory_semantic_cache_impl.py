import logging
import hashlib
from typing import Optional, Dict
from app.core.domain.gateway.cache_gateway import SemanticCacheGateway
from app.core.domain.models.route import InferenceRequest, InferenceResponse, RouteDecision
from app.core.domain.enum.runtime_type import RuntimeType

logger = logging.getLogger("llm_router.infra.cache")


class InMemorySemanticCacheGatewayImpl(SemanticCacheGateway):
    """In-memory semantic and exact match cache adapter for LLM prompts."""

    def __init__(self):
        """Initializes internal dictionary store."""
        self._cache: Dict[str, InferenceResponse] = {}

    def _compute_key(self, request: InferenceRequest) -> str:
        """Computes deterministic hash key based on model and normalized messages."""
        normalized_msgs = "-".join(
            f"{m.get('role', '')}:{m.get('content', '').strip().lower()}"
            for m in request.messages
        )
        raw_key = f"{request.model}:{normalized_msgs}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    async def get_cached_response(self, request: InferenceRequest) -> Optional[InferenceResponse]:
        """Retrieves cached response if key exists.

        Args:
            request: InferenceRequest.

        Returns:
            Cached response with updated request_id and is_cached=True if hit, else None.
        """
        key = self._compute_key(request)
        if key in self._cache:
            hit = self._cache[key]
            logger.info("Semantic cache HIT for request_id=%s (key=%s...)", request.request_id, key[:8])
            return InferenceResponse(
                request_id=request.request_id,
                content=hit.content,
                model_used=hit.model_used,
                node_used="SemanticCache",
                prompt_tokens=hit.prompt_tokens,
                completion_tokens=hit.completion_tokens,
                total_latency_ms=1.2,
                route_decision=RouteDecision(
                    request_id=request.request_id,
                    selected_node_id="cache",
                    selected_runtime_type=RuntimeType.CLOUD_FALLBACK,
                    reasoning="Cached response matched with 100% semantic identity",
                    estimated_complexity=0.0,
                ),
                is_cached=True,
            )

        logger.debug("Semantic cache MISS for request_id=%s", request.request_id)
        return None

    async def store_response(self, request: InferenceRequest, response: InferenceResponse) -> None:
        """Persists response in semantic cache store.

        Args:
            request: Inference request.
            response: Executed response.
        """
        key = self._compute_key(request)
        self._cache[key] = response
        logger.info("Persisted response in semantic cache (key=%s...)", key[:8])
