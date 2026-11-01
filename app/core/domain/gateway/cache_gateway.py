from abc import ABC, abstractmethod
from typing import Optional
from app.core.domain.models.route import InferenceRequest, InferenceResponse


class SemanticCacheGateway(ABC):
    """Abstract interface defining semantic caching for LLM requests."""

    @abstractmethod
    async def get_cached_response(self, request: InferenceRequest) -> Optional[InferenceResponse]:
        """Retrieves a cached response if a semantically equivalent prompt exists.

        Args:
            request: Incoming inference request.

        Returns:
            Cached InferenceResponse if hit, None if miss.
        """
        pass

    @abstractmethod
    async def store_response(self, request: InferenceRequest, response: InferenceResponse) -> None:
        """Stores an inference response in the semantic cache.

        Args:
            request: Originating inference request.
            response: Executed inference response.
        """
        pass
