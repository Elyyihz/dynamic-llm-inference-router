from abc import ABC, abstractmethod
from app.core.domain.models.route import RuntimeNode, InferenceRequest, InferenceResponse, RouteDecision


class RuntimeClientGateway(ABC):
    """Abstract interface for executing inference requests against backend runtimes."""

    @abstractmethod
    async def execute_inference(
        self,
        node: RuntimeNode,
        request: InferenceRequest,
        decision: RouteDecision
    ) -> InferenceResponse:
        """Executes HTTP request to runtime endpoint (vLLM / llama.cpp).

        Args:
            node: Target RuntimeNode selected by router.
            request: InferenceRequest payload.
            decision: RouteDecision context.

        Returns:
            InferenceResponse entity with token metrics and response content.
        """
        pass
