from abc import ABC, abstractmethod
from app.core.domain.models.route import RouteDecision, InferenceResponse


class MetricsGateway(ABC):
    """Abstract interface defining metrics collection and OpenTelemetry emission."""

    @abstractmethod
    def record_route_decision(self, decision: RouteDecision) -> None:
        """Increments Prometheus counter for routing targets."""
        pass

    @abstractmethod
    def record_inference_telemetry(self, response: InferenceResponse) -> None:
        """Records latency histogram, token counts, and error rates."""
        pass
