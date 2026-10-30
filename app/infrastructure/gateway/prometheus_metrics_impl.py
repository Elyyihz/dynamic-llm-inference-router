import logging
from app.core.domain.gateway.metrics_gateway import MetricsGateway
from app.core.domain.models.route import RouteDecision, InferenceResponse

logger = logging.getLogger("llm_router.infra.prometheus_metrics")


class PrometheusMetricsGatewayImpl(MetricsGateway):
    """Prometheus telemetry recorder."""

    def __init__(self):
        """Initializes internal counters."""
        self.routing_counters = {}
        self.total_tokens_generated = 0

    def record_route_decision(self, decision: RouteDecision) -> None:
        """Records route decision."""
        key = decision.selected_runtime_type.value
        self.routing_counters[key] = self.routing_counters.get(key, 0) + 1
        logger.info("Prometheus Metric: router_decisions_total{target='%s'} incremented to %d", key, self.routing_counters[key])

    def record_inference_telemetry(self, response: InferenceResponse) -> None:
        """Records inference telemetry."""
        self.total_tokens_generated += response.completion_tokens
        logger.info(
            "Prometheus Metric: router_latency_ms{node='%s'}=%.1fms, total_tokens=%d",
            response.node_used,
            response.total_latency_ms,
            self.total_tokens_generated,
        )
