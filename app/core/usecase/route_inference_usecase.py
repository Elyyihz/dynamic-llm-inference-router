import logging
from typing import Optional
from app.core.domain.models.route import (
    InferenceRequest,
    InferenceResponse,
    RouteDecision,
    RuntimeNode,
)
from app.core.domain.enum.runtime_type import RuntimeType, RoutingStrategy
from app.core.domain.gateway.health_monitor_gateway import HealthMonitorGateway
from app.core.domain.gateway.runtime_client_gateway import RuntimeClientGateway
from app.core.domain.gateway.metrics_gateway import MetricsGateway
from app.core.domain.gateway.cache_gateway import SemanticCacheGateway

logger = logging.getLogger("llm_router.usecase.route")


class RouteInferenceUseCase:
    """Evaluates request parameters, selects optimal runtime node, and executes inference."""

    def __init__(
        self,
        health_gateway: HealthMonitorGateway,
        runtime_gateway: RuntimeClientGateway,
        metrics_gateway: MetricsGateway,
        cache_gateway: Optional[SemanticCacheGateway] = None,
    ):
        """Initializes router use case with gateway ports.

        Args:
            health_gateway: Cluster topology port.
            runtime_gateway: Inference runtime execution port.
            metrics_gateway: Prometheus / OpenTelemetry telemetry port.
            cache_gateway: Optional semantic caching port.
        """
        self.health_gateway = health_gateway
        self.runtime_gateway = runtime_gateway
        self.metrics_gateway = metrics_gateway
        self.cache_gateway = cache_gateway

    def _estimate_complexity(self, req: InferenceRequest) -> float:
        """Estimates token length and reasoning complexity [0.0 - 1.0]."""
        total_chars = sum(len(m.get("content", "")) for m in req.messages)
        # > 1500 chars or high max_tokens implies heavier reasoning workload
        complexity = min(1.0, total_chars / 3000.0)
        if req.max_tokens > 1024:
            complexity = max(complexity, 0.8)
        return round(complexity, 2)

    async def execute(self, request: InferenceRequest) -> InferenceResponse:
        """Dynamically routes inference request to optimal runtime node.

        Args:
            request: InferenceRequest domain entity.

        Returns:
            InferenceResponse entity.
        """
        logger.info("Step started: Routing inference request_id=%s, model=%s", request.request_id, request.model)

        # 1. Check semantic cache
        if self.cache_gateway:
            cached_resp = await self.cache_gateway.get_cached_response(request)
            if cached_resp:
                logger.info("Step concluded: Returning cached response for request_id=%s (latency=%.1fms)", request.request_id, cached_resp.total_latency_ms)
                return cached_resp

        nodes = await self.health_gateway.list_available_nodes()
        if not nodes:
            logger.error("Decision taken: No healthy runtime nodes available in cluster")
            raise RuntimeError("Nenhum nó de inferência disponível no cluster.")

        complexity = self._estimate_complexity(request)
        logger.info("Estimated prompt complexity score: %.2f", complexity)

        # Dynamic selection policy:
        # High complexity (>0.5) -> Route to vLLM (high throughput, GPU KV-cache)
        # Low complexity (<=0.5) -> Route to llama.cpp (low-cost, instant TTFT)
        target_node = None
        reasoning = ""

        if request.routing_strategy == RoutingStrategy.COMPLEXITY_ADAPTIVE:
            if complexity > 0.5:
                vllm_nodes = [n for n in nodes if n.runtime_type == RuntimeType.VLLM]
                if vllm_nodes:
                    target_node = min(vllm_nodes, key=lambda n: n.active_requests)
                    reasoning = f"High prompt complexity ({complexity:.2f}) routed to vLLM GPU node '{target_node.name}'"
            if not target_node:
                # Fallback to least latency node
                target_node = min(nodes, key=lambda n: n.p95_latency_ms)
                reasoning = f"Routed to lowest latency node '{target_node.name}' (p95: {target_node.p95_latency_ms}ms)"
        else:
            target_node = min(nodes, key=lambda n: n.p95_latency_ms)
            reasoning = f"Policy '{request.routing_strategy.value}' selected node '{target_node.name}'"

        logger.info("Decision taken: %s", reasoning)

        decision = RouteDecision(
            request_id=request.request_id,
            selected_node_id=target_node.node_id,
            selected_runtime_type=target_node.runtime_type,
            reasoning=reasoning,
            estimated_complexity=complexity,
        )

        self.metrics_gateway.record_route_decision(decision)

        try:
            response = await self.runtime_gateway.execute_inference(target_node, request, decision)
            await self.health_gateway.update_node_latency(target_node.node_id, response.total_latency_ms, is_error=False)
            self.metrics_gateway.record_inference_telemetry(response)
            if self.cache_gateway:
                await self.cache_gateway.store_response(request, response)
            logger.info("Step concluded: Inference succeeded on node=%s (latency=%.1fms)", target_node.name, response.total_latency_ms)
            return response
        except Exception as exc:
            logger.warning("Decision taken: Primary node %s failed (%s); triggering circuit breaker fallback", target_node.name, exc)
            fallback_nodes = [n for n in nodes if n.node_id != target_node.node_id]
            if not fallback_nodes:
                raise exc
            fallback_node = fallback_nodes[0]
            fallback_decision = RouteDecision(
                request_id=request.request_id,
                selected_node_id=fallback_node.node_id,
                selected_runtime_type=fallback_node.runtime_type,
                reasoning=f"Circuit breaker fallback to '{fallback_node.name}' following primary failure",
                estimated_complexity=complexity,
                fallback_occurred=True,
            )
            response = await self.runtime_gateway.execute_inference(fallback_node, request, fallback_decision)
            self.metrics_gateway.record_inference_telemetry(response)
            logger.info("Step concluded: Fallback inference succeeded on node=%s", fallback_node.name)
            return response
