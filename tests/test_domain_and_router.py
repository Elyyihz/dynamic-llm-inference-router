import pytest
from app.core.domain.models.route import InferenceRequest, RuntimeNode
from app.core.domain.enum.runtime_type import RuntimeType, RoutingStrategy, RuntimeStatus
from app.infrastructure.gateway.in_memory_node_registry_impl import InMemoryNodeRegistryGatewayImpl
from app.infrastructure.gateway.runtime_client_impl import RuntimeClientGatewayImpl
from app.infrastructure.gateway.prometheus_metrics_impl import PrometheusMetricsGatewayImpl
from app.core.usecase.route_inference_usecase import RouteInferenceUseCase


def test_domain_model_creation():
    """Validates default instantiation of RuntimeNode."""
    node = RuntimeNode(
        name="test-vllm",
        runtime_type=RuntimeType.VLLM,
        base_url="http://localhost:8001/v1",
    )
    assert node.node_id is not None
    assert node.status == RuntimeStatus.HEALTHY
    assert node.runtime_type == RuntimeType.VLLM


@pytest.mark.asyncio
async def test_adaptive_routing_to_vllm_for_complex_prompt():
    """Validates that high complexity prompts route to vLLM."""
    registry = InMemoryNodeRegistryGatewayImpl()
    client = RuntimeClientGatewayImpl()
    metrics = PrometheusMetricsGatewayImpl()

    usecase = RouteInferenceUseCase(registry, client, metrics)

    # Long text prompt
    long_content = "Please analyze the financial balance sheet: " * 60
    req = InferenceRequest(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": long_content}],
        max_tokens=1500,
        routing_strategy=RoutingStrategy.COMPLEXITY_ADAPTIVE,
    )

    resp = await usecase.execute(req)
    assert resp.request_id == req.request_id
    assert resp.route_decision.selected_runtime_type == RuntimeType.VLLM
    assert resp.total_latency_ms > 0.0
