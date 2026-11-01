import pytest
from app.core.domain.models.route import InferenceRequest, RuntimeNode, InferenceResponse, RouteDecision
from app.core.domain.enum.runtime_type import RuntimeType, RoutingStrategy, RuntimeStatus
from app.infrastructure.gateway.in_memory_node_registry_impl import InMemoryNodeRegistryGatewayImpl
from app.infrastructure.gateway.runtime_client_impl import RuntimeClientGatewayImpl
from app.infrastructure.gateway.prometheus_metrics_impl import PrometheusMetricsGatewayImpl
from app.infrastructure.gateway.in_memory_semantic_cache_impl import InMemorySemanticCacheGatewayImpl
from app.infrastructure.gateway.sse_streaming_adapter import generate_openai_sse_stream
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


@pytest.mark.asyncio
async def test_semantic_cache_hit():
    """Validates that identical queries return cached response with near-zero latency."""
    registry = InMemoryNodeRegistryGatewayImpl()
    client = RuntimeClientGatewayImpl()
    metrics = PrometheusMetricsGatewayImpl()
    cache = InMemorySemanticCacheGatewayImpl()

    usecase = RouteInferenceUseCase(registry, client, metrics, cache_gateway=cache)

    req = InferenceRequest(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": "What is clean architecture?"}],
        temperature=0.0,
    )

    first_resp = await usecase.execute(req)
    assert first_resp.is_cached is False

    second_resp = await usecase.execute(req)
    assert second_resp.is_cached is True
    assert second_resp.content == first_resp.content
    assert second_resp.total_latency_ms < 5.0


@pytest.mark.asyncio
async def test_sse_streaming_generation():
    """Validates OpenAI-compatible SSE chunk streaming."""
    decision = RouteDecision(
        request_id="test-req-123",
        selected_node_id="node-vllm-1",
        selected_runtime_type=RuntimeType.VLLM,
        reasoning="Test routing",
        estimated_complexity=0.5,
    )
    resp = InferenceResponse(
        request_id="test-req-123",
        content="Streaming tokens test response",
        model_used="test-model",
        node_used="node-vllm-1",
        prompt_tokens=10,
        completion_tokens=4,
        total_latency_ms=15.0,
        route_decision=decision,
    )

    chunks = []
    async for chunk in generate_openai_sse_stream(resp):
        chunks.append(chunk)

    assert len(chunks) == 5  # 4 words + 1 [DONE]
    assert chunks[0].startswith("data: ")
    assert chunks[-1] == "data: [DONE]\n\n"
