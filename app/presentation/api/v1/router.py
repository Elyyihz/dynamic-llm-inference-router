from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from app.presentation.dtos.router_dto import ChatCompletionRequestDto, ChatCompletionResponseDto
from app.presentation.api.v1.controllers.router_controller import RouterController
from app.infrastructure.gateway.in_memory_node_registry_impl import InMemoryNodeRegistryGatewayImpl
from app.infrastructure.gateway.runtime_client_impl import RuntimeClientGatewayImpl
from app.infrastructure.gateway.prometheus_metrics_impl import PrometheusMetricsGatewayImpl
from app.infrastructure.gateway.in_memory_semantic_cache_impl import InMemorySemanticCacheGatewayImpl
from app.infrastructure.gateway.sse_streaming_adapter import generate_openai_sse_stream
from app.core.usecase.route_inference_usecase import RouteInferenceUseCase
from app.core.usecase.get_cluster_telemetry_usecase import GetClusterTelemetryUseCase

router = APIRouter(tags=["Dynamic LLM Router"])

# Shared registry, metrics, and cache singletons
shared_registry = InMemoryNodeRegistryGatewayImpl()
shared_client = RuntimeClientGatewayImpl()
shared_metrics = PrometheusMetricsGatewayImpl()
shared_cache = InMemorySemanticCacheGatewayImpl()


def get_router_controller() -> RouterController:
    """Dependency injection factory wiring Clean Architecture layers."""
    route_uc = RouteInferenceUseCase(
        shared_registry,
        shared_client,
        shared_metrics,
        cache_gateway=shared_cache,
    )
    telemetry_uc = GetClusterTelemetryUseCase(shared_registry)
    return RouterController(route_uc, telemetry_uc)


@router.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequestDto,
    controller: RouterController = Depends(get_router_controller),
):
    """OpenAI-compatible chat completions proxy that dynamically balances between vLLM and llama.cpp.

    Supports SSE streaming when stream=true.
    """
    if req.stream:
        domain_response = await controller.execute_inference(req)
        return StreamingResponse(
            generate_openai_sse_stream(domain_response),
            media_type="text/event-stream",
        )
    return await controller.handle_chat_completion(req)


@router.get("/api/v1/cluster")
async def get_cluster_status(
    controller: RouterController = Depends(get_router_controller),
):
    """Returns cluster topology, active runtimes, and p95 latency telemetry."""
    return await controller.get_cluster_status()


@router.get("/health")
def health_check():
    """Readiness probe."""
    return {"status": "UP", "service": "dynamic-llm-inference-router"}
