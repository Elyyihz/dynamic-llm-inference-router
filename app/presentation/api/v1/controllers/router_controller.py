import logging
from app.core.usecase.route_inference_usecase import RouteInferenceUseCase
from app.core.usecase.get_cluster_telemetry_usecase import GetClusterTelemetryUseCase
from app.presentation.dtos.router_dto import (
    ChatCompletionRequestDto,
    ChatCompletionResponseDto,
    ChatChoiceDto,
    ChatMessageDto,
    UsageDto,
)
from app.core.domain.models.route import InferenceRequest
from app.core.domain.enum.runtime_type import RoutingStrategy

logger = logging.getLogger("llm_router.presentation.controller")


class RouterController:
    """Handles HTTP requests and response mapping for LLM inference gateway."""

    def __init__(
        self,
        route_usecase: RouteInferenceUseCase,
        telemetry_usecase: GetClusterTelemetryUseCase,
    ):
        """Initializes with use case dependencies."""
        self.route_usecase = route_usecase
        self.telemetry_usecase = telemetry_usecase

    async def execute_inference(self, req: ChatCompletionRequestDto) -> InferenceResponse:
        """Executes domain routing usecase from presentation DTO."""
        logger.info("Step started: Handling chat completion request for model=%s", req.model)

        strategy_str = (req.routing_strategy or "COMPLEXITY_ADAPTIVE").upper()
        strategy = (
            RoutingStrategy[strategy_str]
            if strategy_str in RoutingStrategy.__members__
            else RoutingStrategy.COMPLEXITY_ADAPTIVE
        )

        domain_request = InferenceRequest(
            model=req.model,
            messages=[{"role": m.role, "content": m.content} for m in req.messages],
            temperature=req.temperature or 0.7,
            max_tokens=req.max_tokens or 512,
            routing_strategy=strategy,
            stream=bool(req.stream),
        )

        return await self.route_usecase.execute(domain_request)

    async def handle_chat_completion(self, req: ChatCompletionRequestDto) -> ChatCompletionResponseDto:
        """Processes chat completion with dynamic routing."""
        response = await self.execute_inference(req)

        cached_indicator = " [CACHE HIT]" if response.is_cached else ""
        user_msg = (
            f"Requisição roteada dinamicamente{cached_indicator} para o nó '{response.node_used}' "
            f"({response.route_decision.selected_runtime_type.value}) em {response.total_latency_ms:.1f}ms."
        )

        logger.info("Step concluded: Completed response for request_id=%s", response.request_id)

        return ChatCompletionResponseDto(
            id=response.request_id,
            model=response.model_used,
            choices=[
                ChatChoiceDto(
                    index=0,
                    message=ChatMessageDto(role="assistant", content=response.content),
                    finish_reason="stop",
                )
            ],
            usage=UsageDto(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.prompt_tokens + response.completion_tokens,
            ),
            routed_node=response.node_used,
            runtime_type=response.route_decision.selected_runtime_type.value,
            routing_reason=response.route_decision.reasoning,
            latency_ms=response.total_latency_ms,
            is_cached=response.is_cached,
            user_message=user_msg,
        )

    async def get_cluster_status(self):
        """Queries cluster nodes and health."""
        return await self.telemetry_usecase.execute()
