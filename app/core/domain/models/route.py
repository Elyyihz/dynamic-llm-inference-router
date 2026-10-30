from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import uuid

from app.core.domain.enum.runtime_type import RuntimeType, RuntimeStatus, RoutingStrategy


@dataclass
class RuntimeNode:
    """Represents an active LLM inference server node.

    Attributes:
        node_id: Unique node identifier.
        name: Human-readable label.
        runtime_type: VLLM, LLAMA_CPP, etc.
        base_url: HTTP endpoint of the runtime.
        status: Current health status.
        p95_latency_ms: Rolling 95th percentile latency.
        active_requests: Current in-flight request count.
        supported_models: List of model names hosted by this node.
    """

    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    runtime_type: RuntimeType = RuntimeType.VLLM
    base_url: str = "http://localhost:8001/v1"
    status: RuntimeStatus = RuntimeStatus.HEALTHY
    p95_latency_ms: float = 120.0
    active_requests: int = 0
    supported_models: List[str] = field(default_factory=list)


@dataclass
class InferenceRequest:
    """Incoming user chat completion request.

    Attributes:
        request_id: UUID of request.
        model: Requested model name.
        messages: List of conversation messages (role, content).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        routing_strategy: Strategy override.
    """

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model: str = "default-instruct"
    messages: List[Dict[str, str]] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 512
    routing_strategy: RoutingStrategy = RoutingStrategy.COMPLEXITY_ADAPTIVE


@dataclass
class RouteDecision:
    """Auditable log of the routing decision taken by the gateway.

    Attributes:
        request_id: Linked request UUID.
        selected_node_id: Chosen runtime node.
        selected_runtime_type: Type of runtime.
        reasoning: Algorithmic justification (e.g. latency vs tokens).
        estimated_complexity: Inferred prompt complexity score.
        fallback_occurred: Flag indicating primary failure and reroute.
    """

    request_id: str
    selected_node_id: str
    selected_runtime_type: RuntimeType
    reasoning: str
    estimated_complexity: float
    fallback_occurred: bool = False


@dataclass
class InferenceResponse:
    """Inference output payload with routing and latency metadata.

    Attributes:
        request_id: UUID.
        content: Textual generated output.
        model_used: Executing model.
        node_used: Node label.
        prompt_tokens: Input token count.
        completion_tokens: Output token count.
        total_latency_ms: End-to-end processing latency.
        route_decision: Decision record.
    """

    request_id: str
    content: str
    model_used: str
    node_used: str
    prompt_tokens: int
    completion_tokens: int
    total_latency_ms: float
    route_decision: RouteDecision
