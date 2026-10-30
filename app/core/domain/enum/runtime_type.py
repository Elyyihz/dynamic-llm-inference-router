from enum import Enum


class RuntimeType(str, Enum):
    """Inference engine runtime technology."""

    VLLM = "VLLM"
    LLAMA_CPP = "LLAMA_CPP"
    OLLAMA = "OLLAMA"
    CLOUD_FALLBACK = "CLOUD_FALLBACK"


class RuntimeStatus(str, Enum):
    """Health and operational availability state of a runtime node."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    DRAINING = "DRAINING"


class RoutingStrategy(str, Enum):
    """Dynamic routing algorithm."""

    LEAST_LATENCY = "LEAST_LATENCY"
    LOWEST_COST = "LOWEST_COST"
    COMPLEXITY_ADAPTIVE = "COMPLEXITY_ADAPTIVE"
    ROUND_ROBIN = "ROUND_ROBIN"
