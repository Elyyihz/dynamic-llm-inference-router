import logging
from typing import List, Dict, Optional
from app.core.domain.gateway.health_monitor_gateway import HealthMonitorGateway
from app.core.domain.models.route import RuntimeNode
from app.core.domain.enum.runtime_type import RuntimeType, RuntimeStatus
from app.infrastructure.config.settings import settings

logger = logging.getLogger("llm_router.infra.registry")


class InMemoryNodeRegistryGatewayImpl(HealthMonitorGateway):
    """In-memory active cluster node registry."""

    def __init__(self):
        """Pre-seeds standard runtime nodes (vLLM and llama.cpp)."""
        self._nodes: Dict[str, RuntimeNode] = {}
        self._seed_default_nodes()

    def _seed_default_nodes(self):
        """Seeds default runtime topology."""
        vllm = RuntimeNode(
            node_id="node-vllm-primary",
            name="vLLM-GPU-Cluster-01",
            runtime_type=RuntimeType.VLLM,
            base_url=settings.VLLM_NODE_URL,
            status=RuntimeStatus.HEALTHY,
            p95_latency_ms=95.0,
            active_requests=2,
            supported_models=["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3-8B"],
        )
        llamacpp = RuntimeNode(
            node_id="node-llamacpp-edge",
            name="llama.cpp-CPU-Edge-01",
            runtime_type=RuntimeType.LLAMA_CPP,
            base_url=settings.LLAMACPP_NODE_URL,
            status=RuntimeStatus.HEALTHY,
            p95_latency_ms=145.0,
            active_requests=0,
            supported_models=["mistral-7b-instruct-q4", "phi-3-mini"],
        )
        self._nodes[vllm.node_id] = vllm
        self._nodes[llamacpp.node_id] = llamacpp

    async def list_available_nodes(self) -> List[RuntimeNode]:
        """Returns healthy nodes."""
        return [n for n in self._nodes.values() if n.status == RuntimeStatus.HEALTHY]

    async def register_node(self, node: RuntimeNode) -> None:
        """Registers new node."""
        self._nodes[node.node_id] = node

    async def update_node_latency(self, node_id: str, latency_ms: float, is_error: bool = False) -> None:
        """Updates rolling p95 latency."""
        node = self._nodes.get(node_id)
        if node:
            # Exponential moving average
            node.p95_latency_ms = round((node.p95_latency_ms * 0.8) + (latency_ms * 0.2), 1)
            if is_error:
                node.status = RuntimeStatus.DEGRADED
