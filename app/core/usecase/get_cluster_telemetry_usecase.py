import logging
from typing import List, Dict, Any
from app.core.domain.gateway.health_monitor_gateway import HealthMonitorGateway

logger = logging.getLogger("llm_router.usecase.telemetry")


class GetClusterTelemetryUseCase:
    """Aggregates cluster health, node statuses, and routing metrics."""

    def __init__(self, health_gateway: HealthMonitorGateway):
        """Initializes with health gateway port.

        Args:
            health_gateway: HealthMonitorGateway port.
        """
        self.health_gateway = health_gateway

    async def execute(self) -> Dict[str, Any]:
        """Queries cluster state.

        Returns:
            Dict containing cluster summary and active nodes.
        """
        nodes = await self.health_gateway.list_available_nodes()
        total_active = sum(n.active_requests for n in nodes)
        avg_p95 = sum(n.p95_latency_ms for n in nodes) / len(nodes) if nodes else 0.0

        return {
            "node_count": len(nodes),
            "total_active_requests": total_active,
            "cluster_avg_p95_latency_ms": round(avg_p95, 1),
            "nodes": [
                {
                    "node_id": n.node_id,
                    "name": n.name,
                    "runtime_type": n.runtime_type.value,
                    "status": n.status.value,
                    "base_url": n.base_url,
                    "p95_latency_ms": n.p95_latency_ms,
                    "active_requests": n.active_requests,
                }
                for n in nodes
            ],
        }
