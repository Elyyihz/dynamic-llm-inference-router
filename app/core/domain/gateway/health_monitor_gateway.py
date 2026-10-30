from abc import ABC, abstractmethod
from typing import List, Optional
from app.core.domain.models.route import RuntimeNode


class HealthMonitorGateway(ABC):
    """Abstract interface for querying node health and active topology."""

    @abstractmethod
    async def list_available_nodes(self) -> List[RuntimeNode]:
        """Returns all actively registered and healthy runtime nodes.

        Returns:
            List of healthy RuntimeNode instances.
        """
        pass

    @abstractmethod
    async def register_node(self, node: RuntimeNode) -> None:
        """Registers a new runtime node into the cluster."""
        pass

    @abstractmethod
    async def update_node_latency(self, node_id: str, latency_ms: float, is_error: bool = False) -> None:
        """Updates real-time rolling p95 latency and active request metrics."""
        pass
