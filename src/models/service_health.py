"""Models for service health assessment."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealthSnapshot:
    """Point-in-time health assessment for a service."""

    service_name: str
    timestamp: datetime
    status: HealthStatus
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    active_pods: int = 0
    desired_pods: int = 0
    avg_response_time_ms: float = 0.0
    error_rate: float = 0.0
    request_rate: float = 0.0

    @property
    def pod_availability(self) -> float:
        if self.desired_pods == 0:
            return 0.0
        return self.active_pods / self.desired_pods


@dataclass
class ServiceDependency:
    """Represents a dependency between services."""

    source_service: str
    target_service: str
    dependency_type: str  # "http", "grpc", "message_queue", "database"
    criticality: str  # "required", "optional", "fallback"


@dataclass
class ServiceMap:
    """Map of services and their dependencies for correlation."""

    services: list[str] = field(default_factory=list)
    dependencies: list[ServiceDependency] = field(default_factory=list)

    def get_downstream(self, service_name: str) -> list[str]:
        """Get services that depend on the given service."""
        return [
            dep.source_service
            for dep in self.dependencies
            if dep.target_service == service_name
        ]

    def get_upstream(self, service_name: str) -> list[str]:
        """Get services that the given service depends on."""
        return [
            dep.target_service
            for dep in self.dependencies
            if dep.source_service == service_name
        ]
