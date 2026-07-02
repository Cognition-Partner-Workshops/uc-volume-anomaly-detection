"""Service Health Assessment Agent — correlates anomalies with service health signals."""

import logging
from datetime import datetime
from typing import Optional

from src.models.anomaly import AnomalyEvent
from src.models.service_health import (
    HealthStatus,
    ServiceHealthSnapshot,
    ServiceMap,
)

logger = logging.getLogger(__name__)


class ServiceHealthAgent:
    """Correlates detected anomalies with live service health data."""

    def __init__(self, service_map: Optional[ServiceMap] = None) -> None:
        self.service_map = service_map or ServiceMap()
        self.health_cache: dict[str, ServiceHealthSnapshot] = {}

    def assess(
        self,
        anomaly: AnomalyEvent,
    ) -> ServiceHealthSnapshot:
        """Assess the health of the service associated with an anomaly."""
        snapshot = self.health_cache.get(anomaly.service_name)
        if snapshot is None:
            snapshot = self._fetch_health(anomaly.service_name)
            self.health_cache[anomaly.service_name] = snapshot

        return snapshot

    def correlate(
        self,
        anomaly: AnomalyEvent,
    ) -> list[str]:
        """Find correlated services that may be affected by or causing the anomaly."""
        correlated: list[str] = []

        # Check upstream dependencies
        upstream = self.service_map.get_upstream(anomaly.service_name)
        for svc in upstream:
            health = self._fetch_health(svc)
            if health.status in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY):
                correlated.append(svc)
                logger.info(
                    "Upstream service %s is %s (may be causing anomaly in %s)",
                    svc,
                    health.status.value,
                    anomaly.service_name,
                )

        # Check downstream dependents
        downstream = self.service_map.get_downstream(anomaly.service_name)
        for svc in downstream:
            health = self._fetch_health(svc)
            if health.status in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY):
                correlated.append(svc)

        anomaly.correlated_services = correlated
        return correlated

    @staticmethod
    def _fetch_health(service_name: str) -> ServiceHealthSnapshot:
        """Fetch current health data for a service.

        In production, this queries Prometheus, Datadog, or a health-check endpoint.
        """
        # Stub: return a healthy snapshot
        return ServiceHealthSnapshot(
            service_name=service_name,
            timestamp=datetime.utcnow(),
            status=HealthStatus.HEALTHY,
            cpu_utilization=0.0,
            memory_utilization=0.0,
            active_pods=3,
            desired_pods=3,
            avg_response_time_ms=50.0,
            error_rate=0.001,
            request_rate=100.0,
        )
