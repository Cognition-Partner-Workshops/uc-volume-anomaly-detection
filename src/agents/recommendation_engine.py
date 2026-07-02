"""Knowledge-Based Recommendation Agent — suggests corrective actions from prior incidents."""

import logging
from dataclasses import dataclass
from typing import Optional

from src.models.anomaly import AnomalyEvent, AnomalySeverity, AnomalyType

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """A recommended corrective action."""

    action: str
    confidence: float
    source: str  # "runbook", "historical_incident", "heuristic"
    priority: int = 0
    details: str = ""


@dataclass
class RunbookEntry:
    """A runbook entry mapping anomaly patterns to corrective actions."""

    anomaly_types: list[AnomalyType]
    severity_threshold: AnomalySeverity
    actions: list[str]
    description: str


# Built-in runbook knowledge base
DEFAULT_RUNBOOK: list[RunbookEntry] = [
    RunbookEntry(
        anomaly_types=[AnomalyType.VOLUME_DROP],
        severity_threshold=AnomalySeverity.HIGH,
        actions=[
            "Check upstream service health and connectivity",
            "Verify load balancer configuration",
            "Review recent deployment changes",
            "Check DNS resolution for dependent services",
        ],
        description="Significant volume drop may indicate upstream failure or routing issue",
    ),
    RunbookEntry(
        anomaly_types=[AnomalyType.VOLUME_SPIKE],
        severity_threshold=AnomalySeverity.HIGH,
        actions=[
            "Check for retry storms from downstream services",
            "Verify rate limiting is active",
            "Review auto-scaling policies",
            "Check for batch job or cron job overlap",
        ],
        description="Unexpected volume spike may indicate retry storms or misconfigured batch jobs",
    ),
    RunbookEntry(
        anomaly_types=[AnomalyType.LATENCY_SPIKE],
        severity_threshold=AnomalySeverity.MEDIUM,
        actions=[
            "Check database connection pool utilization",
            "Review slow query logs",
            "Check for resource contention (CPU, memory, disk I/O)",
            "Verify cache hit rates",
        ],
        description="Latency increase often correlates with database or resource contention",
    ),
    RunbookEntry(
        anomaly_types=[AnomalyType.ERROR_RATE_SPIKE],
        severity_threshold=AnomalySeverity.MEDIUM,
        actions=[
            "Review application error logs for root cause",
            "Check dependent service availability",
            "Verify configuration changes from recent deployments",
            "Check certificate expiration dates",
        ],
        description="Error rate spikes typically indicate dependency failures or bad deployments",
    ),
]


class RecommendationEngine:
    """Generates corrective action recommendations based on anomaly patterns."""

    def __init__(
        self,
        runbook: Optional[list[RunbookEntry]] = None,
    ) -> None:
        self.runbook = runbook or DEFAULT_RUNBOOK

    def recommend(
        self,
        anomaly: AnomalyEvent,
        correlated_services: Optional[list[str]] = None,
    ) -> list[Recommendation]:
        """Generate recommendations for a detected anomaly."""
        recommendations: list[Recommendation] = []

        # Match against runbook entries
        for entry in self.runbook:
            if anomaly.anomaly_type not in entry.anomaly_types:
                continue

            severity_rank = _severity_rank(anomaly.severity)
            threshold_rank = _severity_rank(entry.severity_threshold)
            if severity_rank < threshold_rank:
                continue

            for i, action in enumerate(entry.actions):
                recommendations.append(
                    Recommendation(
                        action=action,
                        confidence=0.8 - (i * 0.1),
                        source="runbook",
                        priority=i + 1,
                        details=entry.description,
                    )
                )

        # Add correlation-based recommendations
        if correlated_services:
            for svc in correlated_services:
                recommendations.append(
                    Recommendation(
                        action=f"Investigate correlated service: {svc}",
                        confidence=0.7,
                        source="heuristic",
                        priority=0,
                        details=f"Service {svc} is showing degraded health during this anomaly",
                    )
                )

        anomaly.recommended_actions = [r.action for r in recommendations]
        logger.info(
            "Generated %d recommendations for anomaly %s",
            len(recommendations),
            anomaly.anomaly_id,
        )
        return recommendations


def _severity_rank(severity: AnomalySeverity) -> int:
    return {
        AnomalySeverity.LOW: 0,
        AnomalySeverity.MEDIUM: 1,
        AnomalySeverity.HIGH: 2,
        AnomalySeverity.CRITICAL: 3,
    }.get(severity, 0)
