"""Correlation Engine — detects whether volume anomalies in one service coincide
with changes in upstream services.

Examines temporal proximity of anomalies across related services and scores
correlation strength to help identify root-cause propagation paths.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from src.models.anomaly import AnomalyEvent, AnomalySeverity
from src.models.service_health import (
    HealthStatus,
    ServiceHealthSnapshot,
    ServiceMap,
)
from src.models.transaction import TransactionVolume

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Result of a cross-service anomaly correlation analysis."""

    source_anomaly: AnomalyEvent
    correlated_service: str
    correlation_type: str  # "upstream_cause", "downstream_impact", "co_occurrence"
    confidence: float  # 0.0 - 1.0
    time_offset_seconds: float  # how far apart the events are
    evidence: str

    @property
    def is_likely_cause(self) -> bool:
        return self.correlation_type == "upstream_cause" and self.confidence >= 0.6


@dataclass
class CorrelationReport:
    """Aggregated correlation findings for an anomaly event."""

    anomaly: AnomalyEvent
    correlations: list[CorrelationResult] = field(default_factory=list)
    root_cause_candidates: list[str] = field(default_factory=list)
    impact_scope: list[str] = field(default_factory=list)

    @property
    def has_upstream_cause(self) -> bool:
        return len(self.root_cause_candidates) > 0


class CorrelationEngine:
    """Analyzes cross-service anomaly correlations to identify root causes.

    Given an anomaly detected in one service, the engine:
    1. Looks up upstream/downstream dependencies from the service map.
    2. Checks whether upstream services had recent volume changes (anomalies
       or significant shifts) within a configurable time window.
    3. Scores correlation strength based on temporal proximity, dependency
       criticality, and magnitude of the upstream change.
    """

    def __init__(
        self,
        service_map: ServiceMap,
        correlation_window: timedelta = timedelta(minutes=30),
        min_confidence: float = 0.4,
    ) -> None:
        self.service_map = service_map
        self.correlation_window = correlation_window
        self.min_confidence = min_confidence
        self._anomaly_history: dict[str, list[AnomalyEvent]] = {}
        self._volume_history: dict[str, list[TransactionVolume]] = {}

    def register_anomaly(self, anomaly: AnomalyEvent) -> None:
        """Register a detected anomaly for future correlation lookups."""
        key = anomaly.service_name
        if key not in self._anomaly_history:
            self._anomaly_history[key] = []
        self._anomaly_history[key].append(anomaly)

    def register_volume(self, observation: TransactionVolume) -> None:
        """Register a volume observation for detecting upstream changes."""
        key = observation.service_name
        if key not in self._volume_history:
            self._volume_history[key] = []
        self._volume_history[key].append(observation)

    def correlate(
        self,
        anomaly: AnomalyEvent,
        health_snapshots: Optional[dict[str, ServiceHealthSnapshot]] = None,
    ) -> CorrelationReport:
        """Analyze an anomaly for cross-service correlations.

        Args:
            anomaly: The detected anomaly to investigate.
            health_snapshots: Optional current health state of services.

        Returns:
            A CorrelationReport with all found correlations.
        """
        report = CorrelationReport(anomaly=anomaly)
        health_snapshots = health_snapshots or {}

        # Check upstream services for causal correlations
        upstream_services = self.service_map.get_upstream(anomaly.service_name)
        for upstream_svc in upstream_services:
            correlations = self._check_upstream(anomaly, upstream_svc, health_snapshots)
            report.correlations.extend(correlations)

        # Check downstream services for impact assessment
        downstream_services = self.service_map.get_downstream(anomaly.service_name)
        for downstream_svc in downstream_services:
            correlations = self._check_downstream(anomaly, downstream_svc, health_snapshots)
            report.correlations.extend(correlations)

        # Check co-occurring anomalies in peer services
        peer_correlations = self._check_peers(anomaly)
        report.correlations.extend(peer_correlations)

        # Derive root-cause candidates and impact scope
        report.root_cause_candidates = [
            c.correlated_service
            for c in report.correlations
            if c.is_likely_cause
        ]
        report.impact_scope = [
            c.correlated_service
            for c in report.correlations
            if c.correlation_type == "downstream_impact"
        ]

        # Update the source anomaly
        anomaly.correlated_services = list(
            {c.correlated_service for c in report.correlations}
        )

        logger.info(
            "Correlation analysis for %s: %d correlations, %d root-cause candidates",
            anomaly.anomaly_id,
            len(report.correlations),
            len(report.root_cause_candidates),
        )
        return report

    def _check_upstream(
        self,
        anomaly: AnomalyEvent,
        upstream_svc: str,
        health_snapshots: dict[str, ServiceHealthSnapshot],
    ) -> list[CorrelationResult]:
        """Check if an upstream service had anomalies preceding this one."""
        results: list[CorrelationResult] = []

        # Check anomaly history of upstream service
        upstream_anomalies = self._anomaly_history.get(upstream_svc, [])
        for upstream_anomaly in upstream_anomalies:
            offset = self._time_offset(upstream_anomaly.detected_at, anomaly.detected_at)
            if 0 <= offset <= self.correlation_window.total_seconds():
                confidence = self._compute_confidence(
                    offset, upstream_anomaly.severity, "upstream"
                )
                if confidence >= self.min_confidence:
                    results.append(
                        CorrelationResult(
                            source_anomaly=anomaly,
                            correlated_service=upstream_svc,
                            correlation_type="upstream_cause",
                            confidence=confidence,
                            time_offset_seconds=offset,
                            evidence=(
                                f"Upstream {upstream_svc} had "
                                f"{upstream_anomaly.anomaly_type.value} "
                                f"({upstream_anomaly.severity.value}) "
                                f"{offset:.0f}s before this anomaly"
                            ),
                        )
                    )

        # Check health degradation of upstream service
        health = health_snapshots.get(upstream_svc)
        if health and health.status in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY):
            results.append(
                CorrelationResult(
                    source_anomaly=anomaly,
                    correlated_service=upstream_svc,
                    correlation_type="upstream_cause",
                    confidence=0.7 if health.status == HealthStatus.UNHEALTHY else 0.5,
                    time_offset_seconds=0.0,
                    evidence=(
                        f"Upstream {upstream_svc} is currently "
                        f"{health.status.value} "
                        f"(error_rate={health.error_rate:.3f})"
                    ),
                )
            )

        return results

    def _check_downstream(
        self,
        anomaly: AnomalyEvent,
        downstream_svc: str,
        health_snapshots: dict[str, ServiceHealthSnapshot],
    ) -> list[CorrelationResult]:
        """Check if downstream services are showing impact from this anomaly."""
        results: list[CorrelationResult] = []

        downstream_anomalies = self._anomaly_history.get(downstream_svc, [])
        for ds_anomaly in downstream_anomalies:
            offset = self._time_offset(anomaly.detected_at, ds_anomaly.detected_at)
            if 0 <= offset <= self.correlation_window.total_seconds():
                confidence = self._compute_confidence(
                    offset, ds_anomaly.severity, "downstream"
                )
                if confidence >= self.min_confidence:
                    results.append(
                        CorrelationResult(
                            source_anomaly=anomaly,
                            correlated_service=downstream_svc,
                            correlation_type="downstream_impact",
                            confidence=confidence,
                            time_offset_seconds=offset,
                            evidence=(
                                f"Downstream {downstream_svc} showed "
                                f"{ds_anomaly.anomaly_type.value} "
                                f"{offset:.0f}s after this anomaly"
                            ),
                        )
                    )

        return results

    def _check_peers(
        self,
        anomaly: AnomalyEvent,
    ) -> list[CorrelationResult]:
        """Check for co-occurring anomalies in services sharing upstream deps."""
        results: list[CorrelationResult] = []
        upstream_of_source = set(self.service_map.get_upstream(anomaly.service_name))

        for svc, svc_anomalies in self._anomaly_history.items():
            if svc == anomaly.service_name:
                continue

            upstream_of_peer = set(self.service_map.get_upstream(svc))
            shared_upstream = upstream_of_source & upstream_of_peer
            if not shared_upstream:
                continue

            for peer_anomaly in svc_anomalies:
                offset = abs(
                    self._time_offset(anomaly.detected_at, peer_anomaly.detected_at)
                )
                if offset <= self.correlation_window.total_seconds():
                    confidence = max(0.3, 0.6 - (offset / self.correlation_window.total_seconds()) * 0.3)
                    if confidence >= self.min_confidence:
                        results.append(
                            CorrelationResult(
                                source_anomaly=anomaly,
                                correlated_service=svc,
                                correlation_type="co_occurrence",
                                confidence=confidence,
                                time_offset_seconds=offset,
                                evidence=(
                                    f"Peer {svc} (shared upstream: "
                                    f"{', '.join(shared_upstream)}) had "
                                    f"{peer_anomaly.anomaly_type.value} "
                                    f"within {offset:.0f}s"
                                ),
                            )
                        )

        return results

    @staticmethod
    def _time_offset(earlier: datetime, later: datetime) -> float:
        """Compute seconds between two timestamps (positive if later > earlier)."""
        delta = later - earlier
        return delta.total_seconds()

    @staticmethod
    def _compute_confidence(
        time_offset_seconds: float,
        severity: AnomalySeverity,
        direction: str,
    ) -> float:
        """Compute correlation confidence from temporal proximity and severity.

        Closer in time and higher severity yield higher confidence.
        """
        # Time decay: confidence drops as events are further apart
        max_window = 1800.0  # 30 minutes in seconds
        time_factor = max(0.0, 1.0 - (time_offset_seconds / max_window))

        # Severity boost
        severity_boost = {
            AnomalySeverity.LOW: 0.0,
            AnomalySeverity.MEDIUM: 0.1,
            AnomalySeverity.HIGH: 0.2,
            AnomalySeverity.CRITICAL: 0.3,
        }.get(severity, 0.0)

        # Direction factor: upstream causes are weighted higher
        direction_factor = 0.8 if direction == "upstream" else 0.6

        confidence = (time_factor * direction_factor) + severity_boost
        return min(1.0, max(0.0, confidence))
