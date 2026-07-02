"""Z-score based anomaly detection for transaction volumes."""

import logging
from datetime import datetime
from typing import Optional
from uuid import uuid4

from src.models.anomaly import AnomalyEvent, AnomalySeverity, AnomalyType
from src.models.transaction import TransactionVolume, VolumeBaseline

logger = logging.getLogger(__name__)


class ZScoreDetector:
    """Detects anomalies by computing z-scores against historical baselines."""

    def __init__(
        self,
        warning_threshold: float = 2.0,
        critical_threshold: float = 3.0,
    ) -> None:
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def detect(
        self,
        observation: TransactionVolume,
        baseline: VolumeBaseline,
    ) -> Optional[AnomalyEvent]:
        """Check a single observation against its baseline."""
        if baseline.std_count == 0:
            return None

        z_score = (observation.count - baseline.mean_count) / baseline.std_count
        abs_z = abs(z_score)

        if abs_z < self.warning_threshold:
            return None

        severity = self._classify_severity(abs_z)
        anomaly_type = (
            AnomalyType.VOLUME_SPIKE if z_score > 0 else AnomalyType.VOLUME_DROP
        )

        direction = "above" if z_score > 0 else "below"
        description = (
            f"{observation.service_name}/{observation.endpoint}: "
            f"volume {observation.count} is {abs_z:.1f} std devs {direction} "
            f"expected {baseline.mean_count:.0f} "
            f"(hour={baseline.hour_of_day}, dow={baseline.day_of_week})"
        )

        return AnomalyEvent(
            anomaly_id=f"anom-{uuid4().hex[:8]}",
            anomaly_type=anomaly_type,
            severity=severity,
            service_name=observation.service_name,
            endpoint=observation.endpoint,
            detected_at=datetime.utcnow(),
            observed_value=float(observation.count),
            expected_value=baseline.mean_count,
            deviation_score=abs_z,
            description=description,
        )

    def detect_latency(
        self,
        observation: TransactionVolume,
        baseline: VolumeBaseline,
    ) -> Optional[AnomalyEvent]:
        """Detect latency anomalies against the baseline."""
        if baseline.std_latency_ms == 0:
            return None

        z_score = (
            observation.avg_latency_ms - baseline.mean_latency_ms
        ) / baseline.std_latency_ms

        if z_score < self.warning_threshold:
            return None

        severity = self._classify_severity(z_score)
        description = (
            f"{observation.service_name}/{observation.endpoint}: "
            f"latency {observation.avg_latency_ms:.0f}ms is {z_score:.1f} std devs "
            f"above expected {baseline.mean_latency_ms:.0f}ms"
        )

        return AnomalyEvent(
            anomaly_id=f"anom-{uuid4().hex[:8]}",
            anomaly_type=AnomalyType.LATENCY_SPIKE,
            severity=severity,
            service_name=observation.service_name,
            endpoint=observation.endpoint,
            detected_at=datetime.utcnow(),
            observed_value=observation.avg_latency_ms,
            expected_value=baseline.mean_latency_ms,
            deviation_score=z_score,
            description=description,
        )

    def _classify_severity(self, abs_z: float) -> AnomalySeverity:
        """Map a z-score magnitude to a severity level."""
        if abs_z >= self.critical_threshold * 2:
            return AnomalySeverity.CRITICAL
        if abs_z >= self.critical_threshold:
            return AnomalySeverity.HIGH
        if abs_z >= self.warning_threshold:
            return AnomalySeverity.MEDIUM
        return AnomalySeverity.LOW
