"""Rate-of-change based anomaly detection.

Detects when a metric is changing faster than historical norms, even if the
absolute value is still within bounds. This catches accelerating degradation
or rapid recovery that warrants attention.
"""

import logging
import math
from datetime import datetime
from typing import Optional
from uuid import uuid4

from src.models.anomaly import AnomalyEvent, AnomalySeverity, AnomalyType
from src.models.transaction import TransactionVolume, VolumeTimeSeries

logger = logging.getLogger(__name__)


class RateOfChangeBaseline:
    """Historical rate-of-change statistics for a service endpoint."""

    def __init__(
        self,
        service_name: str,
        endpoint: str,
        mean_rate: float,
        std_rate: float,
        sample_size: int,
    ) -> None:
        self.service_name = service_name
        self.endpoint = endpoint
        self.mean_rate = mean_rate
        self.std_rate = std_rate
        self.sample_size = sample_size


class RateOfChangeDetector:
    """Detects anomalous rates of change in transaction volume.

    Computes the rate of change between consecutive observations and compares
    it against historical norms. Flags observations where the rate of change
    (positive or negative) exceeds a threshold number of standard deviations
    from the historical mean rate.
    """

    def __init__(
        self,
        warning_threshold: float = 2.0,
        critical_threshold: float = 3.0,
        min_history: int = 5,
    ) -> None:
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.min_history = min_history

    def build_baseline(
        self,
        time_series: VolumeTimeSeries,
    ) -> Optional[RateOfChangeBaseline]:
        """Build a rate-of-change baseline from historical observations.

        Computes the percentage change between consecutive observations and
        derives mean/std statistics for the rate distribution.
        """
        observations = time_series.observations
        if len(observations) < self.min_history:
            logger.debug(
                "Insufficient data for rate-of-change baseline: %d < %d",
                len(observations),
                self.min_history,
            )
            return None

        rates = self._compute_rates(observations)
        if not rates:
            return None

        mean_rate = sum(rates) / len(rates)
        if len(rates) < 2:
            return RateOfChangeBaseline(
                service_name=time_series.service_name,
                endpoint=time_series.endpoint,
                mean_rate=mean_rate,
                std_rate=0.0,
                sample_size=len(rates),
            )
        variance = sum((r - mean_rate) ** 2 for r in rates) / (len(rates) - 1)
        std_rate = math.sqrt(variance)

        return RateOfChangeBaseline(
            service_name=time_series.service_name,
            endpoint=time_series.endpoint,
            mean_rate=mean_rate,
            std_rate=std_rate,
            sample_size=len(rates),
        )

    def detect(
        self,
        previous: TransactionVolume,
        current: TransactionVolume,
        baseline: RateOfChangeBaseline,
    ) -> Optional[AnomalyEvent]:
        """Detect if the rate of change between two observations is anomalous.

        Args:
            previous: The prior observation in the time series.
            current: The current observation being evaluated.
            baseline: Historical rate-of-change statistics.

        Returns:
            An AnomalyEvent if the rate of change is anomalous, else None.
        """
        if baseline.std_rate == 0:
            return None

        rate = self._compute_single_rate(previous, current)
        if rate is None:
            return None

        z_score = (rate - baseline.mean_rate) / baseline.std_rate
        abs_z = abs(z_score)

        if abs_z < self.warning_threshold:
            return None

        severity = self._classify_severity(abs_z)
        direction = "acceleration" if rate > 0 else "deceleration"
        anomaly_type = AnomalyType.RATE_OF_CHANGE

        description = (
            f"Rate-of-change anomaly: {current.service_name}/{current.endpoint} "
            f"rate={rate:+.1f}% ({direction}), "
            f"expected range [{baseline.mean_rate - baseline.std_rate:.1f}%, "
            f"{baseline.mean_rate + baseline.std_rate:.1f}%], "
            f"z-score={z_score:.2f}"
        )

        return AnomalyEvent(
            anomaly_id=f"anom-roc-{uuid4().hex[:8]}",
            anomaly_type=anomaly_type,
            severity=severity,
            service_name=current.service_name,
            endpoint=current.endpoint,
            detected_at=datetime.utcnow(),
            observed_value=rate,
            expected_value=baseline.mean_rate,
            deviation_score=abs_z,
            description=description,
        )

    def detect_from_series(
        self,
        time_series: VolumeTimeSeries,
        baseline: RateOfChangeBaseline,
    ) -> list[AnomalyEvent]:
        """Run rate-of-change detection across an entire time series."""
        anomalies: list[AnomalyEvent] = []
        observations = time_series.observations

        for i in range(1, len(observations)):
            anomaly = self.detect(observations[i - 1], observations[i], baseline)
            if anomaly:
                anomalies.append(anomaly)

        return anomalies

    def _compute_rates(
        self,
        observations: list[TransactionVolume],
    ) -> list[float]:
        """Compute percentage rates of change between consecutive observations."""
        rates: list[float] = []
        for i in range(1, len(observations)):
            rate = self._compute_single_rate(observations[i - 1], observations[i])
            if rate is not None:
                rates.append(rate)
        return rates

    @staticmethod
    def _compute_single_rate(
        previous: TransactionVolume,
        current: TransactionVolume,
    ) -> Optional[float]:
        """Compute percentage change from previous to current observation."""
        if previous.count == 0:
            return None
        return ((current.count - previous.count) / previous.count) * 100.0

    def _classify_severity(self, abs_z: float) -> AnomalySeverity:
        """Map z-score magnitude to severity level."""
        if abs_z >= self.critical_threshold * 2:
            return AnomalySeverity.CRITICAL
        if abs_z >= self.critical_threshold:
            return AnomalySeverity.HIGH
        if abs_z >= self.warning_threshold:
            return AnomalySeverity.MEDIUM
        return AnomalySeverity.LOW
