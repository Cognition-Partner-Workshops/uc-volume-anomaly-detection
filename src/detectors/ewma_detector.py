"""EWMA-based anomaly detection for transaction volumes."""

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import uuid4

from src.models.anomaly import AnomalyEvent, AnomalySeverity, AnomalyType
from src.models.transaction import TransactionVolume

logger = logging.getLogger(__name__)


@dataclass
class EWMAState:
    """Running exponentially-weighted mean and variance for one series."""

    mean: float
    variance: float
    sample_size: int

    @property
    def std(self) -> float:
        return math.sqrt(self.variance) if self.variance > 0 else 0.0


class EWMADetector:
    """Detects anomalies by tracking an exponentially-weighted moving average.

    Unlike the baseline-based detectors this one is stateful: it keeps a
    per-``service_name/endpoint`` EWMA of the observed volume and its
    exponentially-weighted standard deviation, so results depend on the order
    in which observations are fed in.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        warning_threshold: float = 2.0,
        critical_threshold: float = 3.0,
        min_observations: int = 5,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.min_observations = min_observations
        self.states: dict[str, EWMAState] = {}

    def reset(self) -> None:
        """Drop all accumulated EWMA state."""
        self.states.clear()

    def detect(self, observation: TransactionVolume) -> Optional[AnomalyEvent]:
        """Score an observation against the current EWMA, then absorb it."""
        key = f"{observation.service_name}/{observation.endpoint}"
        state = self.states.get(key)
        value = float(observation.count)

        if state is None:
            self.states[key] = EWMAState(mean=value, variance=0.0, sample_size=1)
            return None

        expected = state.mean
        std = state.std
        warmed_up = state.sample_size >= self.min_observations
        self._update(state, value)

        if not warmed_up or std == 0:
            return None

        deviation = (value - expected) / std
        abs_deviation = abs(deviation)
        if abs_deviation < self.warning_threshold:
            return None

        anomaly_type = (
            AnomalyType.VOLUME_SPIKE if deviation > 0 else AnomalyType.VOLUME_DROP
        )
        direction = "above" if deviation > 0 else "below"
        description = (
            f"EWMA anomaly: {observation.service_name}/{observation.endpoint} "
            f"volume={observation.count} is {abs_deviation:.1f}σ {direction} "
            f"EWMA={expected:.0f} (alpha={self.alpha}, "
            f"samples={state.sample_size})"
        )

        return AnomalyEvent(
            anomaly_id=f"anom-{uuid4().hex[:8]}",
            anomaly_type=anomaly_type,
            severity=self._classify_severity(abs_deviation),
            service_name=observation.service_name,
            endpoint=observation.endpoint,
            detected_at=datetime.utcnow(),
            observed_value=value,
            expected_value=expected,
            deviation_score=abs_deviation,
            description=description,
        )

    def _update(self, state: EWMAState, value: float) -> None:
        previous_mean = state.mean
        state.mean = self.alpha * value + (1 - self.alpha) * previous_mean
        state.variance = (
            self.alpha * (value - previous_mean) ** 2
            + (1 - self.alpha) * state.variance
        )
        state.sample_size += 1

    def _classify_severity(self, abs_deviation: float) -> AnomalySeverity:
        """Map a deviation magnitude to a severity level."""
        if abs_deviation >= self.critical_threshold * 2:
            return AnomalySeverity.CRITICAL
        if abs_deviation >= self.critical_threshold:
            return AnomalySeverity.HIGH
        if abs_deviation >= self.warning_threshold:
            return AnomalySeverity.MEDIUM
        return AnomalySeverity.LOW
