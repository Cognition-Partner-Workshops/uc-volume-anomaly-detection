"""Seasonal decomposition-based anomaly detection."""

import logging
import math
from collections import defaultdict
from datetime import datetime
from typing import Literal, Optional
from uuid import uuid4

from src.models.anomaly import AnomalyEvent, AnomalySeverity, AnomalyType
from src.models.transaction import TransactionVolume, VolumeBaseline, VolumeTimeSeries

logger = logging.getLogger(__name__)

SeasonalMode = Literal["day_of_week", "time_of_day"]


class SeasonalDetector:
    """Detects anomalies using seasonal pattern decomposition.

    Supports two modes controlled by *seasonal_mode*:

    * ``"day_of_week"`` (default) -- builds hourly-by-day-of-week baselines so
      that, e.g., Monday-10 AM has its own profile distinct from Tuesday-10 AM.
    * ``"time_of_day"`` -- buckets observations by hour only, ignoring the day
      of week.  This is useful for services whose traffic follows a consistent
      daily pattern regardless of which day it is.
    """

    def __init__(
        self,
        deviation_threshold: float = 2.5,
        min_samples: int = 4,
        seasonal_mode: SeasonalMode = "day_of_week",
    ) -> None:
        self.deviation_threshold = deviation_threshold
        self.min_samples = min_samples

        if seasonal_mode not in ("day_of_week", "time_of_day"):
            raise ValueError(
                f"Invalid seasonal_mode '{seasonal_mode}'. "
                "Must be 'day_of_week' or 'time_of_day'."
            )
        self.seasonal_mode: SeasonalMode = seasonal_mode

    def build_baselines(
        self,
        time_series: VolumeTimeSeries,
    ) -> list[VolumeBaseline]:
        """Build baselines from historical observations.

        In ``day_of_week`` mode the key is ``(hour, weekday)``.
        In ``time_of_day`` mode the key is ``(hour,)`` -- all days are pooled.
        """
        buckets: dict[tuple[int, ...], list[TransactionVolume]] = defaultdict(list)

        for obs in time_series.observations:
            if self.seasonal_mode == "day_of_week":
                key: tuple[int, ...] = (obs.timestamp.hour, obs.timestamp.weekday())
            else:
                key = (obs.timestamp.hour,)
            buckets[key].append(obs)

        baselines: list[VolumeBaseline] = []
        for bucket_key, observations in buckets.items():
            if len(observations) < self.min_samples:
                continue

            counts = [obs.count for obs in observations]
            latencies = [obs.avg_latency_ms for obs in observations]

            hour = bucket_key[0]
            dow = bucket_key[1] if self.seasonal_mode == "day_of_week" else None

            baselines.append(
                VolumeBaseline(
                    service_name=time_series.service_name,
                    endpoint=time_series.endpoint,
                    hour_of_day=hour,
                    mean_count=_mean(counts),
                    std_count=_std(counts),
                    mean_latency_ms=_mean(latencies),
                    std_latency_ms=_std(latencies),
                    day_of_week=dow,
                    sample_size=len(observations),
                )
            )

        logger.info(
            "Built %d seasonal baselines (%s mode) for %s/%s",
            len(baselines),
            self.seasonal_mode,
            time_series.service_name,
            time_series.endpoint,
        )
        return baselines

    def detect(
        self,
        observation: TransactionVolume,
        baselines: list[VolumeBaseline],
    ) -> Optional[AnomalyEvent]:
        """Check an observation against its matching seasonal baseline."""
        hour = observation.timestamp.hour
        dow = observation.timestamp.weekday()

        baseline = self._find_baseline(baselines, hour, dow)
        if baseline is None:
            return None

        if baseline.std_count == 0:
            return None

        z_score = (observation.count - baseline.mean_count) / baseline.std_count
        abs_z = abs(z_score)

        if abs_z < self.deviation_threshold:
            return None

        anomaly_type = (
            AnomalyType.VOLUME_SPIKE if z_score > 0 else AnomalyType.VOLUME_DROP
        )
        severity = self._classify_severity(abs_z)

        direction = "above" if z_score > 0 else "below"
        if self.seasonal_mode == "day_of_week":
            context = f"hour={hour}, day={dow}, samples={baseline.sample_size}"
        else:
            context = f"hour={hour}, samples={baseline.sample_size}"

        description = (
            f"Seasonal anomaly: {observation.service_name}/{observation.endpoint} "
            f"volume={observation.count} is {abs_z:.1f}σ {direction} "
            f"seasonal baseline={baseline.mean_count:.0f} "
            f"({context})"
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

    def _find_baseline(
        self,
        baselines: list[VolumeBaseline],
        hour: int,
        dow: int,
    ) -> Optional[VolumeBaseline]:
        """Return the baseline matching the given hour (and optionally day)."""
        for b in baselines:
            if self.seasonal_mode == "day_of_week":
                if b.hour_of_day == hour and b.day_of_week == dow:
                    return b
            else:
                if b.hour_of_day == hour and b.day_of_week is None:
                    return b
        return None

    def _classify_severity(self, abs_z: float) -> AnomalySeverity:
        if abs_z >= self.deviation_threshold * 2.5:
            return AnomalySeverity.CRITICAL
        if abs_z >= self.deviation_threshold * 1.5:
            return AnomalySeverity.HIGH
        if abs_z >= self.deviation_threshold:
            return AnomalySeverity.MEDIUM
        return AnomalySeverity.LOW


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)
