"""Models for transaction volume data."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class TransactionVolume:
    """A single time-bucketed transaction volume observation."""

    timestamp: datetime
    service_name: str
    endpoint: str
    count: int
    error_count: int = 0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    @property
    def error_rate(self) -> float:
        return self.error_count / self.count if self.count > 0 else 0.0


@dataclass
class VolumeBaseline:
    """Statistical baseline for a service endpoint's transaction volume."""

    service_name: str
    endpoint: str
    hour_of_day: int
    mean_count: float
    std_count: float
    mean_latency_ms: float
    std_latency_ms: float
    day_of_week: Optional[int] = None
    sample_size: int = 0

    @property
    def upper_bound(self) -> float:
        """Upper bound at ~3 standard deviations."""
        return self.mean_count + 3.0 * self.std_count

    @property
    def lower_bound(self) -> float:
        """Lower bound at ~3 standard deviations."""
        return max(0.0, self.mean_count - 3.0 * self.std_count)


@dataclass
class VolumeTimeSeries:
    """A time-ordered collection of volume observations for one service."""

    service_name: str
    endpoint: str
    observations: list[TransactionVolume] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        return sum(obs.count for obs in self.observations)

    @property
    def time_range(self) -> Optional[tuple[datetime, datetime]]:
        if not self.observations:
            return None
        return self.observations[0].timestamp, self.observations[-1].timestamp
