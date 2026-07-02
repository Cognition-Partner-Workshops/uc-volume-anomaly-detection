"""Models for detected anomaly events."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AnomalySeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(Enum):
    VOLUME_SPIKE = "volume_spike"
    VOLUME_DROP = "volume_drop"
    LATENCY_SPIKE = "latency_spike"
    ERROR_RATE_SPIKE = "error_rate_spike"
    PATTERN_SHIFT = "pattern_shift"
    RATE_OF_CHANGE = "rate_of_change"


@dataclass
class AnomalyEvent:
    """A detected anomaly in transaction volume or performance."""

    anomaly_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    service_name: str
    endpoint: str
    detected_at: datetime
    observed_value: float
    expected_value: float
    deviation_score: float
    description: str
    correlated_services: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)

    @property
    def deviation_percentage(self) -> float:
        if self.expected_value == 0:
            return 100.0
        return abs(self.observed_value - self.expected_value) / self.expected_value * 100


@dataclass
class AnomalyReport:
    """Consolidated anomaly report for incident insight."""

    report_id: str
    generated_at: datetime
    time_window_start: datetime
    time_window_end: datetime
    anomalies: list[AnomalyEvent] = field(default_factory=list)
    affected_services: list[str] = field(default_factory=list)
    summary: str = ""
    recommended_actions: list[str] = field(default_factory=list)

    @property
    def critical_count(self) -> int:
        return sum(
            1 for a in self.anomalies if a.severity == AnomalySeverity.CRITICAL
        )

    @property
    def high_count(self) -> int:
        return sum(
            1 for a in self.anomalies if a.severity == AnomalySeverity.HIGH
        )
