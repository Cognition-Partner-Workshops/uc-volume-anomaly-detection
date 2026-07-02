"""Anomaly Detection Agent — establishes baselines and identifies volume deviations."""

import csv
import logging
from datetime import datetime
from pathlib import Path

from src.detectors.rate_of_change_detector import RateOfChangeBaseline, RateOfChangeDetector
from src.detectors.seasonal_detector import SeasonalDetector
from src.detectors.zscore_detector import ZScoreDetector
from src.models.anomaly import AnomalyEvent
from src.models.transaction import TransactionVolume, VolumeBaseline, VolumeTimeSeries

logger = logging.getLogger(__name__)


class AnomalyDetectionAgent:
    """Orchestrates anomaly detection using multiple detection algorithms."""

    def __init__(
        self,
        zscore_warning: float = 2.0,
        zscore_critical: float = 3.0,
        seasonal_threshold: float = 2.5,
        roc_warning: float = 2.0,
        roc_critical: float = 3.0,
    ) -> None:
        self.zscore_detector = ZScoreDetector(
            warning_threshold=zscore_warning,
            critical_threshold=zscore_critical,
        )
        self.seasonal_detector = SeasonalDetector(
            deviation_threshold=seasonal_threshold,
        )
        self.roc_detector = RateOfChangeDetector(
            warning_threshold=roc_warning,
            critical_threshold=roc_critical,
        )
        self.baselines: dict[str, list[VolumeBaseline]] = {}
        self.roc_baselines: dict[str, RateOfChangeBaseline] = {}
        self.previous_observations: dict[str, TransactionVolume] = {}
        self.detected_anomalies: list[AnomalyEvent] = []

    def load_historical_data(self, csv_path: str) -> dict[str, VolumeTimeSeries]:
        """Load historical transaction data from CSV."""
        path = Path(csv_path)
        if not path.exists():
            logger.error("Historical data file not found: %s", csv_path)
            return {}

        series_map: dict[str, VolumeTimeSeries] = {}

        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row['service_name']}/{row['endpoint']}"
                if key not in series_map:
                    series_map[key] = VolumeTimeSeries(
                        service_name=row["service_name"],
                        endpoint=row["endpoint"],
                    )

                obs = TransactionVolume(
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    service_name=row["service_name"],
                    endpoint=row["endpoint"],
                    count=int(row["count"]),
                    error_count=int(row.get("error_count", "0")),
                    avg_latency_ms=float(row.get("avg_latency_ms", "0")),
                    p99_latency_ms=float(row.get("p99_latency_ms", "0")),
                )
                series_map[key].observations.append(obs)

        logger.info("Loaded %d time series from %s", len(series_map), csv_path)
        return series_map

    def build_baselines(
        self, historical_data: dict[str, VolumeTimeSeries]
    ) -> None:
        """Build seasonal and rate-of-change baselines from historical data."""
        for key, series in historical_data.items():
            baselines = self.seasonal_detector.build_baselines(series)
            self.baselines[key] = baselines

            roc_baseline = self.roc_detector.build_baseline(series)
            if roc_baseline:
                self.roc_baselines[key] = roc_baseline

        total = sum(len(bl) for bl in self.baselines.values())
        logger.info(
            "Built %d seasonal baselines and %d rate-of-change baselines across %d series",
            total,
            len(self.roc_baselines),
            len(self.baselines),
        )

    def analyze(
        self, observation: TransactionVolume
    ) -> list[AnomalyEvent]:
        """Analyze a single observation for anomalies using all detectors."""
        key = f"{observation.service_name}/{observation.endpoint}"
        anomalies: list[AnomalyEvent] = []

        # Seasonal and z-score detection (requires seasonal baselines)
        baselines = self.baselines.get(key, [])
        if baselines:
            seasonal_anomaly = self.seasonal_detector.detect(observation, baselines)
            if seasonal_anomaly:
                anomalies.append(seasonal_anomaly)

            hour = observation.timestamp.hour
            dow = observation.timestamp.weekday()
            matching_baseline = next(
                (b for b in baselines if b.hour_of_day == hour and b.day_of_week == dow),
                None,
            )
            if matching_baseline:
                zscore_anomaly = self.zscore_detector.detect(observation, matching_baseline)
                if zscore_anomaly:
                    anomalies.append(zscore_anomaly)

                latency_anomaly = self.zscore_detector.detect_latency(
                    observation, matching_baseline
                )
                if latency_anomaly:
                    anomalies.append(latency_anomaly)

        # Rate-of-change detection (independent of seasonal baselines)
        roc_baseline = self.roc_baselines.get(key)
        previous = self.previous_observations.get(key)
        if roc_baseline and previous:
            roc_anomaly = self.roc_detector.detect(previous, observation, roc_baseline)
            if roc_anomaly:
                anomalies.append(roc_anomaly)
        self.previous_observations[key] = observation

        self.detected_anomalies.extend(anomalies)
        return anomalies
