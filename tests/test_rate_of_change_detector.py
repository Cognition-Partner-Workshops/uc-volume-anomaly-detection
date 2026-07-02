"""Tests for the rate-of-change anomaly detector."""

from datetime import datetime

import pytest

from src.detectors.rate_of_change_detector import (
    RateOfChangeBaseline,
    RateOfChangeDetector,
)
from src.models.anomaly import AnomalySeverity, AnomalyType
from src.models.transaction import TransactionVolume, VolumeTimeSeries


class TestRateOfChangeDetector:
    def _make_baseline(self) -> RateOfChangeBaseline:
        return RateOfChangeBaseline(
            service_name="payment-service",
            endpoint="/api/v1/payments",
            mean_rate=5.0,  # average 5% change between observations
            std_rate=3.0,
            sample_size=20,
        )

    def test_no_anomaly_within_threshold(self):
        detector = RateOfChangeDetector(warning_threshold=2.0, critical_threshold=3.0)
        baseline = self._make_baseline()

        prev = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 11, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=1000,
        )
        curr = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 12, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=1080,  # 8% change, z-score = (8-5)/3 = 1.0
        )
        result = detector.detect(prev, curr, baseline)
        assert result is None

    def test_detects_rapid_increase(self):
        detector = RateOfChangeDetector(warning_threshold=2.0, critical_threshold=3.0)
        baseline = self._make_baseline()

        prev = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 11, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=1000,
        )
        curr = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 12, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=1200,  # 20% change, z-score = (20-5)/3 = 5.0
        )
        result = detector.detect(prev, curr, baseline)
        assert result is not None
        assert result.anomaly_type == AnomalyType.RATE_OF_CHANGE
        assert result.deviation_score == pytest.approx(5.0)

    def test_detects_rapid_decrease(self):
        detector = RateOfChangeDetector(warning_threshold=2.0, critical_threshold=3.0)
        baseline = self._make_baseline()

        prev = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 11, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=1000,
        )
        curr = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 12, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=800,  # -20% change, z-score = (-20-5)/3 = -8.33
        )
        result = detector.detect(prev, curr, baseline)
        assert result is not None
        assert result.anomaly_type == AnomalyType.RATE_OF_CHANGE
        assert result.deviation_score > 2.0

    def test_zero_previous_count_returns_none(self):
        detector = RateOfChangeDetector()
        baseline = self._make_baseline()

        prev = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 11, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=0,
        )
        curr = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 12, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=100,
        )
        result = detector.detect(prev, curr, baseline)
        assert result is None

    def test_zero_std_returns_none(self):
        detector = RateOfChangeDetector()
        baseline = RateOfChangeBaseline(
            service_name="payment-service",
            endpoint="/api/v1/payments",
            mean_rate=5.0,
            std_rate=0.0,
            sample_size=20,
        )

        prev = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 11, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=1000,
        )
        curr = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 12, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=1500,
        )
        result = detector.detect(prev, curr, baseline)
        assert result is None

    def test_severity_classification_medium(self):
        detector = RateOfChangeDetector(warning_threshold=2.0, critical_threshold=3.0)
        baseline = self._make_baseline()

        prev = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 11, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=1000,
        )
        # z-score = (12-5)/3 = 2.33 -> MEDIUM
        curr = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 12, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=1120,
        )
        result = detector.detect(prev, curr, baseline)
        assert result is not None
        assert result.severity == AnomalySeverity.MEDIUM

    def test_severity_classification_high(self):
        detector = RateOfChangeDetector(warning_threshold=2.0, critical_threshold=3.0)
        baseline = self._make_baseline()

        prev = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 11, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=1000,
        )
        # z-score = (20-5)/3 = 5.0 -> HIGH (>= 3.0)
        curr = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 12, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=1200,
        )
        result = detector.detect(prev, curr, baseline)
        assert result is not None
        assert result.severity == AnomalySeverity.HIGH

    def test_severity_classification_critical(self):
        detector = RateOfChangeDetector(warning_threshold=2.0, critical_threshold=3.0)
        baseline = self._make_baseline()

        prev = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 11, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=1000,
        )
        # z-score = (30-5)/3 = 8.33 -> CRITICAL (>= 6.0)
        curr = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 12, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=1300,
        )
        result = detector.detect(prev, curr, baseline)
        assert result is not None
        assert result.severity == AnomalySeverity.CRITICAL


class TestBuildBaseline:
    def test_builds_baseline_from_series(self):
        detector = RateOfChangeDetector(min_history=3)
        series = VolumeTimeSeries(
            service_name="payment-service",
            endpoint="/api/v1/payments",
            observations=[
                TransactionVolume(
                    timestamp=datetime(2026, 1, 1, i, 0),
                    service_name="payment-service",
                    endpoint="/api/v1/payments",
                    count=100 + i * 10,
                )
                for i in range(10)
            ],
        )
        baseline = detector.build_baseline(series)
        assert baseline is not None
        assert baseline.service_name == "payment-service"
        assert baseline.endpoint == "/api/v1/payments"
        assert baseline.sample_size == 9  # n-1 rates from n observations
        assert baseline.mean_rate > 0
        assert baseline.std_rate >= 0

    def test_insufficient_data_returns_none(self):
        detector = RateOfChangeDetector(min_history=5)
        series = VolumeTimeSeries(
            service_name="payment-service",
            endpoint="/api/v1/payments",
            observations=[
                TransactionVolume(
                    timestamp=datetime(2026, 1, 1, i, 0),
                    service_name="payment-service",
                    endpoint="/api/v1/payments",
                    count=100 + i * 10,
                )
                for i in range(3)
            ],
        )
        baseline = detector.build_baseline(series)
        assert baseline is None


class TestDetectFromSeries:
    def test_detects_anomalies_in_series(self):
        detector = RateOfChangeDetector(warning_threshold=2.0, critical_threshold=3.0)
        baseline = RateOfChangeBaseline(
            service_name="payment-service",
            endpoint="/api/v1/payments",
            mean_rate=5.0,
            std_rate=3.0,
            sample_size=20,
        )

        # Mostly normal changes with one spike
        observations = [
            TransactionVolume(
                timestamp=datetime(2026, 3, 10, i, 0),
                service_name="payment-service",
                endpoint="/api/v1/payments",
                count=count,
            )
            for i, count in enumerate(
                [1000, 1050, 1100, 1150, 1600, 1650, 1700]
            )
        ]

        series = VolumeTimeSeries(
            service_name="payment-service",
            endpoint="/api/v1/payments",
            observations=observations,
        )

        anomalies = detector.detect_from_series(series, baseline)
        # The jump from 1150 to 1600 is ~39% change -> z = (39-5)/3 = 11.3
        assert len(anomalies) >= 1
        assert any(a.observed_value > 30 for a in anomalies)
