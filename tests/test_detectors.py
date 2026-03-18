"""Tests for anomaly detection algorithms."""

from datetime import datetime

from src.detectors.zscore_detector import ZScoreDetector
from src.detectors.seasonal_detector import SeasonalDetector, _mean, _std
from src.models.anomaly import AnomalySeverity, AnomalyType
from src.models.transaction import TransactionVolume, VolumeBaseline, VolumeTimeSeries


class TestZScoreDetector:
    def _make_baseline(self) -> VolumeBaseline:
        return VolumeBaseline(
            service_name="payment-service",
            endpoint="/api/v1/payments",
            hour_of_day=12,
            mean_count=1000.0,
            std_count=100.0,
            mean_latency_ms=50.0,
            std_latency_ms=10.0,
            day_of_week=1,
            sample_size=30,
        )

    def test_no_anomaly_within_threshold(self):
        detector = ZScoreDetector(warning_threshold=2.0, critical_threshold=3.0)
        baseline = self._make_baseline()
        obs = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 12, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=1050,
        )
        result = detector.detect(obs, baseline)
        assert result is None

    def test_detects_volume_spike(self):
        detector = ZScoreDetector(warning_threshold=2.0, critical_threshold=3.0)
        baseline = self._make_baseline()
        obs = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 12, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=1500,  # 5 std devs above mean
        )
        result = detector.detect(obs, baseline)
        assert result is not None
        assert result.anomaly_type == AnomalyType.VOLUME_SPIKE

    def test_detects_volume_drop(self):
        detector = ZScoreDetector(warning_threshold=2.0, critical_threshold=3.0)
        baseline = self._make_baseline()
        obs = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 12, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=500,  # 5 std devs below mean
        )
        result = detector.detect(obs, baseline)
        assert result is not None
        assert result.anomaly_type == AnomalyType.VOLUME_DROP

    def test_severity_classification(self):
        detector = ZScoreDetector(warning_threshold=2.0, critical_threshold=3.0)
        baseline = self._make_baseline()

        # 2.5 std devs -> MEDIUM
        obs = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 12, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=1250,
        )
        result = detector.detect(obs, baseline)
        assert result is not None
        assert result.severity == AnomalySeverity.MEDIUM

        # 4 std devs -> HIGH
        obs_high = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 12, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=1400,
        )
        result_high = detector.detect(obs_high, baseline)
        assert result_high is not None
        assert result_high.severity == AnomalySeverity.HIGH


class TestSeasonalDetectorDayOfWeekMode:
    """Tests for the default day_of_week seasonal mode."""

    def _make_series(self) -> VolumeTimeSeries:
        """Build a time series with observations across multiple Mondays and Tuesdays."""
        observations = []
        # 3 Mondays at 10am
        for day in (5, 12, 19):  # Jan 2026 Mondays
            observations.append(
                TransactionVolume(
                    timestamp=datetime(2026, 1, day, 10, 0),
                    service_name="test-service",
                    endpoint="/api/test",
                    count=100 + day,
                    avg_latency_ms=50.0,
                )
            )
        # 3 Tuesdays at 10am  (different volume level)
        for day in (6, 13, 20):  # Jan 2026 Tuesdays
            observations.append(
                TransactionVolume(
                    timestamp=datetime(2026, 1, day, 10, 0),
                    service_name="test-service",
                    endpoint="/api/test",
                    count=200 + day,
                    avg_latency_ms=60.0,
                )
            )
        return VolumeTimeSeries(
            service_name="test-service",
            endpoint="/api/test",
            observations=observations,
        )

    def test_build_baselines(self):
        detector = SeasonalDetector(min_samples=2, seasonal_mode="day_of_week")
        series = VolumeTimeSeries(
            service_name="test-service",
            endpoint="/api/test",
            observations=[
                TransactionVolume(
                    timestamp=datetime(2026, 1, 5, 10, 0),  # Monday 10am
                    service_name="test-service",
                    endpoint="/api/test",
                    count=100,
                    avg_latency_ms=50.0,
                ),
                TransactionVolume(
                    timestamp=datetime(2026, 1, 12, 10, 0),  # Monday 10am
                    service_name="test-service",
                    endpoint="/api/test",
                    count=120,
                    avg_latency_ms=55.0,
                ),
                TransactionVolume(
                    timestamp=datetime(2026, 1, 19, 10, 0),  # Monday 10am
                    service_name="test-service",
                    endpoint="/api/test",
                    count=110,
                    avg_latency_ms=52.0,
                ),
            ],
        )
        baselines = detector.build_baselines(series)
        assert len(baselines) == 1
        assert baselines[0].hour_of_day == 10
        assert baselines[0].day_of_week == 0  # Monday

    def test_separate_baselines_per_day(self):
        """day_of_week mode should create separate baselines for Monday vs Tuesday."""
        detector = SeasonalDetector(min_samples=2, seasonal_mode="day_of_week")
        series = self._make_series()
        baselines = detector.build_baselines(series)

        # Should get 2 baselines: one for Monday-10am, one for Tuesday-10am
        assert len(baselines) == 2
        days = {b.day_of_week for b in baselines}
        assert days == {0, 1}  # Monday=0, Tuesday=1

    def test_detect_anomaly_day_of_week(self):
        """An observation far from the Monday baseline should be flagged."""
        detector = SeasonalDetector(
            deviation_threshold=2.0, min_samples=2, seasonal_mode="day_of_week"
        )
        series = self._make_series()
        baselines = detector.build_baselines(series)

        # Observation on a Monday at 10am with a very high count
        obs = TransactionVolume(
            timestamp=datetime(2026, 1, 26, 10, 0),  # Monday
            service_name="test-service",
            endpoint="/api/test",
            count=500,
        )
        result = detector.detect(obs, baselines)
        assert result is not None
        assert result.anomaly_type.name == "VOLUME_SPIKE"
        assert "day=" in result.description

    def test_no_anomaly_within_threshold_day_of_week(self):
        """Normal observation should not trigger anomaly."""
        detector = SeasonalDetector(
            deviation_threshold=2.0, min_samples=2, seasonal_mode="day_of_week"
        )
        series = self._make_series()
        baselines = detector.build_baselines(series)

        obs = TransactionVolume(
            timestamp=datetime(2026, 1, 26, 10, 0),  # Monday
            service_name="test-service",
            endpoint="/api/test",
            count=112,  # Close to Monday mean (~112)
        )
        result = detector.detect(obs, baselines)
        assert result is None


class TestSeasonalDetectorTimeOfDayMode:
    """Tests for the time_of_day seasonal mode."""

    def _make_series(self) -> VolumeTimeSeries:
        """Build a time series with observations across multiple days at the same hours."""
        observations = []
        # Observations at 10am across Mon, Tue, Wed, Thu (different days, same hour)
        for day in (6, 7, 8, 9):  # Jan 2026: Mon-Thu
            observations.append(
                TransactionVolume(
                    timestamp=datetime(2026, 1, day, 10, 0),
                    service_name="test-service",
                    endpoint="/api/test",
                    count=100 + day,
                    avg_latency_ms=50.0,
                )
            )
        # Observations at 14:00 across Mon, Tue, Wed, Thu
        for day in (6, 7, 8, 9):
            observations.append(
                TransactionVolume(
                    timestamp=datetime(2026, 1, day, 14, 0),
                    service_name="test-service",
                    endpoint="/api/test",
                    count=200 + day,
                    avg_latency_ms=60.0,
                )
            )
        return VolumeTimeSeries(
            service_name="test-service",
            endpoint="/api/test",
            observations=observations,
        )

    def test_build_baselines_pools_all_days(self):
        """time_of_day mode should pool all days into one baseline per hour."""
        detector = SeasonalDetector(min_samples=2, seasonal_mode="time_of_day")
        series = self._make_series()
        baselines = detector.build_baselines(series)

        # Should get 2 baselines: one for hour=10, one for hour=14
        assert len(baselines) == 2
        hours = {b.hour_of_day for b in baselines}
        assert hours == {10, 14}
        # day_of_week should be None for all baselines
        for b in baselines:
            assert b.day_of_week is None

    def test_baselines_have_correct_sample_size(self):
        """Each hour bucket should contain observations from all days."""
        detector = SeasonalDetector(min_samples=2, seasonal_mode="time_of_day")
        series = self._make_series()
        baselines = detector.build_baselines(series)

        for b in baselines:
            assert b.sample_size == 4  # 4 days pooled

    def test_detect_anomaly_time_of_day(self):
        """An observation far from the hourly baseline should be flagged."""
        detector = SeasonalDetector(
            deviation_threshold=2.0, min_samples=2, seasonal_mode="time_of_day"
        )
        series = self._make_series()
        baselines = detector.build_baselines(series)

        # Observation on a Friday at 10am with a very high count
        obs = TransactionVolume(
            timestamp=datetime(2026, 1, 10, 10, 0),  # Friday
            service_name="test-service",
            endpoint="/api/test",
            count=500,
        )
        result = detector.detect(obs, baselines)
        assert result is not None
        assert result.anomaly_type.name == "VOLUME_SPIKE"
        # In time_of_day mode the description should NOT contain "day="
        assert "day=" not in result.description

    def test_no_anomaly_within_threshold_time_of_day(self):
        """Normal observation should not trigger anomaly."""
        detector = SeasonalDetector(
            deviation_threshold=2.0, min_samples=2, seasonal_mode="time_of_day"
        )
        series = self._make_series()
        baselines = detector.build_baselines(series)

        obs = TransactionVolume(
            timestamp=datetime(2026, 1, 10, 10, 0),  # Friday
            service_name="test-service",
            endpoint="/api/test",
            count=108,  # Close to the mean (~107.5)
        )
        result = detector.detect(obs, baselines)
        assert result is None

    def test_detect_volume_drop_time_of_day(self):
        """A very low observation should be flagged as a volume drop."""
        detector = SeasonalDetector(
            deviation_threshold=2.0, min_samples=2, seasonal_mode="time_of_day"
        )
        series = self._make_series()
        baselines = detector.build_baselines(series)

        obs = TransactionVolume(
            timestamp=datetime(2026, 1, 10, 14, 0),  # Friday 2pm
            service_name="test-service",
            endpoint="/api/test",
            count=10,  # Far below the mean (~207.5)
        )
        result = detector.detect(obs, baselines)
        assert result is not None
        assert result.anomaly_type.name == "VOLUME_DROP"


class TestSeasonalDetectorInvalidMode:
    def test_invalid_mode_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Invalid seasonal_mode"):
            SeasonalDetector(seasonal_mode="invalid")  # type: ignore[arg-type]


class TestStatHelpers:
    def test_mean(self):
        assert _mean([10, 20, 30]) == 20.0

    def test_std(self):
        std = _std([10, 20, 30])
        assert abs(std - 10.0) < 0.01

    def test_mean_empty(self):
        assert _mean([]) == 0.0

    def test_std_single(self):
        assert _std([42]) == 0.0
