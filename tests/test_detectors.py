"""Tests for anomaly detection algorithms."""

from datetime import datetime, timedelta

from src.detectors.ewma_detector import EWMADetector
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
            day_of_week=1,
            mean_count=1000.0,
            std_count=100.0,
            mean_latency_ms=50.0,
            std_latency_ms=10.0,
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


class TestSeasonalDetector:
    def test_build_baselines(self):
        detector = SeasonalDetector(min_samples=2)
        series = VolumeTimeSeries(
            service_name="test-service",
            endpoint="/api/test",
            observations=[
                TransactionVolume(
                    timestamp=datetime(2026, 1, 6, 10, 0),  # Monday 10am
                    service_name="test-service",
                    endpoint="/api/test",
                    count=100,
                    avg_latency_ms=50.0,
                ),
                TransactionVolume(
                    timestamp=datetime(2026, 1, 13, 10, 0),  # Monday 10am
                    service_name="test-service",
                    endpoint="/api/test",
                    count=120,
                    avg_latency_ms=55.0,
                ),
                TransactionVolume(
                    timestamp=datetime(2026, 1, 20, 10, 0),  # Monday 10am
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


class TestEWMADetector:
    WARM_UP_COUNTS = [100, 104, 96, 102, 98, 101, 99, 103, 97, 100]

    def _make_detector(self) -> EWMADetector:
        return EWMADetector(
            alpha=0.3,
            warning_threshold=2.0,
            critical_threshold=3.0,
            min_observations=5,
        )

    def _observation(self, count: int, index: int) -> TransactionVolume:
        return TransactionVolume(
            timestamp=datetime(2026, 3, 10, 0, 0) + timedelta(hours=index),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=count,
        )

    def _warm_up(self, detector: EWMADetector) -> int:
        for index, count in enumerate(self.WARM_UP_COUNTS):
            assert detector.detect(self._observation(count, index)) is None
        return len(self.WARM_UP_COUNTS)

    def test_no_anomaly_within_threshold(self):
        detector = self._make_detector()
        index = self._warm_up(detector)
        assert detector.detect(self._observation(101, index)) is None

    def test_detects_volume_spike(self):
        detector = self._make_detector()
        index = self._warm_up(detector)
        result = detector.detect(self._observation(300, index))
        assert result is not None
        assert result.anomaly_type == AnomalyType.VOLUME_SPIKE
        assert result.observed_value == 300.0
        assert result.expected_value < 110.0
        assert result.deviation_score > 3.0

    def test_detects_volume_drop(self):
        detector = self._make_detector()
        index = self._warm_up(detector)
        result = detector.detect(self._observation(10, index))
        assert result is not None
        assert result.anomaly_type == AnomalyType.VOLUME_DROP
        assert result.deviation_score > 3.0

    def test_severity_classification(self):
        # ~2.1σ above the EWMA -> MEDIUM
        medium = self._make_detector()
        index = self._warm_up(medium)
        medium_result = medium.detect(self._observation(105, index))
        assert medium_result is not None
        assert medium_result.severity == AnomalySeverity.MEDIUM

        # ~4.1σ above the EWMA -> HIGH
        high = self._make_detector()
        index = self._warm_up(high)
        high_result = high.detect(self._observation(110, index))
        assert high_result is not None
        assert high_result.severity == AnomalySeverity.HIGH

        # far outside the EWMA -> CRITICAL
        critical = self._make_detector()
        index = self._warm_up(critical)
        critical_result = critical.detect(self._observation(200, index))
        assert critical_result is not None
        assert critical_result.severity == AnomalySeverity.CRITICAL

    def test_first_observation_and_warm_up_are_silent(self):
        detector = self._make_detector()
        assert detector.detect(self._observation(100, 0)) is None
        for index, count in enumerate([104, 96, 5000], start=1):
            assert detector.detect(self._observation(count, index)) is None

    def test_order_dependence(self):
        # The same spike is only flagged once the EWMA has warmed up, so the
        # ordering of observations changes the outcome.
        early = self._make_detector()
        assert early.detect(self._observation(100, 0)) is None
        assert early.detect(self._observation(500, 1)) is None

        late = self._make_detector()
        index = self._warm_up(late)
        assert late.detect(self._observation(500, index)) is not None

    def test_sustained_shift_stops_alerting(self):
        detector = self._make_detector()
        index = self._warm_up(detector)
        assert detector.detect(self._observation(300, index)) is not None
        for offset in range(1, 15):
            detector.detect(self._observation(300, index + offset))
        assert detector.detect(self._observation(300, index + 15)) is None

    def test_state_is_per_service_endpoint(self):
        detector = self._make_detector()
        self._warm_up(detector)
        other = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 12, 0),
            service_name="order-service",
            endpoint="/api/v1/orders",
            count=5000,
        )
        assert detector.detect(other) is None
        assert len(detector.states) == 2


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
