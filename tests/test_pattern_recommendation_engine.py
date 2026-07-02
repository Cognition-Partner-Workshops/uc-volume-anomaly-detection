"""Tests for the pattern-aware recommendation engine."""

from datetime import datetime

from src.agents.correlation_engine import CorrelationReport, CorrelationResult
from src.agents.pattern_recommendation_engine import (
    PatternRecommendationEngine,
)
from src.models.anomaly import AnomalyEvent, AnomalySeverity, AnomalyType


def _make_anomaly(
    anomaly_type: AnomalyType = AnomalyType.VOLUME_DROP,
    severity: AnomalySeverity = AnomalySeverity.HIGH,
    service_name: str = "payment-service",
) -> AnomalyEvent:
    return AnomalyEvent(
        anomaly_id="anom-test-001",
        anomaly_type=anomaly_type,
        severity=severity,
        service_name=service_name,
        endpoint="/api/v1/payments",
        detected_at=datetime(2026, 3, 10, 12, 0),
        observed_value=500.0,
        expected_value=1000.0,
        deviation_score=5.0,
        description="Test anomaly",
    )


class TestPatternRecommendationEngine:
    def test_matches_rapid_degradation_pattern(self):
        engine = PatternRecommendationEngine()
        anomaly = _make_anomaly(
            anomaly_type=AnomalyType.VOLUME_DROP,
            severity=AnomalySeverity.HIGH,
        )
        actions = engine.recommend(anomaly)
        assert len(actions) > 0
        action_strings = [a.action for a in actions]
        assert any("deployment" in a.lower() for a in action_strings)

    def test_matches_traffic_surge_pattern(self):
        engine = PatternRecommendationEngine()
        anomaly = _make_anomaly(
            anomaly_type=AnomalyType.VOLUME_SPIKE,
            severity=AnomalySeverity.HIGH,
        )
        actions = engine.recommend(anomaly)
        assert len(actions) > 0
        action_strings = [a.action for a in actions]
        assert any("auto-scaling" in a.lower() or "scaling" in a.lower() for a in action_strings)

    def test_matches_rate_of_change_pattern(self):
        engine = PatternRecommendationEngine()
        anomaly = _make_anomaly(
            anomaly_type=AnomalyType.RATE_OF_CHANGE,
            severity=AnomalySeverity.MEDIUM,
        )
        actions = engine.recommend(anomaly)
        assert len(actions) > 0
        action_strings = [a.action for a in actions]
        assert any("rate of change" in a.lower() or "deployment" in a.lower() for a in action_strings)

    def test_upstream_cascade_with_correlation(self):
        engine = PatternRecommendationEngine()
        anomaly = _make_anomaly(
            anomaly_type=AnomalyType.VOLUME_DROP,
            severity=AnomalySeverity.HIGH,
        )

        correlation_report = CorrelationReport(
            anomaly=anomaly,
            correlations=[
                CorrelationResult(
                    source_anomaly=anomaly,
                    correlated_service="auth-service",
                    correlation_type="upstream_cause",
                    confidence=0.8,
                    time_offset_seconds=60.0,
                    evidence="Upstream auth-service had volume_drop 60s before",
                ),
            ],
            root_cause_candidates=["auth-service"],
            impact_scope=[],
        )

        actions = engine.recommend(anomaly, correlation_report=correlation_report)
        assert len(actions) > 0
        action_strings = [a.action for a in actions]
        assert any("upstream" in a.lower() for a in action_strings)
        assert any("auth-service" in a for a in action_strings)

    def test_multi_service_outage_pattern(self):
        engine = PatternRecommendationEngine()
        anomaly = _make_anomaly(
            anomaly_type=AnomalyType.VOLUME_DROP,
            severity=AnomalySeverity.HIGH,
        )

        correlation_report = CorrelationReport(
            anomaly=anomaly,
            correlations=[
                CorrelationResult(
                    source_anomaly=anomaly,
                    correlated_service="order-service",
                    correlation_type="co_occurrence",
                    confidence=0.7,
                    time_offset_seconds=30.0,
                    evidence="Peer order-service had volume_drop within 30s",
                ),
            ],
            root_cause_candidates=[],
            impact_scope=[],
        )

        actions = engine.recommend(anomaly, correlation_report=correlation_report)
        assert len(actions) > 0

    def test_no_match_below_severity_threshold(self):
        engine = PatternRecommendationEngine()
        anomaly = _make_anomaly(
            anomaly_type=AnomalyType.VOLUME_DROP,
            severity=AnomalySeverity.LOW,
        )
        actions = engine.recommend(anomaly)
        # LOW severity shouldn't match rapid-degradation (requires MEDIUM+)
        assert len(actions) == 0

    def test_correlation_actions_include_root_cause(self):
        engine = PatternRecommendationEngine()
        anomaly = _make_anomaly(
            anomaly_type=AnomalyType.RATE_OF_CHANGE,
            severity=AnomalySeverity.MEDIUM,
        )

        correlation_report = CorrelationReport(
            anomaly=anomaly,
            correlations=[
                CorrelationResult(
                    source_anomaly=anomaly,
                    correlated_service="auth-service",
                    correlation_type="upstream_cause",
                    confidence=0.85,
                    time_offset_seconds=45.0,
                    evidence="test",
                ),
            ],
            root_cause_candidates=["auth-service"],
            impact_scope=["notification-service"],
        )

        actions = engine.recommend(anomaly, correlation_report=correlation_report)
        action_strings = [a.action for a in actions]
        assert any("auth-service" in a and "root cause" in a.lower() for a in action_strings)
        assert any("notification-service" in a for a in action_strings)

    def test_deduplicates_actions(self):
        engine = PatternRecommendationEngine()
        anomaly = _make_anomaly(
            anomaly_type=AnomalyType.VOLUME_DROP,
            severity=AnomalySeverity.CRITICAL,
        )

        actions = engine.recommend(anomaly)
        action_strings = [a.action for a in actions]
        # No duplicates
        assert len(action_strings) == len(set(action_strings))

    def test_updates_anomaly_recommended_actions(self):
        engine = PatternRecommendationEngine()
        anomaly = _make_anomaly(
            anomaly_type=AnomalyType.VOLUME_SPIKE,
            severity=AnomalySeverity.HIGH,
        )
        assert anomaly.recommended_actions == []

        engine.recommend(anomaly)
        assert len(anomaly.recommended_actions) > 0
