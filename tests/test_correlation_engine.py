"""Tests for the correlation engine."""

from datetime import datetime, timedelta


from src.agents.correlation_engine import (
    CorrelationEngine,
    CorrelationResult,
)
from src.models.anomaly import AnomalyEvent, AnomalySeverity, AnomalyType
from src.models.transaction import TransactionVolume
from src.models.service_health import (
    HealthStatus,
    ServiceDependency,
    ServiceHealthSnapshot,
    ServiceMap,
)


def _make_service_map() -> ServiceMap:
    """Create a test service map: auth -> payment -> notification."""
    return ServiceMap(
        services=["auth-service", "payment-service", "notification-service", "order-service"],
        dependencies=[
            ServiceDependency(
                source_service="payment-service",
                target_service="auth-service",
                dependency_type="http",
                criticality="required",
            ),
            ServiceDependency(
                source_service="notification-service",
                target_service="payment-service",
                dependency_type="message_queue",
                criticality="optional",
            ),
            ServiceDependency(
                source_service="order-service",
                target_service="auth-service",
                dependency_type="http",
                criticality="required",
            ),
        ],
    )


def _make_anomaly(
    service_name: str = "payment-service",
    detected_at: datetime = datetime(2026, 3, 10, 12, 0),
    anomaly_type: AnomalyType = AnomalyType.VOLUME_DROP,
    severity: AnomalySeverity = AnomalySeverity.HIGH,
) -> AnomalyEvent:
    return AnomalyEvent(
        anomaly_id="anom-test-001",
        anomaly_type=anomaly_type,
        severity=severity,
        service_name=service_name,
        endpoint="/api/v1/payments",
        detected_at=detected_at,
        observed_value=500.0,
        expected_value=1000.0,
        deviation_score=5.0,
        description="Test anomaly",
    )


class TestCorrelationEngine:
    def test_detects_upstream_anomaly_correlation(self):
        service_map = _make_service_map()
        engine = CorrelationEngine(service_map=service_map)

        # Register an upstream anomaly 5 minutes before
        upstream_anomaly = _make_anomaly(
            service_name="auth-service",
            detected_at=datetime(2026, 3, 10, 11, 55),
            severity=AnomalySeverity.HIGH,
        )
        engine.register_anomaly(upstream_anomaly)

        # Detect correlation for the downstream anomaly
        downstream_anomaly = _make_anomaly(
            service_name="payment-service",
            detected_at=datetime(2026, 3, 10, 12, 0),
        )
        report = engine.correlate(downstream_anomaly)

        assert len(report.correlations) >= 1
        upstream_correlations = [
            c for c in report.correlations
            if c.correlation_type == "upstream_cause"
        ]
        assert len(upstream_correlations) >= 1
        assert upstream_correlations[0].correlated_service == "auth-service"
        assert upstream_correlations[0].time_offset_seconds == 300.0

    def test_no_correlation_outside_window(self):
        service_map = _make_service_map()
        engine = CorrelationEngine(
            service_map=service_map,
            correlation_window=timedelta(minutes=10),
        )

        # Register an upstream anomaly 20 minutes before (outside 10min window)
        upstream_anomaly = _make_anomaly(
            service_name="auth-service",
            detected_at=datetime(2026, 3, 10, 11, 40),
        )
        engine.register_anomaly(upstream_anomaly)

        downstream_anomaly = _make_anomaly(
            service_name="payment-service",
            detected_at=datetime(2026, 3, 10, 12, 0),
        )
        report = engine.correlate(downstream_anomaly)

        upstream_correlations = [
            c for c in report.correlations
            if c.correlation_type == "upstream_cause"
            and c.correlated_service == "auth-service"
            and c.time_offset_seconds > 0  # from anomaly history only
        ]
        assert len(upstream_correlations) == 0

    def test_detects_downstream_impact(self):
        service_map = _make_service_map()
        engine = CorrelationEngine(service_map=service_map)

        # Register a downstream anomaly 2 minutes after
        downstream_anomaly = _make_anomaly(
            service_name="notification-service",
            detected_at=datetime(2026, 3, 10, 12, 2),
            severity=AnomalySeverity.MEDIUM,
        )
        engine.register_anomaly(downstream_anomaly)

        # Check correlation for the upstream anomaly
        source_anomaly = _make_anomaly(
            service_name="payment-service",
            detected_at=datetime(2026, 3, 10, 12, 0),
        )
        report = engine.correlate(source_anomaly)

        downstream_correlations = [
            c for c in report.correlations
            if c.correlation_type == "downstream_impact"
        ]
        assert len(downstream_correlations) >= 1
        assert downstream_correlations[0].correlated_service == "notification-service"

    def test_detects_peer_co_occurrence(self):
        service_map = _make_service_map()
        engine = CorrelationEngine(service_map=service_map)

        # payment-service and order-service share upstream auth-service
        # Register a peer anomaly close in time
        peer_anomaly = _make_anomaly(
            service_name="order-service",
            detected_at=datetime(2026, 3, 10, 12, 1),
            severity=AnomalySeverity.HIGH,
        )
        engine.register_anomaly(peer_anomaly)

        source_anomaly = _make_anomaly(
            service_name="payment-service",
            detected_at=datetime(2026, 3, 10, 12, 0),
        )
        report = engine.correlate(source_anomaly)

        co_occurrence = [
            c for c in report.correlations
            if c.correlation_type == "co_occurrence"
        ]
        assert len(co_occurrence) >= 1
        assert co_occurrence[0].correlated_service == "order-service"

    def test_health_degradation_detected(self):
        service_map = _make_service_map()
        engine = CorrelationEngine(service_map=service_map)

        health_snapshots = {
            "auth-service": ServiceHealthSnapshot(
                service_name="auth-service",
                timestamp=datetime(2026, 3, 10, 12, 0),
                status=HealthStatus.UNHEALTHY,
                error_rate=0.15,
            ),
        }

        anomaly = _make_anomaly(
            service_name="payment-service",
            detected_at=datetime(2026, 3, 10, 12, 0),
        )
        report = engine.correlate(anomaly, health_snapshots=health_snapshots)

        health_correlations = [
            c for c in report.correlations
            if c.correlated_service == "auth-service"
            and "currently" in c.evidence
        ]
        assert len(health_correlations) >= 1
        assert health_correlations[0].confidence == 0.7  # UNHEALTHY = 0.7

    def test_root_cause_candidates(self):
        service_map = _make_service_map()
        engine = CorrelationEngine(service_map=service_map)

        # Register a clear upstream anomaly
        upstream_anomaly = _make_anomaly(
            service_name="auth-service",
            detected_at=datetime(2026, 3, 10, 11, 59),
            severity=AnomalySeverity.CRITICAL,
        )
        engine.register_anomaly(upstream_anomaly)

        downstream_anomaly = _make_anomaly(
            service_name="payment-service",
            detected_at=datetime(2026, 3, 10, 12, 0),
        )
        report = engine.correlate(downstream_anomaly)

        assert report.has_upstream_cause
        assert "auth-service" in report.root_cause_candidates

    def test_correlation_updates_anomaly_correlated_services(self):
        service_map = _make_service_map()
        engine = CorrelationEngine(service_map=service_map)

        upstream_anomaly = _make_anomaly(
            service_name="auth-service",
            detected_at=datetime(2026, 3, 10, 11, 58),
            severity=AnomalySeverity.HIGH,
        )
        engine.register_anomaly(upstream_anomaly)

        anomaly = _make_anomaly(
            service_name="payment-service",
            detected_at=datetime(2026, 3, 10, 12, 0),
        )
        engine.correlate(anomaly)

        assert "auth-service" in anomaly.correlated_services

    def test_empty_service_map_returns_empty_report(self):
        engine = CorrelationEngine(service_map=ServiceMap())

        anomaly = _make_anomaly()
        report = engine.correlate(anomaly)

        assert len(report.correlations) == 0
        assert not report.has_upstream_cause

    def test_register_volume_stored(self):
        engine = CorrelationEngine(service_map=ServiceMap())
        obs = TransactionVolume(
            timestamp=datetime(2026, 3, 10, 12, 0),
            service_name="payment-service",
            endpoint="/api/v1/payments",
            count=1000,
        )
        engine.register_volume(obs)
        assert "payment-service" in engine._volume_history
        assert len(engine._volume_history["payment-service"]) == 1


class TestCorrelationResult:
    def test_is_likely_cause_true(self):
        result = CorrelationResult(
            source_anomaly=_make_anomaly(),
            correlated_service="auth-service",
            correlation_type="upstream_cause",
            confidence=0.8,
            time_offset_seconds=60.0,
            evidence="test",
        )
        assert result.is_likely_cause is True

    def test_is_likely_cause_false_low_confidence(self):
        result = CorrelationResult(
            source_anomaly=_make_anomaly(),
            correlated_service="auth-service",
            correlation_type="upstream_cause",
            confidence=0.4,
            time_offset_seconds=60.0,
            evidence="test",
        )
        assert result.is_likely_cause is False

    def test_is_likely_cause_false_wrong_type(self):
        result = CorrelationResult(
            source_anomaly=_make_anomaly(),
            correlated_service="notification-service",
            correlation_type="downstream_impact",
            confidence=0.9,
            time_offset_seconds=60.0,
            evidence="test",
        )
        assert result.is_likely_cause is False

