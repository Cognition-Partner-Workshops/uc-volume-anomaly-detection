"""Pattern-Aware Recommendation Engine — suggests runbook actions based on
anomaly patterns and correlated services.

Extends the base recommendation engine with pattern matching that considers:
- The type of anomaly detected (rate-of-change vs absolute)
- Which correlated services are involved
- The propagation path (upstream cause vs downstream impact)
- Historical incident patterns
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.agents.correlation_engine import CorrelationReport
from src.models.anomaly import AnomalyEvent, AnomalySeverity, AnomalyType

logger = logging.getLogger(__name__)


@dataclass
class RunbookAction:
    """A specific runbook action to take in response to an anomaly pattern."""

    action: str
    priority: int  # 1 = highest priority
    confidence: float  # 0.0 - 1.0
    source: str  # "pattern_match", "correlation", "historical", "heuristic"
    rationale: str
    estimated_impact: str = ""
    automation_hint: Optional[str] = None


@dataclass
class PatternSignature:
    """Defines a recognizable anomaly pattern and its recommended responses."""

    pattern_id: str
    name: str
    description: str
    anomaly_types: list[AnomalyType]
    requires_correlation: bool
    correlation_types: list[str] = field(default_factory=list)
    severity_minimum: AnomalySeverity = AnomalySeverity.MEDIUM
    actions: list[RunbookAction] = field(default_factory=list)


# Built-in pattern signatures
PATTERN_SIGNATURES: list[PatternSignature] = [
    PatternSignature(
        pattern_id="rapid-degradation",
        name="Rapid Service Degradation",
        description="Volume changing faster than historical norms, indicating accelerating failure",
        anomaly_types=[AnomalyType.VOLUME_DROP],
        requires_correlation=False,
        severity_minimum=AnomalySeverity.MEDIUM,
        actions=[
            RunbookAction(
                action="Check recent deployments for regression (rollback if <30min old)",
                priority=1,
                confidence=0.85,
                source="pattern_match",
                rationale="Rapid volume decline often correlates with bad deployments",
                estimated_impact="Immediate restoration if deployment-related",
                automation_hint="kubectl rollout undo deployment/<service>",
            ),
            RunbookAction(
                action="Inspect error logs for new exception patterns",
                priority=2,
                confidence=0.80,
                source="pattern_match",
                rationale="Accelerating drops indicate cascading failures visible in logs",
            ),
            RunbookAction(
                action="Verify health check endpoints are responding correctly",
                priority=3,
                confidence=0.75,
                source="pattern_match",
                rationale="Load balancers may route away from unhealthy instances",
                automation_hint="curl -f http://<service>/health",
            ),
        ],
    ),
    PatternSignature(
        pattern_id="upstream-cascade",
        name="Upstream Cascade Failure",
        description="Anomaly in service with correlated upstream degradation",
        anomaly_types=[AnomalyType.VOLUME_DROP, AnomalyType.LATENCY_SPIKE],
        requires_correlation=True,
        correlation_types=["upstream_cause"],
        severity_minimum=AnomalySeverity.MEDIUM,
        actions=[
            RunbookAction(
                action="Investigate upstream service health and connectivity",
                priority=1,
                confidence=0.90,
                source="pattern_match",
                rationale="Upstream failures propagate to dependent services",
                estimated_impact="Fix upstream to resolve downstream cascade",
            ),
            RunbookAction(
                action="Enable circuit breaker for degraded upstream dependency",
                priority=2,
                confidence=0.80,
                source="pattern_match",
                rationale="Isolate failure to prevent resource exhaustion",
                automation_hint="Toggle circuit-breaker feature flag for <upstream>",
            ),
            RunbookAction(
                action="Activate fallback/cached responses if available",
                priority=3,
                confidence=0.70,
                source="pattern_match",
                rationale="Reduce customer impact while upstream recovers",
            ),
            RunbookAction(
                action="Scale downstream services to handle retry pressure",
                priority=4,
                confidence=0.60,
                source="pattern_match",
                rationale="Retries from upstream failures increase load",
                automation_hint="kubectl scale deployment/<service> --replicas=<n+2>",
            ),
        ],
    ),
    PatternSignature(
        pattern_id="traffic-surge",
        name="Unexpected Traffic Surge",
        description="Rapid volume increase exceeding capacity planning norms",
        anomaly_types=[AnomalyType.VOLUME_SPIKE],
        requires_correlation=False,
        severity_minimum=AnomalySeverity.MEDIUM,
        actions=[
            RunbookAction(
                action="Verify auto-scaling is active and responding",
                priority=1,
                confidence=0.85,
                source="pattern_match",
                rationale="Surge may exceed current scaling capacity",
                automation_hint="kubectl get hpa <service>",
            ),
            RunbookAction(
                action="Check for retry storms from downstream consumers",
                priority=2,
                confidence=0.75,
                source="pattern_match",
                rationale="Exponential retries amplify traffic beyond organic growth",
            ),
            RunbookAction(
                action="Activate rate limiting at API gateway level",
                priority=3,
                confidence=0.70,
                source="pattern_match",
                rationale="Protect service from overload during spike investigation",
            ),
            RunbookAction(
                action="Review scheduled batch jobs for unintended overlap",
                priority=4,
                confidence=0.60,
                source="pattern_match",
                rationale="Batch executions can cause predictable spikes",
            ),
        ],
    ),
    PatternSignature(
        pattern_id="multi-service-outage",
        name="Multi-Service Coordinated Outage",
        description="Multiple services experiencing anomalies within correlation window",
        anomaly_types=[AnomalyType.VOLUME_DROP, AnomalyType.VOLUME_SPIKE, AnomalyType.LATENCY_SPIKE],
        requires_correlation=True,
        correlation_types=["co_occurrence", "upstream_cause"],
        severity_minimum=AnomalySeverity.HIGH,
        actions=[
            RunbookAction(
                action="Check shared infrastructure (database, message queue, network)",
                priority=1,
                confidence=0.90,
                source="pattern_match",
                rationale="Multi-service impact indicates shared dependency failure",
                estimated_impact="Resolves all affected services simultaneously",
            ),
            RunbookAction(
                action="Verify DNS resolution and service discovery health",
                priority=2,
                confidence=0.80,
                source="pattern_match",
                rationale="DNS failures cause widespread service disruption",
            ),
            RunbookAction(
                action="Check cloud provider status page for regional issues",
                priority=3,
                confidence=0.70,
                source="pattern_match",
                rationale="Regional outages affect multiple services simultaneously",
            ),
            RunbookAction(
                action="Engage incident commander and open war room",
                priority=4,
                confidence=0.85,
                source="pattern_match",
                rationale="Multi-service outages require coordinated response",
            ),
        ],
    ),
    PatternSignature(
        pattern_id="gradual-capacity-exhaustion",
        name="Gradual Capacity Exhaustion",
        description="Rate of change indicates slow resource leak or growing backlog",
        anomaly_types=[AnomalyType.LATENCY_SPIKE],
        requires_correlation=False,
        severity_minimum=AnomalySeverity.LOW,
        actions=[
            RunbookAction(
                action="Check database connection pool utilization trend",
                priority=1,
                confidence=0.80,
                source="pattern_match",
                rationale="Connection leaks cause gradual latency increase",
                automation_hint="SELECT count(*) FROM pg_stat_activity",
            ),
            RunbookAction(
                action="Review memory and GC metrics for leak indicators",
                priority=2,
                confidence=0.75,
                source="pattern_match",
                rationale="Memory pressure causes increased GC pauses",
            ),
            RunbookAction(
                action="Check disk I/O and queue depth metrics",
                priority=3,
                confidence=0.70,
                source="pattern_match",
                rationale="Disk saturation causes write stalls and read delays",
            ),
        ],
    ),
    PatternSignature(
        pattern_id="rate-of-change-anomaly",
        name="Abnormal Rate of Change",
        description="Metric changing faster than historical norms even if absolute value is within bounds",
        anomaly_types=[AnomalyType.RATE_OF_CHANGE],
        requires_correlation=False,
        severity_minimum=AnomalySeverity.MEDIUM,
        actions=[
            RunbookAction(
                action="Compare current rate of change against deployment timeline",
                priority=1,
                confidence=0.85,
                source="pattern_match",
                rationale="Rapid changes often correlate with recent releases",
            ),
            RunbookAction(
                action="Check for upstream dependency latency or error rate changes",
                priority=2,
                confidence=0.80,
                source="pattern_match",
                rationale="Cascading effects from dependencies appear as rate changes first",
            ),
            RunbookAction(
                action="Monitor for next observation to confirm trend direction",
                priority=3,
                confidence=0.70,
                source="pattern_match",
                rationale="Single-point rate anomalies may self-correct; confirm with follow-up",
            ),
            RunbookAction(
                action="Pre-scale resources if rate indicates approaching capacity limits",
                priority=4,
                confidence=0.65,
                source="pattern_match",
                rationale="Proactive scaling prevents threshold breaches",
                automation_hint="kubectl scale deployment/<service> --replicas=<current+1>",
            ),
        ],
    ),
    PatternSignature(
        pattern_id="rate-of-change-with-upstream",
        name="Upstream-Driven Rate Acceleration",
        description="Rate of change anomaly with correlated upstream service changes",
        anomaly_types=[AnomalyType.RATE_OF_CHANGE],
        requires_correlation=True,
        correlation_types=["upstream_cause"],
        severity_minimum=AnomalySeverity.MEDIUM,
        actions=[
            RunbookAction(
                action="Investigate correlated upstream service for root cause",
                priority=1,
                confidence=0.90,
                source="pattern_match",
                rationale="Rate changes driven by upstream propagate predictably",
                estimated_impact="Fixing upstream stabilizes rate of change downstream",
            ),
            RunbookAction(
                action="Enable adaptive rate limiting to match upstream recovery pace",
                priority=2,
                confidence=0.75,
                source="pattern_match",
                rationale="Throttle requests to match upstream's current capacity",
            ),
        ],
    ),
]


class PatternRecommendationEngine:
    """Generates context-aware runbook recommendations by matching anomaly
    patterns against known incident signatures."""

    def __init__(
        self,
        patterns: Optional[list[PatternSignature]] = None,
        min_confidence: float = 0.4,
    ) -> None:
        self.patterns = patterns or PATTERN_SIGNATURES
        self.min_confidence = min_confidence

    def recommend(
        self,
        anomaly: AnomalyEvent,
        correlation_report: Optional[CorrelationReport] = None,
    ) -> list[RunbookAction]:
        """Generate runbook recommendations for an anomaly.

        Matches the anomaly (and optional correlation context) against known
        pattern signatures and returns prioritized actions.

        Args:
            anomaly: The detected anomaly event.
            correlation_report: Optional correlation analysis results.

        Returns:
            A prioritized list of recommended runbook actions.
        """
        all_actions: list[RunbookAction] = []

        for pattern in self.patterns:
            if self._matches_pattern(anomaly, correlation_report, pattern):
                all_actions.extend(pattern.actions)
                logger.info(
                    "Matched pattern '%s' for anomaly %s",
                    pattern.name,
                    anomaly.anomaly_id,
                )

        # Add correlation-specific recommendations
        if correlation_report:
            correlation_actions = self._generate_correlation_actions(
                anomaly, correlation_report
            )
            all_actions.extend(correlation_actions)

        # Deduplicate and sort by priority
        seen_actions: set[str] = set()
        unique_actions: list[RunbookAction] = []
        for action in sorted(all_actions, key=lambda a: (a.priority, -a.confidence)):
            if action.action not in seen_actions and action.confidence >= self.min_confidence:
                seen_actions.add(action.action)
                unique_actions.append(action)

        # Update the anomaly with recommended action strings
        anomaly.recommended_actions = [a.action for a in unique_actions]

        logger.info(
            "Generated %d recommendations for anomaly %s",
            len(unique_actions),
            anomaly.anomaly_id,
        )
        return unique_actions

    def _matches_pattern(
        self,
        anomaly: AnomalyEvent,
        correlation_report: Optional[CorrelationReport],
        pattern: PatternSignature,
    ) -> bool:
        """Check if an anomaly matches a pattern signature."""
        # Check anomaly type
        if anomaly.anomaly_type not in pattern.anomaly_types:
            return False

        # Check severity minimum
        if _severity_rank(anomaly.severity) < _severity_rank(pattern.severity_minimum):
            return False

        # Check correlation requirements
        if pattern.requires_correlation:
            if correlation_report is None:
                return False
            if not correlation_report.correlations:
                return False
            if pattern.correlation_types:
                found_types = {c.correlation_type for c in correlation_report.correlations}
                if not found_types.intersection(pattern.correlation_types):
                    return False

        return True

    def _generate_correlation_actions(
        self,
        anomaly: AnomalyEvent,
        report: CorrelationReport,
    ) -> list[RunbookAction]:
        """Generate recommendations specific to the correlation findings."""
        actions: list[RunbookAction] = []

        for candidate in report.root_cause_candidates:
            actions.append(
                RunbookAction(
                    action=f"Prioritize investigation of {candidate} (likely root cause)",
                    priority=1,
                    confidence=0.85,
                    source="correlation",
                    rationale=(
                        f"{candidate} showed anomalous behavior preceding "
                        f"the current issue in {anomaly.service_name}"
                    ),
                    estimated_impact="Resolving root cause fixes downstream effects",
                )
            )

        for impacted_svc in report.impact_scope:
            actions.append(
                RunbookAction(
                    action=f"Monitor {impacted_svc} for cascading impact",
                    priority=5,
                    confidence=0.65,
                    source="correlation",
                    rationale=(
                        f"{impacted_svc} depends on {anomaly.service_name} and "
                        f"may show degradation"
                    ),
                )
            )

        return actions


def _severity_rank(severity: AnomalySeverity) -> int:
    return {
        AnomalySeverity.LOW: 0,
        AnomalySeverity.MEDIUM: 1,
        AnomalySeverity.HIGH: 2,
        AnomalySeverity.CRITICAL: 3,
    }.get(severity, 0)
