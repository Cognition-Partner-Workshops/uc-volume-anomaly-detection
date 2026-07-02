"""Incident Insight Agent — consolidates findings into actionable summaries."""

import logging
from datetime import datetime
from uuid import uuid4

from src.agents.recommendation_engine import Recommendation
from src.models.anomaly import AnomalyEvent, AnomalyReport, AnomalySeverity
from src.models.service_health import ServiceHealthSnapshot

logger = logging.getLogger(__name__)


class IncidentInsightAgent:
    """Consolidates anomaly detections, health assessments, and recommendations
    into structured incident insight reports for support teams."""

    def generate_report(
        self,
        anomalies: list[AnomalyEvent],
        health_snapshots: dict[str, ServiceHealthSnapshot],
        recommendations: list[Recommendation],
        time_window_start: datetime,
        time_window_end: datetime,
    ) -> AnomalyReport:
        """Generate a consolidated incident insight report."""
        affected_services = list({a.service_name for a in anomalies})
        deduplicated_actions = list(
            dict.fromkeys(r.action for r in recommendations)
        )

        summary = self._build_summary(
            anomalies, affected_services, health_snapshots
        )

        report = AnomalyReport(
            report_id=f"rpt-{uuid4().hex[:8]}",
            generated_at=datetime.utcnow(),
            time_window_start=time_window_start,
            time_window_end=time_window_end,
            anomalies=anomalies,
            affected_services=affected_services,
            summary=summary,
            recommended_actions=deduplicated_actions,
        )

        logger.info(
            "Generated incident report %s: %d anomalies, %d services, %d actions",
            report.report_id,
            len(anomalies),
            len(affected_services),
            len(deduplicated_actions),
        )
        return report

    @staticmethod
    def _build_summary(
        anomalies: list[AnomalyEvent],
        affected_services: list[str],
        health_snapshots: dict[str, ServiceHealthSnapshot],
    ) -> str:
        """Build a human-readable summary of the incident."""
        critical = sum(
            1 for a in anomalies if a.severity == AnomalySeverity.CRITICAL
        )
        high = sum(
            1 for a in anomalies if a.severity == AnomalySeverity.HIGH
        )

        lines = [
            f"Detected {len(anomalies)} anomalies across {len(affected_services)} service(s).",
        ]

        if critical > 0:
            lines.append(f"  - {critical} CRITICAL severity anomaly/anomalies requiring immediate attention.")
        if high > 0:
            lines.append(f"  - {high} HIGH severity anomaly/anomalies.")

        for svc in affected_services:
            snapshot = health_snapshots.get(svc)
            if snapshot:
                lines.append(
                    f"  - {svc}: status={snapshot.status.value}, "
                    f"pods={snapshot.active_pods}/{snapshot.desired_pods}, "
                    f"error_rate={snapshot.error_rate:.3f}"
                )

        return "\n".join(lines)

    def format_for_notification(self, report: AnomalyReport) -> str:
        """Format a report for webhook/Slack/PagerDuty notification."""
        severity_emoji = ""
        if report.critical_count > 0:
            severity_emoji = "[CRITICAL]"
        elif report.high_count > 0:
            severity_emoji = "[HIGH]"

        header = (
            f"{severity_emoji} Anomaly Report {report.report_id}\n"
            f"Time: {report.time_window_start.isoformat()} - "
            f"{report.time_window_end.isoformat()}\n"
        )

        body = f"\n{report.summary}\n"

        if report.recommended_actions:
            actions = "\nRecommended Actions:\n"
            for i, action in enumerate(report.recommended_actions[:5], 1):
                actions += f"  {i}. {action}\n"
            body += actions

        return header + body
