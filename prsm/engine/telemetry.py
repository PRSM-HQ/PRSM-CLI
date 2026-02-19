"""Runtime telemetry collection and export helpers."""
from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat()


def _parse_time_range(time_range: str) -> timedelta:
    match = re.match(r"^\s*(\d+)\s*([mhd])\s*$", time_range)
    if not match:
        return timedelta(hours=24)
    quantity = int(match.group(1))
    unit = match.group(2)
    if unit == "m":
        return timedelta(minutes=quantity)
    if unit == "h":
        return timedelta(hours=quantity)
    return timedelta(days=quantity)


@dataclass
class _MetricRow:
    timestamp: str
    metric_type: str
    value: float
    tags_json: str


class TelemetryCollector:
    """Collects and exports runtime metrics."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    tags_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_type_ts ON metrics(metric_type, timestamp)"
            )
            conn.commit()

    def record_metric(
        self,
        metric_type: str,
        value: float,
        tags: dict | None = None,
    ) -> None:
        """Record a single metric data point."""
        payload = json.dumps(tags or {}, sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO metrics(timestamp, metric_type, value, tags_json)
                VALUES (?, ?, ?, ?)
                """,
                (_iso_utc(_utc_now()), metric_type, float(value), payload),
            )
            conn.commit()

    def _sum_metric_since(self, conn: sqlite3.Connection, metric_type: str, cutoff: str) -> float:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(value), 0.0) AS total
            FROM metrics
            WHERE metric_type = ? AND timestamp >= ?
            """,
            (metric_type, cutoff),
        ).fetchone()
        return float(row["total"]) if row is not None else 0.0

    def _avg_metric_since(self, conn: sqlite3.Connection, metric_type: str, cutoff: str) -> float:
        row = conn.execute(
            """
            SELECT AVG(value) AS avg_value
            FROM metrics
            WHERE metric_type = ? AND timestamp >= ?
            """,
            (metric_type, cutoff),
        ).fetchone()
        if row is None or row["avg_value"] is None:
            return 0.0
        return float(row["avg_value"])

    def _metric_rows_since(
        self,
        conn: sqlite3.Connection,
        metric_type: str,
        cutoff: str,
    ) -> list[_MetricRow]:
        rows = conn.execute(
            """
            SELECT timestamp, metric_type, value, tags_json
            FROM metrics
            WHERE metric_type = ? AND timestamp >= ?
            ORDER BY timestamp ASC
            """,
            (metric_type, cutoff),
        ).fetchall()
        return [
            _MetricRow(
                timestamp=str(row["timestamp"]),
                metric_type=str(row["metric_type"]),
                value=float(row["value"]),
                tags_json=str(row["tags_json"]),
            )
            for row in rows
        ]

    def get_dashboard_data(self, time_range: str = "24h") -> dict:
        """Return aggregated metrics for dashboard display."""
        cutoff_iso = _iso_utc(_utc_now() - _parse_time_range(time_range))
        with self._connect() as conn:
            agent_failure_rate = self._avg_metric_since(conn, "agent_failure", cutoff_iso)
            avg_latency_seconds = self._avg_metric_since(conn, "latency_seconds", cutoff_iso)
            total_cost_estimate = self._sum_metric_since(conn, "cost_estimate", cutoff_iso)
            policy_deny_count = int(self._sum_metric_since(conn, "policy_deny_count", cutoff_iso))

            triage_comparisons = self._sum_metric_since(conn, "triage_comparison", cutoff_iso)
            false_positives = self._sum_metric_since(conn, "triage_false_positive", cutoff_iso)
            false_negatives = self._sum_metric_since(conn, "triage_false_negative", cutoff_iso)
            divisor = triage_comparisons if triage_comparisons > 0 else 1.0

            tool_usage = {}
            for row in self._metric_rows_since(conn, "tool_usage", cutoff_iso):
                tags = json.loads(row.tags_json or "{}")
                tool_name = str(tags.get("tool_name", "unknown"))
                tool_usage[tool_name] = tool_usage.get(tool_name, 0.0) + row.value
            top_tools = sorted(
                (
                    {"tool_name": tool_name, "count": int(count)}
                    for tool_name, count in tool_usage.items()
                ),
                key=lambda item: item["count"],
                reverse=True,
            )

            expert_utilization: dict[str, float] = {}
            for row in self._metric_rows_since(conn, "expert_utilization", cutoff_iso):
                tags = json.loads(row.tags_json or "{}")
                expert_id = str(tags.get("expert_id", "unknown"))
                expert_utilization[expert_id] = (
                    expert_utilization.get(expert_id, 0.0) + row.value
                )

        return {
            "agent_failure_rate": agent_failure_rate,
            "avg_latency_seconds": avg_latency_seconds,
            "total_cost_estimate": total_cost_estimate,
            "policy_deny_count": policy_deny_count,
            "triage_false_positive_rate": false_positives / divisor,
            "triage_false_negative_rate": false_negatives / divisor,
            "top_tools_by_usage": top_tools,
            "expert_utilization": expert_utilization,
        }

    def export_csv(self, metric_type: str, output_path: Path) -> None:
        """Export metric data as CSV for external analysis."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, metric_type, value, tags_json
                FROM metrics
                WHERE metric_type = ?
                ORDER BY timestamp ASC
                """,
                (metric_type,),
            ).fetchall()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "metric_type", "value", "tags_json"])
            for row in rows:
                writer.writerow(
                    [
                        row["timestamp"],
                        row["metric_type"],
                        row["value"],
                        row["tags_json"],
                    ]
                )
