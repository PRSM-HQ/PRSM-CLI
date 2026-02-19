"""Phase 7 tests — telemetry collector."""

from __future__ import annotations

import csv
import sqlite3

from prsm.engine.telemetry import TelemetryCollector


def test_record_metric_persists_row_and_tags(tmp_path):
    db_path = tmp_path / "telemetry.sqlite3"
    collector = TelemetryCollector(db_path)
    collector.record_metric("tool_usage", 1.0, {"tool_name": "Bash"})

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT metric_type, value, tags_json FROM metrics ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert row[0] == "tool_usage"
    assert float(row[1]) == 1.0
    assert "Bash" in row[2]


def test_get_dashboard_data_returns_expected_keys_and_aggregates(tmp_path):
    collector = TelemetryCollector(tmp_path / "telemetry.sqlite3")
    collector.record_metric("agent_failure", 1.0)
    collector.record_metric("agent_failure", 0.0)
    collector.record_metric("latency_seconds", 2.0)
    collector.record_metric("latency_seconds", 4.0)
    collector.record_metric("cost_estimate", 1.5)
    collector.record_metric("cost_estimate", 2.0)
    collector.record_metric("policy_deny_count", 1.0)
    collector.record_metric("policy_deny_count", 1.0)
    collector.record_metric("triage_comparison", 1.0)
    collector.record_metric("triage_comparison", 1.0)
    collector.record_metric("triage_false_positive", 1.0)
    collector.record_metric("tool_usage", 1.0, {"tool_name": "Bash"})
    collector.record_metric("tool_usage", 1.0, {"tool_name": "Bash"})
    collector.record_metric("tool_usage", 1.0, {"tool_name": "Read"})
    collector.record_metric("expert_utilization", 2.0, {"expert_id": "db"})
    collector.record_metric("expert_utilization", 1.0, {"expert_id": "db"})

    dashboard = collector.get_dashboard_data("24h")
    assert set(dashboard) == {
        "agent_failure_rate",
        "avg_latency_seconds",
        "total_cost_estimate",
        "policy_deny_count",
        "triage_false_positive_rate",
        "triage_false_negative_rate",
        "top_tools_by_usage",
        "expert_utilization",
    }
    assert dashboard["agent_failure_rate"] == 0.5
    assert dashboard["avg_latency_seconds"] == 3.0
    assert dashboard["total_cost_estimate"] == 3.5
    assert dashboard["policy_deny_count"] == 2
    assert dashboard["triage_false_positive_rate"] == 0.5
    assert dashboard["triage_false_negative_rate"] == 0.0
    assert dashboard["top_tools_by_usage"][0]["tool_name"] == "Bash"
    assert dashboard["top_tools_by_usage"][0]["count"] == 2
    assert dashboard["expert_utilization"]["db"] == 3.0


def test_export_csv_writes_metric_rows_with_headers(tmp_path):
    collector = TelemetryCollector(tmp_path / "telemetry.sqlite3")
    collector.record_metric("latency_seconds", 2.5, {"route": "publish"})
    collector.record_metric("latency_seconds", 3.0, {"route": "receive"})

    out_path = tmp_path / "latency.csv"
    collector.export_csv("latency_seconds", out_path)

    with out_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == ["timestamp", "metric_type", "value", "tags_json"]
    assert len(rows) == 3
    assert rows[1][1] == "latency_seconds"
    assert rows[2][1] == "latency_seconds"
