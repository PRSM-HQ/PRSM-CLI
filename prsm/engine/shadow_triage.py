"""Shadow-mode triage model for non-enforcing policy comparisons."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ShadowTriageModel:
    """Runs model-like triage in parallel with rules triage.

    This implementation intentionally keeps decisions non-enforcing.
    """

    def __init__(self, model: str = "claude-haiku"):
        self._model = model
        self._comparison_log: list[dict[str, Any]] = []

    async def shadow_evaluate(self, message: Any, subscriber_profile: dict) -> Any:
        """Run shadow triage and return a decision object.

        Decision is heuristic for now; promotion to real model inference can
        happen without changing router integration points.
        """
        from .message_router import TriageDecision

        topic = str(getattr(message, "topic", "") or "")
        urgency = str(getattr(message, "urgency", "normal"))
        threshold = str(subscriber_profile.get("urgency_threshold", "low"))

        if urgency == "high":
            decision = "deliver_now"
            reason = "shadow_high_urgency"
        elif topic.startswith("telemetry.") and urgency == "low":
            decision = "drop"
            reason = "shadow_telemetry_noise"
        elif threshold == "high" and urgency != "high":
            decision = "deliver_digest"
            reason = "shadow_threshold_digest"
        else:
            decision = "deliver_digest"
            reason = "shadow_default_digest"

        return TriageDecision(
            decision=decision,
            reason_code=reason,
            relevance_score=0.5,
            policy_snapshot_id=f"shadow:{self._model}",
        )

    def compare_with_rules(self, rules_decision: Any, model_decision: Any) -> dict:
        """Log comparison between rules and shadow model decisions."""
        rules_action = str(getattr(rules_decision, "decision", ""))
        model_action = str(getattr(model_decision, "decision", ""))
        agreement = rules_action == model_action

        false_positive = (model_action == "deliver_now") and (rules_action != "deliver_now")
        false_negative = (rules_action == "deliver_now") and (model_action != "deliver_now")

        comparison = {
            "timestamp": _utc_iso(),
            "rules_decision": rules_action,
            "model_decision": model_action,
            "agreement": agreement,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "model": self._model,
        }
        self._comparison_log.append(comparison)
        return comparison

    def get_comparison_stats(self) -> dict:
        """Return agreement/disagreement and false-positive/negative rates."""
        total = len(self._comparison_log)
        if total == 0:
            return {
                "total_comparisons": 0,
                "agreement_rate": 0.0,
                "disagreement_rate": 0.0,
                "false_positive_rate": 0.0,
                "false_negative_rate": 0.0,
            }
        agreements = sum(1 for item in self._comparison_log if item.get("agreement"))
        false_positives = sum(1 for item in self._comparison_log if item.get("false_positive"))
        false_negatives = sum(1 for item in self._comparison_log if item.get("false_negative"))
        return {
            "total_comparisons": total,
            "agreement_rate": agreements / total,
            "disagreement_rate": 1.0 - (agreements / total),
            "false_positive_rate": false_positives / total,
            "false_negative_rate": false_negatives / total,
        }
