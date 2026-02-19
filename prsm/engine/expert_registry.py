"""Pluggable expert registry.

Ships with zero built-in experts. Users register their own domain
experts via engine.register_expert() or registry.register().

Example:
    from claude_orchestrator import ExpertProfile

    engine.register_expert(ExpertProfile(
        expert_id="react-frontend",
        name="React Frontend Expert",
        description="React 18, JSX, inline styles",
        system_prompt="You are a React expert...",
        tools=["Read", "Write", "Edit", "Glob", "Grep"],
        model="claude-opus-4-6",
    ))
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .models import ExpertProfile
from .errors import ExpertNotFoundError

logger = logging.getLogger(__name__)


class ExpertRegistry:
    """Registry of available expert profiles.

    Fully pluggable — starts empty. Register domain experts
    for your project before running the engine.
    """

    def __init__(self, max_experts_per_project: int = 20) -> None:
        self._profiles: dict[str, ExpertProfile] = {}
        self._max_experts_per_project = max_experts_per_project
        self._metrics_db_path: Path | None = None

    def set_metrics_db_path(self, path: str | Path | None) -> None:
        """Set optional persistence location for expert performance metrics."""
        self._metrics_db_path = Path(path) if path else None
        self._load_metrics()

    def get(self, expert_id: str) -> ExpertProfile:
        """Get an expert profile by ID.

        Raises ExpertNotFoundError if not found.
        """
        profile = self._profiles.get(expert_id)
        if profile is None:
            raise ExpertNotFoundError(expert_id)
        return profile

    def register(self, profile: ExpertProfile) -> None:
        """Register a new expert profile (or overwrite existing)."""
        if (
            profile.expert_id not in self._profiles
            and self.count >= self._max_experts_per_project
        ):
            raise ValueError(
                f"Maximum experts reached ({self._max_experts_per_project})"
            )
        if profile.created_at is None:
            profile.created_at = datetime.now(timezone.utc)
        self._profiles[profile.expert_id] = profile
        logger.info(
            "Expert registered: %s (%s)",
            profile.expert_id,
            profile.name,
        )
        self._persist_metrics()

    def unregister(self, expert_id: str) -> None:
        """Remove an expert profile."""
        self._profiles.pop(expert_id, None)
        self._persist_metrics()

    def propose(self, profile: ExpertProfile) -> str:
        """Submit expert proposal for approval."""
        profile.lifecycle_state = "proposed"
        if profile.created_at is None:
            profile.created_at = datetime.now(timezone.utc)
        self.register(profile)
        return profile.expert_id

    def approve_proposal(self, expert_id: str) -> None:
        """Approve a proposed expert and activate it."""
        profile = self.get(expert_id)
        if profile.lifecycle_state == "proposed":
            profile.lifecycle_state = "active"
            self._persist_metrics()

    def deprecate(self, expert_id: str, reason: str = "") -> None:
        """Mark expert as deprecated (still available)."""
        profile = self.get(expert_id)
        profile.lifecycle_state = "deprecated"
        profile.deprecated_at = datetime.now(timezone.utc)
        profile.deprecation_reason = reason
        self._persist_metrics()

    def archive(self, expert_id: str) -> None:
        """Fully retire expert from active consultation."""
        profile = self.get(expert_id)
        profile.lifecycle_state = "archived"
        self._persist_metrics()

    def record_consultation(
        self,
        expert_id: str,
        success: bool,
        duration: float,
        confidence: float,
    ) -> None:
        """Track expert consultation outcomes and refresh utility score."""
        profile = self.get(expert_id)
        profile.consultation_count += 1
        if success:
            profile.success_count += 1
        else:
            profile.failure_count += 1

        count = float(profile.consultation_count)
        previous_count = max(count - 1.0, 0.0)
        profile.avg_duration_seconds = (
            ((profile.avg_duration_seconds * previous_count) + max(duration, 0.0))
            / count
        )
        bounded_conf = min(max(confidence, 0.0), 1.0)
        profile.avg_confidence = (
            ((profile.avg_confidence * previous_count) + bounded_conf) / count
        )

        success_rate = profile.success_count / count
        usage_factor = count / (count + 10.0)
        profile.utility_score = success_rate * usage_factor
        self._persist_metrics()

    def get_utility_rankings(self) -> list[tuple[str, float]]:
        """Return expert IDs ranked by utility score descending."""
        rankings = [
            (profile.expert_id, profile.utility_score)
            for profile in self._profiles.values()
        ]
        rankings.sort(key=lambda item: item[1], reverse=True)
        return rankings

    def get_low_utility_experts(self, threshold: float = 0.3) -> list[ExpertProfile]:
        """Return non-archived experts below utility threshold."""
        return [
            profile
            for profile in self._profiles.values()
            if profile.lifecycle_state != "archived"
            and profile.utility_score < threshold
        ]

    def _persist_metrics(self) -> None:
        """Persist expert lifecycle + metrics to JSON if path is configured."""
        if self._metrics_db_path is None:
            return
        data = {
            "experts": {
                expert_id: {
                    "lifecycle_state": profile.lifecycle_state,
                    "created_at": (
                        profile.created_at.isoformat()
                        if profile.created_at else None
                    ),
                    "deprecated_at": (
                        profile.deprecated_at.isoformat()
                        if profile.deprecated_at else None
                    ),
                    "deprecation_reason": profile.deprecation_reason,
                    "evaluation_criteria": profile.evaluation_criteria,
                    "deprecation_policy": profile.deprecation_policy,
                    "consultation_count": profile.consultation_count,
                    "success_count": profile.success_count,
                    "failure_count": profile.failure_count,
                    "avg_duration_seconds": profile.avg_duration_seconds,
                    "avg_confidence": profile.avg_confidence,
                    "utility_score": profile.utility_score,
                }
                for expert_id, profile in self._profiles.items()
            }
        }
        self._metrics_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._metrics_db_path.write_text(json.dumps(data), encoding="utf-8")

    def _load_metrics(self) -> None:
        """Load persisted expert lifecycle + metric metadata."""
        if self._metrics_db_path is None or not self._metrics_db_path.exists():
            return
        try:
            raw = json.loads(self._metrics_db_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to load expert metrics from %s", self._metrics_db_path)
            return
        experts_raw = raw.get("experts", {})
        for expert_id, data in experts_raw.items():
            profile = self._profiles.get(expert_id)
            if profile is None:
                continue
            profile.lifecycle_state = data.get("lifecycle_state", profile.lifecycle_state)
            created_at = data.get("created_at")
            deprecated_at = data.get("deprecated_at")
            profile.created_at = (
                datetime.fromisoformat(created_at) if created_at else profile.created_at
            )
            profile.deprecated_at = (
                datetime.fromisoformat(deprecated_at) if deprecated_at else profile.deprecated_at
            )
            profile.deprecation_reason = data.get("deprecation_reason", profile.deprecation_reason)
            profile.evaluation_criteria = list(data.get("evaluation_criteria", profile.evaluation_criteria))
            profile.deprecation_policy = data.get("deprecation_policy", profile.deprecation_policy)
            profile.consultation_count = int(data.get("consultation_count", profile.consultation_count))
            profile.success_count = int(data.get("success_count", profile.success_count))
            profile.failure_count = int(data.get("failure_count", profile.failure_count))
            profile.avg_duration_seconds = float(data.get("avg_duration_seconds", profile.avg_duration_seconds))
            profile.avg_confidence = float(data.get("avg_confidence", profile.avg_confidence))
            profile.utility_score = float(data.get("utility_score", profile.utility_score))

    def list_profiles(self) -> list[ExpertProfile]:
        """Return all registered profiles."""
        return list(self._profiles.values())

    def list_ids(self) -> list[str]:
        """Return all registered profile IDs."""
        return list(self._profiles.keys())

    @property
    def count(self) -> int:
        return len(self._profiles)
