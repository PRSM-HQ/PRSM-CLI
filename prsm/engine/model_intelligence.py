"""Model intelligence — learned task→model rankings that persist across restarts.

A background research agent periodically evaluates which models excel at
which task categories and writes ranked lists to a JSON file under
~/.prsm/model_intelligence.json. When parents spawn children, the
ModelRegistry consults these rankings so the *best* model for a task
is selected first, with fallbacks if the top choice is unavailable.

The JSON file structure:
{
    "version": 2,
    "last_updated": "2026-02-13T12:00:00",
    "last_research_run": "2026-02-13T12:00:00",
    "rankings": {
        "coding": [
            {"model_id": "MiniMax-M2.5", "score": 0.95, "reason": "..."},
            {"model_id": "claude-opus-4-6", "score": 0.92, "reason": "..."},
            ...
        ],
        "exploration": [...],
        ...
    },
    "meta": {
        "models_evaluated": [...],
        "research_duration_seconds": 12.3
    }
}
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .model_registry import ModelCapability, ModelRegistry, TaskCategory

logger = logging.getLogger(__name__)

# Default storage location
DEFAULT_INTELLIGENCE_PATH = Path.home() / ".prsm" / "model_intelligence.json"

# How often the background researcher runs (24 hours)
RESEARCH_INTERVAL_SECONDS = 24 * 60 * 60

# Current file format version
FILE_VERSION = 2


class RankedModel:
    """A model's ranking for a specific task category."""

    __slots__ = ("model_id", "score", "reason")

    def __init__(self, model_id: str, score: float, reason: str = "") -> None:
        self.model_id = model_id
        self.score = score
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "score": self.score,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RankedModel:
        return cls(
            model_id=data["model_id"],
            score=float(data.get("score", 0.5)),
            reason=data.get("reason", ""),
        )


class ModelIntelligence:
    """Persistent, learned model rankings per task category.

    Stores ranked lists of models for each task category, persisted to
    JSON so they survive restarts. The rankings are updated by a
    background research agent that runs daily.

    Thread-safe for single-event-loop (asyncio) usage.
    """

    def __init__(
        self,
        path: Path | None = None,
    ) -> None:
        self._path = path or DEFAULT_INTELLIGENCE_PATH
        # task_category → list of RankedModel, sorted best-first
        self._rankings: dict[str, list[RankedModel]] = {}
        self._last_updated: datetime | None = None
        self._last_research_run: datetime | None = None
        self._meta: dict[str, Any] = {}
        self._dirty = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def last_updated(self) -> datetime | None:
        return self._last_updated

    @property
    def last_research_run(self) -> datetime | None:
        return self._last_research_run

    @property
    def has_rankings(self) -> bool:
        return bool(self._rankings)

    def needs_research(self) -> bool:
        """Whether the background researcher should run.

        Returns True if:
        - No rankings exist at all
        - The last research run was more than RESEARCH_INTERVAL_SECONDS ago
        """
        if not self._rankings:
            return True
        if self._last_research_run is None:
            return True
        elapsed = (
            datetime.now(timezone.utc) - self._last_research_run
        ).total_seconds()
        return elapsed >= RESEARCH_INTERVAL_SECONDS

    # ── Persistence ──────────────────────────────────────────────

    def load(self) -> bool:
        """Load rankings from the JSON file. Returns True if loaded successfully."""
        if not self._path.exists():
            logger.info("No model intelligence file at %s", self._path)
            return False

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "Failed to load model intelligence from %s: %s",
                self._path, e,
            )
            return False

        version = data.get("version", 1)
        if version < FILE_VERSION:
            logger.info(
                "Model intelligence file version %d < %d, will be refreshed",
                version, FILE_VERSION,
            )
            return False

        raw_rankings = data.get("rankings", {})
        self._rankings = {}
        for category, entries in raw_rankings.items():
            self._rankings[category] = [
                RankedModel.from_dict(e) for e in entries
            ]

        if data.get("last_updated"):
            try:
                self._last_updated = datetime.fromisoformat(
                    data["last_updated"]
                )
            except (ValueError, TypeError):
                pass

        if data.get("last_research_run"):
            try:
                self._last_research_run = datetime.fromisoformat(
                    data["last_research_run"]
                )
            except (ValueError, TypeError):
                pass

        self._meta = data.get("meta", {})
        self._dirty = False

        total_entries = sum(len(v) for v in self._rankings.values())
        logger.info(
            "Loaded model intelligence: %d categories, %d total entries, "
            "last_updated=%s",
            len(self._rankings),
            total_entries,
            self._last_updated,
        )
        return True

    def save(self) -> None:
        """Persist rankings to the JSON file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": FILE_VERSION,
            "last_updated": (
                self._last_updated.isoformat() if self._last_updated else None
            ),
            "last_research_run": (
                self._last_research_run.isoformat()
                if self._last_research_run else None
            ),
            "rankings": {
                cat: [rm.to_dict() for rm in models]
                for cat, models in self._rankings.items()
            },
            "meta": self._meta,
        }

        try:
            self._path.write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8",
            )
            self._dirty = False
            logger.info("Saved model intelligence to %s", self._path)
        except OSError as e:
            logger.error(
                "Failed to save model intelligence to %s: %s",
                self._path, e,
            )

    # ── Query API ────────────────────────────────────────────────

    def get_ranked_models(
        self,
        task_category: str,
    ) -> list[RankedModel]:
        """Get the ranked list of models for a task category, best first.

        Returns empty list if no rankings exist for the category.
        """
        return list(self._rankings.get(task_category, []))

    def get_best_model(
        self,
        task_category: str,
        available_model_ids: set[str] | None = None,
    ) -> RankedModel | None:
        """Get the best available model for a task category.

        Walks the ranked list top-to-bottom, returning the first model
        that is in the available set. If available_model_ids is None,
        returns the top-ranked model unconditionally.
        """
        ranked = self._rankings.get(task_category, [])
        for rm in ranked:
            if available_model_ids is None or rm.model_id in available_model_ids:
                return rm
        return None

    def get_all_categories(self) -> list[str]:
        """Return all categories that have rankings."""
        return list(self._rankings.keys())

    # ── Update API (called by the researcher) ────────────────────

    def update_rankings(
        self,
        rankings: dict[str, list[RankedModel]],
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Replace all rankings with new data from the researcher.

        Args:
            rankings: category → list of RankedModel, sorted best-first.
            meta: Optional metadata about the research run.
        """
        now = datetime.now(timezone.utc)
        self._rankings = rankings
        self._last_updated = now
        self._last_research_run = now
        self._meta = meta or {}
        self._dirty = True
        logger.info(
            "Updated model intelligence: %d categories, %d total entries",
            len(rankings),
            sum(len(v) for v in rankings.values()),
        )

    def to_summary(self) -> str:
        """Human-readable summary of current rankings."""
        if not self._rankings:
            return "No model intelligence data available."

        lines = [
            "Model Intelligence Rankings",
            f"(last updated: {self._last_updated or 'never'})",
            "",
        ]
        for category in sorted(self._rankings.keys()):
            ranked = self._rankings[category]
            lines.append(f"  {category}:")
            for i, rm in enumerate(ranked[:5], 1):  # Top 5
                lines.append(
                    f"    {i}. {rm.model_id} (score={rm.score:.2f})"
                )
            if len(ranked) > 5:
                lines.append(f"    ... and {len(ranked) - 5} more")
        return "\n".join(lines)


# ── Background Research Agent ────────────────────────────────────


def _build_research_prompt(
    model_ids: list[str],
    task_categories: list[str],
) -> str:
    """Build the prompt for the background research agent.

    The agent's job is to evaluate how well each model handles each
    task category and produce a JSON-structured ranking.
    """
    return f"""\
You are a model-evaluation research agent. Your job is to analyze the
current landscape of AI coding models and produce ranked lists showing
which models are best for each type of task.

You MUST evaluate models using recent public information from the web.
Before ranking, you MUST use web search to verify current capabilities,
release notes, and capability updates for each provider family, then
score models using that evidence.

## Models to Evaluate
{json.dumps(model_ids, indent=2)}

## Task Categories to Rank
{json.dumps(task_categories, indent=2)}

## Task Category Definitions
- **architecture**: System design, high-level planning, decomposing complex problems
- **complex-reasoning**: Multi-step logic, debugging root causes, analyzing edge cases
- **code-review**: Reviewing code for correctness, security, style, bugs
- **coding**: Writing/editing code, implementing features, refactoring
- **exploration**: Searching codebases, reading files, understanding implementations
- **simple-tasks**: Simple renames, typo fixes, formatting, trivial edits
- **documentation**: Writing docs, READMEs, docstrings, tutorials
- **agentic**: Multi-step agent workflows, tool coordination, orchestration
- **planning**: Task decomposition, strategy, work breakdown
- **tool-use**: Effective use of tools like bash, file editing, search
- **search**: Web search, code search, finding information
- **general**: General-purpose tasks that don't fit other categories

## Your Task
For each task category, rank ALL the models from best to worst based on
current web-verified capabilities. Give each a score from 0.0 to 1.0
and a brief reason.

If web research is unavailable, use conservative baseline rankings
and state that explicitly in your reasons.

Consider these factors:
- **Quality**: How good is the model's output for this task type?
- **Reliability**: Does it follow instructions well for this task type?
- **Efficiency**: Does it complete the task without unnecessary steps?
- **Cost-effectiveness**: Value for money at this task type

## Output Format
You MUST respond with ONLY a JSON object (no markdown, no explanation
outside the JSON). Use this exact structure:

```json
{{
  "rankings": {{
    "coding": [
      {{"model_id": "model-name", "score": 0.95, "reason": "Brief reason"}},
      ...
    ],
    "exploration": [...],
    ...
  }}
}}
```

Every category must have an entry. Every model must appear in every category.
Models should be sorted by score descending within each category.
Respond with ONLY the JSON — no other text.
"""


async def run_model_research(
    intelligence: ModelIntelligence,
    model_registry: ModelRegistry,
    provider_fn: Any = None,
) -> bool:
    """Run the background research to update model rankings.

    This function synthesizes rankings from:
    1. The hardcoded/YAML affinities already in the model registry
    2. Web research about current model capabilities (if a provider is available)
    3. Any observed performance data

    For the initial version, it uses the existing registry affinities
    combined with web research to produce comprehensive rankings.

    Args:
        intelligence: The ModelIntelligence instance to update.
        model_registry: Current model registry with capabilities.
        provider_fn: Optional async callable that performs research.
            Signature: async (prompt: str) -> str (JSON response).
            If None, uses registry affinities as baseline.

    Returns:
        True if rankings were updated, False on error.
    """
    from .model_registry import TaskCategory

    started = time.monotonic()
    all_models = model_registry.list_models()
    model_ids = [m.model_id for m in all_models]
    categories = [tc.value for tc in TaskCategory]

    logger.info(
        "Starting model intelligence research: %d models, %d categories",
        len(model_ids),
        len(categories),
    )

    # Start with existing registry affinities as baseline
    baseline_rankings: dict[str, list[RankedModel]] = {}
    for cat in categories:
        scored = []
        for model in all_models:
            score = model.score_for_task(cat)
            scored.append(RankedModel(
                model_id=model.model_id,
                score=score,
                reason=f"baseline affinity from registry (tier={model.tier.value})",
            ))
        scored.sort(key=lambda rm: -rm.score)
        baseline_rankings[cat] = scored

    # If we have a research provider, use it to get updated rankings
    research_rankings = None
    if provider_fn is not None:
        try:
            prompt = _build_research_prompt(model_ids, categories)
            logger.info("Sending research prompt to provider...")
            response = await provider_fn(prompt)

            # Parse the JSON response
            research_rankings = _parse_research_response(
                response, model_ids, categories,
            )
            if research_rankings:
                logger.info(
                    "Research provider returned rankings for %d categories",
                    len(research_rankings),
                )
        except Exception as e:
            logger.warning(
                "Research provider failed, using baseline: %s", e,
            )

    # Merge: research rankings override baseline where available
    final_rankings: dict[str, list[RankedModel]] = {}
    for cat in categories:
        if research_rankings and cat in research_rankings:
            # Use research rankings but ensure all models are present
            research_models = {
                rm.model_id for rm in research_rankings[cat]
            }
            merged = list(research_rankings[cat])
            # Add any models missing from research results
            for rm in baseline_rankings.get(cat, []):
                if rm.model_id not in research_models:
                    merged.append(rm)
            merged.sort(key=lambda rm: -rm.score)
            final_rankings[cat] = merged
        else:
            final_rankings[cat] = baseline_rankings.get(cat, [])

    elapsed = time.monotonic() - started
    meta = {
        "models_evaluated": model_ids,
        "categories_evaluated": categories,
        "research_duration_seconds": round(elapsed, 2),
        "used_provider": provider_fn is not None,
        "provider_succeeded": research_rankings is not None,
    }

    intelligence.update_rankings(final_rankings, meta)
    intelligence.save()

    logger.info(
        "Model intelligence research complete: %.1fs, %d categories, %d models",
        elapsed,
        len(final_rankings),
        len(model_ids),
    )
    return True


def _parse_research_response(
    response: str,
    valid_model_ids: list[str],
    valid_categories: list[str],
) -> dict[str, list[RankedModel]] | None:
    """Parse the JSON response from the research agent.

    Validates structure and filters to known models/categories.
    Returns None on parse failure.
    """
    # Strip markdown code fences if present
    text = response.strip()
    if text.startswith("```"):
        # Remove opening fence
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        # Remove closing fence
        if text.endswith("```"):
            text = text[:-3].rstrip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse research response as JSON: %s", e)
        return None

    raw_rankings = data.get("rankings", data)
    if not isinstance(raw_rankings, dict):
        logger.warning("Research response missing 'rankings' dict")
        return None

    valid_models_set = set(valid_model_ids)
    valid_cats_set = set(valid_categories)
    result: dict[str, list[RankedModel]] = {}

    for cat, entries in raw_rankings.items():
        if cat not in valid_cats_set:
            continue
        if not isinstance(entries, list):
            continue

        ranked = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            mid = entry.get("model_id", "")
            if mid not in valid_models_set:
                continue
            score = float(entry.get("score", 0.5))
            score = max(0.0, min(1.0, score))
            reason = str(entry.get("reason", ""))
            ranked.append(RankedModel(mid, score, reason))

        if ranked:
            ranked.sort(key=lambda rm: -rm.score)
            result[cat] = ranked

    return result if result else None


async def run_research_loop(
    intelligence: ModelIntelligence,
    model_registry: ModelRegistry,
    provider_fn: Any = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Background loop that runs model research periodically.

    Runs once on startup if rankings are stale/missing, then every
    RESEARCH_INTERVAL_SECONDS. Hidden from the user.

    Args:
        intelligence: The ModelIntelligence to update.
        model_registry: Current model registry.
        provider_fn: Optional research provider callable.
        stop_event: Set this event to stop the loop.
    """
    _stop = stop_event or asyncio.Event()

    while not _stop.is_set():
        if intelligence.needs_research():
            try:
                logger.info("Background model research starting...")
                await run_model_research(
                    intelligence, model_registry, provider_fn,
                )
            except Exception:
                logger.exception("Background model research failed")

        # Wait for next cycle or stop
        try:
            await asyncio.wait_for(
                _stop.wait(), timeout=RESEARCH_INTERVAL_SECONDS,
            )
            break  # stop_event was set
        except asyncio.TimeoutError:
            pass  # Time for next research cycle
