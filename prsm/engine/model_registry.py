"""Model capability registry — maps task types to optimal models.

Enables intelligent model selection: the orchestrator can automatically
pick the best model for each subtask based on complexity, task type,
and cost efficiency. Simple tasks get smaller/cheaper models, complex
tasks get frontier models.

The registry is populated from YAML config and can be extended at
runtime. Each model has a "tier" (frontier, strong, fast, economy)
and a set of task affinities with strength scores.

Example YAML:
    model_registry:
      claude-opus-4-6:
        tier: frontier
        affinities:
          architecture: 0.95
          complex-reasoning: 0.95
          code-review: 0.90
          planning: 0.95
      claude-sonnet-4-5-20250929:
        tier: strong
        affinities:
          coding: 0.90
          code-review: 0.85
          general: 0.85
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Performance/cost tier for a model."""
    FRONTIER = "frontier"  # Best quality, highest cost (opus, gpt-5.2)
    STRONG = "strong"      # Near-frontier, good value (sonnet-class models)
    FAST = "fast"          # Fast and cheap (flash-class models)
    ECONOMY = "economy"    # Cheapest option for trivial tasks


class TaskCategory(str, Enum):
    """Categories of tasks that models can excel at."""
    ARCHITECTURE = "architecture"       # System design, planning
    COMPLEX_REASONING = "complex-reasoning"  # Multi-step logic
    CODING = "coding"                   # Code writing/generation
    CODE_REVIEW = "code-review"         # Code review and analysis
    PLANNING = "planning"               # Task decomposition, strategy
    TOOL_USE = "tool-use"               # Agentic tool calling
    EXPLORATION = "exploration"         # Codebase search/reading
    SIMPLE_TASKS = "simple-tasks"       # Simple edits, renames
    GENERAL = "general"                 # General-purpose
    AGENTIC = "agentic"                # Multi-step agent workflows
    SEARCH = "search"                  # Web/code search tasks
    DOCUMENTATION = "documentation"     # Writing docs/summaries


@dataclass
class ModelCapability:
    """Describes a model's capabilities and best-use scenarios."""
    model_id: str
    provider: str = "claude"
    tier: ModelTier = ModelTier.STRONG
    affinities: dict[str, float] = field(default_factory=dict)
    # Cost multiplier relative to baseline (1.0 = baseline)
    cost_factor: float = 1.0
    # Speed multiplier relative to baseline (1.0 = baseline)
    speed_factor: float = 1.0
    # Maximum context window tokens
    max_context: int = 200_000
    # Whether this model is currently available
    available: bool = True

    def score_for_task(self, task_category: str) -> float:
        """Return this model's affinity score for a task category.

        Returns 0.5 (neutral) if no specific affinity is registered.
        """
        return self.affinities.get(task_category, 0.5)


class ModelRegistry:
    """Registry of model capabilities for intelligent selection.

    The orchestrator queries this registry to pick the optimal model
    for each child agent based on the task description and required
    capabilities.

    When a ModelIntelligence instance is attached, the registry uses
    learned rankings (from background research) as the primary scoring
    source, falling back to static affinities when no learned data exists.

    **Model aliases** allow short names like ``"claude-sonnet"`` to resolve
    to the latest versioned ID (``"claude-sonnet-4-5-20250929"``).  Aliases
    are populated automatically from built-in defaults, YAML ``models:``
    entries, and explicit ``register_alias()`` calls.  All ``get()`` and
    lookup paths resolve aliases transparently.
    """

    # ── Built-in family aliases ──────────────────────────────────
    # Maps short family names to their latest full model IDs.
    # Updated when new model versions are released.
    CLAUDE_FAMILY_ALIASES: dict[str, str] = {
        # Canonical short names
        "claude-opus": "claude-opus-4-6",
        "claude-sonnet": "claude-sonnet-4-5-20250929",
        "claude-haiku": "claude-3-5-haiku-20241022",
        # Even shorter convenience names
        "opus": "claude-opus-4-6",
        "sonnet": "claude-sonnet-4-5-20250929",
        "haiku": "claude-3-5-haiku-20241022",
    }

    # Non-Claude provider aliases — short name → full model ID.
    PROVIDER_ALIASES: dict[str, str] = {
        "gemini-3": "gemini-3-pro-preview",
        "gemini-3-flash": "gemini-3-flash-preview",
        "gemini-flash": "gemini-2.5-flash",
        "codex": "gpt-5.2-codex",
        "gpt-5-3-spark": "gpt-5.3-spark",
        "spark": "gpt-5.3-spark",
    }

    def __init__(self) -> None:
        self._models: dict[str, ModelCapability] = {}
        self._aliases: dict[str, str] = {}  # alias → canonical model_id
        self._intelligence: object | None = None  # ModelIntelligence
        # Pre-populate with built-in family aliases
        self._aliases.update(self.CLAUDE_FAMILY_ALIASES)
        self._aliases.update(self.PROVIDER_ALIASES)

    def set_intelligence(self, intelligence: object) -> None:
        """Attach a ModelIntelligence instance for learned rankings.

        When set, best_for_task() will consult learned rankings before
        falling back to static affinities.
        """
        self._intelligence = intelligence
        logger.info("Model intelligence attached to registry")

    # ── Alias management ────────────────────────────────────────

    def register_alias(self, alias: str, model_id: str) -> None:
        """Register a short alias that maps to a full model_id.

        Example::

            registry.register_alias("claude-sonnet", "claude-sonnet-4-5-20250929")
        """
        self._aliases[alias] = model_id
        logger.debug("Model alias registered: %s → %s", alias, model_id)

    def resolve_alias(self, name: str) -> str:
        """Resolve a model name, which may be an alias, to a canonical model_id.

        If *name* is already a registered model_id it is returned as-is.
        If *name* is a known alias the target model_id is returned.
        Otherwise *name* is returned unchanged (assumed to be a raw model_id
        that hasn't been registered yet).
        """
        # Direct match takes priority
        if name in self._models:
            return name
        # Try alias lookup
        return self._aliases.get(name, name)

    def resolve_alias_with_provider(
        self, name: str
    ) -> tuple[str, str | None]:
        """Resolve a model name and also return its provider.

        Returns ``(model_id, provider)`` where provider may be ``None``
        if the resolved model_id is not in the registry.
        """
        model_id = self.resolve_alias(name)
        # Prefer exact lookup first so runtime-encoded variants like
        # ``gpt-5-3::reasoning_effort=high`` can carry distinct metadata.
        cap = self._models.get(model_id)
        if cap is None:
            normalized_model_id = self._strip_runtime_model_options(model_id)
            cap = self._models.get(normalized_model_id)
        return (model_id, cap.provider if cap else None)

    def list_aliases(self) -> dict[str, str]:
        """Return a copy of the alias → model_id mapping."""
        return dict(self._aliases)

    # ── Model registration & lookup ───────────────────────────

    def register(self, capability: ModelCapability) -> None:
        """Register a model's capabilities."""
        self._models[capability.model_id] = capability
        logger.info(
            "Model registered: %s (tier=%s, provider=%s, affinities=%d)",
            capability.model_id,
            capability.tier.value,
            capability.provider,
            len(capability.affinities),
        )

    def get(self, model_id: str) -> ModelCapability | None:
        """Get a model's capability profile.

        Resolves aliases transparently — ``get("claude-sonnet")`` works
        the same as ``get("claude-sonnet-4-5-20250929")``.
        """
        resolved = self.resolve_alias(model_id)
        # Preserve exact matches first so registry entries with encoded
        # runtime options (e.g. reasoning effort) resolve correctly.
        cap = self._models.get(resolved)
        if cap is not None:
            return cap
        normalized = self._strip_runtime_model_options(resolved)
        return self._models.get(normalized)

    @staticmethod
    def _strip_runtime_model_options(model_id: str) -> str:
        """Strip encoded runtime options from a model identifier.

        Example:
            gpt-5.2-codex::reasoning_effort=high -> gpt-5.2-codex
        """
        marker = "::reasoning_effort="
        if marker in model_id:
            return model_id.split(marker, 1)[0]
        return model_id

    def list_models(self) -> list[ModelCapability]:
        """Return all registered models."""
        return list(self._models.values())

    def list_by_tier(self, tier: ModelTier) -> list[ModelCapability]:
        """Return models in a specific tier."""
        return [m for m in self._models.values() if m.tier == tier]

    def best_for_task(
        self,
        task_category: str,
        *,
        max_tier: ModelTier | None = None,
        min_tier: ModelTier | None = None,
        preferred_provider: str | None = None,
        available_only: bool = True,
    ) -> ModelCapability | None:
        """Find the best model for a specific task category.

        Args:
            task_category: The type of task (e.g., "coding", "exploration")
            max_tier: Maximum tier (cost ceiling). If FAST, won't use FRONTIER.
            min_tier: Minimum tier (quality floor). If STRONG, won't use FAST.
            preferred_provider: Prefer models from this provider.
            available_only: Only consider available models.

        Returns:
            The best-matching ModelCapability, or None if no models registered.
        """
        tier_order = [
            ModelTier.FRONTIER,
            ModelTier.STRONG,
            ModelTier.FAST,
            ModelTier.ECONOMY,
        ]

        candidates = list(self._models.values())

        if available_only:
            candidates = [m for m in candidates if m.available]

        if max_tier:
            max_idx = tier_order.index(max_tier)
            candidates = [
                m for m in candidates
                if tier_order.index(m.tier) >= max_idx
            ]

        if min_tier:
            min_idx = tier_order.index(min_tier)
            candidates = [
                m for m in candidates
                if tier_order.index(m.tier) <= min_idx
            ]

        if not candidates:
            return None

        # Build a set of candidate model IDs for fast lookup
        candidate_ids = {m.model_id for m in candidates}

        # Try learned rankings first (from ModelIntelligence)
        learned_score_map: dict[str, float] = {}
        if self._intelligence is not None:
            try:
                ranked = self._intelligence.get_ranked_models(task_category)
                if ranked:
                    for rm in ranked:
                        if rm.model_id in candidate_ids:
                            learned_score_map[rm.model_id] = rm.score
            except Exception:
                pass  # Fall back to static affinities

        # Score each candidate
        scored = []
        for model in candidates:
            # Prefer learned score; fall back to static affinity
            if model.model_id in learned_score_map:
                score = learned_score_map[model.model_id]
            else:
                score = model.score_for_task(task_category)

            # Bonus for preferred provider (small tiebreaker)
            if preferred_provider and model.provider == preferred_provider:
                score += 0.05

            scored.append((score, model))

        # Sort by score descending, then by cost ascending for ties
        scored.sort(key=lambda x: (-x[0], x[1].cost_factor))

        return scored[0][1] if scored else None

    def get_ranked_for_task(
        self,
        task_category: str,
        *,
        available_only: bool = True,
    ) -> list[tuple[float, ModelCapability]]:
        """Get ALL models ranked for a task category, best first.

        Returns a list of (score, ModelCapability) tuples. Uses learned
        rankings when available, otherwise static affinities. This gives
        callers visibility into the full ranked list so they can fall
        back to the next-best model if the top choice is unavailable.
        """
        candidates = list(self._models.values())
        if available_only:
            candidates = [m for m in candidates if m.available]

        candidate_ids = {m.model_id for m in candidates}

        # Try learned rankings
        learned_score_map: dict[str, float] = {}
        if self._intelligence is not None:
            try:
                ranked = self._intelligence.get_ranked_models(task_category)
                if ranked:
                    for rm in ranked:
                        if rm.model_id in candidate_ids:
                            learned_score_map[rm.model_id] = rm.score
            except Exception:
                pass

        scored = []
        for model in candidates:
            if model.model_id in learned_score_map:
                score = learned_score_map[model.model_id]
            else:
                score = model.score_for_task(task_category)
            scored.append((score, model))

        scored.sort(key=lambda x: (-x[0], x[1].cost_factor))
        return scored

    def recommend_model(
        self,
        task_description: str,
        *,
        complexity: str = "medium",
        preferred_provider: str | None = None,
    ) -> ModelCapability | None:
        """Recommend a model based on task description and complexity.

        This is the primary API for intelligent model selection. It
        analyzes the task description to infer the task category, then
        uses complexity to set tier constraints.

        Args:
            task_description: Natural language description of the task.
            complexity: "trivial", "simple", "medium", "complex", or "frontier"
            preferred_provider: Prefer models from this provider.

        Returns:
            Recommended ModelCapability, or None if registry is empty.
        """
        # Infer task category from description
        category = self._infer_category(task_description)

        # Map complexity to tier constraints
        tier_map = {
            "trivial": (ModelTier.ECONOMY, ModelTier.FAST),   # max=FAST
            "simple": (ModelTier.FAST, ModelTier.STRONG),     # max=STRONG
            "medium": (ModelTier.STRONG, ModelTier.FRONTIER),  # no constraint
            "complex": (ModelTier.STRONG, ModelTier.FRONTIER), # min=STRONG
            "frontier": (ModelTier.FRONTIER, ModelTier.FRONTIER),  # FRONTIER only
        }
        min_tier, max_tier = tier_map.get(
            complexity, (ModelTier.STRONG, ModelTier.FRONTIER)
        )

        return self.best_for_task(
            category,
            max_tier=max_tier,
            min_tier=min_tier,
            preferred_provider=preferred_provider,
        )

    def _infer_category(self, description: str) -> str:
        """Infer task category from a natural language description.

        Uses keyword matching — intentionally simple and fast.
        The master agent can also explicitly specify the category.
        """
        desc = description.lower()

        # Check patterns in priority order
        patterns: list[tuple[list[str], str]] = [
            # Architecture / planning
            (
                ["architect", "design", "plan", "strategy", "decompos",
                 "break down", "organize", "structure"],
                TaskCategory.ARCHITECTURE.value,
            ),
            # Complex reasoning
            (
                ["complex", "reason", "analyz", "debug", "investigat",
                 "root cause", "diagnos", "figure out"],
                TaskCategory.COMPLEX_REASONING.value,
            ),
            # Code review
            (
                ["review", "check", "audit", "inspect", "correctness",
                 "security", "bug", "vulnerability"],
                TaskCategory.CODE_REVIEW.value,
            ),
            # Coding
            (
                ["implement", "write", "code", "build", "create",
                 "add feature", "function", "class", "module",
                 "refactor", "fix", "patch", "edit"],
                TaskCategory.CODING.value,
            ),
            # Exploration
            (
                ["explore", "search", "find", "grep", "read",
                 "understand", "describe", "summarize", "list",
                 "what is", "how does"],
                TaskCategory.EXPLORATION.value,
            ),
            # Simple tasks
            (
                ["rename", "typo", "comment", "format", "move",
                 "copy", "delete", "remove", "simple", "trivial",
                 "quick", "small"],
                TaskCategory.SIMPLE_TASKS.value,
            ),
            # Documentation
            (
                ["document", "readme", "docstring", "explain",
                 "tutorial", "guide"],
                TaskCategory.DOCUMENTATION.value,
            ),
            # Agentic
            (
                ["agent", "orchestrat", "coordinat", "multi-step",
                 "workflow", "pipeline"],
                TaskCategory.AGENTIC.value,
            ),
        ]

        for keywords, category in patterns:
            if any(kw in desc for kw in keywords):
                return category

        return TaskCategory.GENERAL.value

    def sync_availability(
        self,
        provider_availability: dict[str, bool],
    ) -> dict[str, bool]:
        """Sync model availability with actual provider availability.

        Marks models as unavailable if their provider is not available
        (CLI not installed, etc.). Returns a dict mapping model_id →
        available for all models whose status changed.

        Args:
            provider_availability: Dict mapping provider name → available.
                Typically from ProviderRegistry.get_availability_report().

        Returns:
            Dict of model_id → new available status for models that changed.
        """
        changed: dict[str, bool] = {}
        for model in self._models.values():
            provider_ok = provider_availability.get(
                model.provider, False
            )
            if model.available != provider_ok:
                model.available = provider_ok
                changed[model.model_id] = provider_ok
                if not provider_ok:
                    logger.warning(
                        "Model '%s' marked unavailable "
                        "(provider '%s' not installed)",
                        model.model_id,
                        model.provider,
                    )
                else:
                    logger.info(
                        "Model '%s' marked available "
                        "(provider '%s' found)",
                        model.model_id,
                        model.provider,
                    )
        return changed

    async def probe_claude_models(self) -> dict[str, bool]:
        """Probe which Claude models are actually accessible.

        Runs `claude -p "hi" --model <model>` for each Claude model
        to check if the user's plan/account supports it. Models that
        fail are marked unavailable.

        This is async because it runs subprocesses.

        Returns:
            Dict mapping model_id → available for models that changed.
        """
        import asyncio
        import os
        import shutil

        claude_models = [
            m for m in self._models.values()
            if m.provider == "claude" and m.available
        ]
        if not claude_models:
            return {}

        # Ensure CLAUDECODE doesn't block nested invocations
        probe_env = os.environ.copy()
        probe_env.pop("CLAUDECODE", None)

        changed: dict[str, bool] = {}

        async def _probe(model: ModelCapability) -> None:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "claude", "-p", "hi", "--model", model.model_id,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=probe_env,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=30
                )
                if proc.returncode != 0:
                    # Claude CLI may output error to stdout or stderr
                    combined = (
                        stdout.decode("utf-8", errors="replace")
                        + stderr.decode("utf-8", errors="replace")
                    )
                    if ("not exist" in combined
                            or "not have access" in combined
                            or "issue with the selected model" in combined):
                        model.available = False
                        changed[model.model_id] = False
                        logger.warning(
                            "Model '%s' probed UNAVAILABLE: %s",
                            model.model_id,
                            combined.strip()[:120],
                        )
            except (asyncio.TimeoutError, FileNotFoundError, Exception) as exc:
                logger.debug(
                    "Model probe failed for '%s': %s",
                    model.model_id, exc,
                )

        # Probe all Claude models concurrently (fast: "hi" is a trivial prompt)
        if shutil.which("claude"):
            await asyncio.gather(*[_probe(m) for m in claude_models])
            if changed:
                logger.info(
                    "Claude model probe: %d/%d models unavailable: %s",
                    len(changed),
                    len(claude_models),
                    ", ".join(changed.keys()),
                )
        return changed

    def list_available(self) -> list[ModelCapability]:
        """Return all models whose provider is available."""
        return [m for m in self._models.values() if m.available]

    def list_unavailable(self) -> list[ModelCapability]:
        """Return all models whose provider is NOT available."""
        return [m for m in self._models.values() if not m.available]

    def is_model_available(self, model_id: str) -> bool:
        """Check if a specific model is registered AND available.

        Resolves aliases transparently.
        """
        model = self.get(model_id)
        if model is None:
            return False
        return model.available

    @property
    def count(self) -> int:
        return len(self._models)

    def to_summary(self) -> str:
        """Return a human-readable summary of registered models.

        Used in system prompts to inform the master agent about
        available models and their strengths.
        """
        if not self._models:
            return "No models registered in capability registry."

        # Build reverse alias map: model_id → list of aliases
        reverse_aliases: dict[str, list[str]] = {}
        for alias, target in self._aliases.items():
            reverse_aliases.setdefault(target, []).append(alias)

        lines = ["Available models and their strengths:"]
        # Group by tier
        for tier in ModelTier:
            models = self.list_by_tier(tier)
            if not models:
                continue
            lines.append(f"\n  {tier.value.upper()} tier:")
            for m in models:
                top_affinities = sorted(
                    m.affinities.items(),
                    key=lambda x: -x[1],
                )[:3]
                strengths = ", ".join(
                    f"{k} ({v:.0%})" for k, v in top_affinities
                )
                status = "" if m.available else " [unavailable]"
                aliases = reverse_aliases.get(m.model_id, [])
                alias_str = f" (aliases: {', '.join(sorted(aliases))})" if aliases else ""
                lines.append(
                    f"    - {m.model_id} ({m.provider}): {strengths}{status}{alias_str}"
                )

        return "\n".join(lines)


def build_default_registry() -> ModelRegistry:
    """Build a ModelRegistry with sensible defaults for known models.

    These defaults can be overridden or extended via YAML config.
    """
    registry = ModelRegistry()

    # Claude models
    registry.register(ModelCapability(
        model_id="claude-opus-4-6",
        provider="claude",
        tier=ModelTier.FRONTIER,
        cost_factor=5.0,
        speed_factor=0.5,
        max_context=200_000,
        affinities={
            TaskCategory.ARCHITECTURE.value: 0.95,
            TaskCategory.COMPLEX_REASONING.value: 0.95,
            TaskCategory.CODING.value: 0.90,
            TaskCategory.CODE_REVIEW.value: 0.92,
            TaskCategory.PLANNING.value: 0.95,
            TaskCategory.TOOL_USE.value: 0.90,
            TaskCategory.AGENTIC.value: 0.93,
            TaskCategory.GENERAL.value: 0.92,
            TaskCategory.DOCUMENTATION.value: 0.90,
        },
    ))

    registry.register(ModelCapability(
        model_id="claude-sonnet-4-5-20250929",
        provider="claude",
        tier=ModelTier.STRONG,
        cost_factor=1.0,
        speed_factor=1.0,
        max_context=200_000,
        affinities={
            TaskCategory.CODING.value: 0.88,
            TaskCategory.CODE_REVIEW.value: 0.85,
            TaskCategory.GENERAL.value: 0.85,
            TaskCategory.TOOL_USE.value: 0.85,
            TaskCategory.EXPLORATION.value: 0.83,
            TaskCategory.SIMPLE_TASKS.value: 0.85,
            TaskCategory.DOCUMENTATION.value: 0.85,
            TaskCategory.AGENTIC.value: 0.80,
        },
    ))

    registry.register(ModelCapability(
        model_id="claude-3-5-haiku-20241022",
        provider="claude",
        tier=ModelTier.FAST,
        cost_factor=0.2,
        speed_factor=3.0,
        max_context=200_000,
        affinities={
            TaskCategory.EXPLORATION.value: 0.78,
            TaskCategory.SIMPLE_TASKS.value: 0.82,
            TaskCategory.GENERAL.value: 0.75,
            TaskCategory.CODING.value: 0.72,
            TaskCategory.DOCUMENTATION.value: 0.75,
        },
    ))

    # OpenAI Codex
    registry.register(ModelCapability(
        model_id="gpt-5.3-spark",
        provider="codex",
        tier=ModelTier.FAST,
        cost_factor=0.4,
        speed_factor=2.4,
        max_context=200_000,
        affinities={
            TaskCategory.CODING.value: 0.88,
            TaskCategory.SIMPLE_TASKS.value: 0.92,
            TaskCategory.GENERAL.value: 0.88,
            TaskCategory.TOOL_USE.value: 0.87,
            TaskCategory.EXPLORATION.value: 0.86,
        },
    ))

    registry.register(ModelCapability(
        model_id="gpt-5.2-codex",
        provider="codex",
        tier=ModelTier.FRONTIER,
        cost_factor=4.0,
        speed_factor=0.6,
        max_context=200_000,
        affinities={
            TaskCategory.CODING.value: 0.92,
            TaskCategory.COMPLEX_REASONING.value: 0.90,
            TaskCategory.GENERAL.value: 0.88,
            TaskCategory.TOOL_USE.value: 0.88,
            TaskCategory.AGENTIC.value: 0.85,
        },
    ))

    # Gemini
    registry.register(ModelCapability(
        model_id="gemini-3-pro-preview",
        provider="gemini",
        tier=ModelTier.FRONTIER,
        cost_factor=3.0,
        speed_factor=0.7,
        max_context=1_000_000,
        affinities={
            TaskCategory.CODING.value: 0.88,
            TaskCategory.GENERAL.value: 0.88,
            TaskCategory.SEARCH.value: 0.90,
            TaskCategory.EXPLORATION.value: 0.85,
            TaskCategory.COMPLEX_REASONING.value: 0.85,
        },
    ))

    registry.register(ModelCapability(
        model_id="gemini-3-flash-preview",
        provider="gemini",
        tier=ModelTier.FAST,
        cost_factor=0.3,
        speed_factor=2.0,
        max_context=1_000_000,
        affinities={
            TaskCategory.CODING.value: 0.82,
            TaskCategory.GENERAL.value: 0.80,
            TaskCategory.SEARCH.value: 0.85,
            TaskCategory.EXPLORATION.value: 0.82,
            TaskCategory.SIMPLE_TASKS.value: 0.80,
        },
    ))

    registry.register(ModelCapability(
        model_id="gemini-2.5-flash",
        provider="gemini",
        tier=ModelTier.FAST,
        cost_factor=0.15,
        speed_factor=2.5,
        max_context=1_000_000,
        affinities={
            TaskCategory.EXPLORATION.value: 0.82,
            TaskCategory.SIMPLE_TASKS.value: 0.80,
            TaskCategory.GENERAL.value: 0.78,
            TaskCategory.SEARCH.value: 0.85,
        },
    ))

    return registry


def load_model_registry_from_yaml(
    raw_registry: dict[str, Any],
    base_registry: ModelRegistry | None = None,
    *,
    model_aliases: dict[str, Any] | None = None,
) -> ModelRegistry:
    """Load/override model registry entries from YAML config.

    Args:
        raw_registry: The 'model_registry' section from YAML.
        base_registry: Optional base registry to extend (defaults to
            build_default_registry()).
        model_aliases: Optional 'models' section from YAML. Each entry
            maps a short alias (e.g. ``"sonnet"``) to a dict with
            ``provider`` and ``model_id`` keys.  These are registered as
            aliases in the registry.

    Returns:
        ModelRegistry with YAML overrides applied.
    """
    registry = base_registry or build_default_registry()

    # ── Register YAML model aliases ───────────────────────────
    if model_aliases:
        for alias, cfg in model_aliases.items():
            if isinstance(cfg, dict) and "model_id" in cfg:
                model_id = str(cfg["model_id"])
                provider = str(cfg.get("provider", ""))
                effort = str(cfg.get("reasoning_effort", "")).strip().lower()
                if provider == "codex" and effort in {"low", "medium", "high"}:
                    model_id = f"{model_id}::reasoning_effort={effort}"
                registry.register_alias(alias, model_id)
            elif hasattr(cfg, "model_id"):
                # ModelAlias dataclass
                model_id = str(cfg.model_id)
                provider = str(getattr(cfg, "provider", ""))
                effort = str(getattr(cfg, "reasoning_effort", "") or "").strip().lower()
                if provider == "codex" and effort in {"low", "medium", "high"}:
                    model_id = f"{model_id}::reasoning_effort={effort}"
                registry.register_alias(alias, model_id)

    # ── Register / override model capabilities ────────────────
    for alias, cfg in raw_registry.items():
        if not isinstance(cfg, dict):
            logger.warning(
                "Skipping invalid model_registry entry: %s", alias
            )
            continue

        model_id = str(alias)
        raw_model_id = cfg.get("model_id")
        if isinstance(raw_model_id, str) and raw_model_id.strip():
            model_id = raw_model_id.strip()

        tier_str = cfg.get("tier", "strong")
        try:
            tier = ModelTier(tier_str)
        except ValueError:
            logger.warning(
                "Unknown tier '%s' for model %s (alias=%s), defaulting to 'strong'",
                tier_str, model_id, alias,
            )
            tier = ModelTier.STRONG

        affinities = cfg.get("affinities", {})
        if not isinstance(affinities, dict):
            affinities = {}

        registry.register_alias(str(alias), model_id)

        registry.register(ModelCapability(
            model_id=model_id,
            provider=cfg.get("provider", "claude"),
            tier=tier,
            cost_factor=float(cfg.get("cost_factor", 1.0)),
            speed_factor=float(cfg.get("speed_factor", 1.0)),
            max_context=int(cfg.get("max_context", 200_000)),
            affinities=affinities,
            available=bool(cfg.get("available", True)),
        ))

    return registry
