"""Model discovery — discovers available models from installed CLI tools.

Probes Claude, Codex, and Gemini CLIs to enumerate the models they expose,
optionally updates the CLIs beforehand, and writes newly discovered models
into ``~/.prsm/models.yaml`` so the rest of PRSM can use them immediately.

The module is designed to run at server/TUI startup.  Every operation is
best-effort: failures are logged but never crash the caller.

Typical usage::

    result = await discover_and_update_models(update_clis=True)
    print(result.discovered_models)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

_SUBPROCESS_TIMEOUT = 120  # seconds, for CLI update/discovery commands
_CACHE_STALENESS_SECONDS = 24 * 60 * 60  # 24 hours

DEFAULT_MODELS_YAML_PATH = Path.home() / ".prsm" / "models.yaml"

# Claude model ID regex — matches patterns like claude-opus-4-6,
# claude-sonnet-4-5-20250929, claude-3-5-haiku-20241022
_CLAUDE_MODEL_RE = re.compile(
    r"claude-(?:opus|sonnet|haiku)-\d[\w.-]*"
)

# Legacy models to exclude from discovery results.
_CLAUDE_LEGACY_PREFIXES = (
    "claude-3-opus",
    "claude-3-sonnet",
    "claude-3-haiku",
    "claude-3.0",
)

# Suffixes / patterns to exclude from Claude model discovery.
# -v1 = Bedrock/Vertex API variant; @ = alternate naming format.
_CLAUDE_EXCLUDE_PATTERNS = (
    re.compile(r"-v\d+$"),          # e.g. claude-opus-4-6-v1
    re.compile(r".*@\d+$"),          # e.g. claude-opus-4@20250514
)

# Gemini model IDs to skip (non-chat models).
_GEMINI_SKIP_IDS = {"gemini-embedding-001"}

_CLAUDE_FAMILY_NAMES = {"opus", "sonnet", "haiku"}


def _normalize_model_alias_id(model_id: str) -> str:
    """Normalize model IDs for family/version comparison.

    Dots and repeated separators are converted to hyphens so values like
    ``4-6`` and ``4.6`` are treated as the same version family.
    """
    return model_id.replace(".", "-").replace("_", "-")


def _extract_claude_family_and_version(model_id: str) -> tuple[str, str] | None:
    """Return ``(family, version_value)`` for Claude models.

    Family names are detected within the model ID so both
    ``claude-opus-4-6`` and ``claude-3-5-haiku-20241022`` resolve to
    their family and version buckets.
    """
    normalized = _normalize_model_alias_id(model_id).lower()
    if not normalized.startswith("claude-"):
        return None

    parts = [part for part in normalized.split("-") if part]
    if len(parts) < 3:
        return None

    for idx in range(1, len(parts) - 1):
        family = parts[idx]
        if family in _CLAUDE_FAMILY_NAMES:
            prefix = "-".join(parts[1:idx])
            version = "-".join(parts[idx + 1:])
            if not version:
                return None
            if prefix:
                version = f"{prefix}-{version}"
            return family, version
    return None


def _claude_version_sort_key(version: str) -> tuple:
    """Build a stable version key that compares latest numerically/lexically."""
    if not version:
        return (("",),)

    bits: list[tuple] = []
    for chunk in re.findall(r"[0-9]+|[A-Za-z]+", version):
        if chunk.isdigit():
            bits.append((0, int(chunk)))
        else:
            bits.append((1, chunk.lower()))
    if not bits:
        return ((1, version.lower()),)
    return tuple(bits)


def _pick_latest_claude_family_models(
    discovered: dict[str, DiscoveredModel],
) -> dict[str, str]:
    """Return one canonical model_id per Claude family (opus/sonnet/haiku)."""
    latest: dict[str, str] = {}
    latest_key: dict[str, tuple] = {}

    for model_id in sorted(discovered.keys()):
        family_version = _extract_claude_family_and_version(model_id)
        if not family_version:
            continue
        family, version = family_version
        key = _claude_version_sort_key(version)
        current_key = latest_key.get(family)
        if current_key is None or key > current_key:
            latest[family] = model_id
            latest_key[family] = key

    return latest


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class DiscoveredModel:
    """A single model discovered from a CLI tool."""

    model_id: str
    provider: str  # "claude", "codex", "gemini"
    display_name: str
    description: str
    tier: str  # "frontier", "strong", "fast", "economy"
    visibility: str  # "list", "hide" (from codex); "visible" for others
    reasoning_levels: list[str] = field(default_factory=list)
    source: str = ""  # e.g. "codex_cache", "gemini_core", "claude_binary"


@dataclass
class DiscoveryResult:
    """Aggregated result of the full discovery + update cycle."""

    discovered_models: dict[str, DiscoveredModel] = field(default_factory=dict)
    updated_clis: list[str] = field(default_factory=list)
    models_yaml_updated: bool = False
    errors: list[str] = field(default_factory=list)


# ── Tier inference ───────────────────────────────────────────────────

def _infer_tier(model_id: str, provider: str, description: str = "") -> str:
    """Infer a model's tier from its ID, provider, and description.

    Mapping heuristics by provider:

    * **Claude**: ``opus`` → frontier, ``sonnet`` → strong, ``haiku`` → fast.
    * **Codex**: ``spark`` / ``mini`` → fast, ``codex-max`` → frontier,
      latest numbered (e.g. ``gpt-5.3-codex``) → frontier, older → strong.
    * **Gemini**: ``pro`` (3.x / preview) → frontier, ``pro`` (2.x) → strong,
      ``flash-lite`` → economy, ``flash`` → fast.
    """
    mid = model_id.lower()
    desc = description.lower()

    if provider == "claude":
        if "opus" in mid:
            return "frontier"
        if "sonnet" in mid:
            return "strong"
        if "haiku" in mid:
            return "fast"
        return "strong"

    if provider == "codex":
        if "spark" in mid or "mini" in mid:
            return "fast"
        if "codex-max" in mid:
            return "frontier"
        # Latest numbered series (e.g. gpt-5.3-codex, gpt-5.2-codex)
        m = re.search(r"gpt-(\d+)\.(\d+)", mid)
        if m:
            major, minor = int(m.group(1)), int(m.group(2))
            # Heuristic: the highest numbered model is frontier
            if major >= 5 and minor >= 2:
                return "frontier"
            return "strong"
        # O-series models from OpenAI
        if mid.startswith("o") and re.match(r"o\d", mid):
            if "mini" in mid:
                return "fast"
            return "strong"
        return "strong"

    if provider == "gemini":
        if "flash-lite" in mid:
            return "economy"
        if "flash" in mid:
            return "fast"
        # Pro models: 3.x / preview → frontier, 2.x → strong
        if "pro" in mid:
            if "3" in mid or "preview" in mid:
                return "frontier"
            return "strong"
        return "strong"

    return "strong"


# ── Default affinities ──────────────────────────────────────────────

def _default_affinities(tier: str, provider: str) -> dict[str, float]:
    """Return reasonable default affinity scores for a model.

    Scores range 0.0–1.0 across the 12 standard task categories.
    Higher tiers receive stronger baseline scores.
    """
    # Base scores by tier
    bases: dict[str, dict[str, float]] = {
        "frontier": {
            "architecture": 0.95,
            "complex-reasoning": 0.95,
            "coding": 0.90,
            "code-review": 0.90,
            "planning": 0.95,
            "tool-use": 0.85,
            "exploration": 0.80,
            "simple-tasks": 0.70,
            "general": 0.90,
            "agentic": 0.90,
            "search": 0.80,
            "documentation": 0.85,
        },
        "strong": {
            "architecture": 0.80,
            "complex-reasoning": 0.80,
            "coding": 0.85,
            "code-review": 0.80,
            "planning": 0.80,
            "tool-use": 0.80,
            "exploration": 0.80,
            "simple-tasks": 0.80,
            "general": 0.80,
            "agentic": 0.80,
            "search": 0.75,
            "documentation": 0.80,
        },
        "fast": {
            "architecture": 0.55,
            "complex-reasoning": 0.55,
            "coding": 0.70,
            "code-review": 0.65,
            "planning": 0.60,
            "tool-use": 0.70,
            "exploration": 0.75,
            "simple-tasks": 0.90,
            "general": 0.70,
            "agentic": 0.65,
            "search": 0.75,
            "documentation": 0.70,
        },
        "economy": {
            "architecture": 0.40,
            "complex-reasoning": 0.40,
            "coding": 0.55,
            "code-review": 0.50,
            "planning": 0.45,
            "tool-use": 0.55,
            "exploration": 0.65,
            "simple-tasks": 0.85,
            "general": 0.60,
            "agentic": 0.50,
            "search": 0.65,
            "documentation": 0.60,
        },
    }

    affinities = dict(bases.get(tier, bases["strong"]))

    # Provider-specific tweaks
    if provider == "codex":
        affinities["coding"] = min(1.0, affinities["coding"] + 0.05)
        affinities["tool-use"] = min(1.0, affinities["tool-use"] + 0.05)
    elif provider == "gemini":
        affinities["search"] = min(1.0, affinities["search"] + 0.05)
        affinities["exploration"] = min(1.0, affinities["exploration"] + 0.05)
    elif provider == "claude":
        affinities["agentic"] = min(1.0, affinities["agentic"] + 0.05)
        affinities["complex-reasoning"] = min(
            1.0, affinities["complex-reasoning"] + 0.03
        )

    # Round to avoid floating-point display artefacts (e.g. 0.9500000000000001).
    return {k: round(v, 4) for k, v in affinities.items()}


def _build_default_registry_profile_map() -> dict[str, dict[str, Any]]:
    """Build a lightweight reference map for model capability defaults.

    This pulls only the static defaults from ``build_default_registry()``
    and returns direct model lookup by model id plus fallback aliases
    for each provider/tier family.
    """
    try:
        from prsm.engine.model_registry import build_default_registry
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not load default model registry: %s", exc)
        return {}

    default_registry = build_default_registry()
    if not default_registry or not hasattr(default_registry, "list_models"):
        return {}

    profile_map: dict[str, dict[str, Any]] = {}
    provider_tier_fallback: dict[tuple[str, str], dict[str, Any]] = {}
    provider_family_fallback: dict[tuple[str, str], dict[str, Any]] = {}

    for cap in default_registry.list_models():
        if not cap:
            continue
        key = f"{cap.provider}::{cap.model_id}"
        entry = {
            "model_id": cap.model_id,
            "provider": cap.provider,
            "tier": cap.tier.value if hasattr(cap.tier, "value") else str(cap.tier),
            "affinities": dict(cap.affinities or {}),
            "cost_factor": float(cap.cost_factor),
            "speed_factor": float(cap.speed_factor),
        }
        profile_map[key] = entry

        tier_key = (cap.provider, str(cap.tier.value if hasattr(cap.tier, "value") else cap.tier))
        provider_tier_fallback.setdefault(tier_key, entry)

        # Family heuristics for common providers (Claude/Gemini families).
        if cap.provider == "claude":
            if "opus" in cap.model_id:
                provider_family_fallback[(cap.provider, "opus")] = entry
            elif "sonnet" in cap.model_id:
                provider_family_fallback[(cap.provider, "sonnet")] = entry
            elif "haiku" in cap.model_id:
                provider_family_fallback[(cap.provider, "haiku")] = entry
        elif cap.provider == "gemini":
            if "pro" in cap.model_id:
                provider_family_fallback[(cap.provider, "pro")] = entry
            elif "flash" in cap.model_id:
                provider_family_fallback[(cap.provider, "flash")] = entry
        elif cap.provider == "codex":
            if "spark" in cap.model_id:
                provider_family_fallback[(cap.provider, "spark")] = entry

    profile_map.update(
        {
            "fallback_by_provider_tier": {
                f"{k[0]}::{k[1]}": v for k, v in provider_tier_fallback.items()
            },
            "fallback_by_provider_family": {
                f"{k[0]}::{k[1]}": v for k, v in provider_family_fallback.items()
            },
        }
    )
    return profile_map


def _resolve_model_profile(dm: DiscoveredModel, profile_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Resolve up-to-date capability profile for a discovered model."""
    direct_key = f"{dm.provider}::{dm.model_id}"
    direct = profile_map.get(direct_key)
    if direct:
        return dict(direct)

    norm_model = (dm.model_id or "").lower()
    provider = dm.provider

    family = ""
    if provider == "claude":
        family_info = _extract_claude_family_and_version(norm_model)
        if family_info:
            family = family_info[0]
        elif "opus" in norm_model:
            family = "opus"
        elif "sonnet" in norm_model:
            family = "sonnet"
        elif "haiku" in norm_model:
            family = "haiku"

        if family:
            family_profile = profile_map.get(f"fallback_by_provider_family::{provider}::{family}")
            if family_profile:
                return dict(family_profile)

    if provider == "gemini":
        if "pro" in norm_model:
            fam_profile = profile_map.get(f"fallback_by_provider_family::{provider}::pro")
            if fam_profile:
                return dict(fam_profile)
        if "flash" in norm_model:
            fam_profile = profile_map.get(f"fallback_by_provider_family::{provider}::flash")
            if fam_profile:
                return dict(fam_profile)

    if provider == "codex":
        if "spark" in norm_model or "mini" in norm_model:
            fam_profile = profile_map.get(f"fallback_by_provider_family::{provider}::spark")
            if fam_profile:
                return dict(fam_profile)

    tier_profile = profile_map.get(
        f"fallback_by_provider_tier::{provider}::{dm.tier}"
    )
    if tier_profile:
        return dict(tier_profile)

    return {
        "model_id": dm.model_id,
        "provider": dm.provider,
        "tier": dm.tier,
        "cost_factor": 1.0,
        "speed_factor": 1.0,
        "affinities": _default_affinities(dm.tier, dm.provider),
    }


def _refresh_profiles_for_all_models(
    discovered: dict[str, DiscoveredModel],
) -> dict[str, dict[str, Any]]:
    """Recompute full model profile map for discovered models.

    Called when model list changes so every discovered model gets an
    up-to-date registry profile using current default heuristics.
    """
    profile_map = _build_default_registry_profile_map()
    refreshed: dict[str, dict[str, Any]] = {}

    for model_id, dm in discovered.items():
        profile = _resolve_model_profile(dm, profile_map)
        profile.setdefault("model_id", model_id)
        profile.setdefault("provider", dm.provider)
        profile.setdefault("tier", dm.tier)
        profile.setdefault(
            "affinities",
            _default_affinities(dm.tier, dm.provider),
        )
        refreshed[model_id] = {
            "tier": str(profile["tier"]),
            "provider": str(profile.get("provider", dm.provider)),
            "cost_factor": float(profile.get("cost_factor", 1.0)),
            "speed_factor": float(profile.get("speed_factor", 1.0)),
            "affinities": dict(profile.get("affinities", {}))
            or _default_affinities(dm.tier, dm.provider),
        }

    return refreshed


# ── Subprocess helper ────────────────────────────────────────────────

async def _run_subprocess(
    *args: str,
    timeout: int = _SUBPROCESS_TIMEOUT,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Run a subprocess asynchronously with a timeout.

    Returns ``(returncode, stdout, stderr)``.  On timeout or failure the
    return code is ``-1`` and stderr contains the error message.
    """
    merged_env: dict[str, str] | None = None
    if env is not None:
        merged_env = {**os.environ, **env}

    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=merged_env,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
        return (
            proc.returncode or 0,
            stdout_bytes.decode("utf-8", errors="replace"),
            stderr_bytes.decode("utf-8", errors="replace"),
        )
    except asyncio.TimeoutError:
        # Try to kill the process if it is still alive.
        try:
            proc.kill()  # type: ignore[possibly-undefined]
        except Exception:
            pass
        return (-1, "", f"Command timed out after {timeout}s: {args}")
    except FileNotFoundError:
        return (-1, "", f"Command not found: {args[0]}")
    except Exception as exc:
        return (-1, "", f"Subprocess error: {exc}")


# ── CLI update functions ─────────────────────────────────────────────

async def _update_claude_cli() -> tuple[bool, str]:
    """Update the Claude CLI via ``claude update``.

    The ``CLAUDECODE`` environment variable is explicitly unset to avoid
    interfering with the update process.

    Returns:
        ``(success, message)`` describing what happened.
    """
    if not shutil.which("claude"):
        return False, "Claude CLI not found on PATH"

    # Unset CLAUDECODE so the update doesn't think it's running inside
    # a Claude Code session.
    env_override = {k: v for k, v in os.environ.items()}
    env_override.pop("CLAUDECODE", None)

    rc, stdout, stderr = await _run_subprocess(
        "claude", "update",
        env=env_override,
    )
    output = (stdout + stderr).strip()
    if rc == 0:
        logger.info("Claude CLI updated successfully: %s", output[:200])
        return True, output[:200] or "Updated successfully"
    logger.warning("Claude CLI update failed (rc=%d): %s", rc, output[:300])
    return False, output[:300] or f"Update failed with exit code {rc}"


async def _update_codex_cli() -> tuple[bool, str]:
    """Update the Codex CLI via ``npm update -g @openai/codex``.

    Returns:
        ``(success, message)`` describing what happened.
    """
    npm = shutil.which("npm")
    if not npm:
        return False, "npm not found on PATH"
    if not shutil.which("codex"):
        return False, "Codex CLI not found on PATH"

    rc, stdout, stderr = await _run_subprocess(
        npm, "update", "-g", "@openai/codex",
    )
    output = (stdout + stderr).strip()
    if rc == 0:
        logger.info("Codex CLI updated successfully: %s", output[:200])
        return True, output[:200] or "Updated successfully"
    logger.warning("Codex CLI update failed (rc=%d): %s", rc, output[:300])
    return False, output[:300] or f"Update failed with exit code {rc}"


async def _update_gemini_cli() -> tuple[bool, str]:
    """Update the Gemini CLI via ``npm update -g @google/gemini-cli``.

    Returns:
        ``(success, message)`` describing what happened.
    """
    npm = shutil.which("npm")
    if not npm:
        return False, "npm not found on PATH"
    if not shutil.which("gemini"):
        return False, "Gemini CLI not found on PATH"

    rc, stdout, stderr = await _run_subprocess(
        npm, "update", "-g", "@google/gemini-cli",
    )
    output = (stdout + stderr).strip()
    if rc == 0:
        logger.info("Gemini CLI updated successfully: %s", output[:200])
        return True, output[:200] or "Updated successfully"
    logger.warning("Gemini CLI update failed (rc=%d): %s", rc, output[:300])
    return False, output[:300] or f"Update failed with exit code {rc}"


# ── Model discovery functions ────────────────────────────────────────

async def _discover_codex_models() -> list[DiscoveredModel]:
    """Discover models from the Codex CLI models cache.

    Reads ``~/.codex/models_cache.json``, which is automatically
    maintained by the Codex CLI.  If the cache is stale (>24 h) or
    missing, runs ``codex --help`` to trigger a refresh.

    Returns:
        List of :class:`DiscoveredModel` for all user-facing Codex
        models (``visibility == "list"``).
    """
    cache_path = Path.home() / ".codex" / "models_cache.json"

    # If cache is stale or missing, try refreshing it.
    needs_refresh = False
    if not cache_path.is_file():
        needs_refresh = True
    else:
        age = time.time() - cache_path.stat().st_mtime
        if age > _CACHE_STALENESS_SECONDS:
            needs_refresh = True

    if needs_refresh and shutil.which("codex"):
        logger.info("Codex models cache is stale/missing — refreshing via codex --help")
        await _run_subprocess("codex", "--help", timeout=30)

    if not cache_path.is_file():
        logger.debug("Codex models cache not found at %s", cache_path)
        return []

    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read Codex models cache: %s", exc)
        return []

    models: list[DiscoveredModel] = []
    entries: list[dict[str, Any]] = []

    # Handle both list-at-top-level and dict-with-models-key structures.
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        entries = raw.get("models", raw.get("data", []))
        if isinstance(entries, dict):
            entries = list(entries.values())

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        slug = entry.get("slug") or entry.get("id") or entry.get("model_id", "")
        if not slug:
            continue

        visibility = entry.get("visibility", "list")
        if visibility != "list":
            continue

        description = entry.get("description", "")
        display_name = entry.get("name") or entry.get("display_name") or slug
        raw_reasoning = entry.get("supported_reasoning_levels", [])
        reasoning: list[str] = []
        if isinstance(raw_reasoning, str):
            reasoning = [raw_reasoning]
        elif isinstance(raw_reasoning, list):
            for item in raw_reasoning:
                if isinstance(item, str):
                    reasoning.append(item)
                elif isinstance(item, dict):
                    # Codex cache format: {"effort": "low", "description": "..."}
                    effort = item.get("effort", "")
                    if effort:
                        reasoning.append(effort)

        tier = _infer_tier(slug, "codex", description)

        models.append(DiscoveredModel(
            model_id=slug,
            provider="codex",
            display_name=display_name,
            description=description,
            tier=tier,
            visibility=visibility,
            reasoning_levels=reasoning,
            source="codex_cache",
        ))

    logger.info("Discovered %d Codex models from cache", len(models))
    return models


async def _discover_gemini_models() -> list[DiscoveredModel]:
    """Discover models from the Gemini CLI core package.

    Locates the ``models.js`` file inside
    ``@google/gemini-cli-core/dist/src/config/`` under the global
    npm prefix and parses model constants from the JavaScript source.

    Returns:
        List of :class:`DiscoveredModel` for all found Gemini models.
    """
    npm = shutil.which("npm")
    if not npm:
        logger.debug("npm not found — skipping Gemini model discovery")
        return []

    # Get global npm root.
    rc, stdout, _ = await _run_subprocess(npm, "root", "-g", timeout=15)
    if rc != 0 or not stdout.strip():
        logger.debug("Could not determine npm global root")
        return []

    npm_root = Path(stdout.strip())

    # Possible locations for the models.js file.
    candidates = [
        npm_root / "@google" / "gemini-cli" / "node_modules"
        / "@google" / "gemini-cli-core" / "dist" / "src" / "config" / "models.js",
        npm_root / "@google" / "gemini-cli-core"
        / "dist" / "src" / "config" / "models.js",
    ]

    models_js: str | None = None
    for candidate in candidates:
        if candidate.is_file():
            try:
                models_js = candidate.read_text(encoding="utf-8")
                logger.debug("Found Gemini models.js at %s", candidate)
                break
            except OSError as exc:
                logger.debug("Could not read %s: %s", candidate, exc)

    if models_js is None:
        logger.debug("Gemini models.js not found in any candidate location")
        return []

    model_ids: set[str] = set()

    # Strategy 1: look for VALID_GEMINI_MODELS Set entries.
    # e.g.  new Set(["gemini-2.5-pro", "gemini-2.5-flash", ...])
    set_match = re.search(
        r"(?:VALID_GEMINI_MODELS|validModels|VALID_MODELS)"
        r"\s*=\s*new\s+Set\(\[([^\]]+)\]\)",
        models_js,
    )
    if set_match:
        for m in re.finditer(r'["\']([^"\']+)["\']', set_match.group(1)):
            model_ids.add(m.group(1))

    # Strategy 2: look for named model constants.
    # e.g.  const DEFAULT_GEMINI_MODEL = "gemini-2.5-pro";
    for m in re.finditer(
        r"(?:const|let|var)\s+\w*(?:MODEL|GEMINI)\w*\s*=\s*[\"']"
        r"(gemini-[a-z0-9._-]+)[\"']",
        models_js,
    ):
        model_ids.add(m.group(1))

    # Strategy 3: grab any quoted string that looks like a gemini model id.
    if not model_ids:
        for m in re.finditer(r'["\']((gemini-\d[a-z0-9._-]*))["\']', models_js):
            model_ids.add(m.group(1))

    models: list[DiscoveredModel] = []
    for mid in sorted(model_ids):
        # Skip non-chat models (e.g. embedding models).
        if mid in _GEMINI_SKIP_IDS:
            continue
        tier = _infer_tier(mid, "gemini")
        display = mid.replace("-", " ").title()
        models.append(DiscoveredModel(
            model_id=mid,
            provider="gemini",
            display_name=display,
            description="",
            tier=tier,
            visibility="visible",
            reasoning_levels=[],
            source="gemini_core",
        ))

    logger.info("Discovered %d Gemini models from models.js", len(models))
    return models


async def _discover_claude_models() -> list[DiscoveredModel]:
    """Discover models from the Claude CLI binary.

    Follows the symlink from ``shutil.which("claude")`` to the real
    binary, runs ``strings`` on it, and extracts model IDs matching
    the Claude naming convention.

    Returns:
        List of :class:`DiscoveredModel` for all non-legacy Claude models.
    """
    claude_path = shutil.which("claude")
    if not claude_path:
        logger.debug("Claude CLI not found — skipping Claude model discovery")
        return []

    # Resolve symlinks to the real binary.
    real_path = Path(claude_path).resolve()
    if not real_path.is_file():
        logger.debug("Claude binary not found at resolved path %s", real_path)
        return []

    strings_bin = shutil.which("strings")
    if not strings_bin:
        logger.debug("'strings' utility not found — skipping Claude binary scan")
        return []

    rc, stdout, stderr = await _run_subprocess(
        strings_bin, str(real_path),
        timeout=30,
    )
    if rc != 0:
        logger.warning(
            "'strings' on Claude binary failed (rc=%d): %s", rc, stderr[:200]
        )
        return []

    raw_ids: set[str] = set()
    for line in stdout.splitlines():
        for m in _CLAUDE_MODEL_RE.finditer(line):
            candidate = m.group(0)
            # Skip code-related strings (e.g. "claude-code-*").
            if "code" in candidate.lower():
                continue
            raw_ids.add(candidate)

    # Also try to find a JSON object with model IDs → context sizes.
    # e.g. {"claude-opus-4-6": 200000, "claude-sonnet-4-5-20250929": 200000}
    for m in re.finditer(r'\{[^{}]*"claude-[^{}]+\}', stdout):
        try:
            obj = json.loads(m.group(0))
            for key in obj:
                if _CLAUDE_MODEL_RE.match(key) and "code" not in key.lower():
                    raw_ids.add(key)
        except (json.JSONDecodeError, TypeError):
            pass

    # Filter out legacy/deprecated models, Bedrock/Vertex variants, and
    # alternate naming formats.
    filtered: set[str] = set()
    for mid in raw_ids:
        if any(mid.startswith(prefix) for prefix in _CLAUDE_LEGACY_PREFIXES):
            continue
        if any(pat.search(mid) for pat in _CLAUDE_EXCLUDE_PATTERNS):
            continue
        filtered.add(mid)

    # Deduplicate variants like "claude-opus-4.6" and
    # "claude-opus-4-6" into one canonical normalized form.
    deduped_models: dict[str, str] = {}
    for mid in sorted(filtered):
        normalized = _normalize_model_alias_id(mid)
        current = deduped_models.get(normalized)
        if current is None:
            deduped_models[normalized] = mid
            continue
        # Prefer the hyphen version over dotted punctuation if both exist.
        if "." in current and "." not in mid:
            deduped_models[normalized] = mid

    models: list[DiscoveredModel] = []
    for mid in sorted(deduped_models.values()):
        tier = _infer_tier(mid, "claude")
        # Derive display name: "claude-opus-4-6" → "Claude Opus 4 6"
        display = mid.replace("-", " ").title()
        models.append(DiscoveredModel(
            model_id=mid,
            provider="claude",
            display_name=display,
            description="",
            tier=tier,
            visibility="visible",
            reasoning_levels=[],
            source="claude_binary",
        ))

    logger.info("Discovered %d Claude models from binary", len(models))
    return models


# ── models.yaml update ───────────────────────────────────────────────

async def _update_models_yaml(
    discovered: dict[str, DiscoveredModel],
    yaml_path: Path,
    *,
    force_overwrite: bool = False,
) -> bool:
    """Overwrite discovered model entries in the target ``models.yaml``.

    This intentionally replaces the ``models`` and ``model_registry``
    sections with the latest discovered set, so no duplicates or stale
    entries are left behind.

    When ``force_overwrite`` is True, any existing file payload is
    discarded first so the file is fully regenerated from discovery.

    * **Replaces** the ``models:`` section with short aliases
      (human-readable keys) mapped to discovered ``model_id`` values.
    * **Replaces** the ``model_registry:`` section with discovered model
      capabilities, keyed by alias, plus a ``model_id`` field for each entry.

    Returns:
        ``True`` if the file was written, ``False`` otherwise.
    """
    if yaml_path.is_file() and force_overwrite:
        logger.info("Force-overwriting models.yaml at %s", yaml_path)

    existing: dict[str, Any] = {}
    if not force_overwrite and yaml_path.is_file():
        try:
            with open(yaml_path, encoding="utf-8") as f:
                existing = yaml.safe_load(f) or {}
        except (yaml.YAMLError, OSError) as exc:
            logger.warning(
                "Could not read existing models.yaml at %s: %s", yaml_path, exc
            )
            existing = {}

    next_models_section: dict[str, Any] = {}
    next_registry_section: dict[str, Any] = {}
    refreshed_profiles = _refresh_profiles_for_all_models(discovered)

    if not discovered:
        logger.info("No discovered models to merge into models.yaml")
        return False

    for model_id, dm in sorted(discovered.items()):
        alias = _generate_alias(model_id, dm.provider, next_models_section)

        entry: dict[str, Any] = {
            "provider": dm.provider,
            "model_id": dm.model_id,
        }
        if dm.reasoning_levels:
            # Default to "medium" if available, else first level.
            if "medium" in dm.reasoning_levels:
                entry["reasoning_effort"] = "medium"
            else:
                entry["reasoning_effort"] = dm.reasoning_levels[0]

        next_models_section[alias] = entry
        profile = refreshed_profiles.get(dm.model_id, {})
        next_registry_section[alias] = {
            "model_id": dm.model_id,
            "tier": profile.get("tier", dm.tier),
            "provider": profile.get("provider", dm.provider),
            "cost_factor": profile.get("cost_factor", 1.0),
            "speed_factor": profile.get("speed_factor", 1.0),
            "affinities": profile.get(
                "affinities", _default_affinities(dm.tier, dm.provider)
            ),
        }

    if not next_models_section and not next_registry_section:
        logger.info("No discovered models to write")
        return False

    modified_models = existing.get("models", {}) if isinstance(existing.get("models", {}), dict) else {}
    modified_registry = (
        existing.get("model_registry", {})
        if isinstance(existing.get("model_registry", {}), dict) else {}
    )

    modified = bool(force_overwrite) or (
        modified_models != next_models_section
        or modified_registry != next_registry_section
    )

    if not modified:
        logger.info("models.yaml is already up to date — no changes needed")
        return False

    # Clear any legacy sections and fully rewrite model data.
    existing = {
        "models": next_models_section,
        "model_registry": next_registry_section,
    }

    logger.info(
        "Overwrote models.yaml sections with %d models",
        len(next_models_section),
    )

    # Write back.
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(
                existing,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        logger.info("Updated models.yaml at %s", yaml_path)
        return True
    except OSError as exc:
        logger.error("Failed to write models.yaml to %s: %s", yaml_path, exc)
        return False


def _generate_alias(
    model_id: str,
    provider: str,
    existing_aliases: dict[str, Any],
) -> str:
    """Generate a short alias for a model ID.

    Generates a human-readable, collision-safe alias that does not
    collide with existing keys in *existing_aliases*.
    """

    provider = (provider or "").strip().lower() or "model"
    normalized = model_id.strip().lower()
    if not normalized:
        normalized = provider

    def _titleize(token: str) -> str:
        token = token.strip()
        if not token:
            return ""
        if token.lower() == "gpt":
            return "GPT"
        return token[:1].upper() + token[1:]

    def _split_version_and_date(parts: list[str]) -> tuple[list[str], str | None]:
        if not parts:
            return [], None

        date: str | None = None
        if re.fullmatch(r"\d{6,8}", parts[-1]):
            date = parts[-1]
            parts = parts[:-1]

        version_parts: list[str] = []
        for part in parts:
            if re.fullmatch(r"\d+", part):
                version_parts.append(part)
            elif not version_parts:
                continue
            else:
                break
        return version_parts, date

    def _normalize_version(parts: list[str]) -> str:
        return ".".join(part for part in parts if part)

    def _dedupe_alias(candidate: str, date_suffix: str | None = None) -> str:
        if candidate not in existing_aliases:
            return candidate

        if date_suffix:
            dated = f"{candidate} ({date_suffix})"
            if dated not in existing_aliases:
                return dated

        suffix = 2
        while True:
            fallback = f"{candidate} ({suffix})"
            if fallback not in existing_aliases:
                return fallback
            suffix += 1

    candidate = ""

    if normalized.startswith("claude-"):
        parts = [part for part in re.split(r"[-_]+", normalized) if part]
        family = ""
        version_start = len(parts)
        for idx, part in enumerate(parts[1:], start=1):
            if part in _CLAUDE_FAMILY_NAMES:
                family = _titleize(part)
                version_start = idx + 1
                break

        if family:
            version_parts, date_suffix = _split_version_and_date(parts[version_start:])
            version = _normalize_version(version_parts)
            candidate = f"{family} {version}" if version else family
            candidate = _dedupe_alias(candidate, date_suffix)

    elif provider in {"gpt", "codex"} and normalized.startswith("gpt-"):
        # gpt-5.3-codex-spark => "GPT 5.3 Spark"
        match = re.match(
            r"^gpt-(?P<major>\d+)(?:[.-](?P<minor>\d+))?(?:-(?P<suffix>.+))?$",
            normalized,
        )
        if match:
            major = match.group("major") or ""
            minor = match.group("minor") or ""
            version = f"{major}.{minor}" if minor else major
            raw_suffix = (match.group("suffix") or "").replace("_", "-")
            suffix_parts = [part for part in raw_suffix.split("-") if part]

            if suffix_parts and suffix_parts[0] == "codex":
                suffix_parts = suffix_parts[1:]

            if suffix_parts:
                suffix = " ".join(_titleize(part) for part in suffix_parts)
                candidate = f"GPT {version} {suffix}".strip()
            else:
                candidate = f"GPT {version}".strip()

            candidate = _dedupe_alias(candidate)

    elif provider == "gemini" and normalized.startswith("gemini-"):
        # gemini-2.5-flash => "Gemini 2.5 Flash"
        match = re.match(r"^gemini-(.+)$", normalized)
        if match:
            remainder = match.group(1) or ""
            parts = [part for part in re.split(r"[-_]+", remainder) if part]
            if not parts:
                candidate = "Gemini"
            else:
                version_parts: list[str] = []
                i = 0
                if i < len(parts):
                    if "." in parts[i]:
                        version_parts = [parts[i]]
                        i = 1
                    else:
                        while i < len(parts) and parts[i].isdigit():
                            version_parts.append(parts[i])
                            i += 1
                        if not version_parts:
                            version_parts = [parts[0]]
                            i = 1

                version = ".".join(version_parts).replace(". ", "").strip()
                suffix_parts = parts[i:]
                if suffix_parts:
                    suffix = " ".join(_titleize(part) for part in suffix_parts)
                    candidate = f"Gemini {version} {suffix}".strip()
                else:
                    candidate = f"Gemini {version}".strip()
                candidate = _dedupe_alias(candidate)

    if not candidate:
        candidate = " ".join(
            _titleize(part)
            for part in normalized.replace("_", " ").replace("-", " ").replace(".", " ").split()
        )
        if candidate:
            candidate = _dedupe_alias(candidate)

    if not candidate:
        candidate = provider

    if candidate in existing_aliases:
        return _dedupe_alias(candidate)
    return candidate


def _has_trailing_date_suffix(model_id: str) -> bool:
    normalized = model_id.lower().replace("_", "-").split("::reasoning_effort=")[0]
    tokens = [part for part in normalized.split("-") if part]
    if len(tokens) <= 1:
        return False
    last = tokens[-1]
    return len(last) in {6, 7, 8} and last.isdigit()


def _compare_version_parts(a: str, b: str) -> int:
    a_parts = [part for part in a.split("-") if part]
    b_parts = [part for part in b.split("-") if part]
    max_len = max(len(a_parts), len(b_parts))

    for idx in range(max_len):
        a_part = a_parts[idx] if idx < len(a_parts) else ""
        b_part = b_parts[idx] if idx < len(b_parts) else ""

        if not a_part and b_part:
            return -1
        if a_part and not b_part:
            return 1

        if a_part.isdigit() and b_part.isdigit():
            a_num = int(a_part)
            b_num = int(b_part)
            if a_num != b_num:
                return -1 if a_num < b_num else 1
            continue

        if a_part != b_part:
            return -1 if a_part < b_part else 1

    return 0


def _extract_dedupe_version(model_id: str) -> str:
    normalized = model_id.lower().replace("_", "-").replace(".", "-")
    tokens = [part for part in normalized.split("::reasoning_effort=")[0].split("-") if part]
    if not _has_trailing_date_suffix(model_id):
        filtered_tokens = tokens
    else:
        filtered_tokens = tokens[:-1]

    for idx, token in enumerate(filtered_tokens):
        if token and token[0].isdigit():
            return "-".join(filtered_tokens[idx:])
    return ""


def _is_preferred_duplicate_model(candidate_model_id: str, current_model_id: str) -> bool:
    candidate_has_date = _has_trailing_date_suffix(candidate_model_id)
    current_has_date = _has_trailing_date_suffix(current_model_id)
    if candidate_has_date != current_has_date:
        return not candidate_has_date

    candidate_version = _extract_dedupe_version(candidate_model_id)
    current_version = _extract_dedupe_version(current_model_id)
    if candidate_version != current_version:
        compared = _compare_version_parts(candidate_version, current_version)
        if compared != 0:
            return compared > 0

    return len(candidate_model_id) < len(current_model_id)


def _dedupe_by_display_label(discovered: dict[str, DiscoveredModel]) -> dict[str, DiscoveredModel]:
    provider_groups: dict[str, list[DiscoveredModel]] = {}
    for model in discovered.values():
        provider_groups.setdefault(model.provider, []).append(model)

    filtered: dict[str, DiscoveredModel] = {}

    for provider, models in provider_groups.items():
        by_label: dict[str, DiscoveredModel] = {}
        for model in sorted(models, key=lambda item: item.model_id):
            alias = _generate_alias(model.model_id, provider, {})
            current = by_label.get(alias)
            if not current or _is_preferred_duplicate_model(model.model_id, current.model_id):
                by_label[alias] = model

        for alias, model in by_label.items():
            filtered[model.model_id] = model

    return filtered


# ── Main entry point ─────────────────────────────────────────────────

async def discover_and_update_models(
    update_clis: bool = True,
    models_yaml_path: Path | None = None,
    *,
    force_overwrite: bool = False,
    sync_global_models_yaml: bool = False,
) -> DiscoveryResult:
    """Discover available models and optionally update CLI tools.

    This is the primary entry point for the model discovery system.
    It runs CLI updates in parallel (if requested), then discovers
    models from all available providers in parallel, and finally
    overwrites discovery-managed entries in ``~/.prsm/models.yaml``.

    Args:
        update_clis: Whether to run ``claude update``, ``npm update
            -g @openai/codex``, etc. before discovery.
        models_yaml_path: Path to the models YAML file.  Defaults to
            ``~/.prsm/models.yaml``.
        force_overwrite: Clear and rebuild the discovered models sections
            instead of merging with existing data.
        sync_global_models_yaml: Also write the same discovered set to
            global ``~/.prsm/models.yaml``.

    Returns:
        A :class:`DiscoveryResult` with all discovered models, the
        list of updated CLIs, and any errors encountered.
    """
    if models_yaml_path is None:
        models_yaml_path = DEFAULT_MODELS_YAML_PATH

    result = DiscoveryResult()

    # ── Phase 1: CLI updates (parallel, best-effort) ─────────
    if update_clis:
        update_tasks: list[tuple[str, Any]] = []
        if shutil.which("claude"):
            update_tasks.append(("claude", _update_claude_cli()))
        if shutil.which("codex"):
            update_tasks.append(("codex", _update_codex_cli()))
        if shutil.which("gemini"):
            update_tasks.append(("gemini", _update_gemini_cli()))

        if update_tasks:
            names = [name for name, _ in update_tasks]
            coros = [coro for _, coro in update_tasks]
            logger.info("Updating CLIs: %s", ", ".join(names))

            outcomes = await asyncio.gather(*coros, return_exceptions=True)
            for name, outcome in zip(names, outcomes):
                if isinstance(outcome, BaseException):
                    msg = f"{name} update error: {outcome}"
                    logger.warning(msg)
                    result.errors.append(msg)
                else:
                    success, message = outcome
                    if success:
                        result.updated_clis.append(name)
                    else:
                        logger.info("%s CLI update skipped: %s", name, message)

    # ── Phase 2: Model discovery (parallel, best-effort) ─────
    discovery_tasks: list[tuple[str, Any]] = []
    if shutil.which("codex") or (Path.home() / ".codex" / "models_cache.json").exists():
        discovery_tasks.append(("codex", _discover_codex_models()))
    if shutil.which("npm"):
        discovery_tasks.append(("gemini", _discover_gemini_models()))
    if shutil.which("claude"):
        discovery_tasks.append(("claude", _discover_claude_models()))

    if discovery_tasks:
        names = [name for name, _ in discovery_tasks]
        coros = [coro for _, coro in discovery_tasks]
        logger.info("Discovering models from: %s", ", ".join(names))

        outcomes = await asyncio.gather(*coros, return_exceptions=True)
        for name, outcome in zip(names, outcomes):
            if isinstance(outcome, BaseException):
                msg = f"{name} discovery error: {outcome}"
                logger.warning(msg)
                result.errors.append(msg)
            else:
                for dm in outcome:
                    result.discovered_models[dm.model_id] = dm
    else:
        logger.info("No CLI tools found — skipping model discovery")

    logger.info(
        "Discovery complete: %d models found, %d CLIs updated, %d errors",
        len(result.discovered_models),
        len(result.updated_clis),
        len(result.errors),
    )

    # ── Phase 3: Update models.yaml ──────────────────────────
    if result.discovered_models:
        try:
            filtered_models = _dedupe_by_display_label(result.discovered_models)
            logger.info(
                "Applying dropdown-style dedupe: %d → %d models",
                len(result.discovered_models),
                len(filtered_models),
            )
            result.models_yaml_updated = await _update_models_yaml(
                filtered_models,
                models_yaml_path,
                force_overwrite=force_overwrite,
            )
            if (
                sync_global_models_yaml
                and models_yaml_path.resolve() != DEFAULT_MODELS_YAML_PATH.resolve()
            ):
                global_updated = await _update_models_yaml(
                    result.discovered_models,
                    DEFAULT_MODELS_YAML_PATH,
                    force_overwrite=force_overwrite,
                )
                result.models_yaml_updated = (
                    result.models_yaml_updated or global_updated
                )
        except Exception as exc:
            msg = f"models.yaml update error: {exc}"
            logger.error(msg, exc_info=True)
            result.errors.append(msg)

    return result
