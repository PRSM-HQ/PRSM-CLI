"""Live integration tests for non-Claude master providers.

Tests that GPT-5.2 (Codex), MiniMax, and Gemini-3 can each work
as the master orchestrator with proper tool call event emission.

These tests require the actual CLI tools to be installed and configured:
- codex CLI (for GPT-5.2 and MiniMax)
- gemini CLI (for Gemini-3)
- MINIMAX_API_KEY env var (for MiniMax)

Usage:
    python tests/test_provider_master.py [codex|minimax|gemini|all]
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prsm.engine.config import EngineConfig
from prsm.engine.engine import OrchestrationEngine
from prsm.engine.yaml_config import load_yaml_config
from prsm.engine.providers.registry import build_provider_registry
from prsm.engine.model_registry import build_default_registry, load_model_registry_from_yaml


# Collect events for verification
_events: list[dict] = []


async def _event_collector(event: dict) -> None:
    """Collect all events for post-run analysis."""
    _events.append(event)
    etype = event.get("event", "")
    agent_id = event.get("agent_id", "")[:8]

    if etype == "tool_call_started":
        tool = event.get("tool_name", "")
        args = event.get("arguments", "")[:100]
        print(f"  🔧 [{agent_id}] TOOL_START: {tool}({args})")
    elif etype == "tool_call_completed":
        tool_id = event.get("tool_id", "")[:12]
        is_err = event.get("is_error", False)
        result = event.get("result", "")[:80]
        status = "❌" if is_err else "✅"
        print(f"  {status} [{agent_id}] TOOL_END: {result}")
    elif etype == "stream_chunk":
        text = event.get("text", "").strip()
        if text and len(text) < 200:
            print(f"  📝 [{agent_id}] {text}")
    elif etype == "agent_spawned":
        role = event.get("role", "")
        print(f"  🚀 [{agent_id}] SPAWNED (role={role})")
    elif etype == "agent_result":
        is_err = event.get("is_error", False)
        result = event.get("result", "")[:100]
        status = "❌" if is_err else "✅"
        print(f"  {status} [{agent_id}] RESULT: {result}")


def _analyze_events(provider_name: str) -> dict:
    """Analyze collected events and report on tool call visibility."""
    tool_starts = [e for e in _events if e.get("event") == "tool_call_started"]
    tool_ends = [e for e in _events if e.get("event") == "tool_call_completed"]
    stream_chunks = [e for e in _events if e.get("event") == "stream_chunk"]
    agent_spawns = [e for e in _events if e.get("event") == "agent_spawned"]
    agent_results = [e for e in _events if e.get("event") == "agent_result"]

    # Extract unique tool names from starts
    tools_used = list({e.get("tool_name", "") for e in tool_starts})

    # Check for orchestration tools
    orch_tools = [t for t in tools_used if t in (
        "spawn_child", "spawn_children_parallel", "ask_user", "ask_parent",
        "consult_expert", "wait_for_message", "respond_to_child",
        "task_complete", "report_progress", "get_children_status",
        "check_child_status", "get_child_history", "send_child_prompt",
        "recommend_model",
    )]

    # Check for CLI tools
    cli_tools = [t for t in tools_used if t in (
        "Bash", "Read", "Write", "Edit", "Glob", "Grep",
        "WebSearch", "WebFetch",
    )]

    return {
        "provider": provider_name,
        "total_events": len(_events),
        "tool_call_starts": len(tool_starts),
        "tool_call_ends": len(tool_ends),
        "stream_chunks": len(stream_chunks),
        "agent_spawns": len(agent_spawns),
        "agent_results": len(agent_results),
        "tools_used": tools_used,
        "orch_tools_visible": orch_tools,
        "cli_tools_visible": cli_tools,
        "has_tool_visibility": len(tool_starts) > 0,
        "has_orch_tools": len(orch_tools) > 0,
    }


async def run_provider_test(
    provider_name: str,
    model_alias: str,
    model_id: str,
    task: str,
    config_path: str = ".prism/prsm.yaml",
) -> dict:
    """Test a provider as master and return analysis."""
    global _events
    _events = []

    print(f"\n{'='*60}")
    print(f"Testing {provider_name} (model={model_alias}/{model_id}) as master")
    print(f"Task: {task}")
    print(f"{'='*60}\n")

    # Load YAML config
    yaml_cfg = load_yaml_config(config_path)
    if yaml_cfg is None:
        print(f"❌ Failed to load {config_path}")
        return {"provider": provider_name, "error": "no config"}

    # Build engine config from YAML
    config = yaml_cfg.engine
    config.master_model = model_id
    config.master_provider = provider_name
    config.event_callback = _event_collector
    config.agent_timeout_seconds = 300
    config.tool_call_timeout_seconds = 120

    # Build provider registry from YAML
    provider_registry = build_provider_registry(yaml_cfg.providers)

    # Build model registry
    model_registry = build_default_registry()
    if hasattr(yaml_cfg, "model_registry_raw") and yaml_cfg.model_registry_raw:
        model_registry = load_model_registry_from_yaml(
            yaml_cfg.model_registry_raw, model_registry
        )
    # Sync availability
    provider_report = provider_registry.get_availability_report()
    model_registry.sync_availability(provider_report)
    # Probe which Claude models are actually accessible
    probe_changed = await model_registry.probe_claude_models()
    if probe_changed:
        print(f"  ⚠️ Claude models not accessible: {', '.join(probe_changed.keys())}")
    config.model_registry = model_registry

    engine = OrchestrationEngine(
        config=config,
        provider_registry=provider_registry,
    )

    # Register experts from YAML
    if hasattr(yaml_cfg, "experts"):
        for expert in yaml_cfg.experts:
            engine.register_expert(expert)

    start = time.monotonic()
    elapsed = 0.0
    try:
        result = await asyncio.wait_for(
            engine.run(
                task_definition=task,
                master_model=model_id,
            ),
            timeout=300,
        )
        elapsed = time.monotonic() - start
        print(f"\n✅ {provider_name} completed in {elapsed:.1f}s")
        print(f"Result: {str(result)[:300]}")
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start
        print(f"\n⏱️ {provider_name} timed out after {elapsed:.1f}s")
    except Exception as exc:
        elapsed = time.monotonic() - start
        print(f"\n❌ {provider_name} failed after {elapsed:.1f}s: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        await engine.shutdown()

    analysis = _analyze_events(provider_name)
    analysis["elapsed_seconds"] = round(elapsed, 1)

    print(f"\n--- Event Analysis for {provider_name} ---")
    print(f"  Total events: {analysis['total_events']}")
    print(f"  Tool starts: {analysis['tool_call_starts']}")
    print(f"  Tool ends: {analysis['tool_call_ends']}")
    print(f"  Stream chunks: {analysis['stream_chunks']}")
    print(f"  Agent spawns: {analysis['agent_spawns']}")
    print(f"  Tools visible: {analysis['tools_used']}")
    print(f"  Orch tools: {analysis['orch_tools_visible']}")
    print(f"  CLI tools: {analysis['cli_tools_visible']}")
    print(f"  Has tool visibility: {analysis['has_tool_visibility']}")

    return analysis


# Task that requires orchestration tools
READ_TASK = (
    "Read the file .prism/prsm.yaml and summarize what providers are configured. "
    "Spawn a child agent to read the file, then summarize the findings with task_complete."
)


async def main():
    providers_to_test = sys.argv[1] if len(sys.argv) > 1 else "all"

    results = {}

    if providers_to_test in ("codex", "all"):
        results["codex"] = await run_provider_test(
            provider_name="codex",
            model_alias="codex",
            model_id="gpt-5.2-codex",
            task=READ_TASK,
        )

    if providers_to_test in ("minimax", "all"):
        results["minimax"] = await run_provider_test(
            provider_name="minimax",
            model_alias="minimax",
            model_id="MiniMax-M2.5",
            task=READ_TASK,
        )

    if providers_to_test in ("gemini", "all"):
        results["gemini"] = await run_provider_test(
            provider_name="gemini",
            model_alias="gemini-flash",
            model_id="gemini-2.5-flash",
            task=READ_TASK,
        )

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, analysis in results.items():
        if "error" in analysis:
            print(f"  {name}: ❌ ERROR: {analysis['error']}")
            continue
        visibility = "✅ VISIBLE" if analysis["has_tool_visibility"] else "❌ OPAQUE"
        orch = "✅" if analysis["has_orch_tools"] else "❌"
        print(
            f"  {name}: {visibility} | "
            f"orch_tools={orch} | "
            f"tool_starts={analysis['tool_call_starts']} | "
            f"elapsed={analysis['elapsed_seconds']}s"
        )


if __name__ == "__main__":
    asyncio.run(main())
