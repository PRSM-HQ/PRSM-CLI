"""Master-agent MCP tool definitions for Claude Code.

These tools are the master-specific subset exposed via the stdio
MCP server. The user's Claude Code session IS the master agent.

Exposed:
    spawn_child, spawn_children_parallel, restart_child,
    consult_expert, wait_for_message, respond_to_child,
    list_agents, kill_agent, list_experts

NOT exposed (child-only):
    ask_parent, report_progress, task_complete
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP, Context


def _extract_text(result: dict[str, Any]) -> str:
    """Convert OrchestrationTools response to plain string.

    OrchestrationTools returns {"content": [{"type": "text", "text": ...}]}.
    FastMCP tool functions return plain strings.
    If is_error is set, raise ValueError so FastMCP marks it as error.
    """
    text = result["content"][0]["text"]
    if result.get("is_error"):
        raise ValueError(text.removeprefix("ERROR: "))
    return text


def _get_orch(ctx: Context):
    """Get OrchestrationTools from lifespan context."""
    return ctx.request_context.lifespan_context["orch_tools"]


def _get_manager(ctx: Context):
    """Get AgentManager from lifespan context."""
    return ctx.request_context.lifespan_context["manager"]


def _get_registry(ctx: Context):
    """Get ExpertRegistry from lifespan context."""
    return ctx.request_context.lifespan_context["expert_registry"]


def register_tools(mcp: FastMCP) -> None:
    """Register all master-agent tools with the FastMCP instance."""

    @mcp.tool(
        name="spawn_child",
        description=(
            "Spawn a child agent to work on a task. Tasks are NOT limited "
            "to writing code — a task can be deep exploration and analysis "
            "of existing code, researching a question, describing an "
            "implementation, reviewing for bugs, or any focused work that "
            "returns a summary. Delegating research and exploration to "
            "children preserves your context quality. "
            "The child runs as a headless Claude session with its own tools. "
            "Child launches are always non-blocking and return immediately. "
            "The wait parameter is accepted for compatibility but ignored. "
            "Use wait_for_message + respond_to_child to communicate with "
            "interactive children. "
            "Use 'complexity' for automatic model selection: 'trivial' uses "
            "the cheapest model, 'simple' uses fast models, 'complex' uses "
            "strong models, 'frontier' uses the best available."
        ),
    )
    async def spawn_child(
        prompt: str,
        wait: bool = False,
        tools: list[str] | None = None,
        model: str | None = None,
        cwd: str | None = None,
        complexity: str | None = None,
        ctx: Context = None,
    ) -> str:
        orch = _get_orch(ctx)
        result = await orch.spawn_child(
            prompt, wait, tools, model, cwd, complexity=complexity,
        )
        return _extract_text(result)

    @mcp.tool(
        name="spawn_children_parallel",
        description=(
            "Spawn multiple child agents in parallel (non-blocking). "
            "Returns immediately with child IDs. Each child spec is a dict "
            "with: prompt (required), tools, model, cwd. "
            "Children can perform any focused task: code writing, exploration, "
            "analysis, research, review, or documentation."
        ),
    )
    async def spawn_children_parallel(
        children: list[dict[str, Any]],
        ctx: Context = None,
    ) -> str:
        orch = _get_orch(ctx)
        result = await orch.spawn_children_parallel(children)
        return _extract_text(result)

    @mcp.tool(
        name="restart_child",
        description=(
            "Restart a completed or failed child agent with a new task. "
            "Reuses the child's identity (ID, model, tools, working "
            "directory) but runs a fresh session with the new prompt. "
            "Use this instead of spawn_child when you want to send "
            "follow-up work to the same agent. Restarts are always "
            "non-blocking and return immediately. The wait parameter is "
            "accepted for compatibility but ignored."
        ),
    )
    async def restart_child(
        child_agent_id: str,
        prompt: str,
        wait: bool = False,
        ctx: Context = None,
    ) -> str:
        orch = _get_orch(ctx)
        result = await orch.restart_child(child_agent_id, prompt, wait)
        return _extract_text(result)

    @mcp.tool(
        name="consult_expert",
        description=(
            "Consult a specialist expert agent. The expert runs as a "
            "separate Claude session with domain-specific system prompt "
            "and tools. Experts can answer questions, explore code in "
            "their domain, or provide analysis — not just write code. "
            "Use list_experts to see available expert IDs."
        ),
    )
    async def consult_expert(
        expert_id: str,
        question: str,
        ctx: Context = None,
    ) -> str:
        orch = _get_orch(ctx)
        result = await orch.consult_expert(expert_id, question)
        return _extract_text(result)

    @mcp.tool(
        name="wait_for_message",
        description=(
            "Wait for the next incoming message from any child agent. "
            "Returns the message type, source agent ID, correlation ID, "
            "and payload. Use after spawning children with wait=false. "
            "Message types: question (child asking you something), "
            "task_result (child finished), progress_update (status update). "
            "If the timeout expires, returns a timeout notice — this is "
            "NORMAL and means children are still working. You MUST call "
            "wait_for_message() again in a loop until you receive "
            "task_result from each child. Do NOT start doing the work "
            "yourself on timeout."
        ),
    )
    async def wait_for_message(
        timeout_seconds: float = 0.0,
        ctx: Context = None,
    ) -> str:
        orch = _get_orch(ctx)
        result = await orch.wait_for_message(timeout_seconds)
        return _extract_text(result)

    @mcp.tool(
        name="respond_to_child",
        description=(
            "Send a response to a child agent's question. Use the "
            "child_agent_id and correlation_id from the message received "
            "via wait_for_message."
        ),
    )
    async def respond_to_child(
        child_agent_id: str,
        correlation_id: str,
        response: str,
        ctx: Context = None,
    ) -> str:
        orch = _get_orch(ctx)
        result = await orch.respond_to_child(
            child_agent_id, correlation_id, response
        )
        return _extract_text(result)

    @mcp.tool(
        name="list_agents",
        description=(
            "List all active agent sessions with their state, role, "
            "depth, and parent. Useful for monitoring orchestration."
        ),
    )
    async def list_agents(ctx: Context = None) -> str:
        manager = _get_manager(ctx)
        descriptors = manager.get_all_descriptors()
        if not descriptors:
            return "No agents currently registered."

        lines = []
        for d in descriptors:
            parent = (
                d.parent_id[:8] + "..."
                if d.parent_id
                else "none"
            )
            lines.append(
                f"- {d.agent_id[:12]}  "
                f"role={d.role.value:<8} "
                f"state={d.state.value:<20} "
                f"depth={d.depth} "
                f"parent={parent} "
                f"children={len(d.children)}"
            )
        return (
            f"Active agents ({len(descriptors)}):\n"
            + "\n".join(lines)
        )

    @mcp.tool(
        name="kill_agent",
        description=(
            "Force-kill an agent and all its children. Use when an "
            "agent is stuck or no longer needed."
        ),
    )
    async def kill_agent(
        agent_id: str,
        ctx: Context = None,
    ) -> str:
        manager = _get_manager(ctx)
        desc = manager.get_descriptor(agent_id)
        if desc is None:
            raise ValueError(f"Agent {agent_id} not found")
        await manager.kill_agent(agent_id)
        return (
            f"Agent {agent_id[:12]} and its children have been killed."
        )

    @mcp.tool(
        name="list_experts",
        description=(
            "List all registered expert profiles with their IDs, "
            "names, and descriptions. Use expert_id with consult_expert."
        ),
    )
    async def list_experts(ctx: Context = None) -> str:
        registry = _get_registry(ctx)
        profiles = registry.list_profiles()
        if not profiles:
            return "No experts registered."

        lines = []
        for p in profiles:
            lines.append(
                f"- **{p.expert_id}** ({p.name}): "
                f"{p.description}\n"
                f"  model={p.model}, tools={p.tools}"
            )
        return (
            f"Available experts ({len(profiles)}):\n"
            + "\n".join(lines)
        )

    @mcp.tool(
        name="get_child_history",
        description=(
            "Review a child agent's conversation history. Returns "
            "structured transcript as JSON. Use detail_level='summary' "
            "for text + tool names only, or 'full' for everything "
            "including thinking blocks and tool arguments."
        ),
    )
    async def get_child_history(
        child_agent_id: str,
        detail_level: str = "full",
        ctx: Context = None,
    ) -> str:
        orch = _get_orch(ctx)
        result = await orch.get_child_history(child_agent_id, detail_level)
        return _extract_text(result)

    @mcp.tool(
        name="check_child_status",
        description=(
            "Check a child agent's current state, error info, "
            "timestamps, and children count. Works for both active "
            "and completed agents."
        ),
    )
    async def check_child_status(
        child_agent_id: str,
        ctx: Context = None,
    ) -> str:
        orch = _get_orch(ctx)
        result = await orch.check_child_status(child_agent_id)
        return _extract_text(result)

    @mcp.tool(
        name="send_child_prompt",
        description=(
            "Send a prompt or instruction to a running child agent. "
            "The prompt is delivered via the message router."
        ),
    )
    async def send_child_prompt(
        child_agent_id: str,
        prompt: str,
        ctx: Context = None,
    ) -> str:
        orch = _get_orch(ctx)
        result = await orch.send_child_prompt(child_agent_id, prompt)
        return _extract_text(result)

    @mcp.tool(
        name="consult_peer",
        description=(
            "Consult a peer AI provider (e.g. OpenAI Codex, Gemini, MiniMax) "
            "for a second opinion on design decisions, architecture, or "
            "implementation approaches. Supports multi-turn conversations: "
            "the first call returns a thread_id, pass it back for "
            "follow-up questions to continue the dialogue. "
            "Pass peer='list' to see available peer models, or "
            "peer='alias' to consult a specific peer."
        ),
    )
    async def consult_peer(
        question: str,
        thread_id: str | None = None,
        peer: str | None = None,
        ctx: Context = None,
    ) -> str:
        orch = _get_orch(ctx)
        result = await orch.consult_peer(question, thread_id, peer)
        return _extract_text(result)

    @mcp.tool(
        name="recommend_model",
        description=(
            "Get a model recommendation for a task from the capability "
            "registry. Analyzes the task description and complexity to "
            "suggest the optimal model. Use this to decide which model "
            "to pass to spawn_child, or use the 'complexity' parameter "
            "in spawn_child directly for automatic selection."
        ),
    )
    async def recommend_model(
        task_description: str,
        complexity: str = "medium",
        ctx: Context = None,
    ) -> str:
        orch = _get_orch(ctx)
        result = await orch.recommend_model(task_description, complexity)
        return _extract_text(result)

    @mcp.tool(
        name="get_model_rankings",
        description=(
            "View the learned model rankings from the background research "
            "agent. Shows which models are best for each task category, "
            "ranked with scores and fallback options. The rankings are "
            "automatically updated daily and persist across restarts. "
            "Optionally filter to a specific task_category like 'coding', "
            "'exploration', 'architecture', etc."
        ),
    )
    async def get_model_rankings(
        task_category: str | None = None,
        ctx: Context = None,
    ) -> str:
        orch = _get_orch(ctx)
        result = await orch.get_model_rankings(task_category)
        return _extract_text(result)

    @mcp.tool(
        name="list_available_models",
        description=(
            "List all registered models and their availability status. "
            "Shows which models can be used (provider installed) and "
            "which are unavailable. Use this to check what models you "
            "can pass to spawn_child. Only models configured in "
            "'defaults.peer_models' will be shown if peer models are configured."
        ),
    )
    async def list_available_models(ctx: Context = None) -> str:
        orch = _get_orch(ctx)
        model_registry = orch._model_registry
        if not model_registry:
            return "No model registry configured."

        # Get allowed peer model IDs
        allowed_peer_ids = orch._get_allowed_peer_model_ids()

        available = model_registry.list_available()
        unavailable = model_registry.list_unavailable()

        # Filter by peer_models if configured
        if allowed_peer_ids is not None:
            available = [m for m in available if m.model_id in allowed_peer_ids]
            unavailable = [m for m in unavailable if m.model_id in allowed_peer_ids]

        lines = []
        if allowed_peer_ids is not None:
            lines.append("NOTE: Only models configured in 'defaults.peer_models' are shown.\n")

        if available:
            lines.append(f"Available models for child agents ({len(available)}):")
            for m in available:
                top = sorted(m.affinities.items(), key=lambda x: -x[1])[:3]
                strengths = ", ".join(f"{k}" for k, v in top)
                lines.append(
                    f"  - {m.model_id} ({m.provider}, {m.tier.value}) "
                    f"— best for: {strengths}"
                )

        if unavailable:
            lines.append(f"\nUnavailable models ({len(unavailable)}):")
            for m in unavailable:
                lines.append(
                    f"  - {m.model_id} ({m.provider}) "
                    f"— provider not installed"
                )

        if not available and not unavailable:
            if allowed_peer_ids:
                return "No models from the configured peer_models are registered."
            else:
                return "No models registered."

        return "\n".join(lines)
