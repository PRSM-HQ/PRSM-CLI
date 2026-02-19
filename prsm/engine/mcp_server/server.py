"""Per-agent MCP server factory.

Each agent gets its own in-process MCP server instance with orchestration
tools bound to that agent's ID. Tool handlers are async closures that
route through the MessageRouter.

The MCP server is passed directly to ClaudeAgentOptions.mcp_servers
as an McpSdkServerConfig — no subprocess, no HTTP, no port allocation.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .tools import OrchestrationTools, ToolTimeTracker

if TYPE_CHECKING:
    from ..agent_manager import AgentManager
    from ..message_router import MessageRouter
    from ..expert_registry import ExpertRegistry


# Tool names that agents can use (must match @tool decorator names)
ORCHESTRATION_TOOL_NAMES = [
    "ask_parent",
    "ask_user",
    "spawn_child",
    "spawn_children_parallel",
    "restart_child",
    "consult_expert",
    "report_progress",
    "task_complete",
    "wait_for_message",
    "respond_to_child",
    "consult_peer",
    "get_child_history",
    "check_child_status",
    "send_child_prompt",
    "get_children_status",
    "recommend_model",
    "run_bash",
]


def build_agent_mcp_config(
    agent_id: str,
    manager: AgentManager,
    router: MessageRouter,
    expert_registry: ExpertRegistry,
    tool_call_timeout: float = 7200.0,
    default_model: str = "claude-opus-4-6",
    default_provider: str = "claude",
    provider_registry: object | None = None,
    peer_model: str | None = None,
    peer_provider: object | None = None,
    user_question_callback: object | None = None,
    conversation_store: object | None = None,
    peer_models: dict | None = None,
    model_registry: object | None = None,
    permission_callback: object | None = None,
    cwd: str | None = None,
    event_callback: object | None = None,
) -> tuple[dict[str, Any], OrchestrationTools]:
    """Build MCP server config for ClaudeAgentOptions.mcp_servers.

    Returns a tuple of:
    - mcp_servers dict suitable for ClaudeAgentOptions(mcp_servers=...)
    - OrchestrationTools instance (for accessing ToolTimeTracker)
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool

    orch = OrchestrationTools(
        agent_id=agent_id,
        manager=manager,
        router=router,
        expert_registry=expert_registry,
        tool_call_timeout=tool_call_timeout,
        default_model=default_model,
        default_provider=default_provider,
        provider_registry=provider_registry,
        peer_model=peer_model,
        peer_provider=peer_provider,
        user_question_callback=user_question_callback,
        conversation_store=conversation_store,
        peer_models=peer_models,
        model_registry=model_registry,
        permission_callback=permission_callback,
        cwd=cwd,
        event_callback=event_callback,
    )

    # Build @tool-decorated wrappers that delegate to OrchestrationTools

    @tool("ask_parent", "Ask your parent agent a question and wait for an answer. Use when you need clarification or information only your parent has.", {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question to ask your parent agent"},
        },
        "required": ["question"],
    })
    async def ask_parent(args: dict[str, Any]) -> dict[str, Any]:
        return await orch.ask_parent(args["question"])

    @tool("spawn_child", "Spawn a child agent for a focused task. Child launches are always non-blocking and return immediately with child_id. Use wait_for_message() to collect results and respond_to_child() to answer questions. The wait parameter is accepted for compatibility but ignored. Use mcp_servers to give the child specific MCP plugins, or exclude_plugins to remove global ones.", {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Task description for the child agent"},
            "wait": {"type": "boolean", "description": "Compatibility flag only. Ignored: child launches are always non-blocking and return immediately with child_id.", "default": False},
            "tools": {"type": "array", "items": {"type": "string"}, "description": "Tool allowlist for the child (default: Read, Write, Edit, Bash, Glob, Grep)"},
            "model": {"type": "string", "description": "Model override for the child agent. If not specified, the system will auto-select based on complexity."},
            "cwd": {"type": "string", "description": "Working directory for the child agent"},
            "mcp_servers": {"type": "object", "description": "MCP server configs for this child only. Keys are server names, values are config objects ({type, command/url, args, headers}).", "additionalProperties": True},
            "exclude_plugins": {"type": "array", "items": {"type": "string"}, "description": "Plugin names to exclude from the global set for this child."},
            "complexity": {"type": "string", "description": "Task complexity for smart model selection: 'trivial' (use cheapest), 'simple' (use fast), 'medium' (default), 'complex' (use strong+), 'frontier' (use best). When set, the system auto-selects the optimal model.", "enum": ["trivial", "simple", "medium", "complex", "frontier"]},
        },
        "required": ["prompt"],
    })
    async def spawn_child(args: dict[str, Any]) -> dict[str, Any]:
        return await orch.spawn_child(
            prompt=args["prompt"],
            wait=args.get("wait", False),
            tools=args.get("tools"),
            model=args.get("model"),
            cwd=args.get("cwd"),
            mcp_servers=args.get("mcp_servers"),
            exclude_plugins=args.get("exclude_plugins"),
            complexity=args.get("complexity"),
        )

    @tool("spawn_children_parallel", "Spawn multiple child agents simultaneously in the background (non-blocking). Returns immediately with child IDs. You MUST then use wait_for_message() in a loop to collect TASK_RESULT messages from each child and handle any QUESTION messages with respond_to_child(). Each child spec is a dict with: prompt (required), tools, model, cwd, mcp_servers, exclude_plugins, complexity.", {
        "type": "object",
        "properties": {
            "children": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "tools": {"type": "array", "items": {"type": "string"}},
                        "model": {"type": "string"},
                        "cwd": {"type": "string"},
                        "mcp_servers": {"type": "object", "description": "MCP server configs for this child.", "additionalProperties": True},
                        "exclude_plugins": {"type": "array", "items": {"type": "string"}, "description": "Plugin names to exclude."},
                        "complexity": {"type": "string", "description": "Task complexity for smart model selection: trivial/simple/medium/complex/frontier", "enum": ["trivial", "simple", "medium", "complex", "frontier"]},
                    },
                    "required": ["prompt"],
                },
                "description": "List of child agent specifications",
            },
        },
        "required": ["children"],
    })
    async def spawn_children_parallel(args: dict[str, Any]) -> dict[str, Any]:
        return await orch.spawn_children_parallel(args["children"])

    @tool("restart_child", "Restart a completed or failed child agent with a new task. Reuses the child's identity, model, tools, and working directory. Use this instead of spawn_child when you want to send follow-up work to the same agent.", {
        "type": "object",
        "properties": {
            "child_agent_id": {"type": "string", "description": "ID of the completed/failed child agent to restart"},
            "prompt": {"type": "string", "description": "New task description for the child"},
            "wait": {"type": "boolean", "description": "Compatibility flag only. Ignored: restart launches are always non-blocking and return immediately.", "default": False},
        },
        "required": ["child_agent_id", "prompt"],
    })
    async def restart_child(args: dict[str, Any]) -> dict[str, Any]:
        return await orch.restart_child(
            child_agent_id=args["child_agent_id"],
            prompt=args["prompt"],
            wait=args.get("wait", False),
        )

    @tool("consult_expert", "Consult a specialist expert agent for domain-specific advice. Use list_experts to see available expert IDs.", {
        "type": "object",
        "properties": {
            "expert_id": {"type": "string", "description": "ID of the expert to consult"},
            "question": {"type": "string", "description": "Question for the expert"},
        },
        "required": ["expert_id", "question"],
    })
    async def consult_expert(args: dict[str, Any]) -> dict[str, Any]:
        return await orch.consult_expert(args["expert_id"], args["question"])

    @tool("report_progress", "Send a non-blocking progress update to your parent agent.", {
        "type": "object",
        "properties": {
            "status": {"type": "string", "description": "Progress status message"},
            "percent_complete": {"type": "integer", "description": "Percentage complete (0-100)", "default": 0},
        },
        "required": ["status"],
    })
    async def report_progress(args: dict[str, Any]) -> dict[str, Any]:
        return await orch.report_progress(
            args["status"], args.get("percent_complete", 0)
        )

    @tool("task_complete", "Signal that your assigned task is complete. ALWAYS call this when your task is done. For substantial work, include a comprehensive user-facing summary of what you did, where you changed things, what you verified, remaining risks, and recommended next steps.", {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "User-facing completion summary. For substantial tasks include: what changed, where, verification, risks/gaps, and recommended next steps."},
            "artifacts": {"type": "object", "description": "Optional artifacts (file paths, data, etc.)"},
            "steps": {"type": "array", "items": {"type": "string"}, "description": "Optional implementation steps completed"},
            "assumptions": {"type": "array", "items": {"type": "string"}, "description": "Optional assumptions made"},
            "risks": {"type": "array", "items": {"type": "string"}, "description": "Optional remaining risks"},
            "rollback_plan": {"type": "string", "description": "Optional rollback plan"},
            "confidence": {"type": "number", "description": "Optional confidence score (0.0-1.0)"},
            "verification_results": {"type": "array", "items": {"type": "object"}, "description": "Optional verification check results"},
        },
        "required": ["summary"],
    })
    async def task_complete(args: dict[str, Any]) -> dict[str, Any]:
        return await orch.task_complete(
            summary=args["summary"],
            artifacts=args.get("artifacts"),
            steps=args.get("steps"),
            assumptions=args.get("assumptions"),
            risks=args.get("risks"),
            rollback_plan=args.get("rollback_plan"),
            confidence=args.get("confidence"),
            verification_results=args.get("verification_results"),
        )

    @tool("wait_for_message", "Wait for the next incoming routed message (from children if you're a master, or from your parent if you're a worker). Returns message type, source agent ID, correlation ID, and payload. If the timeout expires, returns a timeout notice — this is NORMAL and means children are still working. You MUST call wait_for_message() again in a loop until you receive TASK_RESULT from each child. Do NOT start doing the work yourself on timeout.", {
        "type": "object",
        "properties": {
            "timeout_seconds": {"type": "number", "description": "Max seconds to wait (0 disables timeout)", "default": 0},
        },
    })
    async def wait_for_message(args: dict[str, Any]) -> dict[str, Any]:
        return await orch.wait_for_message(
            args.get("timeout_seconds", 0.0)
        )

    @tool("respond_to_child", "Send a response to a child agent's ask_parent question. Use the child_agent_id and correlation_id from the message received via wait_for_message.", {
        "type": "object",
        "properties": {
            "child_agent_id": {"type": "string", "description": "ID of the child agent to respond to"},
            "correlation_id": {"type": "string", "description": "Correlation ID from the child's question"},
            "response": {"type": "string", "description": "Your response to the child's question"},
        },
        "required": ["child_agent_id", "correlation_id", "response"],
    })
    async def respond_to_child(args: dict[str, Any]) -> dict[str, Any]:
        return await orch.respond_to_child(
            args["child_agent_id"],
            args["correlation_id"],
            args["response"],
        )

    @tool("consult_peer", "Consult a peer AI provider (e.g. OpenAI Codex, Gemini, MiniMax) for a second opinion. Supports multi-turn via thread_id. Pass peer='list' to see available peers, or peer='alias' to consult a specific peer.", {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Question for the peer provider"},
            "thread_id": {"type": "string", "description": "Thread ID for follow-up questions"},
            "peer": {"type": "string", "description": "Peer alias to consult (e.g. 'codex', 'gemini', 'minimax'). Use 'list' to see available peers. Default: primary peer."},
        },
        "required": ["question"],
    })
    async def consult_peer(args: dict[str, Any]) -> dict[str, Any]:
        return await orch.consult_peer(
            args["question"], args.get("thread_id"), args.get("peer"),
        )

    @tool("ask_user", "Ask the user a question with clickable options. Use when you need the user to make a decision, choose between approaches, or provide input. Options should have label and description fields.", {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question to ask the user"},
            "options": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "Short option label (1-5 words)"},
                        "description": {"type": "string", "description": "What this option means"},
                    },
                    "required": ["label", "description"],
                },
                "description": "Clickable options for the user to choose from (2-5 options)",
            },
        },
        "required": ["question"],
    })
    async def ask_user(args: dict[str, Any]) -> dict[str, Any]:
        return await orch.ask_user(
            args["question"], args.get("options", []),
        )

    @tool("get_child_history", "Review a child agent's conversation history. Returns structured transcript as JSON. Use detail_level='summary' for text + tool names only, or 'full' for everything including thinking and tool arguments.", {
        "type": "object",
        "properties": {
            "child_agent_id": {"type": "string", "description": "ID of the child agent whose history to retrieve"},
            "detail_level": {"type": "string", "description": "Level of detail: 'full' or 'summary'", "default": "full", "enum": ["full", "summary"]},
        },
        "required": ["child_agent_id"],
    })
    async def get_child_history(args: dict[str, Any]) -> dict[str, Any]:
        return await orch.get_child_history(
            args["child_agent_id"],
            args.get("detail_level", "full"),
        )

    @tool("check_child_status", "Check a child agent's current state, error, timestamps, and children count.", {
        "type": "object",
        "properties": {
            "child_agent_id": {"type": "string", "description": "ID of the child agent to check"},
        },
        "required": ["child_agent_id"],
    })
    async def check_child_status(args: dict[str, Any]) -> dict[str, Any]:
        return await orch.check_child_status(args["child_agent_id"])

    @tool("send_child_prompt", "Send a prompt or instruction to a child agent.", {
        "type": "object",
        "properties": {
            "child_agent_id": {"type": "string", "description": "ID of the child agent to send the prompt to"},
            "prompt": {"type": "string", "description": "The prompt or instruction to send"},
        },
        "required": ["child_agent_id", "prompt"],
    })
    async def send_child_prompt(args: dict[str, Any]) -> dict[str, Any]:
        return await orch.send_child_prompt(
            args["child_agent_id"], args["prompt"],
        )

    @tool("get_children_status", "Get the current status of all children spawned by you. Non-blocking — returns each child's state (running, completed, failed) and a summary count. Use this to check progress without waiting.", {
        "type": "object",
        "properties": {},
    })
    async def get_children_status(args: dict[str, Any]) -> dict[str, Any]:
        return await orch.get_children_status()

    @tool("recommend_model", "Get a model recommendation for a task from the capability registry. Returns the best model based on task type and complexity.", {
        "type": "object",
        "properties": {
            "task_description": {"type": "string", "description": "Description of the task to find a model for"},
            "complexity": {"type": "string", "description": "Task complexity: 'trivial', 'simple', 'medium', 'complex', or 'frontier'", "default": "medium", "enum": ["trivial", "simple", "medium", "complex", "frontier"]},
        },
        "required": ["task_description"],
    })
    async def recommend_model(args: dict[str, Any]) -> dict[str, Any]:
        return await orch.recommend_model(
            args["task_description"], args.get("complexity", "medium"),
        )

    @tool("run_bash", "Execute a bash command with permission checking. Dangerous commands (rm, sudo, etc.) require explicit user approval before execution. Use this for ALL shell commands.", {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The bash command to execute"},
            "timeout": {"type": "integer", "description": "Optional timeout in seconds (default 120)"},
            "cwd": {"type": "string", "description": "Optional working directory"},
        },
        "required": ["command"],
    })
    async def run_bash(args: dict[str, Any]) -> dict[str, Any]:
        return await orch.run_bash(
            args["command"],
            timeout=args.get("timeout"),
            cwd=args.get("cwd"),
            tool_call_id=args.get("tool_call_id") or args.get("_tool_call_id"),
        )

    # Create the in-process MCP server
    sdk_tools = [
        ask_parent, ask_user, spawn_child, spawn_children_parallel,
        restart_child, consult_expert, report_progress, task_complete,
        wait_for_message, respond_to_child, consult_peer,
        get_child_history, check_child_status, send_child_prompt,
        get_children_status, recommend_model, run_bash,
    ]
    server_config = create_sdk_mcp_server("orchestrator", tools=sdk_tools)

    return {"orchestrator": server_config}, orch
