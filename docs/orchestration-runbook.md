# Orchestration Runbook

## Non-Blocking Child Execution Checklist

Use this checklist whenever delegating work to child agents.

1. Assume child work is interactive unless proven otherwise.
2. Spawn with `wait=false`.
3. Use `wait_for_message(timeout_seconds=0)` in a loop (`0` disables timeout).
4. If child sends a question, answer with `respond_to_child(...)`.
5. Continue until `task_result` is received.
6. Only use blocking `wait=true` for short, guaranteed non-interactive tasks.

## Timeout Guardrail

- Do not rely on short tool-call deadlines for interactive children.
- If a child may ask questions or run multi-step work, non-blocking orchestration is required.
- Prefer `timeout_seconds=0` for parent/child coordination waits. Use positive timeouts only when you intentionally want polling behavior.
