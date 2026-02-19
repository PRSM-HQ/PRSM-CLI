"""Tests for prsm.shared.formatters.tool_call — the rich tool call formatting system."""

import json
import pytest

from prsm.shared.formatters.tool_call import (
    FormattedToolCall,
    Section,
    parse_args,
    format_tool_call,
    render_collapsed_rich,
    render_expanded_rich,
    _normalize_tool_name,
)


# ── parse_args tests ──


class TestParseArgs:
    def test_valid_json_dict(self):
        args = json.dumps({"file_path": "/foo/bar.py", "command": "ls"})
        result = parse_args(args)
        assert result == {"file_path": "/foo/bar.py", "command": "ls"}

    def test_python_repr_dict(self):
        args = "{'file_path': '/foo/bar.py', 'old_string': 'hello'}"
        result = parse_args(args)
        assert result == {"file_path": "/foo/bar.py", "old_string": "hello"}

    def test_garbage_input(self):
        args = "not valid at all {{{"
        result = parse_args(args)
        assert result == {"_raw": "not valid at all {{{"}

    def test_empty_string(self):
        assert parse_args("") == {}

    def test_none_like(self):
        assert parse_args("") == {}

    def test_json_array_falls_back(self):
        """JSON arrays aren't dicts, should fall to _raw."""
        args = json.dumps([1, 2, 3])
        result = parse_args(args)
        assert result == {"_raw": "[1, 2, 3]"}

    def test_json_string_falls_back(self):
        args = json.dumps("hello")
        result = parse_args(args)
        assert result == {"_raw": '"hello"'}


# ── format_tool_call tests (individual formatters) ──


class TestFormatEdit:
    def test_basic_edit(self):
        args = json.dumps({
            "file_path": "/home/user/project/src/app.py",
            "old_string": "def foo():\n    pass",
            "new_string": "def foo():\n    return 42",
        })
        fmt = format_tool_call("Edit", args, None, True)
        assert fmt.icon == "\u270f"
        assert fmt.label == "Edit"
        assert "app.py" in fmt.summary
        # Should have path + diff sections
        kinds = [s.kind for s in fmt.sections]
        assert "path" in kinds
        assert "diff" in kinds

    def test_edit_diff_content(self):
        args = json.dumps({
            "file_path": "/foo.py",
            "old_string": "old line 1\nold line 2",
            "new_string": "new line 1\nnew line 2",
        })
        fmt = format_tool_call("Edit", args, None, True)
        diff = next(s for s in fmt.sections if s.kind == "diff")
        assert diff.content["old_lines"] == ["old line 1", "old line 2"]
        assert diff.content["new_lines"] == ["new line 1", "new line 2"]


class TestFormatBash:
    def test_basic_bash(self):
        args = json.dumps({"command": "npm install --save-dev typescript"})
        fmt = format_tool_call("Bash", args, "added 15 packages", True)
        assert fmt.icon == "$"
        assert fmt.label == "Bash"
        assert "npm install" in fmt.summary
        # Should have code section + result section
        kinds = [s.kind for s in fmt.sections]
        assert "code" in kinds
        assert "plain" in kinds

    def test_bash_with_description(self):
        args = json.dumps({"command": "npm run build", "description": "Build project"})
        fmt = format_tool_call("Bash", args, None, True)
        assert any(s.kind == "plain" and s.title == "Description" for s in fmt.sections)

    def test_long_command_truncated_in_summary(self):
        cmd = "a" * 200
        args = json.dumps({"command": cmd})
        fmt = format_tool_call("Bash", args, None, True)
        assert len(fmt.summary) <= 63  # 60 + "..."

    def test_bash_rg_maps_to_grep_formatter(self):
        args = json.dumps({"command": "rg TODO src"})
        fmt = format_tool_call("Bash", args, "src/a.py:1:# TODO", True)
        assert fmt.label == "Grep"
        assert '"TODO"' in fmt.summary

    def test_bash_sed_maps_to_read_formatter(self):
        args = json.dumps({"command": "sed -n '12,20p' src/app.py"})
        fmt = format_tool_call("Bash", args, "line content", True)
        assert fmt.label == "Read"
        assert "app.py" in fmt.summary
        assert "L12" in fmt.summary

    def test_bash_ls_maps_to_glob_formatter(self):
        args = json.dumps({"command": "ls -la src"})
        fmt = format_tool_call("Bash", args, "a.py\nb.py", True)
        assert fmt.label == "Glob"
        assert "src" in fmt.summary

    def test_bash_cat_maps_to_read_formatter(self):
        args = json.dumps({"command": "cat src/app.py"})
        fmt = format_tool_call("Bash", args, "print('x')", True)
        assert fmt.label == "Read"
        assert "app.py" in fmt.summary

    def test_bash_head_maps_to_read_formatter(self):
        args = json.dumps({"command": "head -n 25 src/app.py"})
        fmt = format_tool_call("Bash", args, "line1", True)
        assert fmt.label == "Read"
        assert "+25" in fmt.summary

    def test_bash_tail_maps_to_read_formatter(self):
        args = json.dumps({"command": "tail -n 40 src/app.py"})
        fmt = format_tool_call("Bash", args, "line-last", True)
        assert fmt.label == "Read"
        assert "+40" in fmt.summary

    def test_bash_find_maps_to_glob_formatter(self):
        args = json.dumps({"command": "find src -name '*.py'"})
        fmt = format_tool_call("Bash", args, "src/a.py", True)
        assert fmt.label == "Glob"
        assert "*.py" in fmt.summary

    def test_bash_sed_inplace_maps_to_edit_formatter(self):
        args = json.dumps({"command": "sed -i 's/old/new/g' src/app.py"})
        fmt = format_tool_call("Bash", args, "done", True)
        assert fmt.label == "Edit"
        kinds = [s.kind for s in fmt.sections]
        assert "diff" in kinds

    def test_bash_tee_maps_to_write_formatter(self):
        args = json.dumps({"command": "tee src/generated.txt"})
        fmt = format_tool_call("Bash", args, "written", True)
        assert fmt.label == "Write"
        assert "generated.txt" in fmt.summary

    def test_bash_redirect_maps_to_write_formatter(self):
        args = json.dumps({"command": "echo hello > src/generated.txt"})
        fmt = format_tool_call("Bash", args, "ok", True)
        assert fmt.label == "Write"
        assert "generated.txt" in fmt.summary

    def test_bash_git_diff_maps_to_git_formatter(self):
        args = json.dumps({"command": "git diff -- src/app.py"})
        fmt = format_tool_call("Bash", args, "diff output", True)
        assert fmt.label == "Git"
        assert "diff" in fmt.summary

    def test_bash_pytest_maps_to_test_formatter(self):
        args = json.dumps({"command": "pytest -q tests/test_tool_formatters.py"})
        fmt = format_tool_call("Bash", args, "1 passed", True)
        assert fmt.label == "Test"

    def test_bash_npm_build_maps_to_build_formatter(self):
        args = json.dumps({"command": "npm run build"})
        fmt = format_tool_call("Bash", args, "build ok", True)
        assert fmt.label == "Build"

    def test_bash_process_control_maps(self):
        args = json.dumps({"command": "pgrep -af prsm"})
        fmt = format_tool_call("Bash", args, "1234", True)
        assert fmt.label == "Process"

    def test_bash_curl_maps_to_http_formatter(self):
        args = json.dumps({"command": "curl -X POST https://example.com/api"})
        fmt = format_tool_call("Bash", args, "{\"ok\":true}", True)
        assert fmt.label == "HTTP"
        assert "POST" in fmt.summary

    def test_bash_jq_maps_to_json_formatter(self):
        args = json.dumps({"command": "jq '.items[]' data.json"})
        fmt = format_tool_call("Bash", args, "value", True)
        assert fmt.label == "JSON"

    def test_bash_wc_maps_to_stats_formatter(self):
        args = json.dumps({"command": "wc -l src/app.py"})
        fmt = format_tool_call("Bash", args, "20 src/app.py", True)
        assert fmt.label == "Stats"

    def test_bash_pipeline_maps_to_shell_workflow(self):
        args = json.dumps({"command": "rg TODO src | wc -l"})
        fmt = format_tool_call("Bash", args, "5", True)
        assert fmt.label == "Shell Workflow"
        assert "steps" in fmt.summary
        checklist = next(s for s in fmt.sections if s.kind == "checklist")
        assert len(checklist.content) >= 2


class TestFormatRead:
    def test_basic_read(self):
        args = json.dumps({"file_path": "/home/user/README.md"})
        fmt = format_tool_call("Read", args, "# Hello World", True)
        assert fmt.label == "Read"
        assert "README.md" in fmt.summary

    def test_read_with_offset(self):
        args = json.dumps({"file_path": "/foo.py", "offset": 10, "limit": 20})
        fmt = format_tool_call("Read", args, None, True)
        assert "L10" in fmt.summary
        assert "+20" in fmt.summary

    def test_read_content_renders_as_code_section(self):
        args = json.dumps({"file_path": "/foo.py"})
        fmt = format_tool_call("Read", args, "print('hello')", True)
        code = next(s for s in fmt.sections if s.kind == "code")
        assert code.title == "Content"

    def test_read_structured_result_extracts_text_content(self):
        args = json.dumps({"file_path": "/foo.py"})
        wrapped = json.dumps({
            "content": [{"type": "text", "text": "line1\nline2"}],
        })
        fmt = format_tool_call("Read", args, wrapped, True)
        code = next(s for s in fmt.sections if s.kind == "code")
        assert code.content["text"] == "line1\nline2"

    def test_read_truncated_json_wrapper_extracts_text_content(self):
        args = json.dumps({"file_path": "/foo.py"})
        wrapped = '{"content":[{"type":"text","text":"line1\\nline2"}'
        fmt = format_tool_call("Read", args, wrapped, True)
        code = next(s for s in fmt.sections if s.kind == "code")
        assert code.content["text"] == "line1\nline2"


class TestFormatWrite:
    def test_basic_write(self):
        args = json.dumps({"file_path": "/home/user/new_file.py", "content": "print('hi')"})
        fmt = format_tool_call("Write", args, "File written", True)
        assert fmt.label == "Write"
        assert "new_file.py" in fmt.summary


class TestFormatGlob:
    def test_basic_glob(self):
        args = json.dumps({"pattern": "**/*.tsx"})
        fmt = format_tool_call("Glob", args, "src/App.tsx\nsrc/index.tsx", True)
        assert fmt.summary == "**/*.tsx"
        assert fmt.label == "Glob"

    def test_glob_with_path(self):
        args = json.dumps({"pattern": "*.py", "path": "/home/user/project/src"})
        fmt = format_tool_call("Glob", args, None, True)
        assert "in" in fmt.summary


class TestFormatGrep:
    def test_basic_grep(self):
        args = json.dumps({"pattern": "TODO", "glob": "*.py"})
        fmt = format_tool_call("Grep", args, "src/app.py:10: # TODO fix", True)
        assert '"TODO"' in fmt.summary
        assert "*.py" in fmt.summary


class TestFormatTask:
    def test_basic_task(self):
        args = json.dumps({"description": "Build the project", "subagent_type": "Bash"})
        fmt = format_tool_call("Task", args, None, True)
        assert fmt.label == "Task"
        assert "Build the project" in fmt.summary
        kv = next(s for s in fmt.sections if s.kind == "kv")
        assert kv.content["type"] == "Bash"


class TestFormatTodoWrite:
    def test_basic_todo(self):
        args = json.dumps({
            "todos": [
                {"content": "Fix bug", "status": "completed", "activeForm": "Fixing bug"},
                {"content": "Add tests", "status": "pending", "activeForm": "Adding tests"},
                {"content": "Deploy", "status": "in_progress", "activeForm": "Deploying"},
            ]
        })
        fmt = format_tool_call("TodoWrite", args, None, True)
        assert fmt.label == "TodoWrite"
        assert "1/3 done" in fmt.summary
        checklist = next(s for s in fmt.sections if s.kind == "checklist")
        assert len(checklist.content) == 3
        assert checklist.content[0]["done"] is True
        assert checklist.content[1]["done"] is False


class TestFormatWebFetch:
    def test_basic_web_fetch(self):
        args = json.dumps({"url": "https://example.com/api/data", "prompt": "Extract the title"})
        fmt = format_tool_call("WebFetch", args, "Title: Example", True)
        assert "example.com" in fmt.summary


class TestFormatWebSearch:
    def test_basic_web_search(self):
        args = json.dumps({"query": "python async tutorial"})
        fmt = format_tool_call("WebSearch", args, "Result 1...", True)
        assert "python async tutorial" in fmt.summary


class TestFormatDefault:
    def test_unknown_tool(self):
        args = json.dumps({"foo": "bar", "baz": 42})
        fmt = format_tool_call("UnknownTool", args, None, True)
        assert fmt.label == "UnknownTool"
        assert fmt.sections  # Should have some sections

    def test_raw_args_fallback(self):
        fmt = format_tool_call("WeirdTool", "not json at all", None, True)
        assert fmt.label == "WeirdTool"
        assert fmt.summary  # Should have something


# ── file_path field tests ──


class TestFilePathField:
    """Verify that file-related formatters populate the file_path field."""

    def test_edit_has_file_path(self):
        args = json.dumps({"file_path": "/home/user/project/src/app.py", "old_string": "a", "new_string": "b"})
        fmt = format_tool_call("Edit", args, None, True)
        assert fmt.file_path == "/home/user/project/src/app.py"

    def test_read_has_file_path(self):
        args = json.dumps({"file_path": "/tmp/readme.md"})
        fmt = format_tool_call("Read", args, None, True)
        assert fmt.file_path == "/tmp/readme.md"

    def test_write_has_file_path(self):
        args = json.dumps({"file_path": "/tmp/output.txt"})
        fmt = format_tool_call("Write", args, None, True)
        assert fmt.file_path == "/tmp/output.txt"

    def test_glob_has_file_path_when_path_set(self):
        args = json.dumps({"pattern": "*.py", "path": "/home/user/project"})
        fmt = format_tool_call("Glob", args, None, True)
        assert fmt.file_path == "/home/user/project"

    def test_glob_empty_file_path_when_no_path(self):
        args = json.dumps({"pattern": "*.py"})
        fmt = format_tool_call("Glob", args, None, True)
        assert fmt.file_path == ""

    def test_grep_has_file_path(self):
        args = json.dumps({"pattern": "TODO", "path": "/home/user/project/src"})
        fmt = format_tool_call("Grep", args, None, True)
        assert fmt.file_path == "/home/user/project/src"

    def test_notebook_edit_has_file_path(self):
        args = json.dumps({"notebook_path": "/home/user/analysis.ipynb", "new_source": "x=1"})
        fmt = format_tool_call("NotebookEdit", args, None, True)
        assert fmt.file_path == "/home/user/analysis.ipynb"

    def test_bash_no_file_path(self):
        args = json.dumps({"command": "npm run test"})
        fmt = format_tool_call("Bash", args, None, True)
        assert fmt.file_path == ""

    def test_task_no_file_path(self):
        args = json.dumps({"description": "search code", "prompt": "find bugs", "subagent_type": "general"})
        fmt = format_tool_call("Task", args, None, True)
        assert fmt.file_path == ""


# ── Rich renderer tests ──


class TestRenderCollapsedRich:
    def test_pending_status(self):
        fmt = FormattedToolCall(icon="$", label="Bash", summary="npm install", sections=[])
        result = render_collapsed_rich(fmt, "pending")
        assert "Bash" in result
        assert "npm install" in result
        assert "pending" in result

    def test_done_status(self):
        fmt = FormattedToolCall(icon="\u270f", label="Edit", summary="app.py", sections=[])
        result = render_collapsed_rich(fmt, "done")
        assert "Edit" in result
        assert "done" in result
        assert "green" in result

    def test_error_status(self):
        fmt = FormattedToolCall(icon="$", label="Bash", summary="rm -rf /", sections=[])
        result = render_collapsed_rich(fmt, "error")
        assert "red" in result


class TestRenderExpandedRich:
    def test_diff_section(self):
        fmt = FormattedToolCall(
            icon="\u270f",
            label="Edit",
            summary="app.py",
            sections=[
                Section(kind="path", content="/src/app.py"),
                Section(
                    kind="diff",
                    content={
                        "old_lines": ["old line"],
                        "new_lines": ["new line"],
                    },
                ),
            ],
        )
        result = render_expanded_rich(fmt, "done", "12:00:00")
        assert "red" in result  # red for removed
        assert "green" in result  # green for added
        assert "old line" in result
        assert "new line" in result

    def test_code_section(self):
        fmt = FormattedToolCall(
            icon="$",
            label="Bash",
            summary="ls",
            sections=[
                Section(kind="code", title="Command", content={"language": "bash", "text": "ls -la"}),
            ],
        )
        result = render_expanded_rich(fmt, "done")
        assert "ls -la" in result
        assert "$" in result  # prompt marker

    def test_checklist_section(self):
        fmt = FormattedToolCall(
            icon="\u2611",
            label="TodoWrite",
            summary="2/3 done",
            sections=[
                Section(
                    kind="checklist",
                    content=[
                        {"text": "Task 1", "done": True},
                        {"text": "Task 2", "done": False},
                    ],
                ),
            ],
        )
        result = render_expanded_rich(fmt, "done")
        assert "Task 1" in result
        assert "Task 2" in result
        assert "\u2713" in result  # checkmark
        assert "\u25cb" in result  # open circle

    def test_kv_section(self):
        fmt = FormattedToolCall(
            icon="\U0001f500",
            label="Task",
            summary="do stuff",
            sections=[
                Section(kind="kv", content={"type": "Bash", "description": "build it"}),
            ],
        )
        result = render_expanded_rich(fmt, "done")
        assert "type" in result
        assert "Bash" in result

    def test_plain_section_truncation(self):
        long_text = "\n".join(f"line {i}" for i in range(50))
        fmt = FormattedToolCall(
            icon="$",
            label="Bash",
            summary="big output",
            sections=[Section(kind="plain", content=long_text)],
        )
        result = render_expanded_rich(fmt, "done")
        assert "more lines" in result  # should show truncation notice

    def test_no_crash_on_empty(self):
        fmt = FormattedToolCall()
        result = render_expanded_rich(fmt, "pending")
        assert isinstance(result, str)

    def test_escapes_rich_markup(self):
        """Ensure [brackets] in content don't break Rich markup."""
        fmt = FormattedToolCall(
            icon="$",
            label="Bash",
            summary="echo [test]",
            sections=[Section(kind="plain", content="output [with] brackets")],
        )
        result = render_expanded_rich(fmt, "done")
        assert "\\[test]" in result or "\\[with]" in result


# ── Name normalization tests ──


class TestNormalizeToolName:
    def test_mcp_orchestrator_prefix(self):
        assert _normalize_tool_name("mcp__orchestrator__task_complete") == "task_complete"

    def test_mcp_orchestrator_spawn_child(self):
        assert _normalize_tool_name("mcp__orchestrator__spawn_child") == "spawn_child"

    def test_bare_name_unchanged(self):
        assert _normalize_tool_name("Edit") == "Edit"

    def test_single_underscore_unchanged(self):
        assert _normalize_tool_name("mcp_something") == "mcp_something"

    def test_nested_double_underscores(self):
        assert _normalize_tool_name("mcp__server__nested__tool") == "nested__tool"

    def test_run_bash_alias_maps_to_bash(self):
        assert _normalize_tool_name("run_bash") == "Bash"

    def test_prefixed_edit_alias_maps_to_edit(self):
        assert _normalize_tool_name("mcp__orchestrator__edit_file") == "Edit"

    def test_format_dispatches_with_prefix(self):
        """format_tool_call should find the task_complete formatter even with mcp__ prefix."""
        args = json.dumps({"summary": "All done!", "artifacts": {"file": "output.txt"}})
        fmt = format_tool_call("mcp__orchestrator__task_complete", args, None, True)
        assert fmt.label == "Task Complete"
        assert fmt.icon == "✅"

    def test_format_dispatches_spawn_child_with_prefix(self):
        args = json.dumps({"prompt": "Do something", "complexity": "simple"})
        fmt = format_tool_call("mcp__orchestrator__spawn_child", args, None, True)
        assert fmt.label == "Spawn Agent"
        assert fmt.icon == "🚀"

    def test_format_dispatches_bash_formatter_for_alias(self):
        args = json.dumps({"command": "echo hi"})
        fmt = format_tool_call("run_bash", args, None, True)
        assert fmt.label == "Bash"


# ── Orchestrator tool formatter tests ──


class TestFormatTaskComplete:
    def test_basic_task_complete(self):
        args = json.dumps({"summary": "Implemented the feature successfully"})
        fmt = format_tool_call("task_complete", args, None, True)
        assert fmt.icon == "✅"
        assert fmt.label == "Task Complete"
        assert "Implemented" in fmt.summary
        assert any(s.kind == "result_block" and s.title == "Agent Summary" for s in fmt.sections)

    def test_with_artifacts(self):
        args = json.dumps({
            "summary": "Done",
            "artifacts": {"output_file": "/tmp/result.json", "lines_changed": 42},
        })
        fmt = format_tool_call("task_complete", args, None, True)
        kv_section = next(s for s in fmt.sections if s.kind == "kv")
        assert "output_file" in kv_section.content


class TestFormatSpawnChild:
    def test_basic_spawn(self):
        args = json.dumps({"prompt": "Search for all TODO comments", "complexity": "simple"})
        fmt = format_tool_call("spawn_child", args, None, True)
        assert fmt.icon == "🚀"
        assert fmt.label == "Spawn Agent"
        assert "[simple]" in fmt.summary
        assert "TODO" in fmt.summary

    def test_spawn_with_model(self):
        args = json.dumps({"prompt": "Analyze architecture", "model": "opus"})
        fmt = format_tool_call("spawn_child", args, None, True)
        assert "[opus]" in fmt.summary

    def test_spawn_with_tools(self):
        args = json.dumps({
            "prompt": "Do work",
            "tools": ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Task"],
        })
        fmt = format_tool_call("spawn_child", args, None, True)
        kv = next(s for s in fmt.sections if s.kind == "kv")
        assert "tools" in kv.content
        assert "+1 more" in kv.content["tools"]


class TestFormatSpawnChildrenParallel:
    def test_basic_parallel(self):
        args = json.dumps({
            "children": [
                {"prompt": "Task A", "complexity": "simple"},
                {"prompt": "Task B", "complexity": "medium"},
                {"prompt": "Task C"},
            ]
        })
        fmt = format_tool_call("spawn_children_parallel", args, None, True)
        assert fmt.label == "Spawn Parallel"
        assert "3 agents" in fmt.summary
        agent_prompts = [s for s in fmt.sections if s.kind == "agent_prompt"]
        assert len(agent_prompts) == 3
        assert agent_prompts[0].content["complexity"] == "simple"
        assert agent_prompts[0].content["prompt"] == "Task A"

    def test_count_falls_back_to_result_when_args_unparseable(self):
        # Simulates truncated JSON args that parse as {"_raw": "..."}.
        args = '{"children": [{"prompt": "Task A"}, {"prompt": "Task B"}'
        result = (
            "Spawned 2 children in parallel (non-blocking).\n"
            "- child_id: child-a\n"
            "- child_id: child-b\n"
        )
        fmt = format_tool_call("spawn_children_parallel", args, result, True)
        assert "2 agents" in fmt.summary


class TestFormatRestartChild:
    def test_basic_restart(self):
        args = json.dumps({"child_agent_id": "abc123def456ghi", "prompt": "Fix the failing test"})
        fmt = format_tool_call("restart_child", args, None, True)
        assert fmt.icon == "🔄"
        assert fmt.label == "Restart Agent"
        assert "→ abc123def456..." in fmt.summary


class TestFormatAskParent:
    def test_basic_ask(self):
        args = json.dumps({"question": "Should I use TypeScript or JavaScript?"})
        fmt = format_tool_call("ask_parent", args, "Use TypeScript", True)
        assert fmt.icon == "💬"
        assert fmt.label == "Ask Parent"
        assert "TypeScript" in fmt.summary
        assert any(s.title == "💬 Answer" for s in fmt.sections)


class TestFormatAskUser:
    def test_with_options(self):
        args = json.dumps({
            "question": "Which database should we use?",
            "options": [
                {"label": "PostgreSQL", "description": "Relational DB"},
                {"label": "MongoDB", "description": "Document DB"},
            ],
        })
        fmt = format_tool_call("ask_user", args, None, True)
        assert fmt.icon == "💬"
        assert fmt.label == "Ask User"
        checklist = next(s for s in fmt.sections if s.kind == "checklist")
        assert len(checklist.content) == 2
        assert "PostgreSQL" in checklist.content[0]["text"]


class TestFormatWaitForMessage:
    def test_with_timeout(self):
        args = json.dumps({"timeout_seconds": 120})
        fmt = format_tool_call("wait_for_message", args, None, True)
        assert fmt.icon == "⏳"
        assert fmt.summary == "waiting..."


class TestFormatRespondToChild:
    def test_basic_respond(self):
        args = json.dumps({
            "child_agent_id": "abc123def456ghi",
            "correlation_id": "corr-1",
            "response": "Yes, proceed with option A",
        })
        fmt = format_tool_call("respond_to_child", args, None, True)
        assert fmt.label == "Reply to Agent"
        assert "abc123def456..." in fmt.summary


class TestFormatConsultExpert:
    def test_basic_consult(self):
        args = json.dumps({"expert_id": "code-reviewer", "question": "Is this pattern safe?"})
        fmt = format_tool_call("consult_expert", args, "Yes, it's safe", True)
        assert fmt.icon == "🎓"
        assert "code-reviewer" in fmt.summary


class TestFormatConsultPeer:
    def test_basic_peer(self):
        args = json.dumps({"peer": "codex", "question": "How would you solve this?"})
        fmt = format_tool_call("consult_peer", args, None, True)
        assert fmt.icon == "🤝"
        assert "codex" in fmt.summary


class TestFormatReportProgress:
    def test_with_percent(self):
        args = json.dumps({"status": "Building project", "percent_complete": 75})
        fmt = format_tool_call("report_progress", args, None, True)
        assert fmt.icon == "📊"
        assert "75%" in fmt.summary
        assert "Building" in fmt.summary

    def test_without_percent(self):
        args = json.dumps({"status": "Starting analysis"})
        fmt = format_tool_call("report_progress", args, None, True)
        assert "Starting analysis" in fmt.summary


class TestFormatGetChildHistory:
    def test_basic(self):
        args = json.dumps({"child_agent_id": "abc123def456", "detail_level": "summary"})
        fmt = format_tool_call("get_child_history", args, None, True)
        assert fmt.icon == "📜"
        assert fmt.label == "Agent History"


class TestFormatCheckChildStatus:
    def test_basic(self):
        args = json.dumps({"child_agent_id": "abc123def456"})
        fmt = format_tool_call("check_child_status", args, None, True)
        assert fmt.label == "Check Agent"


class TestFormatSendChildPrompt:
    def test_basic(self):
        args = json.dumps({"child_agent_id": "abc123", "prompt": "Continue with step 2"})
        fmt = format_tool_call("send_child_prompt", args, None, True)
        assert fmt.icon == "📨"
        assert "abc123" in fmt.summary


class TestFormatGetChildrenStatus:
    def test_basic(self):
        fmt = format_tool_call("get_children_status", "{}", "3 running, 2 completed", True)
        assert fmt.icon == "👥"
        assert fmt.label == "All Agents Status"


class TestFormatRecommendModel:
    def test_basic(self):
        args = json.dumps({"task_description": "Analyze code architecture", "complexity": "complex"})
        fmt = format_tool_call("recommend_model", args, None, True)
        assert fmt.icon == "🧠"
        assert "[complex]" in fmt.summary


# ── Orchestrator Result Formatting Tests ──


class TestSpawnChildResult:
    def test_wait_true_result(self):
        """When wait=true, spawn_child returns a structured result with '--- Result ---' section."""
        args = json.dumps({"prompt": "Search for TODO comments", "wait": True})
        result = (
            "Child completed (success=True).\n"
            "child_id: abc123def456\n"
            "duration: 3.2s\n\n"
            "--- Result ---\n"
            "Found 12 TODO comments across 5 files."
        )
        fmt = format_tool_call("spawn_child", args, result, True)
        # Should have structured sections, not raw text dump
        kinds = [s.kind for s in fmt.sections]
        assert "kv" in kinds  # child_id, duration parsed as kv
        assert any(s.title == "Result" and "12 TODO" in s.content for s in fmt.sections)

    def test_wait_false_result(self):
        """When wait=false, spawn_child returns a background spawn confirmation."""
        args = json.dumps({"prompt": "Explore codebase", "wait": False})
        result = (
            "Child agent spawned in background.\n"
            "child_id: abc123def456\n"
            "model: claude-opus-4-6\n"
            "Use wait_for_message to receive its result."
        )
        fmt = format_tool_call("spawn_child", args, result, True)
        kinds = [s.kind for s in fmt.sections]
        assert "kv" in kinds  # child_id, model parsed as kv

    def test_with_error_result(self):
        """spawn_child result with error section."""
        args = json.dumps({"prompt": "Do something"})
        result = (
            "Child completed (success=False).\n"
            "child_id: xyz789\n"
            "duration: 1.0s\n\n"
            "--- Result ---\n"
            "Partial output\n\n"
            "--- Error ---\n"
            "Something went wrong"
        )
        fmt = format_tool_call("spawn_child", args, result, False)
        titles = [s.title for s in fmt.sections]
        assert "Result" in titles
        assert "Error" in titles


class TestWaitForMessageResult:
    def test_task_result_payload(self):
        """wait_for_message with a TASK_RESULT payload."""
        args = json.dumps({"timeout_seconds": 120})
        payload = json.dumps({"summary": "Found the bug in auth.py", "artifacts": {"file": "auth.py"}})
        result = (
            "Message received:\n"
            "  type: task_result\n"
            "  from: child-abc123\n"
            "  correlation_id: msg-xyz789\n"
            f"  payload: {payload}"
        )
        fmt = format_tool_call("wait_for_message", args, result, True)
        # Should have Message kv, Child Result block, and Artifacts sections
        kinds = [s.kind for s in fmt.sections]
        titles = [s.title for s in fmt.sections]
        assert "kv" in kinds  # timeout and/or message fields
        assert "Child Result" in titles
        assert "Artifacts" in titles
        # Summary should include the child's summary preview
        assert "child-ab" in fmt.summary  # short agent ID
        assert "Found the bug" in fmt.summary  # payload preview

    def test_question_payload(self):
        """wait_for_message with a QUESTION payload."""
        args = json.dumps({"timeout_seconds": 60})
        result = (
            "Message received:\n"
            "  type: question\n"
            "  from: child-def456\n"
            "  correlation_id: msg-abc123\n"
            "  payload: Should I use TypeScript?"
        )
        fmt = format_tool_call("wait_for_message", args, result, True)
        assert "question from child-def456" in fmt.summary

    def test_timeout_result(self):
        """wait_for_message that timed out."""
        args = json.dumps({"timeout_seconds": 30})
        result = "No messages received within 30.0s. This is normal — your child agents are likely still working. Call wait_for_message() again to continue waiting for results. Do NOT start doing the work yourself. Use get_children_status() to check if children are still running."
        fmt = format_tool_call("wait_for_message", args, result, True)
        assert "timed out" in fmt.summary


class TestCheckChildStatusResult:
    def test_json_status(self):
        """check_child_status with JSON result."""
        args = json.dumps({"child_agent_id": "abc123def456"})
        result = json.dumps({
            "agent_id": "abc123def456ghijklmno",
            "state": "completed",
            "role": "worker",
            "model": "claude-opus-4-6",
            "depth": 1,
            "children_count": 2,
            "children_ids": ["child1", "child2"],
            "created_at": "2025-01-01T00:00:00",
            "completed_at": "2025-01-01T00:01:30",
            "error": None,
            "prompt_preview": "Search for TODO comments in the codebase"
        })
        fmt = format_tool_call("check_child_status", args, result, True)
        kinds = [s.kind for s in fmt.sections]
        titles = [s.title for s in fmt.sections]
        assert "kv" in kinds  # Agent Status kv
        assert "Agent Status" in titles
        # Should have Prompt and Children sections
        assert "Prompt" in titles
        assert "Children" in titles


class TestGetChildrenStatusResult:
    def test_structured_result(self):
        """get_children_status with structured output."""
        result = (
            "Children status: 2 completed, 0 failed, 1 running (total: 3)\n\n"
            "- abc123: state=completed\n"
            "- def456: state=completed\n"
            "- ghi789: state=running"
        )
        fmt = format_tool_call("get_children_status", "{}", result, True)
        kinds = [s.kind for s in fmt.sections]
        assert "checklist" in kinds
        # Summary should contain the count info
        assert "completed" in fmt.summary.lower() or "2" in fmt.summary


class TestTaskCompleteResult:
    def test_skips_redundant_confirmation(self):
        """task_complete skips the 'Task marked complete' boilerplate."""
        args = json.dumps({"summary": "Implemented dark mode"})
        result = "Task marked complete. Session will end."
        fmt = format_tool_call("task_complete", args, result, True)
        # Should NOT have a section with the redundant confirmation
        for section in fmt.sections:
            if section.kind == "plain":
                assert "Task marked complete" not in (section.content or "")


class TestConsultExpertResult:
    def test_structured_response(self):
        """consult_expert with structured result."""
        args = json.dumps({"expert_id": "code-reviewer", "question": "Is this pattern safe?"})
        result = (
            "Expert 'Code Reviewer' responded (success=True, duration=5.2s):\n\n"
            "The pattern is safe because it uses immutable data structures.\n"
            "However, consider adding validation for edge cases."
        )
        fmt = format_tool_call("consult_expert", args, result, True)
        titles = [s.title for s in fmt.sections]
        assert "Status" in titles
        assert "Response" in titles


class TestConsultPeerResult:
    def test_structured_response_with_thread(self):
        """consult_peer with thread_id and other peers."""
        args = json.dumps({"peer": "codex", "question": "Best approach for caching?"})
        result = (
            "Peer response (provider=openai, model=codex-mini):\n\n"
            "I'd recommend using Redis for distributed caching.\n\n"
            "thread_id: thread-abc123\n"
            "(Pass this thread_id to consult_peer for follow-up questions)\n\n"
            "Other available peers: gemini, minimax"
        )
        fmt = format_tool_call("consult_peer", args, result, True)
        titles = [s.title for s in fmt.sections]
        assert "Status" in titles
        assert "Response" in titles
        assert "Info" in titles


class TestRecommendModelResult:
    def test_structured_recommendation(self):
        """recommend_model with structured result."""
        args = json.dumps({"task_description": "Analyze architecture", "complexity": "complex"})
        result = (
            "Recommended model: claude-opus-4-6\n"
            "Provider: claude\n"
            "Tier: frontier\n"
            "Task category: architecture\n"
            "Affinity score: 95%\n"
            "Cost factor: 3x\n"
            "Speed factor: 0.5x\n\n"
            "Fallback options (if top choice unavailable):\n"
            "  2. gemini-2.5-pro (google, score=90%)\n"
            "  3. codex-mini (openai, score=80%)\n\n"
            "Use model='claude-opus-4-6' in spawn_child to use this model."
        )
        fmt = format_tool_call("recommend_model", args, result, True)
        kinds = [s.kind for s in fmt.sections]
        titles = [s.title for s in fmt.sections]
        assert "Recommendation" in titles
        assert "Fallback Options" in titles
        # Check the kv section has parsed fields
        kv_sections = [s for s in fmt.sections if s.kind == "kv"]
        assert any("claude-opus-4-6" in str(s.content) for s in kv_sections)


class TestRespondToChildResult:
    def test_skips_redundant_confirmation(self):
        """respond_to_child skips 'Response delivered' boilerplate."""
        args = json.dumps({"child_agent_id": "abc123", "response": "Use TypeScript"})
        result = "Response delivered to child abc123..."
        fmt = format_tool_call("respond_to_child", args, result, True)
        for section in fmt.sections:
            if section.kind == "plain":
                assert "Response delivered" not in (section.content or "")


class TestReportProgressResult:
    def test_skips_redundant_confirmation(self):
        """report_progress skips 'Progress reported' boilerplate."""
        args = json.dumps({"status": "Building project", "percent_complete": 75})
        result = "Progress reported: 75% - Building project"
        fmt = format_tool_call("report_progress", args, result, True)
        for section in fmt.sections:
            if section.kind == "plain":
                assert "Progress reported" not in (section.content or "")
