from __future__ import annotations

from pathlib import Path

from prsm.shared.models.message import MessageRole
from prsm.shared.services.transcript_import.providers.claude import ClaudeTranscriptAdapter
from prsm.shared.services.transcript_import.providers.codex import CodexTranscriptAdapter
from prsm.shared.services.transcript_import.service import TranscriptImportService


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "transcript_import"


def _write_codex_fixture(root: Path) -> str:
    session_id = "019c0000-0000-7000-8000-000000000001"
    dst = (
        root
        / "sessions"
        / "2026"
        / "02"
        / "18"
        / f"rollout-2026-02-18T10-00-00-{session_id}.jsonl"
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text((FIXTURES_DIR / "codex_rollout.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    return session_id


def _write_claude_fixture(root: Path) -> str:
    session_id = "217df94b-a1f0-43b4-b457-764295a557ec"
    dst = root / "projects" / "demo" / f"{session_id}.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text((FIXTURES_DIR / "claude_session.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    return session_id


def test_codex_adapter_parses_messages_and_tool_calls(tmp_path: Path) -> None:
    codex_root = tmp_path / ".codex"
    session_id = _write_codex_fixture(codex_root)

    adapter = CodexTranscriptAdapter(root=codex_root)
    summaries = adapter.list_sessions()
    assert len(summaries) == 1
    assert summaries[0].source_session_id == session_id
    assert summaries[0].turn_count == 3

    transcript = adapter.load_session(session_id)
    assert [turn.role for turn in transcript.turns] == ["user", "assistant", "assistant"]
    tool_turn = transcript.turns[1]
    assert len(tool_turn.tool_calls) == 1
    assert tool_turn.tool_calls[0].name == "Bash"
    assert tool_turn.tool_calls[0].result
    assert tool_turn.tool_calls[0].success is True


def test_claude_adapter_parses_user_assistant_and_tool_use(tmp_path: Path) -> None:
    claude_root = tmp_path / ".claude"
    session_id = _write_claude_fixture(claude_root)

    adapter = ClaudeTranscriptAdapter(root=claude_root)
    summaries = adapter.list_sessions()
    assert len(summaries) == 1
    assert summaries[0].source_session_id == session_id
    assert summaries[0].turn_count == 3

    transcript = adapter.load_session(session_id)
    assert [turn.role for turn in transcript.turns] == ["user", "assistant", "assistant"]
    assert "duplication" in transcript.turns[1].content
    assert transcript.turns[1].tool_calls[0].name == "Read"


def test_import_service_converts_transcript_to_prsm_session(tmp_path: Path) -> None:
    codex_root = tmp_path / ".codex"
    session_id = _write_codex_fixture(codex_root)

    service = TranscriptImportService(codex_root=codex_root, claude_root=tmp_path / ".missing")
    result = service.import_to_session("codex", session_id, session_name="Imported session")

    assert result.session.name == "Imported session"
    assert result.session.active_agent_id == "root"
    messages = result.session.get_messages("root")
    assert len(messages) == 3
    assert messages[0].role == MessageRole.USER
    assert messages[1].role == MessageRole.ASSISTANT
    assert messages[1].tool_calls[0].name == "Bash"
    assert messages[2].role == MessageRole.ASSISTANT

    limited = service.import_to_session("codex", session_id, max_turns=2)
    assert limited.imported_turns == 2
    assert limited.dropped_turns == 1

