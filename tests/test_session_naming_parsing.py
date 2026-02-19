from prsm.adapters.session_naming import (
    _extract_name_from_model_output as extract_adapter,
    _fallback_session_name as adapter_fallback,
)
from prsm.shared.services.session_naming import (
    _extract_name_from_model_output as extract_shared,
    _fallback_session_name as shared_fallback,
)


def test_extracts_title_from_json_dict() -> None:
    raw = '{"title":"Fix spark title parsing"}'
    assert extract_shared(raw) == "Fix spark title parsing"
    assert extract_adapter(raw) == "Fix spark title parsing"


def test_extracts_title_from_fenced_json() -> None:
    raw = """```json
{"name":"Add robust title extractor"}
```"""
    assert extract_shared(raw) == "Add robust title extractor"
    assert extract_adapter(raw) == "Add robust title extractor"


def test_extracts_title_from_python_dict_repr() -> None:
    raw = "{'session_name': 'Handle spark json output'}"
    assert extract_shared(raw) == "Handle spark json output"
    assert extract_adapter(raw) == "Handle spark json output"


def test_extracts_title_from_nested_content_blocks() -> None:
    raw = (
        '{"content":[{"type":"text","text":"{\\"title\\":\\"Parse nested title\\"}"}],'
        '"metadata":{"threadId":"abc"}}'
    )
    assert extract_shared(raw) == "Parse nested title"
    assert extract_adapter(raw) == "Parse nested title"


def test_ignores_jsonrpc_metadata_values() -> None:
    raw = (
        '{"jsonrpc":"2.0","id":7,"result":{"content":[{"type":"text",'
        '"text":"{\\"title\\":\\"Fix session title parsing\\"}"}]}}'
    )
    assert extract_shared(raw) == "Fix session title parsing"
    assert extract_adapter(raw) == "Fix session title parsing"


def test_metadata_only_jsonrpc_payload_returns_empty_title() -> None:
    raw = '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"codex"}}'
    assert extract_shared(raw) == ""
    assert extract_adapter(raw) == ""


def test_extracts_title_from_nested_unknown_containers() -> None:
    raw = (
        '{"jsonrpc":"2.0","wrapper":{"unexpected":{"text":"Fix parser metadata regression"}}}'
    )
    assert extract_shared(raw) == "Fix parser metadata regression"
    assert extract_adapter(raw) == "Fix parser metadata regression"


def test_rejects_model_like_title_payloads() -> None:
    raw = '{"title":"gpt-5-3-spark"}'
    assert extract_shared(raw) == ""
    assert extract_adapter(raw) == ""


def test_rejects_titles_with_too_few_words() -> None:
    raw = '{"title":"Fix login"}'
    assert extract_shared(raw) == ""
    assert extract_adapter(raw) == ""


def test_rejects_workspace_path_from_structured_output() -> None:
    raw = '{"cwd":"~/prsm-cli","title":"Refactor session title parser"}'
    assert extract_shared(raw) == "Refactor session title parser"
    assert extract_adapter(raw) == "Refactor session title parser"


def test_rejects_raw_path_like_output() -> None:
    raw = "~/prsm-cli"
    assert extract_shared(raw) == ""
    assert extract_adapter(raw) == ""


def test_fallback_name_produces_three_to_seven_words() -> None:
    prompt = "gpt-5-3-spark reasoning_effort medium fix flaky auth tests and update docs quickly"
    shared_name = shared_fallback(prompt)
    adapter_name = adapter_fallback(prompt)
    assert 3 <= len(shared_name.split()) <= 7
    assert 3 <= len(adapter_name.split()) <= 7
    assert "gpt-5-3-spark" not in shared_name.lower()
    assert "gpt-5-3-spark" not in adapter_name.lower()
