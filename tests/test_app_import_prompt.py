from __future__ import annotations

from types import SimpleNamespace

from prsm.app import _prompt_import_depth_cli


def test_prompt_import_depth_cli_non_interactive_defaults_to_200(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))
    assert _prompt_import_depth_cli() == 200


def test_prompt_import_depth_cli_interactive_full(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda _prompt: "3")
    assert _prompt_import_depth_cli() is None


def test_prompt_import_depth_cli_interactive_blank_defaults_to_recommended(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    assert _prompt_import_depth_cli() == 200

