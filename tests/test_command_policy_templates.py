from __future__ import annotations

import re

from prsm.shared.services.command_policy_store import CommandPolicyStore


def test_destructive_pattern_is_strict_on_path_root() -> None:
    pattern = CommandPolicyStore.build_command_pattern(
        "rm -rf ./build/cache-123",
        allow=True,
    )
    assert re.search(pattern, "rm -rf ./build/cache-999/subdir")
    assert not re.search(pattern, "rm -rf ./tmp/cache-999/subdir")


def test_exploratory_pattern_generalizes_arguments() -> None:
    pattern = CommandPolicyStore.build_command_pattern(
        "ls -la src/components",
        allow=True,
    )
    assert re.search(pattern, "ls -la docs")
    assert re.search(pattern, "ls -la src/components/button")


def test_grep_pattern_keeps_flags_and_generalizes_rest() -> None:
    pattern = CommandPolicyStore.build_command_pattern(
        "grep -R \"needle\" src",
        allow=True,
    )
    assert re.search(pattern, "grep -R another_term docs")
    assert not re.search(pattern, "grep -n another_term docs")


def test_deny_pattern_remains_exact() -> None:
    pattern = CommandPolicyStore.build_command_pattern(
        "rm -rf ./build/cache-123",
        allow=False,
    )
    assert re.search(pattern, "rm -rf ./build/cache-123")
    assert not re.search(pattern, "rm -rf ./build/cache-456")
