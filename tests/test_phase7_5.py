"""Phase 7.5 tests — @ file reference with autocomplete."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from prsm.tui.widgets.file_completer import (
    FileAttachment,
    FileEntry,
    FileIndex,
    build_tree_outline,
    resolve_references,
)

_DEMO_PATCH = patch("prsm.adapters.orchestrator.shutil.which", return_value=None)


# ── FileIndex Tests ──────────────────────────────────────────


class TestFileIndex:
    def test_scan_finds_files(self):
        """FileIndex discovers files in the directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "hello.py").write_text("print('hi')")
            (Path(tmpdir) / "README.md").write_text("# readme")

            idx = FileIndex(Path(tmpdir))
            results = idx.search("")
            paths = [e.path for e in results]
            assert "hello.py" in paths
            assert "README.md" in paths

    def test_scan_finds_nested(self):
        """FileIndex discovers nested files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = Path(tmpdir) / "src" / "lib"
            sub.mkdir(parents=True)
            (sub / "utils.py").write_text("pass")

            idx = FileIndex(Path(tmpdir))
            results = idx.search("src")
            paths = [e.path for e in results]
            assert any("src/" in p for p in paths)
            assert any("utils.py" in p for p in paths)

    def test_skip_dirs_excluded(self):
        """SKIP_DIRS like .git, __pycache__ are not indexed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / ".git").mkdir()
            (Path(tmpdir) / ".git" / "config").write_text("x")
            (Path(tmpdir) / "__pycache__").mkdir()
            (Path(tmpdir) / "__pycache__" / "mod.pyc").write_bytes(b"\x00")
            (Path(tmpdir) / "real.py").write_text("pass")

            idx = FileIndex(Path(tmpdir))
            results = idx.search("")
            paths = [e.path for e in results]
            assert "real.py" in paths
            assert not any(".git" in p for p in paths)
            assert not any("__pycache__" in p for p in paths)

    def test_gitignore_respected(self):
        """Entries matching .gitignore patterns are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / ".gitignore").write_text("*.pyc\nbuild/\n")
            (Path(tmpdir) / "main.py").write_text("pass")
            (Path(tmpdir) / "main.pyc").write_bytes(b"\x00")
            build_dir = Path(tmpdir) / "build"
            build_dir.mkdir()
            (build_dir / "out.js").write_text("x")

            idx = FileIndex(Path(tmpdir))
            results = idx.search("")
            paths = [e.path for e in results]
            assert "main.py" in paths
            assert "main.pyc" not in paths
            assert not any("build" in p for p in paths)

    def test_search_prefix_filtering(self):
        """search() narrows by prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "alpha.py").write_text("a")
            (Path(tmpdir) / "beta.py").write_text("b")
            (Path(tmpdir) / "gamma.py").write_text("g")

            idx = FileIndex(Path(tmpdir))
            results = idx.search("al")
            paths = [e.path for e in results]
            assert "alpha.py" in paths
            assert "beta.py" not in paths

    def test_search_substring_matching(self):
        """search() finds files by substring, not just prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = Path(tmpdir) / "src" / "screens"
            sub.mkdir(parents=True)
            (sub / "main.py").write_text("pass")
            (Path(tmpdir) / "other.py").write_text("pass")

            idx = FileIndex(Path(tmpdir))
            results = idx.search("main")
            paths = [e.path for e in results]
            assert any("main.py" in p for p in paths)
            assert "other.py" not in paths

    def test_search_prefix_before_substring(self):
        """Prefix matches appear before substring matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "app.py").write_text("a")
            sub = Path(tmpdir) / "src"
            sub.mkdir()
            (sub / "myapp.py").write_text("b")

            idx = FileIndex(Path(tmpdir))
            results = idx.search("app")
            paths = [e.path for e in results]
            # "app.py" is a prefix match, should come first
            assert paths.index("app.py") < paths.index("src/myapp.py")

    def test_dirs_sorted_first(self):
        """Directories appear before files in results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "zfile.py").write_text("z")
            (Path(tmpdir) / "adir").mkdir()

            idx = FileIndex(Path(tmpdir))
            results = idx.search("")
            # Filter to just our entries (gitignore may not exist)
            assert results[0].is_dir  # directory first

    def test_cache_staleness(self):
        """Index refreshes after TTL expires."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "initial.py").write_text("x")

            idx = FileIndex(Path(tmpdir))
            idx.CACHE_TTL = 0.0  # Expire immediately
            results1 = idx.search("")

            # Add new file
            (Path(tmpdir) / "new_file.py").write_text("y")
            results2 = idx.search("")
            paths2 = [e.path for e in results2]
            assert "new_file.py" in paths2

    def test_max_depth_respected(self):
        """Files deeper than MAX_DEPTH are not indexed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            deep = Path(tmpdir)
            for i in range(6):
                deep = deep / f"level{i}"
            deep.mkdir(parents=True)
            (deep / "deep_file.py").write_text("x")

            idx = FileIndex(Path(tmpdir))
            idx.MAX_DEPTH = 3
            # Force re-scan with new depth
            idx._last_scan = 0.0
            results = idx.search("")
            paths = [e.path for e in results]
            assert not any("deep_file.py" in p for p in paths)


# ── Tree Outline Tests ───────────────────────────────────────


class TestTreeOutline:
    def test_basic_tree(self):
        """Generates correct tree structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("x")
            (Path(tmpdir) / "b.py").write_text("y")
            sub = Path(tmpdir) / "sub"
            sub.mkdir()
            (sub / "c.py").write_text("z")

            outline = build_tree_outline(Path(tmpdir))
            assert "a.py" in outline
            assert "b.py" in outline
            assert "sub/" in outline
            assert "c.py" in outline

    def test_depth_limit(self):
        """Tree respects max_depth parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            deep = Path(tmpdir) / "a" / "b" / "c" / "d"
            deep.mkdir(parents=True)
            (deep / "hidden.py").write_text("x")

            outline = build_tree_outline(Path(tmpdir), max_depth=2)
            assert "hidden.py" not in outline

    def test_empty_directory(self):
        """Empty directory produces empty or minimal output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outline = build_tree_outline(Path(tmpdir))
            assert isinstance(outline, str)
            assert outline == ""

    def test_skip_dirs_in_tree(self):
        """.git, __pycache__ etc excluded from tree."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / ".git").mkdir()
            (Path(tmpdir) / ".git" / "HEAD").write_text("ref")
            (Path(tmpdir) / "real.py").write_text("x")

            outline = build_tree_outline(Path(tmpdir))
            assert ".git" not in outline
            assert "real.py" in outline


# ── Reference Resolution Tests ───────────────────────────────


class TestResolveReferences:
    def test_single_file_reference(self):
        """Single @file.py resolves to file content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "hello.py").write_text("print('hello')")

            text = "Look at @hello.py please"
            resolved, attachments = resolve_references(text, Path(tmpdir))
            assert len(attachments) == 1
            assert attachments[0].path == "hello.py"
            assert "print('hello')" in attachments[0].content
            assert not attachments[0].is_directory

    def test_single_dir_reference(self):
        """Single @dir/ resolves to tree outline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = Path(tmpdir) / "src"
            sub.mkdir()
            (sub / "main.py").write_text("x")

            text = "Check @src/ for me"
            resolved, attachments = resolve_references(text, Path(tmpdir))
            assert len(attachments) == 1
            assert attachments[0].is_directory
            assert "main.py" in attachments[0].content

    def test_dir_reference_has_no_default_depth_limit(self):
        """Directory @references include deeply nested files by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            deep = Path(tmpdir) / "a" / "b" / "c" / "d" / "e"
            deep.mkdir(parents=True)
            (deep / "deep_file.py").write_text("x")

            text = "Check @a/"
            resolved, attachments = resolve_references(text, Path(tmpdir))
            assert len(attachments) == 1
            assert attachments[0].is_directory
            assert "deep_file.py" in attachments[0].content

    def test_multiple_references(self):
        """Multiple @references in one prompt all resolve."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("aaa")
            (Path(tmpdir) / "b.py").write_text("bbb")

            text = "Compare @a.py with @b.py"
            resolved, attachments = resolve_references(text, Path(tmpdir))
            assert len(attachments) == 2

    def test_missing_reference_skipped(self):
        """@nonexistent.py produces no attachment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            text = "Look at @nonexistent.py"
            resolved, attachments = resolve_references(text, Path(tmpdir))
            assert len(attachments) == 0

    def test_file_size_limit(self):
        """Files exceeding MAX_FILE_SIZE are truncated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            big = Path(tmpdir) / "big.txt"
            big.write_text("x" * 200_000)  # 200KB > 100KB limit

            text = "Read @big.txt"
            resolved, attachments = resolve_references(text, Path(tmpdir))
            assert len(attachments) == 1
            assert attachments[0].truncated
            assert len(attachments[0].content) <= 100 * 1024 + 100

    def test_binary_file_skipped(self):
        """Binary files (with null bytes) are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "binary.bin").write_bytes(b"\x00\x01\x02\x03")

            text = "Look at @binary.bin"
            resolved, attachments = resolve_references(text, Path(tmpdir))
            assert len(attachments) == 0

    def test_at_inside_backticks_ignored(self):
        """@ inside backtick code spans is not treated as reference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "real.py").write_text("x")

            text = "Use `@decorator` but also @real.py"
            resolved, attachments = resolve_references(text, Path(tmpdir))
            assert len(attachments) == 1
            assert attachments[0].path == "real.py"

    def test_at_inside_code_block_ignored(self):
        """@ inside triple-backtick code blocks is not treated as reference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "real.py").write_text("x")

            text = "```\n@not_a_ref\n```\nBut @real.py is"
            resolved, attachments = resolve_references(text, Path(tmpdir))
            assert len(attachments) == 1
            assert attachments[0].path == "real.py"

    def test_at_start_of_line(self):
        """@reference at start of text works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file.py").write_text("content")

            text = "@file.py has a bug"
            resolved, attachments = resolve_references(text, Path(tmpdir))
            assert len(attachments) == 1

    def test_no_references_returns_empty(self):
        """Text without @ returns empty attachments list."""
        text = "Just a normal prompt"
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved, attachments = resolve_references(text, Path(tmpdir))
            assert len(attachments) == 0
            assert resolved == text


# ── ThinkingIndicator Tests ─────────────────────────────────


class TestThinkingIndicator:
    def test_thinking_indicator_has_verbs(self):
        """ThinkingIndicator has a list of verbs to cycle through."""
        from prsm.tui.widgets.thinking import THINKING_VERBS

        assert len(THINKING_VERBS) >= 10
        assert "Thinking" in THINKING_VERBS
        assert "Pondering" in THINKING_VERBS

    def test_thinking_indicator_creates(self):
        """ThinkingIndicator can be instantiated."""
        from prsm.tui.widgets.thinking import ThinkingIndicator

        indicator = ThinkingIndicator()
        assert indicator is not None
        assert "thinking-indicator" in indicator.classes


# ── Headless TUI Integration Tests ──────────────────────────


@pytest.mark.asyncio
async def test_at_shows_completer():
    """CompletionRequested message activates completion mode."""
    from prsm.tui.app import PrsmApp
    from prsm.tui.widgets.input_bar import InputBar, PromptInput

    app = PrsmApp()
    with _DEMO_PATCH:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            inp = screen.query_one("#input-bar", InputBar)
            editor = inp.query_one("#prompt-input", PromptInput)
            editor.focus()
            await pilot.pause()

            # Simulate what _on_key does when @ is typed:
            # insert text and post CompletionRequested
            editor.insert("Check @")
            editor.post_message(
                PromptInput.CompletionRequested(anchor=(0, 6)),
            )
            await pilot.pause()

            # PromptInput should be in completion mode
            assert editor.completion_active


@pytest.mark.asyncio
async def test_escape_dismisses_completer():
    """Pressing Escape while completer is open dismisses it."""
    from prsm.tui.app import PrsmApp
    from prsm.tui.widgets.input_bar import InputBar, PromptInput
    from prsm.tui.widgets.file_completer import FileCompleter

    app = PrsmApp()
    with _DEMO_PATCH:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            inp = screen.query_one("#input-bar", InputBar)
            editor = inp.query_one("#prompt-input", PromptInput)
            completer = inp.query_one("#file-completer", FileCompleter)
            editor.focus()
            await pilot.pause()

            # Activate completion
            editor.insert("@")
            await pilot.pause()

            # Now manually dismiss
            inp._dismiss_completer()
            await pilot.pause()

            assert not editor.completion_active
            assert not completer.has_class("visible")


@pytest.mark.asyncio
async def test_submit_clears_editor():
    """Submitting a prompt with @file clears the editor."""
    from prsm.tui.app import PrsmApp
    from prsm.tui.widgets.input_bar import InputBar, PromptInput

    app = PrsmApp()
    with _DEMO_PATCH:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            inp = screen.query_one("#input-bar", InputBar)
            editor = inp.query_one("#prompt-input", PromptInput)
            editor.focus()
            await pilot.pause()

            # Type a prompt and submit directly via _submit
            editor.insert("Read @pyproject.toml")
            inp._submit()
            await pilot.pause()

            # Editor should be cleared after submit
            assert editor.text.strip() == ""
