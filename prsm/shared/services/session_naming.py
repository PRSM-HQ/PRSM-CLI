"""Auto-generate descriptive session names from user prompts.

Uses a lightweight LLM call to produce a short title.
Non-blocking: falls back to a deterministic title derived from the prompt.
"""
from __future__ import annotations

import asyncio
import ast
import json
import logging
import re

logger = logging.getLogger(__name__)

NAMING_PROMPT = (
    "Generate a short, descriptive title (3-7 words) for a coding session "
    "that starts with this user message. Return ONLY the title, nothing else. "
    "Use imperative form (e.g., 'Fix auth login bug', 'Add dark mode toggle', "
    "'Refactor database queries'). Do not use quotes.\n\n"
    "User message: {message}"
)

_PRIMARY_TITLE_KEYS = ("title", "name", "session_name", "sessiontitle")
_SECONDARY_TEXT_KEYS = (
    "content",
    "output",
    "text",
    "message",
    "response",
    "result",
    "payload",
    "data",
    "choices",
)
_METADATA_KEYS = {
    "jsonrpc",
    "id",
    "method",
    "params",
    "type",
    "role",
    "threadid",
    "thread_id",
    "metadata",
    "usage",
    "created",
    "timestamp",
    "status",
    "success",
    "finish_reason",
    "event",
    "event_type",
    "tool",
    "tool_name",
    "model",
    "model_id",
    "provider",
    "cwd",
    "path",
    "workspace",
    "workdir",
    "directory",
    "project_dir",
    "repo_root",
    "root",
}


async def generate_session_name(
    user_message: str,
    model_id: str = "gpt-5-3-spark",
) -> str | None:
    """Generate a session name from the user's first message.

    Returns a deterministic fallback title if model-based naming fails.
    """
    fallback = _fallback_session_name(user_message)
    try:
        from prsm.engine.providers.codex_provider import CodexProvider

        provider = CodexProvider(default_model=model_id)
        if not provider.is_available():
            return fallback

        prompt = NAMING_PROMPT.format(message=user_message[:500])
        result = await asyncio.wait_for(
            provider.send_message(prompt, model_id=model_id),
            timeout=15.0,
        )

        if result.success and result.text:
            # Clean up the result — strip quotes, newlines, limit length
            raw = result.text.strip()
            name = _extract_name_from_model_output(raw)
            # Limit to ~60 chars
            if len(name) > 60:
                name = name[:57] + "..."
            return name if name else fallback
        return fallback

    except asyncio.TimeoutError:
        logger.debug("Session naming timed out")
        return fallback
    except Exception:
        logger.debug("Session naming failed", exc_info=True)
        return fallback


async def generate_session_metadata(
    user_message: str,
    model_id: str = "gpt-5-3-spark",
) -> tuple[str | None, str]:
    """Generate both session title and summary metadata."""
    name = await generate_session_name(user_message, model_id=model_id)
    summary = _fallback_session_summary(user_message)
    return name, summary


def _fallback_session_name(user_message: str) -> str:
    """Generate a stable fallback title when model naming is unavailable."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", user_message or "")
    filtered = [w for w in words if not _looks_like_model_token(w)]
    chosen = filtered[:7]
    if len(chosen) < 3:
        return "Implement requested feature changes"
    title = " ".join(chosen).strip()
    if len(title) > 60:
        title = title[:57].rstrip() + "..."
    return title or "Implement requested feature changes"


def _fallback_session_summary(user_message: str) -> str:
    """Generate a compact one-line summary of the initial prompt."""
    text = " ".join((user_message or "").strip().split())
    if not text:
        return ""
    words = text.split(" ")
    if len(words) <= 30:
        return text
    return " ".join(words[:30]).rstrip() + "..."


def _extract_name_from_model_output(raw_text: str) -> str:
    """Normalize model output into a plain session title string."""
    if not raw_text:
        return ""

    trimmed = _strip_code_fence(raw_text.strip()).strip().strip('"\'').strip()
    if not trimmed:
        return ""

    parsed = _parse_structured_output(trimmed)
    if parsed is not None:
        extracted = _extract_title_from_structure(parsed).strip()
        return extracted if _is_plausible_title(extracted) else ""

    return trimmed if _is_plausible_title(trimmed) else ""


def _strip_code_fence(text: str) -> str:
    """Extract fenced content when models wrap JSON in Markdown blocks."""
    fence = re.match(r"^```(?:json|JSON)?\s*\n(?P<body>[\s\S]*?)\n```$", text)
    if fence:
        return fence.group("body").strip()
    return text


def _parse_structured_output(text: str) -> object | None:
    """Parse JSON or Python-literal shaped model output."""
    if not text:
        return None
    if text[0] not in "{[":
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def _extract_title_from_structure(value: object, *, key_hint: str | None = None) -> str:
    """Recursively find a human title in structured model output."""
    if isinstance(value, str):
        cleaned = value.strip().strip('"\'').strip()
        reparsed = _parse_structured_output(cleaned)
        if reparsed is not None and reparsed is not value:
            nested = _extract_title_from_structure(reparsed, key_hint=key_hint)
            if nested:
                return nested
        if _is_metadata_key(key_hint):
            return ""
        return cleaned if _is_plausible_title(cleaned) else ""

    if isinstance(value, dict):
        items: list[tuple[str, object]] = []
        for raw_key, item in value.items():
            key = raw_key if isinstance(raw_key, str) else str(raw_key)
            items.append((key, item))

        for preferred in _PRIMARY_TITLE_KEYS:
            for key, item in items:
                if key.lower() != preferred:
                    continue
                nested = _extract_title_from_structure(item, key_hint=key)
                if nested:
                    return nested

        for preferred in _SECONDARY_TEXT_KEYS:
            for key, item in items:
                if key.lower() != preferred:
                    continue
                nested = _extract_title_from_structure(item, key_hint=key)
                if nested:
                    return nested

        # Prefer nested containers before scalar metadata.
        for key, item in items:
            key_l = key.lower()
            if key_l in _PRIMARY_TITLE_KEYS or key_l in _SECONDARY_TEXT_KEYS:
                continue
            if not isinstance(item, (dict, list)):
                continue
            nested = _extract_title_from_structure(item, key_hint=key)
            if nested:
                return nested

        for key, item in items:
            key_l = key.lower()
            if key_l in _PRIMARY_TITLE_KEYS or key_l in _SECONDARY_TEXT_KEYS:
                continue
            nested = _extract_title_from_structure(item, key_hint=key)
            if nested:
                return nested
        return ""

    if isinstance(value, list):
        for item in value:
            nested = _extract_title_from_structure(item)
            if nested:
                return nested
        return ""

    return ""


def _is_plausible_title(text: str) -> bool:
    """Reject obvious metadata values and keep short human-readable titles."""
    if not text:
        return False
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return False
    if len(normalized) > 80:
        return False
    if re.fullmatch(r"\d+(?:\.\d+)?", normalized):
        return False
    if not any(ch.isalpha() for ch in normalized):
        return False
    if _looks_like_model_identifier(normalized):
        return False
    if _looks_like_filesystem_path(normalized):
        return False
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", normalized)
    if len(words) < 3:
        return False
    if len(words) > 7:
        return False
    if all(_looks_like_model_token(word) for word in words):
        return False
    return True


def _looks_like_filesystem_path(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    if value.startswith("~"):
        return True
    if "/" in value or "\\" in value:
        return True
    if re.fullmatch(r"[A-Za-z]:[\\/].+", value):
        return True
    return False


def _is_metadata_key(key: str | None) -> bool:
    if not key:
        return False
    return key.lower() in _METADATA_KEYS


def _looks_like_model_token(token: str) -> bool:
    t = (token or "").strip().lower()
    if not t:
        return False
    if "::reasoning_effort" in t:
        return True
    return bool(re.search(r"(gpt|claude|gemini|minimax|spark|reasoning_effort)", t))


def _looks_like_model_identifier(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    if lowered.startswith("gpt-") or lowered.startswith("claude-") or lowered.startswith("gemini-"):
        return True
    if "::reasoning_effort" in lowered:
        return True
    return bool(re.fullmatch(r"[a-z0-9._:-]{3,}", lowered) and _looks_like_model_token(lowered))
