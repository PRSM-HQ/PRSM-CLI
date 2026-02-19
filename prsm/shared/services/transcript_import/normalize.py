"""Normalization helpers for provider transcript import."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def parse_timestamp(value: Any) -> datetime | None:
    """Parse common provider timestamp formats into aware UTC datetimes."""
    if not value or not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def coerce_text(value: Any) -> str:
    """Render arbitrary values into stable text for transcript storage."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def extract_text_from_blocks(blocks: Any, *, include_thinking: bool = False) -> str:
    """Extract visible text from provider content blocks."""
    if blocks is None:
        return ""
    if isinstance(blocks, str):
        return blocks
    if isinstance(blocks, dict):
        if isinstance(blocks.get("text"), str):
            return blocks["text"]
        if isinstance(blocks.get("content"), str):
            return blocks["content"]
        return coerce_text(blocks)
    if not isinstance(blocks, list):
        return coerce_text(blocks)

    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            if block:
                parts.append(block)
            continue
        if not isinstance(block, dict):
            text = coerce_text(block)
            if text:
                parts.append(text)
            continue

        block_type = str(block.get("type") or "").lower()
        if block_type in {"input_text", "output_text", "text", "summary_text"}:
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
            continue
        if block_type == "thinking":
            if include_thinking:
                thinking = block.get("thinking")
                if isinstance(thinking, str) and thinking:
                    parts.append(thinking)
            continue
        if block_type == "tool_result":
            content = block.get("content")
            if isinstance(content, str) and content:
                parts.append(content)
            continue
        if isinstance(block.get("text"), str):
            parts.append(block["text"])
            continue
        if isinstance(block.get("content"), str):
            parts.append(block["content"])

    return "\n\n".join(part for part in parts if part).strip()

