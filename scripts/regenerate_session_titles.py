#!/usr/bin/env python3
"""Regenerate persisted PRSM session titles from each session's first user prompt."""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prsm.shared.services.session_naming import (
    _fallback_session_name,
    generate_session_name,
)


@dataclass
class SessionRename:
    path: Path
    session_id: str
    old_name: str
    new_name: str
    prompt: str


def _first_user_prompt(data: dict[str, Any]) -> str:
    messages = data.get("messages")
    if not isinstance(messages, dict):
        return ""
    candidates: list[tuple[str, str]] = []
    for thread in messages.values():
        if not isinstance(thread, list):
            continue
        for msg in thread:
            if not isinstance(msg, dict):
                continue
            if str(msg.get("role", "")).lower() != "user":
                continue
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            timestamp = str(msg.get("timestamp") or "")
            candidates.append((timestamp, content))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


async def _name_for_prompt(prompt: str) -> str:
    if not prompt:
        return "Implement requested feature changes"
    try:
        generated = await generate_session_name(prompt)
    except Exception:
        generated = None
    if generated:
        return generated
    return _fallback_session_name(prompt)


def _iter_session_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("**/*.json"))


async def _collect_renames(files: list[Path]) -> list[SessionRename]:
    renames: list[SessionRename] = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        prompt = _first_user_prompt(data)
        if not prompt:
            continue
        new_name = await _name_for_prompt(prompt)
        old_name = str(data.get("name") or "").strip()
        session_id = str(data.get("session_id") or path.stem)
        if new_name and new_name != old_name:
            renames.append(
                SessionRename(
                    path=path,
                    session_id=session_id,
                    old_name=old_name,
                    new_name=new_name,
                    prompt=prompt,
                )
            )
    return renames


def _apply_renames(renames: list[SessionRename]) -> int:
    changed = 0
    for rename in renames:
        try:
            data = json.loads(rename.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            data["name"] = rename.new_name
            rename.path.write_text(
                json.dumps(data, indent=2) + "\n",
                encoding="utf-8",
            )
            changed += 1
        except Exception:
            continue
    return changed


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sessions-root",
        default=str(Path.home() / ".prsm" / "sessions"),
        help="Root directory containing persisted session JSON files.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write regenerated titles back to session JSON files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of files to process (0 means all).",
    )
    args = parser.parse_args()

    files = _iter_session_files(Path(args.sessions_root))
    if args.limit > 0:
        files = files[: args.limit]

    renames = await _collect_renames(files)
    print(f"Scanned files: {len(files)}")
    print(f"Title updates needed: {len(renames)}")
    for row in renames:
        print(f"{row.session_id} | {row.old_name!r} -> {row.new_name!r}")

    if args.apply and renames:
        changed = _apply_renames(renames)
        print(f"Applied updates: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
