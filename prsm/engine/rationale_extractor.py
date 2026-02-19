"""Extract change rationale from agent conversation history.

Analyzes thinking blocks and text output around tool calls to extract
the reasoning behind code changes. Used for generating commit messages
and change documentation that explain WHY changes were made.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conversation_store import ConversationStore


def extract_change_rationale(
    agent_id: str,
    tool_call_id: str,
    conversation_store: ConversationStore,
    max_sentences: int = 3,
) -> str:
    """Extract the rationale for a specific tool call.

    Analyzes the conversation history around a tool call to find thinking
    blocks and text that explain WHY the change was made. Focuses on
    extracting motivation and intent rather than implementation details.

    Args:
        agent_id: The agent whose conversation to analyze
        tool_call_id: The specific tool call ID to find rationale for
        conversation_store: The conversation history store
        max_sentences: Maximum number of sentences to extract (default: 3)

    Returns:
        A clean string containing 1-3 sentences explaining the rationale,
        or an empty string if no clear rationale is found.

    Example:
        >>> rationale = extract_change_rationale(
        ...     "agent-123",
        ...     "tool-456",
        ...     conversation_store
        ... )
        >>> print(rationale)
        "The authentication flow needed to handle expired tokens more gracefully.
        Users were experiencing unexpected logouts during long sessions."
    """
    # Get full conversation history
    history = conversation_store.get_history(agent_id, detail_level="full")

    if not history:
        return ""

    # Find the index of the target tool call
    tool_call_index = -1
    for i, entry in enumerate(history):
        if (entry.get("type") == "tool_call" and
            entry.get("tool_id") == tool_call_id):
            tool_call_index = i
            break

    if tool_call_index == -1:
        return ""

    # Look backward from the tool call to find relevant thinking/text
    # We'll examine up to 10 entries before the tool call
    lookback_window = min(10, tool_call_index)
    start_index = tool_call_index - lookback_window

    rationale_candidates: list[str] = []

    # Extract thinking blocks and text before the tool call
    for i in range(start_index, tool_call_index):
        entry = history[i]
        entry_type = entry.get("type")
        content = entry.get("content", "")

        if not content:
            continue

        if entry_type == "thinking":
            # Extract rationale from thinking blocks
            sentences = _extract_rationale_sentences(content)
            rationale_candidates.extend(sentences)
        elif entry_type == "text":
            # Extract rationale from text output (agent explaining what it's doing)
            sentences = _extract_rationale_sentences(content)
            rationale_candidates.extend(sentences)

    # Filter and rank the candidates
    filtered = _filter_rationale_candidates(rationale_candidates)

    # Return top N sentences
    selected = filtered[:max_sentences]

    if not selected:
        return ""

    return " ".join(selected)


def extract_structured_evidence(
    agent_id: str,
    tool_call_id: str,
    conversation_store: ConversationStore,
) -> dict[str, object]:
    """Extract structured evidence (assumptions/risks/verification) when present."""
    history = conversation_store.get_history(agent_id, detail_level="full")
    if not history:
        return {}

    for entry in history:
        if (
            entry.get("type") == "tool_call"
            and entry.get("tool_id") == tool_call_id
            and entry.get("tool_name") == "task_complete"
        ):
            parsed = _parse_tool_args(entry.get("tool_args", ""))
            if not isinstance(parsed, dict):
                return {}
            evidence: dict[str, object] = {}
            for key in (
                "steps",
                "assumptions",
                "risks",
                "rollback_plan",
                "confidence",
                "verification_results",
            ):
                if key in parsed:
                    evidence[key] = parsed[key]
            return evidence
    return {}


def extract_rationale_with_evidence(
    agent_id: str,
    tool_call_id: str,
    conversation_store: ConversationStore,
    max_sentences: int = 3,
) -> dict[str, object]:
    """Return legacy rationale plus structured evidence when available."""
    return {
        "rationale": extract_change_rationale(
            agent_id=agent_id,
            tool_call_id=tool_call_id,
            conversation_store=conversation_store,
            max_sentences=max_sentences,
        ),
        "structured_evidence": extract_structured_evidence(
            agent_id=agent_id,
            tool_call_id=tool_call_id,
            conversation_store=conversation_store,
        ),
    }


def _extract_rationale_sentences(text: str) -> list[str]:
    """Extract sentences that explain reasoning from text.

    Looks for sentences containing rationale keywords and filters out
    implementation details.
    """
    # Split into sentences (simple approach)
    sentences = re.split(r'[.!?]+\s+', text.strip())

    # Keywords that indicate rationale/motivation
    rationale_keywords = [
        "need", "require", "should", "must", "want",
        "improve", "fix", "address", "solve", "handle",
        "ensure", "prevent", "avoid", "allow", "enable",
        "because", "since", "so that", "in order to",
        "issue", "problem", "bug", "error",
        "user", "customer", "experience",
        "performance", "security", "maintainability",
    ]

    # Keywords that indicate implementation details (avoid these)
    implementation_keywords = [
        "create a", "add a", "update the", "modify the",
        "write a", "implement", "use the", "call the",
        "first", "then", "next", "finally",
        "let me", "i will", "i'll", "i'm going to",
    ]

    candidates = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 20:
            continue

        lower_sentence = sentence.lower()

        # Skip if it's primarily implementation details
        if any(keyword in lower_sentence for keyword in implementation_keywords):
            # Unless it also has strong rationale keywords
            if not any(keyword in lower_sentence for keyword in rationale_keywords[:10]):
                continue

        # Prefer sentences with rationale keywords
        has_rationale = any(keyword in lower_sentence for keyword in rationale_keywords)

        if has_rationale:
            # Clean up the sentence
            cleaned = _clean_sentence(sentence)
            if cleaned:
                candidates.append(cleaned)

    return candidates


def _clean_sentence(sentence: str) -> str:
    """Clean up a sentence for use in rationale.

    Removes agent-speak and meta-commentary.
    """
    # Remove common agent prefixes
    prefixes_to_remove = [
        r"^(I'll|I will|I'm going to|Let me|First,?|Next,?|Now,?|Then,?)\s+",
        r"^(I need to|I should|I must)\s+",
        r"^(The task|This task|We need to|We should)\s+",
    ]

    cleaned = sentence
    for pattern in prefixes_to_remove:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    # Remove markdown formatting
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)  # **bold**
    cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)      # *italic*
    cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)        # `code`

    # Capitalize first letter
    cleaned = cleaned.strip()
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned


def _filter_rationale_candidates(candidates: list[str]) -> list[str]:
    """Filter and rank rationale candidates.

    Returns the most relevant sentences in priority order.
    """
    if not candidates:
        return []

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for candidate in candidates:
        # Normalize for comparison
        normalized = candidate.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(candidate)

    # Score each candidate
    scored = []
    for candidate in unique:
        score = _score_rationale(candidate)
        scored.append((score, candidate))

    # Sort by score (descending)
    scored.sort(reverse=True, key=lambda x: x[0])

    # Return just the sentences
    return [sentence for _, sentence in scored]


def _score_rationale(sentence: str) -> float:
    """Score a rationale sentence for relevance.

    Higher scores indicate better rationale sentences.
    """
    score = 0.0
    lower = sentence.lower()

    # Strong rationale indicators (worth more points)
    strong_indicators = [
        ("fix", 3.0),
        ("bug", 3.0),
        ("error", 2.5),
        ("issue", 2.5),
        ("problem", 2.5),
        ("improve", 2.0),
        ("performance", 2.0),
        ("security", 2.0),
        ("user", 1.5),
        ("experience", 1.5),
        ("ensure", 1.5),
        ("prevent", 1.5),
        ("handle", 1.5),
    ]

    for keyword, points in strong_indicators:
        if keyword in lower:
            score += points

    # Penalize vague or meta sentences
    penalties = [
        ("i'll", -2.0),
        ("let me", -2.0),
        ("first", -1.0),
        ("then", -1.0),
        ("next", -1.0),
    ]

    for keyword, points in penalties:
        if keyword in lower:
            score += points

    # Bonus for reasonable length (not too short, not too long)
    word_count = len(sentence.split())
    if 10 <= word_count <= 30:
        score += 1.0
    elif word_count < 5:
        score -= 2.0

    return score


def _parse_tool_args(tool_args: str) -> dict[str, object] | None:
    if not tool_args:
        return None
    try:
        parsed = json.loads(tool_args)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(parsed, dict):
        return parsed
    return None
