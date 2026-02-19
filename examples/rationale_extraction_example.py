#!/usr/bin/env python3
"""Example usage of the rationale extractor.

This demonstrates how to extract change rationale from agent conversation
history for use in commit messages, change logs, or documentation.
"""

from prsm.orchestrator.conversation_store import ConversationStore, ConversationEntry, EntryType
from prsm.orchestrator.rationale_extractor import extract_change_rationale


def example_commit_message_generation():
    """Example: Generate a commit message with rationale."""
    print("Example: Generating commit message with rationale")
    print("=" * 60)

    # Simulate an agent fixing a bug
    store = ConversationStore()
    agent_id = "worker-123"

    # Build up conversation history
    conversation = [
        ("user_message", "Fix the memory leak in the cache manager"),
        ("thinking", (
            "Looking at the cache manager code, I can see that the cache entries "
            "are never properly cleaned up when they expire. This causes memory "
            "to grow unbounded over time, especially on long-running servers. "
            "The issue is that we're adding entries to the cache but not removing "
            "stale ones. I need to implement automatic cleanup of expired entries."
        )),
        ("text", (
            "I'll add a cleanup mechanism to remove expired cache entries and "
            "prevent memory leaks."
        )),
        ("tool_call", "Edit", "edit-001", "cache_manager.py"),
        ("tool_result", "edit-001", "Successfully added cleanup logic"),
        ("thinking", (
            "Now I should add a test to ensure the cleanup works correctly and "
            "prevent regressions."
        )),
        ("text", "Adding tests to verify cleanup behavior."),
        ("tool_call", "Edit", "edit-002", "test_cache_manager.py"),
        ("tool_result", "edit-002", "Test added"),
    ]

    # Populate the store
    for entry_data in conversation:
        if entry_data[0] == "user_message":
            store.append(agent_id, ConversationEntry(
                entry_type=EntryType.USER_MESSAGE,
                content=entry_data[1]
            ))
        elif entry_data[0] == "thinking":
            store.append(agent_id, ConversationEntry(
                entry_type=EntryType.THINKING,
                content=entry_data[1]
            ))
        elif entry_data[0] == "text":
            store.append(agent_id, ConversationEntry(
                entry_type=EntryType.TEXT,
                content=entry_data[1]
            ))
        elif entry_data[0] == "tool_call":
            store.append(agent_id, ConversationEntry(
                entry_type=EntryType.TOOL_CALL,
                tool_name=entry_data[1],
                tool_id=entry_data[2],
                tool_args=entry_data[3]
            ))
        elif entry_data[0] == "tool_result":
            store.append(agent_id, ConversationEntry(
                entry_type=EntryType.TOOL_RESULT,
                tool_id=entry_data[1],
                content=entry_data[2]
            ))

    # Extract rationale for the cache manager edit
    rationale = extract_change_rationale(agent_id, "edit-001", store)

    print("\nExtracted rationale:")
    print(f"  {rationale}")

    # Generate commit message
    print("\nGenerated commit message:")
    print("─" * 60)
    print("fix: Implement automatic cleanup of expired cache entries")
    print()
    print(rationale)
    print()
    print("Co-Authored-By: Claude Agent <agent@prsm.dev>")
    print("─" * 60)
    print()


def example_change_log_entry():
    """Example: Generate a changelog entry."""
    print("Example: Generating changelog entry")
    print("=" * 60)

    store = ConversationStore()
    agent_id = "worker-456"

    # Simulate adding a new feature
    conversation = [
        ("thinking", (
            "Users have been requesting the ability to export data in CSV format. "
            "Currently we only support JSON exports, which is not ideal for "
            "importing into spreadsheet applications. Adding CSV export will "
            "improve the user experience significantly."
        )),
        ("text", "I'll add CSV export functionality to the export module."),
        ("tool_call", "Write", "write-001", "export.py"),
        ("tool_result", "write-001", "CSV export added"),
    ]

    for entry_data in conversation:
        if entry_data[0] == "thinking":
            store.append(agent_id, ConversationEntry(
                entry_type=EntryType.THINKING,
                content=entry_data[1]
            ))
        elif entry_data[0] == "text":
            store.append(agent_id, ConversationEntry(
                entry_type=EntryType.TEXT,
                content=entry_data[1]
            ))
        elif entry_data[0] == "tool_call":
            store.append(agent_id, ConversationEntry(
                entry_type=EntryType.TOOL_CALL,
                tool_name=entry_data[1],
                tool_id=entry_data[2],
                tool_args=entry_data[3]
            ))
        elif entry_data[0] == "tool_result":
            store.append(agent_id, ConversationEntry(
                entry_type=EntryType.TOOL_RESULT,
                tool_id=entry_data[1],
                content=entry_data[2]
            ))

    rationale = extract_change_rationale(agent_id, "write-001", store)

    print("\nExtracted rationale:")
    print(f"  {rationale}")

    print("\nChangelog entry:")
    print("─" * 60)
    print("### Added")
    print("- CSV export functionality")
    print(f"  - {rationale}")
    print("─" * 60)
    print()


def example_multi_file_change():
    """Example: Extract rationale for multi-file refactoring."""
    print("Example: Multi-file refactoring rationale")
    print("=" * 60)

    store = ConversationStore()
    agent_id = "worker-789"

    conversation = [
        ("thinking", (
            "The current authentication code is scattered across multiple modules, "
            "making it hard to maintain and test. Centralizing the auth logic "
            "into a dedicated module will improve code organization and make it "
            "easier to add new authentication methods in the future."
        )),
        ("tool_call", "Write", "write-001", "auth/core.py"),
        ("tool_result", "write-001", "Created core auth module"),
        ("tool_call", "Edit", "edit-001", "api/routes.py"),
        ("tool_result", "edit-001", "Updated to use new auth module"),
        ("tool_call", "Edit", "edit-002", "api/middleware.py"),
        ("tool_result", "edit-002", "Updated to use new auth module"),
    ]

    for entry_data in conversation:
        if entry_data[0] == "thinking":
            store.append(agent_id, ConversationEntry(
                entry_type=EntryType.THINKING,
                content=entry_data[1]
            ))
        elif entry_data[0] == "tool_call":
            store.append(agent_id, ConversationEntry(
                entry_type=EntryType.TOOL_CALL,
                tool_name=entry_data[1],
                tool_id=entry_data[2],
                tool_args=entry_data[3]
            ))
        elif entry_data[0] == "tool_result":
            store.append(agent_id, ConversationEntry(
                entry_type=EntryType.TOOL_RESULT,
                tool_id=entry_data[1],
                content=entry_data[2]
            ))

    # Same rationale applies to all edits in this refactoring
    rationale = extract_change_rationale(agent_id, "write-001", store)

    print("\nExtracted rationale:")
    print(f"  {rationale}")

    print("\nCommit message for refactoring:")
    print("─" * 60)
    print("refactor: Centralize authentication logic into dedicated module")
    print()
    print(rationale)
    print()
    print("Files changed:")
    print("  - auth/core.py (new)")
    print("  - api/routes.py")
    print("  - api/middleware.py")
    print("─" * 60)
    print()


if __name__ == "__main__":
    print("\nRationale Extraction Examples")
    print("=" * 60)
    print()

    example_commit_message_generation()
    print()

    example_change_log_entry()
    print()

    example_multi_file_change()
    print()

    print("=" * 60)
    print("These examples demonstrate how the rationale extractor")
    print("can be used to generate meaningful commit messages and")
    print("documentation that explain WHY changes were made.")
