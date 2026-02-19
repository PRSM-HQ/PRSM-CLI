#!/usr/bin/env python3
"""Demo script showing settings-menu bash whitelist/blacklist functionality."""

import tempfile
from pathlib import Path

from prsm.shared.services.command_policy_store import CommandPolicyStore


def main():
    print("=== PRSM Settings Menu - Bash Command Policy Demo ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        store = CommandPolicyStore(workspace)

        print(f"Workspace: {workspace}")
        print(f"Policy files will be created at: {workspace / '.prism'}\n")

        # Initialize
        store.ensure_files()
        print("✓ Created .prism/command_whitelist.txt and command_blacklist.txt\n")

        # Add whitelist patterns
        print("Adding whitelist patterns (auto-allow commands):")
        patterns_wl = [r"npm\s+test", r"git\s+status", r"ls\s+-la"]
        for pattern in patterns_wl:
            store.add_whitelist_pattern(pattern)
            print(f"  ✓ Added: {pattern}")

        print()

        # Add blacklist patterns
        print("Adding blacklist patterns (always prompt):")
        patterns_bl = [r"rm\s+-rf", r"docker\s+volume\s+rm", r"sudo\s+.*"]
        for pattern in patterns_bl:
            store.add_blacklist_pattern(pattern)
            print(f"  ✓ Added: {pattern}")

        print()

        # Read back
        whitelist = store.read_whitelist()
        blacklist = store.read_blacklist()

        print(f"Current whitelist ({len(whitelist)} patterns):")
        for pat in whitelist:
            print(f"  - {pat}")

        print()
        print(f"Current blacklist ({len(blacklist)} patterns):")
        for pat in blacklist:
            print(f"  - {pat}")

        print()

        # Remove a pattern
        print("Removing 'ls\\s+-la' from whitelist:")
        removed = store.remove_whitelist_pattern(r"ls\s+-la")
        print(f"  {'✓ Removed' if removed else '✗ Not found'}")

        print()

        # Show updated
        whitelist = store.read_whitelist()
        print(f"Updated whitelist ({len(whitelist)} patterns):")
        for pat in whitelist:
            print(f"  - {pat}")

        print()

        # Show compiled rules
        print("Loading and compiling regex patterns:")
        rules = store.load_compiled()
        print(f"  ✓ Compiled {len(rules.whitelist)} whitelist patterns")
        print(f"  ✓ Compiled {len(rules.blacklist)} blacklist patterns")

        print("\n=== Persistence ===")
        print(f"Files written to:")
        print(f"  - {store.whitelist_path}")
        print(f"  - {store.blacklist_path}")

        print("\n=== UI Access ===")
        print("In the TUI, press F2 to open Settings menu")
        print("Settings screen shows:")
        print("  - Whitelist section with Add/Remove buttons")
        print("  - Blacklist section with Add/Remove buttons")
        print("  - All changes are persisted to .prism/ directory")
        print("  - Patterns are regex and matched against bash commands")


if __name__ == "__main__":
    main()
