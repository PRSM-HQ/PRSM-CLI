#!/usr/bin/env bash
#
# publish.sh — Push the current branch to upstream-deze:master as a single
# squashed commit, without modifying local history.
#
# Uses a temporary orphan branch to create a fresh commit with only
# the initial-deze identity, then pushes it and cleans up.
#
set -euo pipefail

REMOTE="upstream-deze"
REMOTE_BRANCH="master"
AUTHOR_NAME="initial-deze"
AUTHOR_EMAIL="initial-deze@users.noreply.github.com"
COMMIT_MSG="Initial release of PRSM CLI

AI-powered development assistant with VS Code extension, multi-provider LLM engine, and MCP integration."

# ── Helpers ──

info()  { printf '\033[1;34m=>\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m=>\033[0m %s\n' "$*"; }
err()   { printf '\033[1;31m=>\033[0m %s\n' "$*" >&2; }

# ── Validation ──

if [ "$#" -ne 0 ]; then
    echo "Usage: $0" >&2
    echo "" >&2
    echo "Pushes current branch to $REMOTE:$REMOTE_BRANCH as a single squashed commit." >&2
    echo "Local history is NOT modified." >&2
    exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" = "HEAD" ]; then
    err "Cannot publish from detached HEAD. Check out a branch first."
    exit 1
fi

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
    err "Remote '$REMOTE' is not configured."
    err "Add it with: git remote add $REMOTE git@github-deze:PRSM-HQ/PRSM-CLI.git"
    exit 1
fi

# ── Create squashed commit on a temporary branch ──

TEMP_BRANCH="_publish_squash_$$"

info "Creating squashed commit from '$BRANCH'..."

# Create an orphan branch with the current tree
git checkout --orphan "$TEMP_BRANCH" >/dev/null 2>&1

# Stage everything (the working tree already has all files from the real branch)
git add -A >/dev/null 2>&1

# Commit with deze identity
GIT_AUTHOR_NAME="$AUTHOR_NAME" \
GIT_AUTHOR_EMAIL="$AUTHOR_EMAIL" \
GIT_COMMITTER_NAME="$AUTHOR_NAME" \
GIT_COMMITTER_EMAIL="$AUTHOR_EMAIL" \
git commit -m "$COMMIT_MSG" --quiet

SQUASHED_SHA="$(git rev-parse HEAD)"

ok "Squashed commit: $SQUASHED_SHA"

# ── Push ──

info "Pushing to $REMOTE $REMOTE_BRANCH..."
if git push "$REMOTE" "$TEMP_BRANCH:$REMOTE_BRANCH" --force; then
    ok "Pushed to $REMOTE $REMOTE_BRANCH"
else
    err "Push failed."
    # Clean up before exiting
    git checkout "$BRANCH" --quiet 2>/dev/null || true
    git branch -D "$TEMP_BRANCH" --quiet 2>/dev/null || true
    exit 1
fi

# ── Clean up: switch back to original branch and delete temp ──

git checkout "$BRANCH" --quiet 2>/dev/null
git branch -D "$TEMP_BRANCH" --quiet 2>/dev/null

ok "Done. Local branch '$BRANCH' is unchanged."
