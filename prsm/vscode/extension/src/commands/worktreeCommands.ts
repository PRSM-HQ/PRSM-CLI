/**
 * VSCode commands for managing git worktrees.
 *
 * - prsm.worktree.list    — List all worktrees in a QuickPick
 * - prsm.worktree.create  — Create a new worktree with branch name
 * - prsm.worktree.remove  — Remove a worktree (with confirmation)
 * - prsm.worktree.switch  — Switch to a different worktree (opens new window)
 */
import * as vscode from "vscode";
import * as path from "path";
import { WorktreeManager, WorktreeInfo } from "../git/worktreeManager";

/**
 * Register all worktree management commands.
 */
export function registerWorktreeCommands(
  context: vscode.ExtensionContext,
  worktreeManager: WorktreeManager
): void {
  context.subscriptions.push(
    vscode.commands.registerCommand(
      "prsm.worktree.list",
      () => listWorktrees(worktreeManager)
    ),
    vscode.commands.registerCommand(
      "prsm.worktree.create",
      () => createWorktree(worktreeManager)
    ),
    vscode.commands.registerCommand(
      "prsm.worktree.remove",
      () => removeWorktree(worktreeManager)
    ),
    vscode.commands.registerCommand(
      "prsm.worktree.switch",
      () => switchWorktree(worktreeManager)
    )
  );
}

// ── List Worktrees ──

async function listWorktrees(
  manager: WorktreeManager
): Promise<void> {
  const ctx = await manager.getContext();
  if (!ctx.isGitRepo) {
    vscode.window.showWarningMessage(
      "Not inside a git repository."
    );
    return;
  }

  const worktrees = await manager.list();
  if (worktrees.length === 0) {
    vscode.window.showInformationMessage("No worktrees found.");
    return;
  }

  const picks = worktrees.map((wt) => formatWorktreePick(wt));

  const selected = await vscode.window.showQuickPick(picks, {
    title: "Git Worktrees",
    placeHolder: "Select a worktree to open in a new window",
  });

  if (selected && !selected.isCurrent) {
    const uri = vscode.Uri.file(selected.worktreePath);
    await vscode.commands.executeCommand("vscode.openFolder", uri, {
      forceNewWindow: true,
    });
  }
}

// ── Create Worktree ──

async function createWorktree(
  manager: WorktreeManager
): Promise<void> {
  const ctx = await manager.getContext();
  if (!ctx.isGitRepo) {
    vscode.window.showWarningMessage(
      "Not inside a git repository."
    );
    return;
  }

  // Step 1: Branch name
  const branchName = await vscode.window.showInputBox({
    title: "Create Worktree (1/2): Branch Name",
    prompt: "Name for the new branch (or existing branch to checkout)",
    placeHolder: "e.g., feature/new-feature",
    validateInput: (value) => {
      if (!value.trim()) {
        return "Branch name is required";
      }
      if (/\s/.test(value)) {
        return "Branch name cannot contain spaces";
      }
      if (/\.\./.test(value)) {
        return "Branch name cannot contain '..'";
      }
      return null;
    },
  });
  if (!branchName) return;

  // Step 2: Target path
  // Default: sibling directory with branch name as folder
  const mainPath = ctx.mainWorktreePath ?? ctx.worktreePath;
  const parentDir = mainPath ? path.dirname(mainPath) : undefined;
  const safeName = branchName.replace(/\//g, "-");
  const defaultPath = parentDir
    ? path.join(parentDir, safeName)
    : undefined;

  const targetPath = await vscode.window.showInputBox({
    title: "Create Worktree (2/2): Directory",
    prompt: "Path for the new worktree directory",
    value: defaultPath,
    placeHolder: "/path/to/new/worktree",
    validateInput: (value) => {
      if (!value.trim()) {
        return "Path is required";
      }
      return null;
    },
  });
  if (!targetPath) return;

  // Optional: base branch
  const baseBranch = await vscode.window.showInputBox({
    title: "Base Branch (optional)",
    prompt:
      "Create the new branch from this base (leave empty for current HEAD)",
    placeHolder: "e.g., main",
  });

  try {
    await manager.create(
      targetPath,
      branchName,
      baseBranch || undefined
    );
    const openNow = await vscode.window.showInformationMessage(
      `Worktree created at ${targetPath} (branch: ${branchName})`,
      "Open in New Window",
      "Stay Here"
    );
    if (openNow === "Open in New Window") {
      const uri = vscode.Uri.file(targetPath);
      await vscode.commands.executeCommand(
        "vscode.openFolder",
        uri,
        { forceNewWindow: true }
      );
    }
  } catch (err) {
    vscode.window.showErrorMessage(
      `Failed to create worktree: ${(err as Error).message}`
    );
  }
}

// ── Remove Worktree ──

async function removeWorktree(
  manager: WorktreeManager
): Promise<void> {
  const ctx = await manager.getContext();
  if (!ctx.isGitRepo) {
    vscode.window.showWarningMessage(
      "Not inside a git repository."
    );
    return;
  }

  const worktrees = await manager.list();
  // Can't remove the current worktree or the main one
  const removable = worktrees.filter(
    (wt) => !wt.isCurrent && !wt.isBare
  );

  if (removable.length === 0) {
    vscode.window.showInformationMessage(
      "No removable worktrees. Cannot remove the current or main worktree."
    );
    return;
  }

  const picks = removable.map((wt) => formatWorktreePick(wt));

  const selected = await vscode.window.showQuickPick(picks, {
    title: "Remove Worktree",
    placeHolder: "Select a worktree to remove",
  });
  if (!selected) return;

  const confirm = await vscode.window.showWarningMessage(
    `Remove worktree at "${selected.worktreePath}"?\n\nBranch: ${selected.branchName ?? "detached"}\n\nThis will delete the worktree directory.`,
    { modal: true },
    "Remove",
    "Force Remove"
  );
  if (!confirm) return;

  try {
    await manager.remove(
      selected.worktreePath,
      confirm === "Force Remove"
    );
    vscode.window.showInformationMessage(
      `Worktree removed: ${selected.worktreePath}`
    );
  } catch (err) {
    vscode.window.showErrorMessage(
      `Failed to remove worktree: ${(err as Error).message}`
    );
  }
}

// ── Switch Worktree ──

async function switchWorktree(
  manager: WorktreeManager
): Promise<void> {
  const ctx = await manager.getContext();
  if (!ctx.isGitRepo) {
    vscode.window.showWarningMessage(
      "Not inside a git repository."
    );
    return;
  }

  const worktrees = await manager.list();
  // Don't show the current worktree
  const switchable = worktrees.filter((wt) => !wt.isCurrent);

  if (switchable.length === 0) {
    vscode.window.showInformationMessage(
      "No other worktrees to switch to. Create one first with PRSM: Create Worktree."
    );
    return;
  }

  const picks = switchable.map((wt) => formatWorktreePick(wt));

  const selected = await vscode.window.showQuickPick(picks, {
    title: "Switch Worktree",
    placeHolder: "Select a worktree to open",
  });
  if (!selected) return;

  const openMode = await vscode.window.showQuickPick(
    [
      {
        label: "Open in New Window",
        description: "Keep the current window open",
        mode: "new" as const,
      },
      {
        label: "Replace Current Window",
        description: "Close this window and open the worktree",
        mode: "replace" as const,
      },
    ],
    {
      title: "How to Open?",
      placeHolder: "Choose how to open the worktree",
    }
  );
  if (!openMode) return;

  const uri = vscode.Uri.file(selected.worktreePath);
  await vscode.commands.executeCommand("vscode.openFolder", uri, {
    forceNewWindow: openMode.mode === "new",
  });
}

// ── Helpers ──

interface WorktreeQuickPickItem extends vscode.QuickPickItem {
  worktreePath: string;
  branchName: string | null;
  isCurrent: boolean;
}

function formatWorktreePick(
  wt: WorktreeInfo
): WorktreeQuickPickItem {
  const branchLabel = wt.branch ?? "detached HEAD";
  const icon = wt.isCurrent
    ? "$(check) "
    : wt.isBare
      ? "$(repo) "
      : "$(git-branch) ";

  return {
    label: `${icon}${branchLabel}`,
    description: wt.isCurrent
      ? "current"
      : wt.isBare
        ? "main"
        : "",
    detail: wt.path,
    worktreePath: wt.path,
    branchName: wt.branch,
    isCurrent: wt.isCurrent,
  };
}
