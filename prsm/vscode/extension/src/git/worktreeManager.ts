/**
 * Git worktree detection and management.
 *
 * Detects whether the current workspace is a git worktree,
 * provides worktree listing/creation/removal, and exposes
 * branch/path information for the UI.
 */
import * as vscode from "vscode";
import { execFile } from "child_process";
import * as path from "path";

/** Metadata for a single git worktree. */
export interface WorktreeInfo {
  /** Absolute path to the worktree directory. */
  path: string;
  /** HEAD commit hash. */
  head: string;
  /** Branch name (e.g. "main"), or null if detached HEAD. */
  branch: string | null;
  /** Whether this is the bare/main worktree. */
  isBare: boolean;
  /** Whether this is the current workspace worktree. */
  isCurrent: boolean;
}

/** Result of worktree detection for the current workspace. */
export interface WorktreeContext {
  /** Whether the workspace is inside a git repository at all. */
  isGitRepo: boolean;
  /** Whether the workspace is a linked worktree (not the main checkout). */
  isWorktree: boolean;
  /** Current branch name, or null if detached. */
  branch: string | null;
  /** Path to the main (bare/primary) worktree. */
  mainWorktreePath: string | null;
  /** Absolute path of the current worktree. */
  worktreePath: string | null;
}

export class WorktreeManager {
  private readonly outputChannel: vscode.OutputChannel;
  private readonly cwd: string;

  /** Cached context; refreshed on demand. */
  private _context: WorktreeContext | null = null;

  private _onDidChange = new vscode.EventEmitter<WorktreeContext>();
  readonly onDidChange = this._onDidChange.event;

  constructor(cwd: string, outputChannel: vscode.OutputChannel) {
    this.cwd = cwd;
    this.outputChannel = outputChannel;
  }

  /** Get the cached worktree context, or detect it fresh. */
  async getContext(): Promise<WorktreeContext> {
    if (!this._context) {
      this._context = await this.detect();
    }
    return this._context;
  }

  /** Force re-detection and fire change event. */
  async refresh(): Promise<WorktreeContext> {
    this._context = await this.detect();
    this._onDidChange.fire(this._context);
    return this._context;
  }

  // ── Detection ──

  /** Detect whether the current workspace is inside a git worktree. */
  private async detect(): Promise<WorktreeContext> {
    const empty: WorktreeContext = {
      isGitRepo: false,
      isWorktree: false,
      branch: null,
      mainWorktreePath: null,
      worktreePath: null,
    };

    // Check if we're in a git repo at all
    const topLevel = await this.git(["rev-parse", "--show-toplevel"]);
    if (topLevel === null) {
      return empty;
    }

    // Get current branch
    const branch = await this.git([
      "rev-parse",
      "--abbrev-ref",
      "HEAD",
    ]);

    // Get the git common dir (shared across all worktrees)
    const commonDir = await this.git(["rev-parse", "--git-common-dir"]);
    // Get the git dir for this worktree
    const gitDir = await this.git(["rev-parse", "--git-dir"]);

    // Resolve to absolute paths for comparison
    const resolvedCommon = commonDir
      ? path.resolve(this.cwd, commonDir)
      : null;
    const resolvedGit = gitDir
      ? path.resolve(this.cwd, gitDir)
      : null;

    // If git-dir !== git-common-dir, we're in a linked worktree
    const isWorktree =
      resolvedCommon !== null &&
      resolvedGit !== null &&
      resolvedCommon !== resolvedGit;

    // Main worktree path: parent of the common .git dir
    const mainWorktreePath = resolvedCommon
      ? path.dirname(resolvedCommon)
      : null;

    const ctx: WorktreeContext = {
      isGitRepo: true,
      isWorktree,
      branch: branch === "HEAD" ? null : branch, // detached
      mainWorktreePath,
      worktreePath: topLevel,
    };

    this.outputChannel.appendLine(
      `[worktree] detected: isWorktree=${ctx.isWorktree} branch=${ctx.branch ?? "detached"} path=${ctx.worktreePath}`
    );

    return ctx;
  }

  // ── Worktree operations ──

  /** List all worktrees in the repository. */
  async list(): Promise<WorktreeInfo[]> {
    const raw = await this.git(["worktree", "list", "--porcelain"]);
    if (raw === null) {
      return [];
    }
    return this.parsePorcelainOutput(raw);
  }

  /**
   * Create a new worktree.
   * @param targetPath  Absolute path for the new worktree directory.
   * @param branchName  Branch to create/checkout. If it doesn't exist, -b is used.
   * @param baseBranch  Optional base branch to create from (defaults to HEAD).
   */
  async create(
    targetPath: string,
    branchName: string,
    baseBranch?: string
  ): Promise<void> {
    // Check if branch already exists
    const branchExists = await this.git([
      "show-ref",
      "--verify",
      "--quiet",
      `refs/heads/${branchName}`,
    ]);

    const args = ["worktree", "add"];
    if (branchExists === null) {
      // Branch doesn't exist — create it
      args.push("-b", branchName);
    }
    args.push(targetPath);
    if (branchExists !== null) {
      // Checkout existing branch
      args.push(branchName);
    } else if (baseBranch) {
      args.push(baseBranch);
    }

    const result = await this.gitRaw(args);
    if (result.error) {
      throw new Error(result.error);
    }

    this.outputChannel.appendLine(
      `[worktree] created: ${targetPath} (branch: ${branchName})`
    );

    // Refresh context
    await this.refresh();
  }

  /**
   * Remove a worktree.
   * @param worktreePath  Absolute path of the worktree to remove.
   * @param force         Whether to force removal (--force).
   */
  async remove(worktreePath: string, force = false): Promise<void> {
    const args = ["worktree", "remove"];
    if (force) {
      args.push("--force");
    }
    args.push(worktreePath);

    const result = await this.gitRaw(args);
    if (result.error) {
      throw new Error(result.error);
    }

    this.outputChannel.appendLine(
      `[worktree] removed: ${worktreePath}`
    );

    await this.refresh();
  }

  /** Prune stale worktree references. */
  async prune(): Promise<void> {
    await this.git(["worktree", "prune"]);
    await this.refresh();
  }

  // ── Helpers ──

  /**
   * Parse `git worktree list --porcelain` output into structured data.
   */
  private parsePorcelainOutput(raw: string): WorktreeInfo[] {
    const worktrees: WorktreeInfo[] = [];
    const blocks = raw.split("\n\n").filter((b) => b.trim());

    for (const block of blocks) {
      const lines = block.split("\n");
      let wtPath = "";
      let head = "";
      let branch: string | null = null;
      let isBare = false;

      for (const line of lines) {
        if (line.startsWith("worktree ")) {
          wtPath = line.slice("worktree ".length);
        } else if (line.startsWith("HEAD ")) {
          head = line.slice("HEAD ".length);
        } else if (line.startsWith("branch ")) {
          // e.g. "branch refs/heads/main"
          const ref = line.slice("branch ".length);
          branch = ref.startsWith("refs/heads/")
            ? ref.slice("refs/heads/".length)
            : ref;
        } else if (line === "bare") {
          isBare = true;
        }
        // "detached" line means branch stays null
      }

      if (wtPath) {
        worktrees.push({
          path: wtPath,
          head,
          branch,
          isBare,
          isCurrent:
            path.resolve(wtPath) === path.resolve(this.cwd),
        });
      }
    }

    return worktrees;
  }

  /**
   * Run a git command and return trimmed stdout, or null on failure.
   */
  private git(args: string[]): Promise<string | null> {
    return new Promise((resolve) => {
      execFile(
        "git",
        args,
        { cwd: this.cwd, timeout: 10000 },
        (err, stdout) => {
          if (err) {
            resolve(null);
          } else {
            resolve(stdout.trim());
          }
        }
      );
    });
  }

  /**
   * Run a git command and return both stdout and error info.
   */
  private gitRaw(
    args: string[]
  ): Promise<{ stdout: string; error: string | null }> {
    return new Promise((resolve) => {
      execFile(
        "git",
        args,
        { cwd: this.cwd, timeout: 30000 },
        (err, stdout, stderr) => {
          if (err) {
            resolve({
              stdout: stdout?.trim() ?? "",
              error: stderr?.trim() || (err as Error).message,
            });
          } else {
            resolve({ stdout: stdout.trim(), error: null });
          }
        }
      );
    });
  }

  dispose(): void {
    this._onDidChange.dispose();
  }
}
