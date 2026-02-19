/**
 * TreeDataProvider showing files modified by agents with accept/reject actions.
 *
 * Tree structure (grouped by session):
 *   Refactor auth
 *     src/server.py (3 changes)
 *       > Edit by Fix auth bug... at 14:23
 *       > Write by Add tests... at 14:25
 *     src/models/foo.py (new file)
 *       > Write by Refactor models... at 14:24
 *   Add dark mode
 *     src/theme.ts (1 change)
 *       > Write by Implement theme... at 15:00
 *
 * Agent names match those shown in the agent tree. Clicking a change
 * opens the agent's conversation and scrolls to the tool call.
 */
import * as vscode from "vscode";
import * as path from "path";
import {
  FileChangeTracker,
  TrackedFileChange,
} from "../tracking/fileChangeTracker";
import { SessionStore } from "../state/sessionStore";

type FileChangesTreeNode = SessionTreeItem | DirectoryTreeItem | FileTreeItem | ChangeTreeItem;

function stripTmpSessionPrefix(filePath: string, sessionId?: string): string {
  if (!sessionId) return filePath;
  const normalized = filePath.replace(/\\/g, "/");
  const prefixes = [`/tmp/${sessionId}/`, `/private/tmp/${sessionId}/`];
  for (const prefix of prefixes) {
    const idx = normalized.indexOf(prefix);
    if (idx >= 0) {
      return normalized.slice(idx + prefix.length);
    }
  }
  return filePath;
}

function normalizeRelativePath(filePath: string): string {
  const normalized = filePath.replace(/\\/g, "/");
  return normalized.replace(/^\.\/+/, "");
}

function workspaceRelativePath(filePath: string, sessionId?: string): string {
  const maybeStripped = stripTmpSessionPrefix(filePath, sessionId);
  if (maybeStripped !== filePath) return normalizeRelativePath(maybeStripped);
  if (!path.isAbsolute(filePath)) return normalizeRelativePath(filePath);
  const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
  if (!workspaceFolder) return path.basename(filePath);
  const rel = path.relative(workspaceFolder.uri.fsPath, filePath);
  if (!rel || rel.startsWith("..") || path.isAbsolute(rel)) {
    return path.basename(filePath);
  }
  return normalizeRelativePath(rel);
}

function sessionIdCopyLink(sessionId: string): string {
  const args = encodeURIComponent(JSON.stringify([sessionId]));
  return `[${sessionId} $(copy)](command:prsm.copySessionId?${args})`;
}

/**
 * Session-level tree item grouping all file changes for a single session.
 */
class SessionTreeItem extends vscode.TreeItem {
  constructor(
    public readonly sessionId: string,
    public readonly sessionName: string,
    public readonly pendingCount: number,
  ) {
    super(sessionName, vscode.TreeItemCollapsibleState.Expanded);

    this.contextValue = "prsmChangedSession";
    this.iconPath = new vscode.ThemeIcon(
      "symbol-event",
      new vscode.ThemeColor("editorOverviewRuler.modifiedForeground")
    );
    this.description = `${pendingCount} pending`;
    const tooltip = new vscode.MarkdownString(
      `**${sessionName}**\n\n` +
      `- Pending changes: ${pendingCount}\n` +
      `- Session ID: ${sessionIdCopyLink(sessionId)}`
    );
    tooltip.isTrusted = true;
    tooltip.supportThemeIcons = true;
    this.tooltip = tooltip;
  }
}

/**
 * File-level tree item grouping all changes for a single file within a session.
 */
class FileTreeItem extends vscode.TreeItem {
  constructor(
    public readonly filePath: string,
    public readonly sessionId: string,
    public readonly pendingCount: number,
    public readonly hasNewFile: boolean
  ) {
    super(
      path.basename(filePath),
      vscode.TreeItemCollapsibleState.Collapsed
    );

    this.contextValue = "prsmChangedFile";
    this.resourceUri = vscode.Uri.file(filePath);

    // Description shows relative path + count
    const relPath = workspaceRelativePath(filePath, sessionId);
    this.description = hasNewFile
      ? `${relPath} (new file)`
      : `${relPath} (${pendingCount} change${pendingCount !== 1 ? "s" : ""})`;

    this.iconPath = new vscode.ThemeIcon(
      hasNewFile ? "new-file" : "diff-modified",
      new vscode.ThemeColor(
        hasNewFile
          ? "editorOverviewRuler.addedForeground"
          : "editorOverviewRuler.modifiedForeground"
      )
    );

    this.tooltip = `${pendingCount} pending change${pendingCount !== 1 ? "s" : ""}`;

    // Click to open file
    this.command = {
      command: "vscode.open",
      title: "Open File",
      arguments: [vscode.Uri.file(filePath)],
    };
  }
}

/** Directory tree item for rooted file-change browsing. */
class DirectoryTreeItem extends vscode.TreeItem {
  constructor(
    public readonly sessionId: string,
    public readonly dirPath: string,
    public readonly labelPath: string,
  ) {
    super(labelPath, vscode.TreeItemCollapsibleState.Collapsed);
    this.contextValue = "prsmChangedDirectory";
    this.iconPath = new vscode.ThemeIcon("folder");
    this.tooltip = labelPath;
  }
}

/**
 * Individual change tree item with accept/reject/go-to-agent actions.
 * Clicking opens the agent's conversation and scrolls to the tool call.
 */
class ChangeTreeItem extends vscode.TreeItem {
  constructor(
    public readonly change: TrackedFileChange,
    public readonly changeIndex: number,
    public readonly filePath: string,
    store?: SessionStore,
  ) {
    const ts = new Date(change.timestamp);
    const timeStr = ts.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });

    // Resolve the agent label to match the agent tree display.
    // The orchestrator appears in the tree as the session name (e.g. "Refactor auth"),
    // so use that instead of the generic "Orchestrator" label.
    const liveAgent = store?.getAgent(change.sessionId, change.agentId);
    let agentLabel = liveAgent?.name || change.agentName || change.agentId;
    if (liveAgent?.role === "orchestrator") {
      const session = store?.getSession(change.sessionId);
      if (session?.name) {
        agentLabel = session.name;
      }
    }

    super(
      `${change.toolName} by ${agentLabel}`,
      vscode.TreeItemCollapsibleState.None
    );

    this.contextValue = "prsmFileChange";
    this.description = `at ${timeStr}`;

    // Icon based on status
    if (change.status === "accepted") {
      this.iconPath = new vscode.ThemeIcon(
        "check",
        new vscode.ThemeColor("testing.iconPassed")
      );
    } else if (change.status === "rejected") {
      this.iconPath = new vscode.ThemeIcon(
        "close",
        new vscode.ThemeColor("testing.iconFailed")
      );
    } else {
      this.iconPath = new vscode.ThemeIcon(
        "circle-filled",
        new vscode.ThemeColor("editorOverviewRuler.addedForeground")
      );
    }

    this.tooltip = new vscode.MarkdownString(
      `**${change.toolName}** by \`${agentLabel}\`\n\n` +
        `- Type: ${change.changeType}\n` +
        `- Time: ${ts.toLocaleString()}\n` +
        `- Status: ${change.status}\n` +
        `- Tool call: \`${change.toolCallId}\`\n` +
        `- Message index: ${change.messageIndex}`
    );

    // Click to open agent conversation and scroll to the tool call
    this.command = {
      command: "prsm.goToAgent",
      title: "Go to Agent",
      arguments: [
        {
          sessionId: change.sessionId,
          agentId: change.agentId,
          toolCallId: change.toolCallId,
          messageIndex: change.messageIndex,
        },
      ],
    };
  }
}

export class FileChangesTreeProvider
  implements vscode.TreeDataProvider<FileChangesTreeNode>
{
  private static readonly ROOT_DIR_KEY = "__root__";

  private _onDidChangeTreeData = new vscode.EventEmitter<
    FileChangesTreeNode | undefined | void
  >();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  constructor(
    private readonly tracker: FileChangeTracker,
    private readonly store?: SessionStore,
  ) {
    tracker.onDidChange(() => {
      this._onDidChangeTreeData.fire();
    });

    // Also refresh when the agent tree changes so names stay in sync
    store?.onDidChangeTree(() => {
      this._onDidChangeTreeData.fire();
    });
  }

  refresh(): void {
    this._onDidChangeTreeData.fire();
  }

  getTreeItem(element: FileChangesTreeNode): vscode.TreeItem {
    return element;
  }

  getChildren(element?: FileChangesTreeNode): FileChangesTreeNode[] {
    if (!element) {
      // Root level: sessions that have non-accepted changes
      return this._getSessionItems();
    }

    if (element instanceof SessionTreeItem) {
      return this._getDirectoryChildren(element.sessionId, FileChangesTreeProvider.ROOT_DIR_KEY);
    }

    if (element instanceof DirectoryTreeItem) {
      return this._getDirectoryChildren(element.sessionId, element.dirPath);
    }

    if (element instanceof FileTreeItem) {
      // Children of a file: individual non-accepted changes for that file+session
      return this._getChangeItemsForFile(element.filePath, element.sessionId);
    }

    return [];
  }

  getParent(
    element: FileChangesTreeNode
  ): vscode.ProviderResult<FileChangesTreeNode> {
    if (element instanceof ChangeTreeItem) {
      const allChanges = this.tracker.getAllChanges(element.filePath);
      const sessionChanges = allChanges.filter(
        (c) => c.sessionId === element.change.sessionId && c.status === "pending"
      );
      const pendingCount = sessionChanges.filter(
        (c) => c.status === "pending"
      ).length;
      const hasNewFile = sessionChanges.some(
        (c) => c.changeType === "create" && c.status !== "accepted"
      );
      return new FileTreeItem(
        element.filePath,
        element.change.sessionId,
        pendingCount,
        hasNewFile
      );
    }

    if (element instanceof FileTreeItem) {
      const sessionName = this._resolveSessionName(element.sessionId);
      const pendingCount = this._getPendingCountForSession(element.sessionId);
      return new SessionTreeItem(element.sessionId, sessionName, pendingCount);
    }

    return undefined;
  }

  dispose(): void {
    this._onDidChangeTreeData.dispose();
  }

  // ── Private helpers ──

  /**
   * Build session-level tree items. Groups all tracked changes by sessionId,
   * filters to sessions with at least one non-accepted change.
   */
  private _getSessionItems(): SessionTreeItem[] {
    const sessionIds = new Set<string>();
    for (const filePath of this.tracker.getAllTrackedFiles()) {
      for (const change of this.tracker.getAllChanges(filePath)) {
        if (change.status === "pending") {
          sessionIds.add(change.sessionId);
        }
      }
    }

    return Array.from(sessionIds).map((sessionId) => {
      const sessionName = this._resolveSessionName(sessionId);
      const pendingCount = this._getPendingCountForSession(sessionId);
      return new SessionTreeItem(sessionId, sessionName, pendingCount);
    });
  }

  private _buildFileMapForSession(sessionId: string): Map<string, TrackedFileChange[]> {
    const fileMap = new Map<string, TrackedFileChange[]>();
    for (const filePath of this.tracker.getAllTrackedFiles()) {
      const sessionChanges = this.tracker
        .getAllChanges(filePath)
        .filter((c) => c.sessionId === sessionId && c.status === "pending");
      if (sessionChanges.length > 0) {
        fileMap.set(filePath, sessionChanges);
      }
    }
    return fileMap;
  }

  private _compactFoldersEnabled(): boolean {
    return vscode.workspace
      .getConfiguration("explorer")
      .get<boolean>("compactFolders", true);
  }

  private _relSegments(filePath: string, sessionId: string): string[] {
    const safe = workspaceRelativePath(filePath, sessionId);
    return safe.split(/[\\/]+/).filter((part) => part.length > 0);
  }

  private _getDirectoryChildren(sessionId: string, dirPath: string): FileChangesTreeNode[] {
    const fileMap = this._buildFileMapForSession(sessionId);
    const compact = this._compactFoldersEnabled();
    const dirToChildren = new Map<string, Set<string>>();
    const dirToFiles = new Map<string, string[]>();

    for (const filePath of fileMap.keys()) {
      const segments = this._relSegments(filePath, sessionId);
      const dirs = segments.slice(0, -1);
      let current = FileChangesTreeProvider.ROOT_DIR_KEY;
      for (const dir of dirs) {
        const childKey = current === FileChangesTreeProvider.ROOT_DIR_KEY ? dir : `${current}/${dir}`;
        if (!dirToChildren.has(current)) dirToChildren.set(current, new Set());
        dirToChildren.get(current)!.add(childKey);
        current = childKey;
      }
      if (!dirToFiles.has(current)) dirToFiles.set(current, []);
      dirToFiles.get(current)!.push(filePath);
    }

    const collapsedDirs: Array<{ key: string; label: string }> = [];
    for (const child of Array.from(dirToChildren.get(dirPath) ?? []).sort()) {
      let key = child;
      let label = key.split("/").slice(-1)[0];
      while (compact) {
        const nested = Array.from(dirToChildren.get(key) ?? []);
        const filesAtNode = dirToFiles.get(key) ?? [];
        if (nested.length !== 1 || filesAtNode.length > 0) break;
        const next = nested[0];
        label = `${label}/${next.split("/").slice(-1)[0]}`;
        key = next;
      }
      collapsedDirs.push({ key, label });
    }

    const items: FileChangesTreeNode[] = collapsedDirs.map(
      ({ key, label }) => new DirectoryTreeItem(sessionId, key, label)
    );

    const filesHere = (dirToFiles.get(dirPath) ?? []).sort();
    for (const filePath of filesHere) {
      const changes = fileMap.get(filePath) ?? [];
      const pendingCount = changes.filter((c) => c.status === "pending").length;
      const hasNewFile = changes.some(
        (c) => c.changeType === "create" && c.status !== "accepted"
      );
      items.push(new FileTreeItem(filePath, sessionId, pendingCount, hasNewFile));
    }
    return items;
  }

  /**
   * Build change-level tree items for a specific file within a session.
   */
  private _getChangeItemsForFile(
    filePath: string,
    sessionId: string
  ): ChangeTreeItem[] {
    const allChanges = this.tracker.getAllChanges(filePath);
    return allChanges
      .map((change, index) => ({ change, index }))
      .filter(
        ({ change }) =>
          change.sessionId === sessionId && change.status === "pending"
      )
      .map(
        ({ change, index }) =>
          new ChangeTreeItem(change, index, filePath, this.store)
      );
  }

  /**
   * Resolve a session name from the store, falling back to a truncated ID.
   */
  private _resolveSessionName(sessionId: string): string {
    const session = this.store?.getSession(sessionId);
    return session?.name ?? `Untitled Session (${sessionId.slice(0, 8)})`;
  }

  /**
   * Count pending (non-accepted, non-rejected) changes across all files for a session.
   */
  private _getPendingCountForSession(sessionId: string): number {
    let count = 0;
    for (const filePath of this.tracker.getAllTrackedFiles()) {
      for (const change of this.tracker.getAllChanges(filePath)) {
        if (change.sessionId === sessionId && change.status === "pending") {
          count++;
        }
      }
    }
    return count;
  }
}
