/**
 * Tracks file modifications made by agents during a session.
 * Central store for change data used by decorations, tree views, and CodeLens.
 */
import * as vscode from "vscode";
import * as fs from "fs";
import { FileChangedData } from "../protocol/types";

/** Minimal interface for the transport methods we need (avoids circular deps). */
export interface FileChangeTransport {
  acceptFileChange(sessionId: string, toolCallId: string): Promise<{ status: string }>;
  rejectFileChange(sessionId: string, toolCallId: string): Promise<unknown>;
  acceptAllFileChanges(sessionId: string): Promise<{ status: string; count: number }>;
  rejectAllFileChanges(sessionId: string): Promise<unknown>;
}

export type ChangeStatus = "pending" | "accepted" | "rejected";

export interface TrackedFileChange {
  filePath: string;
  agentId: string;
  agentName: string;
  sessionId: string;
  toolCallId: string;
  toolName: string;
  messageIndex: number;
  changeType: "create" | "modify" | "delete";
  timestamp: string;
  oldContent: string | null;
  newContent: string | null; // For Edit: new_string; for Write: new file content
  preToolContent: string | null; // Full file content before tool ran
  addedRanges: Array<{ startLine: number; endLine: number }>;
  removedRanges: Array<{ startLine: number; endLine: number }>;
  status: ChangeStatus;
}

export class FileChangeTracker extends vscode.Disposable {
  private changes: Map<string, TrackedFileChange[]> = new Map();
  private transport: FileChangeTransport | null = null;

  private readonly _onDidChange = new vscode.EventEmitter<string | undefined>();
  readonly onDidChange = this._onDidChange.event;

  constructor() {
    super(() => {
      this._onDidChange.dispose();
    });
  }

  private isIgnoredTmpPath(filePath: string): boolean {
    const normalized = filePath.replace(/\\/g, "/");
    const segments = normalized.split("/").filter(Boolean);
    if (segments.some((segment) => segment.startsWith(".tmp"))) {
      return true;
    }
    const name = segments.length > 0 ? segments[segments.length - 1] : "";
    if (name === ".file_index.txt") return true;
    if (/^\.grep_.*\.txt$/.test(name)) return true;
    if (/^\.pytest_.*\.txt$/.test(name)) return true;
    if (/^\.diff_.*\.txt$/.test(name)) return true;
    if (/^\.compile_.*\.txt$/.test(name)) return true;
    // Filter patch artifacts (.rej / .orig) left by patch(1).
    if (name.endsWith(".rej") || name.endsWith(".orig")) return true;
    return false;
  }

  /**
   * Set the transport used to sync accept/reject to the server.
   * Called once the transport is available after connection.
   */
  setTransport(transport: FileChangeTransport | null): void {
    this.transport = transport;
  }

  /**
   * Sync a single change's status to the server (fire-and-forget).
   */
  private syncToServer(change: TrackedFileChange): void {
    if (!this.transport) return;
    const t = this.transport;
    if (change.status === "accepted") {
      t.acceptFileChange(change.sessionId, change.toolCallId).catch(() => {});
    } else if (change.status === "rejected") {
      t.rejectFileChange(change.sessionId, change.toolCallId).catch(() => {});
    }
  }

  /**
   * Sync bulk accept-all for a session to the server (fire-and-forget).
   */
  private syncAcceptAllToServer(sessionId: string): void {
    if (!this.transport) return;
    this.transport.acceptAllFileChanges(sessionId).catch(() => {});
  }

  /**
   * Track a new file change from a file_changed SSE event.
   */
  trackChange(data: FileChangedData, agentName?: string): void {
    if (this.isIgnoredTmpPath(data.file_path)) {
      return;
    }
    const change: TrackedFileChange = {
      filePath: data.file_path,
      agentId: data.agent_id,
      agentName: agentName ?? data.agent_id,
      sessionId: data.session_id,
      toolCallId: data.tool_call_id,
      toolName: data.tool_name,
      messageIndex: data.message_index,
      changeType: data.change_type as "create" | "modify" | "delete",
      timestamp: data.timestamp,
      oldContent: data.old_content,
      newContent: data.new_content ?? null,
      preToolContent: data.pre_tool_content ?? null,
      addedRanges: data.added_ranges ?? [],
      removedRanges: data.removed_ranges ?? [],
      status: "pending",
    };

    const existing = this.changes.get(data.file_path) ?? [];
    existing.push(change);
    this.changes.set(data.file_path, existing);

    this._onDidChange.fire(data.file_path);
  }

  /**
   * Get all pending changes for a file.
   */
  getPendingChanges(filePath: string): TrackedFileChange[] {
    return (this.changes.get(filePath) ?? []).filter(
      (c) => c.status === "pending"
    );
  }

  /**
   * Get all changes for a file (any status).
   */
  getAllChanges(filePath: string): TrackedFileChange[] {
    return this.changes.get(filePath) ?? [];
  }

  /**
   * Get all files with pending changes.
   */
  getChangedFiles(): string[] {
    const files: string[] = [];
    for (const [filePath, changes] of this.changes) {
      if (this.isIgnoredTmpPath(filePath)) {
        continue;
      }
      if (changes.some((c) => c.status === "pending")) {
        files.push(filePath);
      }
    }
    return files;
  }

  /**
   * Get all files with any tracked changes.
   */
  getAllTrackedFiles(): string[] {
    return Array.from(this.changes.keys());
  }

  /**
   * Accept a specific change (marks it as accepted, decorations cleared).
   */
  acceptChange(filePath: string, changeIndex: number): void {
    const changes = this.changes.get(filePath);
    if (changes && changeIndex < changes.length) {
      changes[changeIndex].status = "accepted";
      this.syncToServer(changes[changeIndex]);
      this._onDidChange.fire(filePath);
    }
  }

  /**
   * Reject a specific change (reverts the file modification).
   */
  async rejectChange(filePath: string, changeIndex: number): Promise<void> {
    const changes = this.changes.get(filePath);
    if (!changes || changeIndex >= changes.length) {
      return;
    }

    const change = changes[changeIndex];
    const reverted = await this._revertChange(change);
    if (!reverted) {
      return;
    }
    change.status = "rejected";
    this.syncToServer(change);
    this._onDidChange.fire(filePath);
  }

  /**
   * Accept all pending changes in a file.
   */
  acceptAll(filePath: string): void {
    const changes = this.changes.get(filePath);
    if (!changes) return;
    for (const c of changes) {
      if (c.status === "pending") {
        c.status = "accepted";
        this.syncToServer(c);
      }
    }
    this._onDidChange.fire(filePath);
  }

  /**
   * Reject all pending changes in a file (reverts all).
   */
  async rejectAll(filePath: string): Promise<void> {
    const changes = this.changes.get(filePath);
    if (!changes) return;

    // Reject in reverse order (most recent first)
    for (let i = changes.length - 1; i >= 0; i--) {
      if (changes[i].status === "pending") {
        const reverted = await this._revertChange(changes[i]);
        if (!reverted) {
          continue;
        }
        changes[i].status = "rejected";
        this.syncToServer(changes[i]);
      }
    }
    this._onDidChange.fire(filePath);
  }

  /**
   * Accept all pending changes for a specific session across all files.
   */
  acceptAllForSession(sessionId: string): void {
    let anyChanged = false;
    for (const [filePath, changes] of this.changes) {
      let changed = false;
      for (const c of changes) {
        if (c.sessionId === sessionId && c.status === "pending") {
          c.status = "accepted";
          changed = true;
          anyChanged = true;
        }
      }
      if (changed) {
        this._onDidChange.fire(filePath);
      }
    }
    // Use bulk accept endpoint for the session
    if (anyChanged) {
      this.syncAcceptAllToServer(sessionId);
    }
  }

  /**
   * Reject all pending changes for a specific session across all files (reverts all).
   */
  async rejectAllForSession(sessionId: string): Promise<void> {
    for (const [filePath, changes] of this.changes) {
      let changed = false;
      for (let i = changes.length - 1; i >= 0; i--) {
        if (changes[i].sessionId === sessionId && changes[i].status === "pending") {
          const reverted = await this._revertChange(changes[i]);
          if (!reverted) {
            continue;
          }
          changes[i].status = "rejected";
          this.syncToServer(changes[i]);
          changed = true;
        }
      }
      if (changed) {
        this._onDidChange.fire(filePath);
      }
    }
  }

  /**
   * Accept all pending changes across all files.
   */
  acceptAllGlobal(): void {
    // Collect session IDs that had pending changes for bulk server sync
    const sessionsWithChanges = new Set<string>();
    for (const [filePath, changes] of this.changes) {
      for (const c of changes) {
        if (c.status === "pending") {
          c.status = "accepted";
          sessionsWithChanges.add(c.sessionId);
        }
      }
      this._onDidChange.fire(filePath);
    }
    for (const sessionId of sessionsWithChanges) {
      this.syncAcceptAllToServer(sessionId);
    }
  }

  /**
   * Reject all pending changes across all files (reverts all).
   */
  async rejectAllGlobal(): Promise<void> {
    for (const [filePath, changes] of this.changes) {
      for (let i = changes.length - 1; i >= 0; i--) {
        if (changes[i].status === "pending") {
          const reverted = await this._revertChange(changes[i]);
          if (!reverted) {
            continue;
          }
          changes[i].status = "rejected";
          this.syncToServer(changes[i]);
        }
      }
      this._onDidChange.fire(filePath);
    }
  }

  /**
   * Get total count of pending changes across all files.
   */
  getPendingCount(): number {
    let count = 0;
    for (const [filePath, changes] of this.changes.entries()) {
      if (this.isIgnoredTmpPath(filePath)) {
        continue;
      }
      count += changes.filter(c => c.status === "pending").length;
    }
    return count;
  }

  /**
   * Find a tracked change by its tool call ID.
   */
  getChangeByToolCallId(toolCallId: string): { change: TrackedFileChange; filePath: string; index: number } | undefined {
    for (const [filePath, changes] of this.changes) {
      for (let i = 0; i < changes.length; i++) {
        if (changes[i].toolCallId === toolCallId) {
          return { change: changes[i], filePath, index: i };
        }
      }
    }
    return undefined;
  }

  /**
   * Bulk-load file changes from the server REST API response.
   * Used to restore file change state after VSCode or server restart.
   * The data format matches GET /sessions/{id}/file-changes.
   */
  restoreFromServer(
    sessionId: string,
    fileChanges: Record<
      string,
      Array<{
        file_path: string;
        agent_id: string;
        change_type: string;
        tool_call_id: string;
        tool_name: string;
        message_index: number;
        old_content: string | null;
        new_content: string | null;
        pre_tool_content: string | null;
        added_ranges: Array<{ startLine: number; endLine: number }>;
        removed_ranges: Array<{ startLine: number; endLine: number }>;
        timestamp: string;
        status: string;
      }>
    >,
    agentNameResolver?: (agentId: string) => string | undefined
  ): void {
    for (const [filePath, records] of Object.entries(fileChanges)) {
      for (const r of records) {
        const canonicalPath = r.file_path || filePath;
        if (this.isIgnoredTmpPath(canonicalPath)) {
          continue;
        }
        const existing = this.changes.get(canonicalPath) ?? [];
        // Skip duplicates (by tool_call_id)
        if (existing.some((c) => c.toolCallId === r.tool_call_id)) continue;

        existing.push({
          filePath: canonicalPath,
          agentId: r.agent_id,
          agentName: agentNameResolver?.(r.agent_id) ?? r.agent_id,
          sessionId,
          toolCallId: r.tool_call_id,
          toolName: r.tool_name,
          messageIndex: r.message_index,
          changeType: r.change_type as "create" | "modify" | "delete",
          timestamp: r.timestamp,
          oldContent: r.old_content,
          newContent: r.new_content ?? null,
          preToolContent: r.pre_tool_content ?? null,
          addedRanges: r.added_ranges ?? [],
          removedRanges: r.removed_ranges ?? [],
          status: (r.status as ChangeStatus) ?? "pending",
        });
        this.changes.set(canonicalPath, existing);
      }
    }
    this._onDidChange.fire(undefined);
  }

  /**
   * Clear all tracked changes (e.g., on session reset or snapshot restore).
   */
  clear(): void {
    this.changes.clear();
    this._onDidChange.fire(undefined);
  }

  /**
   * Check if a file has pending changes.
   */
  hasPendingChanges(filePath: string): boolean {
    return (this.changes.get(filePath) ?? []).some(
      (c) => c.status === "pending"
    );
  }

  /**
   * Revert a single change by applying the reverse edit.
   */
  private async _revertChange(change: TrackedFileChange): Promise<boolean> {
    const uri = vscode.Uri.file(change.filePath);

    if (change.changeType === "create") {
      try {
        await vscode.workspace.fs.delete(uri);
        return true;
      } catch {
        try {
          await vscode.workspace.fs.stat(uri);
          vscode.window.showErrorMessage(`Failed to revert ${change.filePath}`);
          return false;
        } catch {
          // Already deleted is considered reverted.
          return true;
        }
      }
    }

    if (change.changeType === "modify" && change.oldContent !== null) {
      if (change.toolName === "Edit" || change.toolName === "edit") {
        // For Edit: find newContent in current file and replace with oldContent
        if (change.newContent) {
          try {
            const doc = await vscode.workspace.openTextDocument(uri);
            const content = doc.getText();
            const idx = content.indexOf(change.newContent);
            if (idx >= 0) {
              const edit = new vscode.WorkspaceEdit();
              const startPos = doc.positionAt(idx);
              const endPos = doc.positionAt(idx + change.newContent.length);
              edit.replace(uri, new vscode.Range(startPos, endPos), change.oldContent);
              const applied = await vscode.workspace.applyEdit(edit);
              if (!applied) {
                vscode.window.showErrorMessage(`Failed to revert ${change.filePath}`);
                return false;
              }
              await doc.save();
              return true;
            }
          } catch {
            // Fall through to pre-tool content fallback
          }
        }
        // Fallback: restore full pre-tool content
        if (change.preToolContent !== null) {
          try {
            fs.writeFileSync(change.filePath, change.preToolContent, "utf-8");
            return true;
          } catch {
            // Ignore
          }
        }
        vscode.window.showErrorMessage(
          `Failed to revert ${change.filePath} — no revert data available`
        );
        return false;
      } else if (change.toolName === "Write" || change.toolName === "write") {
        // For Write: preToolContent or oldContent is the complete pre-tool file content
        const restoreContent = change.preToolContent ?? change.oldContent;
        if (restoreContent !== null) {
          try {
            fs.writeFileSync(change.filePath, restoreContent, "utf-8");
            return true;
          } catch {
            vscode.window.showErrorMessage(
              `Failed to revert ${change.filePath}`
            );
            return false;
          }
        }
      }
    }
    vscode.window.showErrorMessage(
      `Failed to revert ${change.filePath} — no revert data available`
    );
    return false;
  }
}
