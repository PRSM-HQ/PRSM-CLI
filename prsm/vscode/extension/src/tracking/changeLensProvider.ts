/**
 * CodeLens provider that shows Accept/Reject/Inspect actions
 * above each agent-modified code block in the editor.
 */
import * as vscode from "vscode";
import { FileChangeTracker } from "./fileChangeTracker";

export class ChangeLensProvider implements vscode.CodeLensProvider {
  private readonly _onDidChange = new vscode.EventEmitter<void>();
  readonly onDidChangeCodeLenses = this._onDidChange.event;

  private readonly disposables: vscode.Disposable[] = [];

  constructor(private readonly tracker: FileChangeTracker) {
    this.disposables.push(
      tracker.onDidChange(() => {
        this._onDidChange.fire();
      })
    );
  }

  provideCodeLenses(
    document: vscode.TextDocument,
    _token: vscode.CancellationToken
  ): vscode.CodeLens[] {
    const filePath = document.uri.fsPath;
    const pendingChanges = this.tracker.getPendingChanges(filePath);

    if (pendingChanges.length === 0) {
      return [];
    }

    const lenses: vscode.CodeLens[] = [];

    for (let i = 0; i < pendingChanges.length; i++) {
      const change = pendingChanges[i];

      // Find the first added range to place the lens
      const firstRange = change.addedRanges[0] ?? change.removedRanges[0];
      if (!firstRange) continue;

      const line = Math.max(0, firstRange.startLine);
      if (line >= document.lineCount) continue;

      const range = new vscode.Range(line, 0, line, 0);
      // Find the actual index in allChanges (not just pending)
      const allChanges = this.tracker.getAllChanges(filePath);
      const actualIndex = allChanges.indexOf(change);

      // Accept lens
      lenses.push(
        new vscode.CodeLens(range, {
          title: "$(check) Accept",
          command: "prsm.acceptChange",
          arguments: [filePath, actualIndex],
        })
      );

      // Reject lens
      lenses.push(
        new vscode.CodeLens(range, {
          title: "$(close) Reject",
          command: "prsm.rejectChange",
          arguments: [filePath, actualIndex],
        })
      );

      if (actualIndex < 0) continue;

      // Inspect lens: jump to the exact tool call in chat and expand it
      lenses.push(
        new vscode.CodeLens(range, {
          title: "$(search) Inspect",
          command: "prsm.inspectChange",
          arguments: [
            {
              filePath,
              changeIndex: actualIndex,
            },
          ],
        })
      );

      // Keep quick context access for additional detail
      lenses.push(
        new vscode.CodeLens(range, {
          title: "$(info) Context",
          command: "prsm.showAgentContext",
          arguments: [
            {
              sessionId: change.sessionId,
              agentId: change.agentId,
              toolCallId: change.toolCallId,
              messageIndex: change.messageIndex,
            },
          ],
        })
      );
    }

    return lenses;
  }

  dispose(): void {
    this._onDidChange.dispose();
    for (const d of this.disposables) {
      d.dispose();
    }
  }
}
