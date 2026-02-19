/**
 * FileDecorationProvider that shows visual indicators on files modified by agents.
 * Shows badges in the Explorer sidebar and editor tabs, similar to Git's M/U/A badges.
 */
import * as vscode from "vscode";
import { FileChangeTracker } from "./fileChangeTracker";

export class AgentFileDecorationProvider
  implements vscode.FileDecorationProvider
{
  private readonly _onDidChangeFileDecorations =
    new vscode.EventEmitter<vscode.Uri | vscode.Uri[] | undefined>();
  readonly onDidChangeFileDecorations = this._onDidChangeFileDecorations.event;

  private readonly disposables: vscode.Disposable[] = [];

  constructor(private readonly tracker: FileChangeTracker) {
    this.disposables.push(
      tracker.onDidChange((filePath) => {
        if (filePath) {
          this._onDidChangeFileDecorations.fire(vscode.Uri.file(filePath));
        } else {
          // All changes cleared
          this._onDidChangeFileDecorations.fire(undefined);
        }
      })
    );
  }

  provideFileDecoration(
    uri: vscode.Uri
  ): vscode.FileDecoration | undefined {
    const filePath = uri.fsPath;
    const allChanges = this.tracker.getAllChanges(filePath);
    if (allChanges.length === 0) {
      return undefined;
    }

    const pendingCount = allChanges.filter(
      (c) => c.status === "pending"
    ).length;
    const acceptedCount = allChanges.filter(
      (c) => c.status === "accepted"
    ).length;
    const rejectedCount = allChanges.filter(
      (c) => c.status === "rejected"
    ).length;

    if (pendingCount > 0) {
      return new vscode.FileDecoration(
        "P",
        `${pendingCount} pending agent change${pendingCount !== 1 ? "s" : ""}`,
        new vscode.ThemeColor("editorOverviewRuler.modifiedForeground")
      );
    }

    if (rejectedCount > 0 && acceptedCount === 0) {
      return new vscode.FileDecoration(
        "R",
        `${rejectedCount} rejected agent change${rejectedCount !== 1 ? "s" : ""}`,
        new vscode.ThemeColor("editorOverviewRuler.deletedForeground")
      );
    }

    if (acceptedCount > 0) {
      return new vscode.FileDecoration(
        "A",
        `${acceptedCount} accepted agent change${acceptedCount !== 1 ? "s" : ""}`,
        new vscode.ThemeColor("editorOverviewRuler.addedForeground")
      );
    }

    return undefined;
  }

  dispose(): void {
    this._onDidChangeFileDecorations.dispose();
    for (const d of this.disposables) {
      d.dispose();
    }
  }
}
