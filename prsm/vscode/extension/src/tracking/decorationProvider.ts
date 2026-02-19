/**
 * Applies inline editor decorations for agent-made file changes.
 * Shows green highlights for added lines and red for removed lines,
 * similar to Cursor's approach.
 */
import * as vscode from "vscode";
import { FileChangeTracker } from "./fileChangeTracker";

export class AgentEditDecorationProvider implements vscode.Disposable {
  private readonly addedDecoration: vscode.TextEditorDecorationType;
  private readonly removedDecoration: vscode.TextEditorDecorationType;
  private readonly deletedIndicator: vscode.TextEditorDecorationType;
  private readonly disposables: vscode.Disposable[] = [];

  constructor(private readonly tracker: FileChangeTracker) {
    // Green background for added/modified lines
    this.addedDecoration = vscode.window.createTextEditorDecorationType({
      backgroundColor: new vscode.ThemeColor(
        "diffEditor.insertedTextBackground"
      ),
      isWholeLine: true,
      overviewRulerColor: new vscode.ThemeColor(
        "editorOverviewRuler.addedForeground"
      ),
      overviewRulerLane: vscode.OverviewRulerLane.Left,
    });

    // Red background for removed lines
    this.removedDecoration = vscode.window.createTextEditorDecorationType({
      backgroundColor: new vscode.ThemeColor(
        "diffEditor.removedTextBackground"
      ),
      isWholeLine: true,
      overviewRulerColor: new vscode.ThemeColor(
        "editorOverviewRuler.modifiedForeground"
      ),
      overviewRulerLane: vscode.OverviewRulerLane.Left,
    });

    // Indicator for deleted content (collapsed view showing what was removed)
    this.deletedIndicator = vscode.window.createTextEditorDecorationType({
      isWholeLine: true,
      overviewRulerColor: new vscode.ThemeColor(
        "editorOverviewRuler.deletedForeground"
      ),
      overviewRulerLane: vscode.OverviewRulerLane.Left,
    });

    // Update decorations when changes are tracked
    this.disposables.push(
      tracker.onDidChange((filePath) => {
        if (filePath) {
          this.updateDecorationsForFile(filePath);
        } else {
          // All changes cleared — update all visible editors
          this.updateAllEditors();
        }
      })
    );

    // Update when active editor changes
    this.disposables.push(
      vscode.window.onDidChangeActiveTextEditor((editor) => {
        if (editor) {
          this.updateDecorations(editor);
        }
      })
    );

    // Update when visible editors change
    this.disposables.push(
      vscode.window.onDidChangeVisibleTextEditors((editors) => {
        for (const editor of editors) {
          this.updateDecorations(editor);
        }
      })
    );

    // Initial decoration for any open editors
    this.updateAllEditors();
  }

  /**
   * Update decorations for a specific file across all visible editors.
   */
  updateDecorationsForFile(filePath: string): void {
    for (const editor of vscode.window.visibleTextEditors) {
      if (editor.document.uri.fsPath === filePath) {
        this.updateDecorations(editor);
      }
    }
  }

  /**
   * Update decorations on a specific editor.
   */
  updateDecorations(editor: vscode.TextEditor): void {
    const filePath = editor.document.uri.fsPath;
    const pendingChanges = this.tracker.getPendingChanges(filePath);

    if (pendingChanges.length === 0) {
      // Clear decorations
      editor.setDecorations(this.addedDecoration, []);
      editor.setDecorations(this.removedDecoration, []);
      editor.setDecorations(this.deletedIndicator, []);
      return;
    }

    const addedRanges: vscode.DecorationOptions[] = [];
    const removedRanges: vscode.DecorationOptions[] = [];

    for (const change of pendingChanges) {
      // Process added ranges (green)
      for (const range of change.addedRanges) {
        const startLine = Math.max(0, range.startLine);
        const endLine = Math.min(
          editor.document.lineCount - 1,
          range.endLine
        );
        if (startLine <= endLine) {
          const agentLabel = change.agentName || change.agentId;
          addedRanges.push({
            range: new vscode.Range(startLine, 0, endLine, 0),
            hoverMessage: new vscode.MarkdownString(
              `**PRSM**: Modified by \`${agentLabel}\` using \`${change.toolName}\`\n\n` +
                `[Go to Agent](command:prsm.goToAgent?${encodeURIComponent(
                  JSON.stringify({
                    sessionId: change.sessionId,
                    agentId: change.agentId,
                    toolCallId: change.toolCallId,
                    messageIndex: change.messageIndex,
                  })
                )})`
            ),
          });
        }
      }

      // Process removed ranges (red) — these indicate where old content was
      // Only show if this is a modify (not create)
      if (change.changeType === "modify") {
        for (const range of change.removedRanges) {
          const startLine = Math.max(0, range.startLine);
          const endLine = Math.min(
            editor.document.lineCount - 1,
            range.endLine
          );
          if (startLine <= endLine) {
            removedRanges.push({
              range: new vscode.Range(startLine, 0, endLine, 0),
            });
          }
        }
      }
    }

    const deletedIndicators: vscode.DecorationOptions[] = [];

    for (const change of pendingChanges) {
      // Show deleted content indicators for modify changes with old content
      if (change.changeType === "modify" && change.oldContent && change.removedRanges.length > 0) {
        const firstRemoved = change.removedRanges[0];
        const indicatorLine = Math.max(0, Math.min(firstRemoved.startLine, editor.document.lineCount - 1));
        const oldLines = change.oldContent.split("\n");
        const lineCount = oldLines.length;
        const preview = oldLines[0]?.substring(0, 60) || "";

        deletedIndicators.push({
          range: new vscode.Range(indicatorLine, 0, indicatorLine, 0),
          renderOptions: {
            after: {
              contentText: ` ⊟ ${lineCount} line${lineCount !== 1 ? "s" : ""} removed: ${preview}${preview.length >= 60 ? "..." : ""}`,
              color: new vscode.ThemeColor("diffEditor.removedTextForeground"),
              fontStyle: "italic",
            },
          },
        });
      }
    }

    // Enable command URIs in hover messages
    for (const opt of addedRanges) {
      if (opt.hoverMessage instanceof vscode.MarkdownString) {
        opt.hoverMessage.isTrusted = true;
      }
    }

    editor.setDecorations(this.addedDecoration, addedRanges);
    editor.setDecorations(this.removedDecoration, removedRanges);
    editor.setDecorations(this.deletedIndicator, deletedIndicators);
  }

  /**
   * Update all visible editors.
   */
  updateAllEditors(): void {
    for (const editor of vscode.window.visibleTextEditors) {
      this.updateDecorations(editor);
    }
  }

  /**
   * Clear all decorations.
   */
  clearAll(): void {
    for (const editor of vscode.window.visibleTextEditors) {
      editor.setDecorations(this.addedDecoration, []);
      editor.setDecorations(this.removedDecoration, []);
      editor.setDecorations(this.deletedIndicator, []);
    }
  }

  dispose(): void {
    this.addedDecoration.dispose();
    this.removedDecoration.dispose();
    this.deletedIndicator.dispose();
    for (const d of this.disposables) {
      d.dispose();
    }
  }
}
