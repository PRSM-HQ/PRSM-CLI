/**
 * TextDocumentContentProvider that serves pre-edit file content.
 * Used with vscode.diff() to show original vs current file content.
 * URI scheme: prsm-original:///path/to/file?toolCallId=xxx
 */
import * as vscode from "vscode";
import { FileChangeTracker } from "./fileChangeTracker";

export const ORIGINAL_SCHEME = "prsm-original";

export class OriginalContentProvider implements vscode.TextDocumentContentProvider {
  private readonly _onDidChange = new vscode.EventEmitter<vscode.Uri>();
  readonly onDidChange = this._onDidChange.event;

  constructor(private readonly tracker: FileChangeTracker) {
    // Refresh when changes are updated (accept/reject)
    tracker.onDidChange(() => {
      this._onDidChange.fire(vscode.Uri.parse(`${ORIGINAL_SCHEME}://refresh`));
    });
  }

  provideTextDocumentContent(uri: vscode.Uri): string {
    const params = new URLSearchParams(uri.query);
    const toolCallId = params.get("toolCallId");
    const filePath = uri.path;

    if (toolCallId) {
      // Get specific change's pre-tool content
      const found = this.tracker.getChangeByToolCallId(toolCallId);
      if (found) {
        // For Write tool: preToolContent is the full file before write
        if (found.change.preToolContent !== null) {
          return found.change.preToolContent;
        }
        // For Edit tool: reconstruct original from preToolContent
        if (found.change.preToolContent !== null) {
          return found.change.preToolContent;
        }
        // Fallback: return oldContent if it's the full file (Write case)
        if (found.change.oldContent !== null) {
          return found.change.oldContent;
        }
      }
    }

    // Fallback: try to find the earliest change for this file path
    // and return its pre-tool content
    const allChanges = this.tracker.getAllChanges(filePath);
    if (allChanges.length > 0) {
      const earliest = allChanges[0];
      if (earliest.preToolContent !== null) {
        return earliest.preToolContent;
      }
      if (earliest.oldContent !== null) {
        return earliest.oldContent;
      }
    }

    return "// No original content available";
  }

  /**
   * Build a URI for the original content of a specific change.
   */
  static buildUri(filePath: string, toolCallId: string): vscode.Uri {
    return vscode.Uri.parse(
      `${ORIGINAL_SCHEME}://${filePath}?toolCallId=${encodeURIComponent(toolCallId)}`
    );
  }

  /**
   * Build a URI for the original content of a file (earliest change).
   */
  static buildFileUri(filePath: string): vscode.Uri {
    return vscode.Uri.parse(`${ORIGINAL_SCHEME}://${filePath}`);
  }

  dispose(): void {
    this._onDidChange.dispose();
  }
}
