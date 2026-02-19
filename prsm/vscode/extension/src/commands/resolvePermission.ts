/**
 * Routes permission requests to the inline question-form UI in chat panels.
 */
import * as vscode from "vscode";
import { SessionStore } from "../state/sessionStore";
import { AgentWebviewManager } from "../views/agentWebviewManager";

export function registerPermissionHandler(
  store: SessionStore,
  webviewManager: AgentWebviewManager,
): vscode.Disposable {
  return store.onPermissionRequest((data) => {
    webviewManager.showPermissionRequest(data);
  });
}
