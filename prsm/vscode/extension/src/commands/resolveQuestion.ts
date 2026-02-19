/**
 * Routes user_question_request events to the inline question card
 * inside the agent's chat webview panel.
 *
 * Previously this module showed a VS Code QuickPick/InputBox popup.
 * Now it delegates entirely to AgentWebviewManager.showQuestion(),
 * which renders the question inline in the correct agent chat panel
 * and falls back to a native notification when that panel is inactive.
 */
import * as vscode from "vscode";
import { SessionStore } from "../state/sessionStore";
import { AgentWebviewManager } from "../views/agentWebviewManager";
import { UserQuestionData } from "../protocol/types";

export function registerQuestionHandler(
  store: SessionStore,
  webviewManager: AgentWebviewManager
): vscode.Disposable {
  return store.onUserQuestion((data: UserQuestionData) => {
    webviewManager.showQuestion(data);
  });
}
