/**
 * Manages webview panels for agent conversations.
 *
 * Two types of panels:
 * - Session panels (orchestrator): has chat input bar, shows master agent conversation
 * - Agent panels (child agents): conversation view with input
 *
 * Message queue system: input is never disabled. When the user sends a message
 * while an agent is busy, an action picker lets them choose queue/inject/interrupt.
 */
import * as vscode from "vscode";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { SessionStore } from "../state/sessionStore";
import { PrsmTransport } from "../protocol/transport";
import {
  ModelSwitchedData,
  PermissionRequestData,
  StreamChunkData,
  UserQuestionData,
} from "../protocol/types";
import { toModelAliasLabel } from "../utils/modelLabel";

const DEBUG_LOG = path.join(os.homedir(), ".prsm", "logs", "webview-debug.log");
function debugLog(msg: string): void {
  try {
    fs.appendFileSync(DEBUG_LOG, `${new Date().toISOString()} ${msg}\n`);
  } catch { /* ignore */ }
}

interface QueuedMessage {
  id: string;
  prompt: string;
  mode: "queue" | "inject";
}

interface ThinkingVerbSettings {
  enableNsfw: boolean;
  customVerbs: string[];
  /** Verb lists loaded from the server (prsm/shared_ui/*.txt files). */
  safeVerbs: string[];
  nsfwVerbs: string[];
}

type ImportProvider = "codex" | "claude" | "prsm";
type ImportSessionListEntry = {
  provider: string;
  source_id: string;
  title: string | null;
  turn_count: number;
  source_path: string;
  started_at: string | null;
  updated_at: string | null;
};

export type ActiveChatTarget =
  | { kind: "session"; sessionId: string }
  | { kind: "agent"; sessionId: string; agentId: string };

export class AgentWebviewManager {
  /** Agent-specific panels keyed by `sessionId:agentId`. */
  private agentPanels = new Map<string, vscode.WebviewPanel>();
  /** Session/orchestrator panels keyed by `sessionId`. */
  private sessionPanels = new Map<string, vscode.WebviewPanel>();
  /** Per-session message queues. */
  private messageQueues = new Map<string, QueuedMessage[]>();
  /** Track last master ID sent to each session panel for change detection. */
  private lastMasterIds = new Map<string, string>();

  /** Pending (unanswered) questions keyed by requestId. */
  private pendingQuestions = new Map<string, UserQuestionData>();
  /** Pending permission prompts keyed by requestId, shown via inline question cards. */
  private pendingPermissionRequests = new Map<string, PermissionRequestData>();
  /** Currently active PRSM chat panel key (`agent:<session>:<agent>` or `session:<session>`). */
  private activeChatKey: string | undefined;
  private readonly _onDidChangeActiveChat =
    new vscode.EventEmitter<ActiveChatTarget | undefined>();
  readonly onDidChangeActiveChat = this._onDidChangeActiveChat.event;

  /** Polling intervals for continuous webview refresh while session is running. */
  private pollingIntervals = new Map<string, ReturnType<typeof setInterval>>();
  /** Latest generated plan file per session (if any). */
  private planFileBySession = new Map<string, string>();
  /** Cached verb lists from server (loaded from prsm/shared_ui/*.txt). */
  private cachedSafeVerbs: string[] | null = null;
  private cachedNsfwVerbs: string[] | null = null;
  private verbFetchPromise: Promise<void> | null = null;

  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly store: SessionStore,
    private readonly getTransport: () => PrsmTransport | undefined,
    private readonly outputChannel?: vscode.OutputChannel,
  ) {
    debugLog(`AgentWebviewManager constructor called`);
    // Update open agent webviews when messages change
    store.onDidChangeMessages(({ sessionId, agentId }) => {
      try {
        debugLog(`onDidChangeMessages session=${sessionId.slice(0, 8)} agent=${agentId.slice(0, 8)} sessionPanels=[${[...this.sessionPanels.keys()].map(s => s.slice(0, 8)).join(",")}]`);
        this.updateAgentWebview(sessionId, agentId);
        this.updateSessionWebview(sessionId);
      } catch (err) {
        debugLog(`ERROR onDidChangeMessages: ${(err as Error).message}\n${(err as Error).stack}`);
      }
    });

    // Push stream chunks to open webviews
    store.onStreamChunk((data) => {
      this.pushStreamChunk(data);
      this.pushSessionStreamChunk(data);
    });

    // When tree changes, update session panels (e.g., master agent appeared)
    store.onDidChangeTree(() => {
      try {
        this.outputChannel?.appendLine(
          `[trigger] onDidChangeTree panels=${[...this.sessionPanels.keys()].map(s => s.slice(0, 8)).join(",")}`
        );
        for (const sessionId of this.sessionPanels.keys()) {
          this.updateSessionWebview(sessionId);
        }
      } catch (err) {
        this.outputChannel?.appendLine(
          `[ERROR] onDidChangeTree threw: ${(err as Error).message}\n${(err as Error).stack}`
        );
      }
    });

    // Continuous polling: when engine starts, poll every 2s to ensure the
    // webview stays up-to-date even if individual SSE events are missed.
    store.onEngineStarted((data) => {
      this.outputChannel?.appendLine(
        `[trigger] onEngineStarted session=${data.session_id.slice(0, 8)}`
      );
      this.startPolling(data.session_id);
    });

    // Handle orchestration completion: clear busy state, send final update,
    // and process queue
    store.onEngineFinished((data) => {
      this.stopPolling(data.session_id);
      const panel = this.sessionPanels.get(data.session_id);
      if (panel) {
        panel.webview.postMessage({ type: "setBusyState", busy: false });
      }
      this.broadcastQueueState(data.session_id);
      // Also clear busy on agent panels for this session
      for (const [key, agentPanel] of this.agentPanels) {
        if (key.startsWith(data.session_id + ":")) {
          agentPanel.webview.postMessage({ type: "setBusyState", busy: false });
        }
      }
      // Final fullUpdate so conversation shows completed state
      this.updateSessionWebview(data.session_id);
      // Process next queued message
      this.processNextQueuedMessage(data.session_id);
    });

    // Forward thinking events to session panels
    store.onThinking((data) => {
      // Forward to session panel if from master agent
      const panel = this.sessionPanels.get(data.session_id);
      if (panel) {
        const master = this.store.getMasterAgent(data.session_id);
        if (master && master.id === data.agent_id) {
          panel.webview.postMessage({
            type: "thinking",
            agentId: data.agent_id,
            text: data.text,
          });
        }
      }
      // Forward to agent-specific panel
      const agentKey = `${data.session_id}:${data.agent_id}`;
      const agentPanel = this.agentPanels.get(agentKey);
      if (agentPanel) {
        agentPanel.webview.postMessage({
          type: "thinking",
          agentId: data.agent_id,
          text: data.text,
        });
      }
    });

    store.onPlanFileUpdated((data) => {
      this.planFileBySession.set(data.session_id, data.file_path);
      const panel = this.sessionPanels.get(data.session_id);
      if (panel) {
        panel.webview.postMessage({
          type: "showPlanLink",
          filePath: data.file_path,
        });
      }
    });

    // Forward model switch events to session panels so they can show
    // a visual indicator in the conversation and update the model button.
    store.onModelSwitched((data: ModelSwitchedData) => {
      const panel = this.sessionPanels.get(data.session_id);
      if (panel) {
        const master = this.store.getMasterAgent(data.session_id);
        const atMessageIndex = master
          ? this.store.getMessages(data.session_id, master.id).length
          : 0;
        panel.webview.postMessage({
          type: "modelSwitched",
          oldModel: data.old_model,
          newModel: data.new_model,
          oldModelLabel: toModelAliasLabel(data.old_model) ?? data.old_model,
          newModelLabel: toModelAliasLabel(data.new_model) ?? data.new_model,
          provider: data.provider,
          atMessageIndex,
        });
      }
    });
  }

  /** Get the currently active chat key, if any. */
  getActiveChatKey(): string | undefined {
    return this.activeChatKey;
  }

  private isSessionEffectivelyRunning(sessionId: string): boolean {
    const session = this.store.getSession(sessionId);
    if (!session) return false;
    if (session.running) return true;
    for (const agent of session.agents.values()) {
      if (agent.state !== "completed" && agent.state !== "error") {
        return true;
      }
    }
    return false;
  }

  private startPolling(sessionId: string): void {
    this.stopPolling(sessionId);
    // Immediate update + every 2s while running
    this.updateSessionWebview(sessionId);
    const interval = setInterval(() => {
      if (!this.isSessionEffectivelyRunning(sessionId)) {
        this.stopPolling(sessionId);
        return;
      }
      this.outputChannel?.appendLine(
        `[poll] refresh session=${sessionId.slice(0, 8)}`
      );
      this.updateSessionWebview(sessionId);
    }, 2000);
    this.pollingIntervals.set(sessionId, interval);
  }

  private stopPolling(sessionId: string): void {
    const interval = this.pollingIntervals.get(sessionId);
    if (interval) {
      clearInterval(interval);
      this.pollingIntervals.delete(sessionId);
    }
  }

  /** Open an agent-specific conversation panel. */
  showAgent(sessionId: string, agentId: string): void {
    const key = `${sessionId}:${agentId}`;
    const existing = this.agentPanels.get(key);
    if (existing) {
      existing.reveal();
      this.setActiveChat({
        kind: "agent",
        sessionId,
        agentId,
      });
      return;
    }

    const agent = this.store.getAgent(sessionId, agentId);
    if (!agent) return;

    const session = this.store.getSession(sessionId);
    const sessionName = session?.name ?? sessionId;

    const panel = vscode.window.createWebviewPanel(
      "prsmAgent",
      `${agent.name} - ${sessionName}`,
      {
        viewColumn: this.getPreferredViewColumn(),
        preserveFocus: false,
      },
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [
          vscode.Uri.joinPath(
            this.context.extensionUri,
            "media",
            "webview"
          ),
        ],
      }
    );

    panel.webview.html = this.getWebviewHtml(panel.webview);
    this.agentPanels.set(key, panel);

    panel.onDidDispose(() => {
      if (this.activeChatKey === this.toAgentChatKey(sessionId, agentId)) {
        this.setActiveChat(undefined);
      }
      this.agentPanels.delete(key);
    });
    panel.onDidChangeViewState((event) => {
      if (event.webviewPanel.active) {
        this.setActiveChat({
          kind: "agent",
          sessionId,
          agentId,
        });
      } else if (this.activeChatKey === this.toAgentChatKey(sessionId, agentId)) {
        this.setActiveChat(undefined);
      }
    });

    // Handle messages from webview (prompt submission for agent chat)
    // Wait for "ready" from webview before sending initial state to avoid
    // the race condition where postMessage is lost before JS loads.
    let ready = false;
    panel.webview.onDidReceiveMessage((msg) => {
      if (msg.type === "ready" && !ready) {
        ready = true;
        this.sendAgentState(panel, sessionId, agentId);
        return;
      }
      if (msg.type === "sendPrompt" && msg.prompt) {
        this.handleAgentPromptSubmission(sessionId, agentId, panel, msg.prompt);
      } else if (msg.type === "confirmStopRun") {
        void this.confirmAndStopRun(sessionId, panel);
      } else if (msg.type === "stopRun") {
        this.handleStopRun(sessionId, panel);
      } else if (msg.type === "promptAction" && msg.prompt && msg.action) {
        this.handleAgentPromptAction(
          sessionId,
          agentId,
          panel,
          msg.prompt,
          msg.action,
        );
      } else if (msg.type === "answerQuestion" && msg.requestId && msg.answer) {
        this.handleQuestionAnswer(sessionId, msg.requestId, msg.answer);
      } else if (msg.type === "openFile" && msg.filePath) {
        this.handleOpenFile(msg.filePath);
      } else if (msg.type === "openAgent" && msg.agentId) {
        this.showAgent(sessionId, msg.agentId);
      } else if (msg.type === "fileComplete") {
        this.handleFileComplete(panel, msg.prefix ?? "", msg.limit ?? 10, msg.requestId);
      }
    });
  }

  /** Open the orchestrator conversation panel for a session (with chat input). */
  showSession(sessionId: string): void {
    const existing = this.sessionPanels.get(sessionId);
    if (existing) {
      existing.reveal();
      this.setActiveChat({ kind: "session", sessionId });
      // Focus the prompt input when switching to an existing session tab
      existing.webview.postMessage({ type: "focusInput" });
      return;
    }

    const session = this.store.getSession(sessionId);
    if (!session) return;

    const panel = vscode.window.createWebviewPanel(
      "prsmSession",
      session.name,
      {
        viewColumn: this.getPreferredViewColumn(),
        preserveFocus: false,
      },
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [
          vscode.Uri.joinPath(
            this.context.extensionUri,
            "media",
            "webview"
          ),
        ],
      }
    );

    panel.webview.html = this.getWebviewHtml(panel.webview);
    this.sessionPanels.set(sessionId, panel);
    debugLog(`showSession PANEL CREATED session=${sessionId.slice(0, 8)} panels=[${[...this.sessionPanels.keys()].map(s => s.slice(0, 8)).join(",")}]`);

    panel.onDidDispose(() => {
      if (this.activeChatKey === this.toSessionChatKey(sessionId)) {
        this.setActiveChat(undefined);
      }
      debugLog(`showSession PANEL DISPOSED session=${sessionId.slice(0, 8)}`);
      this.sessionPanels.delete(sessionId);
    });
    panel.onDidChangeViewState((event) => {
      if (event.webviewPanel.active) {
        this.setActiveChat({ kind: "session", sessionId });
      } else if (this.activeChatKey === this.toSessionChatKey(sessionId)) {
        this.setActiveChat(undefined);
      }
    });

    // Handle messages from webview.
    // Wait for "ready" from webview before sending initial state to avoid
    // the race condition where postMessage is lost before JS loads.
    let ready = false;
    panel.webview.onDidReceiveMessage((msg) => {
      debugLog(`webview msg: ${JSON.stringify(msg).slice(0, 200)}`);
      if (msg.type === "ready" && !ready) {
        ready = true;
        debugLog(`webview READY for session=${sessionId.slice(0, 8)} — calling sendSessionState`);
        this.sendSessionState(sessionId);
        // Auto-focus the prompt input so the user can start typing immediately
        panel.webview.postMessage({ type: "focusInput" });
        return;
      }
      if (msg.type === "debugFullUpdate") {
        this.outputChannel?.appendLine(
          `[webview] fullUpdate received #${msg.updateNum ?? "?"} agent=${msg.agentId} state=${msg.agentState} msgs=${msg.msgCount} busy=${msg.busy}`
        );
      } else if (msg.type === "requestRefresh") {
        // Webview is requesting a refresh — it hasn't received updates in a while
        this.outputChannel?.appendLine(
          `[webview] requestRefresh for session=${sessionId.slice(0, 8)} — webview hasn't received updates`
        );
        this.sendSessionState(sessionId);
      } else if (msg.type === "sendPrompt" && msg.prompt) {
        this.handlePromptSubmission(sessionId, panel, msg.prompt);
      } else if (msg.type === "confirmStopRun") {
        void this.confirmAndStopRun(sessionId, panel);
      } else if (msg.type === "stopRun") {
        this.handleStopRun(sessionId, panel);
      } else if (msg.type === "promptAction" && msg.prompt && msg.action) {
        this.handleWebviewAction(sessionId, panel, msg);
      } else if (msg.type === "cancelQueued" && msg.queueId) {
        this.handleCancelQueued(sessionId, msg.queueId);
      } else if (msg.type === "answerQuestion" && msg.requestId && msg.answer) {
        this.handleQuestionAnswer(sessionId, msg.requestId, msg.answer);
      } else if (msg.type === "openFile" && msg.filePath) {
        this.handleOpenFile(msg.filePath);
      } else if (msg.type === "openAgent" && msg.agentId) {
        this.showAgent(sessionId, msg.agentId);
      } else if (msg.type === "fileComplete") {
        this.handleFileComplete(panel, msg.prefix ?? "", msg.limit ?? 10, msg.requestId);
      } else if (msg.type === "selectModel") {
        this.handleModelSelection(sessionId);
      }
    });
  }

  private getPreferredViewColumn(): vscode.ViewColumn {
    const activeGroupColumn = vscode.window.tabGroups.activeTabGroup.viewColumn;
    if (activeGroupColumn !== undefined) {
      return activeGroupColumn;
    }

    return vscode.window.activeTextEditor?.viewColumn ?? vscode.ViewColumn.Active;
  }

  /**
   * Scroll an open agent webview to a specific tool call.
   * Used by the "Go to Agent" command to highlight the edit that made a file change.
   */
  scrollToToolCall(
    sessionId: string,
    agentId: string,
    toolCallId: string,
    messageIndex?: number,
    preferSessionPanel = false,
  ): void {
    const sessionPanel = this.sessionPanels.get(sessionId);
    const key = `${sessionId}:${agentId}`;
    const agentPanel = this.agentPanels.get(key);

    if (preferSessionPanel && sessionPanel) {
      sessionPanel.webview.postMessage({
        type: "scrollToToolCall",
        toolCallId,
        messageIndex,
      });
      return;
    }

    if (agentPanel) {
      agentPanel.webview.postMessage({
        type: "scrollToToolCall",
        toolCallId,
        messageIndex,
      });
      return;
    }

    if (sessionPanel) {
      sessionPanel.webview.postMessage({
        type: "scrollToToolCall",
        toolCallId,
        messageIndex,
      });
    }
  }

  /**
   * Scroll an open agent webview to a specific snapshot.
   * Used by the "Go to Snapshot" command.
   */
  scrollToSnapshot(
    sessionId: string,
    agentId: string,
    snapshotId: string,
    preferSessionPanel = false,
  ): void {
    const sessionPanel = this.sessionPanels.get(sessionId);
    const key = `${sessionId}:${agentId}`;
    const agentPanel = this.agentPanels.get(key);

    const message = {
      type: "scrollToSnapshot",
      snapshotId,
    };

    if (preferSessionPanel && sessionPanel) {
      sessionPanel.webview.postMessage(message);
      return;
    }

    if (agentPanel) {
      agentPanel.webview.postMessage(message);
      return;
    }

    if (sessionPanel) {
      sessionPanel.webview.postMessage(message);
    }
  }

  // ── Private: state updates ──

  /**
   * Fetch thinking verb lists from the server (loaded from prsm/shared_ui/*.txt).
   * Caches the result so subsequent calls are instant.
   */
  fetchThinkingVerbs(): void {
    if (this.verbFetchPromise) return; // already in progress or done
    this.verbFetchPromise = (async () => {
      try {
        const transport = this.getTransport();
        if (!transport) return;
        const result = await transport.getThinkingVerbs();
        if (Array.isArray(result.safe) && result.safe.length > 0) {
          this.cachedSafeVerbs = result.safe;
        }
        if (Array.isArray(result.nsfw)) {
          this.cachedNsfwVerbs = result.nsfw;
        }
        debugLog(`fetchThinkingVerbs: loaded ${this.cachedSafeVerbs?.length ?? 0} safe, ${this.cachedNsfwVerbs?.length ?? 0} nsfw verbs`);
      } catch (err) {
        debugLog(`fetchThinkingVerbs error: ${(err as Error).message}`);
        // Will fall back to hardcoded lists in the webview
      }
    })();
  }

  private getThinkingVerbSettings(): ThinkingVerbSettings {
    // Trigger async fetch if not yet started (non-blocking)
    this.fetchThinkingVerbs();

    const cfg = vscode.workspace.getConfiguration("prsm");
    const enableNsfw = cfg.get<boolean>("thinkingVerbs.enableNsfw", true);
    const rawCustomVerbs = cfg.get<string[]>("thinkingVerbs.custom", []);
    const customVerbs = Array.isArray(rawCustomVerbs)
      ? rawCustomVerbs
        .filter((verb): verb is string => typeof verb === "string")
        .map((verb) => verb.trim())
        .filter((verb, idx, arr) => verb.length > 0 && arr.indexOf(verb) === idx)
      : [];
    return {
      enableNsfw,
      customVerbs,
      safeVerbs: this.cachedSafeVerbs ?? [],
      nsfwVerbs: this.cachedNsfwVerbs ?? [],
    };
  }

  private resolveSessionModelLabel(
    currentModelDisplay?: string | null,
    currentModel?: string | null,
    fallbackModel?: string | null,
  ): string {
    return (
      toModelAliasLabel(currentModelDisplay) ??
      toModelAliasLabel(currentModel ?? fallbackModel ?? "") ??
      toModelAliasLabel(fallbackModel ?? "") ??
      currentModelDisplay ??
      currentModel ??
      fallbackModel ??
      ""
    );
  }

  private sendAgentState(
    panel: vscode.WebviewPanel,
    sessionId: string,
    agentId: string,
  ): void {
    const agent = this.store.getAgent(sessionId, agentId);
    const messages = this.store.getMessages(sessionId, agentId);
    const session = this.store.getSession(sessionId);
    const contextUsage = this.store.getContextUsage(sessionId, agentId);
    const modelId = agent?.model ?? session?.currentModel ?? "";
    const modelLabel = this.resolveSessionModelLabel(
      session?.currentModelDisplay,
      modelId,
    );

    const sessionRunning = this.isSessionEffectivelyRunning(sessionId);
    const agentBusy = sessionRunning || (!!agent && agent.state !== "completed" && agent.state !== "error");

    panel.webview.postMessage({
      type: "fullUpdate",
      agent,
      messages,
      sessionName: session?.name ?? sessionId,
      sessionSummary: session?.summary ?? "",
      showInput: true,
      busy: agentBusy,
      sessionRunning,
      contextUsage,
      currentModel: modelId,
      currentModelLabel: modelLabel,
      isThinking: this.store.isAgentThinking(agentId),
      thinkingVerbSettings: this.getThinkingVerbSettings(),
    });
    this.postPendingQuestionsForAgent(panel, sessionId, agentId);
  }

  private updateAgentWebview(sessionId: string, agentId: string): void {
    const key = `${sessionId}:${agentId}`;
    const panel = this.agentPanels.get(key);
    if (!panel) return;
    this.sendAgentState(panel, sessionId, agentId);
  }

  private sendSessionState(sessionId: string): void {
    const panel = this.sessionPanels.get(sessionId);
    if (!panel) {
      debugLog(`sendSessionState NO PANEL for session=${sessionId.slice(0, 8)} panels=[${[...this.sessionPanels.keys()].map(s => s.slice(0, 8)).join(",")}]`);
      return;
    }

    const session = this.store.getSession(sessionId);
    if (!session) {
      debugLog(`sendSessionState NO SESSION for session=${sessionId.slice(0, 8)}`);
      return;
    }

    const master = this.store.getMasterAgent(sessionId);
    const sessionRunning = this.isSessionEffectivelyRunning(sessionId);
    debugLog(`sendSessionState session=${sessionId.slice(0, 8)} master=${master?.id?.slice(0, 8) ?? "NONE"} running=${sessionRunning}`);

    if (master) {
      // Note: we always send the fullUpdate even if the master is from a
      // previous run (completed/errored) while a new run is starting.
      // The webview's pendingUserMessage mechanism (main.js) preserves
      // the user's follow-up prompt across renderFull() calls.
      // Previously a guard here blocked ALL updates in this state, which
      // could permanently prevent the webview from updating if the new
      // master's agent_spawned event was lost or delayed.
      if (
        sessionRunning &&
        (master.state === "completed" || master.state === "error")
      ) {
        this.outputChannel?.appendLine(
          `[sendSessionState] STALE MASTER: session=${sessionId.slice(0, 8)} master=${master.id.slice(0, 8)} state=${master.state} running=${sessionRunning} — sending anyway (webview preserves pending prompt)`
        );
      }

      const prevMasterId = this.lastMasterIds.get(sessionId);
      if (prevMasterId && prevMasterId !== master.id) {
        this.outputChannel?.appendLine(
          `[sendSessionState] MASTER CHANGED: ${prevMasterId.slice(0, 8)} → ${master.id.slice(0, 8)}`
        );
      }
      this.lastMasterIds.set(sessionId, master.id);

      const messages = this.store.getMessages(sessionId, master.id);
      const contextUsage = this.store.getContextUsage(sessionId, master.id);
      const masterBusy =
        sessionRunning ||
        (master.state !== "completed" && master.state !== "error");
      const modelLabel = this.resolveSessionModelLabel(
        session.currentModelDisplay,
        session.currentModel,
      );
      debugLog(`sendSessionState SEND master=${master.id.slice(0, 8)} state=${master.state} msgs=${messages.length} busy=${masterBusy} panelVisible=${panel.visible}`);
      const sent = panel.webview.postMessage({
        type: "fullUpdate",
        agent: master,
        messages,
        sessionName: session.name,
        sessionSummary: session.summary ?? "",
        showInput: true,
        busy: masterBusy,
        sessionRunning,
        contextUsage,
        isThinking: this.store.isAgentThinking(master.id),
        thinkingVerbSettings: this.getThinkingVerbSettings(),
        currentModel: session.currentModel,
        currentModelLabel: modelLabel,
      });
      sent.then(ok => debugLog(`postMessage result=${ok}`), err => debugLog(`postMessage error=${err}`));
      const planFile = this.planFileBySession.get(sessionId);
      if (planFile) {
        panel.webview.postMessage({ type: "showPlanLink", filePath: planFile });
      }
      this.postPendingQuestionsForSession(panel, sessionId);
    } else if (sessionRunning) {
      // Session is running but master agent hasn't spawned yet.
      // Keep existing conversation visible if we already have history.
      const preservedMessages = this.store.getAllMasterMessages(sessionId);
      if (preservedMessages.length > 0) {
        panel.webview.postMessage({
          type: "fullUpdate",
          agent: {
            name: "Orchestrator",
            state: "running",
            role: "orchestrator",
            model: "",
            promptPreview: "",
          },
          messages: preservedMessages,
          sessionName: session.name,
          sessionSummary: session.summary ?? "",
          showInput: true,
          busy: true,
          sessionRunning,
          contextUsage: null,
          isThinking: false,
          thinkingVerbSettings: this.getThinkingVerbSettings(),
          currentModel: session.currentModel,
          currentModelLabel: this.resolveSessionModelLabel(
            session.currentModelDisplay,
            session.currentModel,
          ),
        });
        const planFile = this.planFileBySession.get(sessionId);
        if (planFile) {
          panel.webview.postMessage({ type: "showPlanLink", filePath: planFile });
        }
        this.postPendingQuestionsForSession(panel, sessionId);
        return;
      }

      // No history to preserve, send a minimal busy state.
      panel.webview.postMessage({
        type: "fullUpdate",
        agent: {
          name: "Orchestrator",
          state: "running",
          role: "orchestrator",
          model: "",
          promptPreview: "",
        },
        messages: [],
        sessionName: session.name,
        sessionSummary: session.summary ?? "",
        showInput: true,
        busy: true,
        sessionRunning,
        contextUsage: null,
        isThinking: false,
        thinkingVerbSettings: this.getThinkingVerbSettings(),
        currentModel: session.currentModel,
        currentModelLabel: this.resolveSessionModelLabel(
          session.currentModelDisplay,
          session.currentModel,
        ),
      });
      const planFile = this.planFileBySession.get(sessionId);
      if (planFile) {
        panel.webview.postMessage({ type: "showPlanLink", filePath: planFile });
      }
      this.postPendingQuestionsForSession(panel, sessionId);
    } else {
      // Not running, no active master. Preserve history when available.
      const preservedMessages = this.store.getAllMasterMessages(sessionId);
      if (preservedMessages.length > 0) {
        panel.webview.postMessage({
          type: "fullUpdate",
          agent: {
            name: "Orchestrator",
            state: "completed",
            role: "orchestrator",
            model: "",
            promptPreview: "",
          },
          messages: preservedMessages,
          sessionName: session.name,
          sessionSummary: session.summary ?? "",
          showInput: true,
          busy: false,
          sessionRunning,
          contextUsage: null,
          isThinking: false,
          thinkingVerbSettings: this.getThinkingVerbSettings(),
          currentModel: session.currentModel,
          currentModelLabel: this.resolveSessionModelLabel(
            session.currentModelDisplay,
            session.currentModel,
          ),
        });
        const planFile = this.planFileBySession.get(sessionId);
        if (planFile) {
          panel.webview.postMessage({ type: "showPlanLink", filePath: planFile });
        }
        this.postPendingQuestionsForSession(panel, sessionId);
        return;
      }

      // No history available — show welcome state.
      panel.webview.postMessage({
        type: "fullUpdate",
        agent: {
          name: "Orchestrator",
          state: "idle",
          role: "orchestrator",
          model: "",
          promptPreview: "",
        },
        messages: [],
        sessionName: session.name,
        sessionSummary: session.summary ?? "",
        showInput: true,
        busy: false,
        sessionRunning,
        contextUsage: null,
        isThinking: false,
        thinkingVerbSettings: this.getThinkingVerbSettings(),
        currentModel: session.currentModel,
        currentModelLabel: this.resolveSessionModelLabel(
          session.currentModelDisplay,
          session.currentModel,
        ),
        });
      const planFile = this.planFileBySession.get(sessionId);
      if (planFile) {
        panel.webview.postMessage({ type: "showPlanLink", filePath: planFile });
      }
      this.postPendingQuestionsForSession(panel, sessionId);
    }
    this.broadcastQueueState(sessionId);
  }

  private updateSessionWebview(sessionId: string): void {
    const panel = this.sessionPanels.get(sessionId);
    if (!panel) return;
    this.sendSessionState(sessionId);
  }

  private pushStreamChunk(data: StreamChunkData): void {
    const key = `${data.session_id}:${data.agent_id}`;
    const panel = this.agentPanels.get(key);
    if (!panel) return;

    panel.webview.postMessage({
      type: "streamChunk",
      agentId: data.agent_id,
      text: data.text,
    });
  }

  private pushSessionStreamChunk(data: StreamChunkData): void {
    const panel = this.sessionPanels.get(data.session_id);
    if (!panel) return;

    // Only forward chunks from the master agent
    const master = this.store.getMasterAgent(data.session_id);
    if (!master || master.id !== data.agent_id) return;

    panel.webview.postMessage({
      type: "streamChunk",
      agentId: data.agent_id,
      text: data.text,
    });
  }

  // ── Prompt handling ──

  private async handlePromptSubmission(
    sessionId: string,
    panel: vscode.WebviewPanel,
    prompt: string,
  ): Promise<void> {
    const rawPrompt = String(prompt || "").trim();
    const slash = this.parseSlashCommand(prompt);
    if (slash) {
      await this.handleSlashCommand(sessionId, panel, slash.name, slash.args);
      return;
    }
    if (rawPrompt.startsWith("/")) {
      await this.showSlashCommandPicker(sessionId, panel, rawPrompt);
      return;
    }

    const transport = this.getTransport();
    if (!transport?.isConnected) {
      panel.webview.postMessage({
        type: "errorMessage",
        content: "Not connected to PRSM server.",
      });
      return;
    }

    const session = this.store.getSession(sessionId);
    const master = this.store.getMasterAgent(sessionId);
    const sessionRunning = this.isSessionEffectivelyRunning(sessionId);
    debugLog(`handlePromptSubmission ENTRY session=${sessionId.slice(0, 8)} running=${sessionRunning} master=${master?.id?.slice(0, 8) ?? "NONE"} masterState=${master?.state ?? "N/A"}`);

    // If session is NOT running and there's an existing master agent,
    // send the message to it directly (restarts the agent and continues
    // its conversation) rather than starting a new orchestration run.
    if (!sessionRunning && master) {
      debugLog(`handlePromptSubmission: RESTARTING existing master=${master.id.slice(0, 8)}`);
      this.outputChannel?.appendLine(
        `[handlePromptSubmission] session=${sessionId.slice(0, 8)} restart master=${master.id.slice(0, 8)} prompt="${prompt.slice(0, 40)}"`
      );
      panel.webview.postMessage({ type: "userMessage", content: prompt });
      panel.webview.postMessage({ type: "setBusyState", busy: true });
      try {
        await transport.sendAgentMessage(sessionId, master.id, prompt);
      } catch (err) {
        debugLog(`handlePromptSubmission: sendAgentMessage FAILED: ${(err as Error).message}`);
        panel.webview.postMessage({
          type: "errorMessage",
          content: `Failed: ${(err as Error).message}`,
        });
        panel.webview.postMessage({ type: "setBusyState", busy: false });
      }
      return;
    }

    // Session is running — show delivery mode picker
    if (sessionRunning) {
      debugLog(`handlePromptSubmission: session RUNNING, showing delivery mode picker`);
      const mode = await this.showDeliveryModePicker();
      if (!mode) return; // User cancelled
      debugLog(`handlePromptSubmission: delivery mode=${mode} master=${master?.id?.slice(0, 8) ?? "NONE"}`);
      if (mode === "queue") {
        this.addToQueue(sessionId, prompt, "queue");
        this.broadcastQueueState(sessionId);
        return;
      }

      // Find master agent for inject target
      const agentId = master?.id ?? "";

      await this.deliverPromptWithMode(sessionId, agentId, panel, prompt, mode);
      return;
    }

    // Not running, no master — start fresh orchestration
    debugLog(`handlePromptSubmission: NO MASTER, starting new run`);
    this.outputChannel?.appendLine(
      `[handlePromptSubmission] session=${sessionId.slice(0, 8)} new run prompt="${prompt.slice(0, 40)}"`
    );
    panel.webview.postMessage({ type: "userMessage", content: prompt });
    panel.webview.postMessage({ type: "setBusyState", busy: true });
    try {
      await transport.runPrompt(sessionId, prompt);
    } catch (err) {
      panel.webview.postMessage({
        type: "errorMessage",
        content: `Failed: ${(err as Error).message}`,
      });
      panel.webview.postMessage({ type: "setBusyState", busy: false });
    }
  }

  private async handleAgentPromptSubmission(
    sessionId: string,
    agentId: string,
    panel: vscode.WebviewPanel,
    prompt: string,
  ): Promise<void> {
    const rawPrompt = String(prompt || "").trim();
    const slash = this.parseSlashCommand(prompt);
    if (slash) {
      await this.handleSlashCommand(sessionId, panel, slash.name, slash.args);
      return;
    }
    if (rawPrompt.startsWith("/")) {
      await this.showSlashCommandPicker(sessionId, panel, rawPrompt);
      return;
    }

    const transport = this.getTransport();
    if (!transport?.isConnected) {
      panel.webview.postMessage({
        type: "errorMessage",
        content: "Not connected to PRSM server.",
      });
      return;
    }

    // Check if session is running — if so, show delivery mode picker
    const session = this.store.getSession(sessionId);
    if (this.isSessionEffectivelyRunning(sessionId)) {
      const mode = await this.showDeliveryModePicker();
      if (!mode) return; // User cancelled
      if (mode === "queue") {
        this.addToQueue(sessionId, prompt, "queue");
        this.broadcastQueueState(sessionId);
        return;
      }

      await this.deliverPromptWithMode(sessionId, agentId, panel, prompt, mode);
      return;
    }

    // Optimistically mark agent as running — the server will restart it,
    // and this prevents stale "completed" state from overriding busy=true
    // when SSE events trigger a fullUpdate before agent_restarted arrives
    const agent = this.store.getAgent(sessionId, agentId);
    if (agent && (agent.state === "completed" || agent.state === "error")) {
      agent.state = "running";
    }

    // Show user message immediately and set busy
    panel.webview.postMessage({ type: "userMessage", content: prompt });
    panel.webview.postMessage({ type: "setBusyState", busy: true });

    try {
      await transport.sendAgentMessage(sessionId, agentId, prompt);
    } catch (err) {
      panel.webview.postMessage({
        type: "errorMessage",
        content: `Failed: ${(err as Error).message}`,
      });
      panel.webview.postMessage({ type: "setBusyState", busy: false });
    }
  }

  private parseSlashCommand(
    rawPrompt: string
  ): { name: string; args: string[] } | null {
    const raw = String(rawPrompt || "").trim();
    if (!raw.startsWith("/")) return null;
    const tokens = raw.match(/"([^"\\]|\\.)*"|'([^'\\]|\\.)*'|\S+/g) ?? [];
    const first = tokens[0];
    if (!first) return null;
    const name = first.slice(1).toLowerCase().trim();
    if (!name) return null;
    const args = tokens.slice(1).map((t) => {
      if (
        (t.startsWith('"') && t.endsWith('"')) ||
        (t.startsWith("'") && t.endsWith("'"))
      ) {
        return t.slice(1, -1);
      }
      return t;
    });
    return { name, args };
  }

  private postSystemMessage(panel: vscode.WebviewPanel, content: string): void {
    panel.webview.postMessage({ type: "systemMessage", content });
  }

  private slashHelpText(): string {
    return [
      "Available slash commands in VS Code chat:",
      "/help",
      "/import",
      "/import list [all|codex|claude|prsm]",
      "/import preview PROVIDER SOURCE_ID",
      "/import run PROVIDER SOURCE_ID [SESSION NAME] [--max-turns N]",
    ].join("\n");
  }

  private async showSlashCommandPicker(
    sessionId: string,
    panel: vscode.WebviewPanel,
    rawInput?: string
  ): Promise<void> {
    const allItems: Array<vscode.QuickPickItem & { value: "help" | "import" }> = [
      {
        label: "/help",
        description: "List supported slash commands",
        value: "help",
      },
      {
        label: "/import",
        description: "Interactive transcript import (provider/session/depth)",
        value: "import",
      },
    ];
    const normalized = String(rawInput || "").trim().toLowerCase();
    let items = allItems;
    if (normalized && normalized !== "/") {
      const filtered = allItems.filter((item) =>
        item.label.toLowerCase().startsWith(normalized)
      );
      if (filtered.length > 0) {
        items = filtered;
      }
    }

    const pick = await vscode.window.showQuickPick(items, {
      title: "PRSM Slash Commands",
      placeHolder: "Select a slash command",
      matchOnDescription: true,
    });
    if (!pick) return;

    if (pick.value === "help") {
      this.postSystemMessage(panel, this.slashHelpText());
      return;
    }
    await this.runInteractiveImport(sessionId, panel);
  }

  private parseImportProvider(raw: string): ImportProvider | null {
    const provider = String(raw || "").trim().toLowerCase();
    if (provider === "codex" || provider === "claude" || provider === "prsm") {
      return provider;
    }
    return null;
  }

  private formatImportTimestamp(raw: string | null): string {
    if (!raw) return "unknown";
    const parsed = new Date(raw);
    if (Number.isNaN(parsed.getTime())) return raw;
    return parsed.toLocaleString();
  }

  private sortImportSessionsByRecency(
    sessions: ImportSessionListEntry[]
  ): ImportSessionListEntry[] {
    return [...sessions].sort((a, b) => {
      const aRaw = a.updated_at || a.started_at;
      const bRaw = b.updated_at || b.started_at;
      const aTs = aRaw ? Date.parse(aRaw) : 0;
      const bTs = bRaw ? Date.parse(bRaw) : 0;
      return bTs - aTs;
    });
  }

  private async promptImportProvider(
    transport: PrsmTransport
  ): Promise<ImportProvider | undefined> {
    const all = await transport.listImportSessions("all", 200);
    const codexCount = all.sessions.filter((s) => s.provider === "codex").length;
    const claudeCount = all.sessions.filter((s) => s.provider === "claude").length;
    const prsmCount = all.sessions.filter((s) => s.provider === "prsm").length;

    const pick = await vscode.window.showQuickPick(
      [
        {
          label: "Codex",
          description: `${codexCount} importable session${codexCount === 1 ? "" : "s"}`,
          value: "codex" as ImportProvider,
        },
        {
          label: "Claude",
          description: `${claudeCount} importable session${claudeCount === 1 ? "" : "s"}`,
          value: "claude" as ImportProvider,
        },
        {
          label: "PRSM",
          description: `${prsmCount} importable session${prsmCount === 1 ? "" : "s"}`,
          value: "prsm" as ImportProvider,
        },
      ],
      {
        title: "Import Transcript",
        placeHolder: "Select provider",
      }
    );
    return pick?.value;
  }

  private async promptImportSession(
    provider: ImportProvider,
    sessions: ImportSessionListEntry[]
  ): Promise<ImportSessionListEntry | undefined> {
    const picks = sessions.map((session) => {
      const title = (session.title || "").trim() || "(untitled)";
      const updated = this.formatImportTimestamp(session.updated_at || session.started_at);
      return {
        label: title,
        description: `${session.turn_count} turns • updated ${updated}`,
        detail: `${session.provider}:${session.source_id}`,
        value: session,
      };
    });

    const pick = await vscode.window.showQuickPick(picks, {
      title: "Import Transcript",
      placeHolder: `Select ${provider} session (sorted by recency)`,
      matchOnDescription: true,
      matchOnDetail: true,
    });
    return pick?.value;
  }

  private async runInteractiveImport(
    sessionId: string,
    panel: vscode.WebviewPanel,
    preferredProvider?: ImportProvider
  ): Promise<void> {
    const transport = this.getTransport();
    if (!transport?.isConnected) {
      panel.webview.postMessage({
        type: "errorMessage",
        content: "Not connected to PRSM server.",
      });
      return;
    }

    let provider = preferredProvider;
    if (!provider) {
      try {
        provider = await this.promptImportProvider(transport);
      } catch (err) {
        panel.webview.postMessage({
          type: "errorMessage",
          content: `Import provider selection failed: ${(err as Error).message}`,
        });
        return;
      }
      if (!provider) {
        this.postSystemMessage(panel, "Import canceled.");
        return;
      }
    }

    let sessions: ImportSessionListEntry[];
    try {
      const response = await transport.listImportSessions(provider, 200);
      sessions = this.sortImportSessionsByRecency(
        response.sessions as ImportSessionListEntry[]
      );
    } catch (err) {
      panel.webview.postMessage({
        type: "errorMessage",
        content: `Import session lookup failed: ${(err as Error).message}`,
      });
      return;
    }

    if (sessions.length === 0) {
      this.postSystemMessage(panel, `No importable ${provider} sessions found.`);
      return;
    }

    // Build QuickPick items with "Import ALL" as the first option
    const importAllItem = {
      label: "$(cloud-download) Import ALL Sessions",
      description: `Import all ${sessions.length} ${provider} session${sessions.length === 1 ? "" : "s"}`,
      detail: "Bulk-import every session from this provider",
      value: null as ImportSessionListEntry | null,
    };
    const sessionItems = sessions.map((session) => {
      const title = (session.title || "").trim() || "(untitled)";
      const updated = this.formatImportTimestamp(session.updated_at || session.started_at);
      return {
        label: title,
        description: `${session.turn_count} turns • updated ${updated}`,
        detail: `${session.provider}:${session.source_id}`,
        value: session as ImportSessionListEntry | null,
      };
    });
    const allItems = [importAllItem, ...sessionItems];

    const pick = await vscode.window.showQuickPick(allItems, {
      title: "Import Transcript",
      placeHolder: `Select ${provider} session (sorted by recency)`,
      matchOnDescription: true,
      matchOnDetail: true,
    });
    if (!pick) {
      this.postSystemMessage(panel, "Import canceled.");
      return;
    }

    // Handle "Import ALL" selection
    if (pick.value === null) {
      const maxTurns = await this.promptImportDepth();
      if (maxTurns === undefined) {
        this.postSystemMessage(panel, "Import canceled.");
        return;
      }
      try {
        const response = await transport.runImportAll(
          sessionId,
          provider,
          { maxTurns: maxTurns === null ? null : maxTurns }
        );
        this.postSystemMessage(
          panel,
          `${response.message}\nImported ${response.imported_count} session${response.imported_count === 1 ? "" : "s"}.`
        );
      } catch (err) {
        panel.webview.postMessage({
          type: "errorMessage",
          content: `Import all failed: ${(err as Error).message}`,
        });
      }
      return;
    }

    const selected = pick.value;

    const maxTurns = await this.promptImportDepth();
    if (maxTurns === undefined) {
      this.postSystemMessage(panel, "Import canceled.");
      return;
    }

    try {
      const sessionName = (selected.title || "").trim() || undefined;
      const response = await transport.runImport(
        sessionId,
        provider,
        selected.source_id,
        {
          sessionName,
          maxTurns: maxTurns === null ? null : maxTurns,
        }
      );
      this.applyImportedSessionSnapshot(response.session);
      this.sendSessionState(sessionId);
      const warningLine = response.warnings?.length
        ? `\nWarnings: ${response.warnings.length}`
        : "";
      this.postSystemMessage(panel, `${response.message}${warningLine}`);
    } catch (err) {
      panel.webview.postMessage({
        type: "errorMessage",
        content: `Import run failed: ${(err as Error).message}`,
      });
    }
  }

  private async handleSlashCommand(
    sessionId: string,
    panel: vscode.WebviewPanel,
    name: string,
    args: string[]
  ): Promise<void> {
    if (name === "help") {
      this.postSystemMessage(panel, this.slashHelpText());
      return;
    }
    if (name === "import") {
      await this.handleImportSlashCommand(sessionId, panel, args);
      return;
    }
    await this.showSlashCommandPicker(sessionId, panel, `/${name}`);
  }

  private async handleImportSlashCommand(
    sessionId: string,
    panel: vscode.WebviewPanel,
    args: string[]
  ): Promise<void> {
    const transport = this.getTransport();
    if (!transport?.isConnected) {
      panel.webview.postMessage({
        type: "errorMessage",
        content: "Not connected to PRSM server.",
      });
      return;
    }

    const action = (args[0] || "").toLowerCase();
    if (!action) {
      await this.runInteractiveImport(sessionId, panel);
      return;
    }

    if (action === "list") {
      const provider = (args[1] || "all").toLowerCase();
      if (!["all", "codex", "claude", "prsm"].includes(provider)) {
        this.postSystemMessage(panel, `Invalid provider: ${provider}`);
        return;
      }
      try {
        const resp = await transport.listImportSessions(
          provider as "all" | "codex" | "claude" | "prsm",
          25
        );
        if (!resp.sessions || resp.sessions.length === 0) {
          this.postSystemMessage(panel, "No importable sessions found.");
          return;
        }
        const lines = [`Importable sessions (${resp.sessions.length}):`];
        for (const s of resp.sessions) {
          lines.push(
            `- ${s.provider}:${s.source_id} -- ${s.title || "(untitled)"} `
            + `(${s.turn_count} turns, updated ${s.updated_at || "unknown"})`
          );
        }
        this.postSystemMessage(panel, lines.join("\n"));
      } catch (err) {
        panel.webview.postMessage({
          type: "errorMessage",
          content: `Import list failed: ${(err as Error).message}`,
        });
      }
      return;
    }

    if (action === "preview") {
      if (args.length < 3) {
        this.postSystemMessage(panel, "Usage: /import preview PROVIDER SOURCE_ID");
        return;
      }
      const provider = args[1].toLowerCase();
      const sourceId = args[2];
      if (!["codex", "claude", "prsm"].includes(provider)) {
        this.postSystemMessage(panel, "Provider must be codex, claude, or prsm.");
        return;
      }
      try {
        const resp = await transport.previewImportSession(
          provider as "codex" | "claude" | "prsm",
          sourceId
        );
        const lines = [
          `Preview: ${resp.summary.provider}:${resp.summary.source_id}`,
          `title: ${resp.summary.title || "(untitled)"}`,
          `turns: ${resp.summary.turn_count}`,
        ];
        if (resp.summary.updated_at) {
          lines.push(`updated: ${resp.summary.updated_at}`);
        }
        for (const turn of resp.preview_turns) {
          const content = (turn.content || "").replace(/\s+/g, " ").trim();
          const short = content.length > 120 ? `${content.slice(0, 117)}...` : content;
          const toolSuffix = turn.tool_call_count ? ` +${turn.tool_call_count} tools` : "";
          lines.push(`- (${turn.role}) ${short}${toolSuffix}`);
        }
        if (resp.warnings?.length) {
          lines.push(`Warnings: ${resp.warnings.length}`);
        }
        this.postSystemMessage(panel, lines.join("\n"));
      } catch (err) {
        panel.webview.postMessage({
          type: "errorMessage",
          content: `Import preview failed: ${(err as Error).message}`,
        });
      }
      return;
    }

    if (action === "run") {
      if (args.length === 1) {
        await this.runInteractiveImport(sessionId, panel);
        return;
      }
      if (args.length === 2) {
        const selectedProvider = this.parseImportProvider(args[1]);
        if (selectedProvider) {
          await this.runInteractiveImport(sessionId, panel, selectedProvider);
          return;
        }
      }

      const parsed = this.parseImportRunArgs(args.slice(1));
      if (parsed.error) {
        this.postSystemMessage(
          panel,
          `${parsed.error}\nUsage: /import run PROVIDER SOURCE_ID [SESSION NAME] [--max-turns N]`
        );
        return;
      }
      if (parsed.cleaned.length < 2) {
        this.postSystemMessage(
          panel,
          "Usage: /import run PROVIDER SOURCE_ID [SESSION NAME] [--max-turns N]"
        );
        return;
      }

      const provider = parsed.cleaned[0].toLowerCase();
      const sourceId = parsed.cleaned[1];
      if (!["codex", "claude", "prsm"].includes(provider)) {
        this.postSystemMessage(panel, "Provider must be codex, claude, or prsm.");
        return;
      }
      const sessionName = parsed.cleaned.slice(2).join(" ").trim() || undefined;
      let maxTurns = parsed.maxTurns;
      if (maxTurns === undefined) {
        const selected = await this.promptImportDepth();
        if (selected === undefined) {
          this.postSystemMessage(panel, "Import canceled.");
          return;
        }
        maxTurns = selected;
      }

      try {
        const resp = await transport.runImport(
          sessionId,
          provider as "codex" | "claude" | "prsm",
          sourceId,
          {
            sessionName,
            maxTurns: maxTurns === null ? null : maxTurns,
          }
        );
        this.applyImportedSessionSnapshot(resp.session);
        this.sendSessionState(sessionId);
        const warningLine = resp.warnings?.length
          ? `\nWarnings: ${resp.warnings.length}`
          : "";
        this.postSystemMessage(panel, `${resp.message}${warningLine}`);
      } catch (err) {
        panel.webview.postMessage({
          type: "errorMessage",
          content: `Import run failed: ${(err as Error).message}`,
        });
      }
      return;
    }

    this.postSystemMessage(
      panel,
      "Usage: /import list [provider] | /import preview PROVIDER SOURCE_ID | /import run PROVIDER SOURCE_ID [NAME] [--max-turns N]"
    );
  }

  private parseImportRunArgs(
    args: string[]
  ): { cleaned: string[]; maxTurns: number | null | undefined; error?: string } {
    const cleaned: string[] = [];
    let maxTurns: number | null | undefined;
    let i = 0;
    while (i < args.length) {
      const token = args[i];
      if (token === "--max-turns") {
        if (i + 1 >= args.length) {
          return { cleaned: [], maxTurns: undefined, error: "Missing value for --max-turns" };
        }
        const parsed = Number.parseInt(args[i + 1], 10);
        if (!Number.isFinite(parsed) || parsed <= 0) {
          return { cleaned: [], maxTurns: undefined, error: "--max-turns must be a positive integer" };
        }
        maxTurns = parsed;
        i += 2;
        continue;
      }
      if (token.startsWith("--max-turns=")) {
        const value = token.split("=", 2)[1];
        const parsed = Number.parseInt(value, 10);
        if (!Number.isFinite(parsed) || parsed <= 0) {
          return { cleaned: [], maxTurns: undefined, error: "--max-turns must be a positive integer" };
        }
        maxTurns = parsed;
        i += 1;
        continue;
      }
      cleaned.push(token);
      i += 1;
    }
    return { cleaned, maxTurns };
  }

  private async promptImportDepth(): Promise<number | null | undefined> {
    const pick = await vscode.window.showQuickPick(
      [
        { label: "Recent 200 (Recommended)", value: 200 as number | null },
        { label: "Recent 500", value: 500 as number | null },
        { label: "Full Transcript", value: null as number | null },
      ],
      {
        placeHolder: "Choose transcript import depth",
      }
    );
    if (!pick) return undefined;
    return pick.value;
  }

  private applyImportedSessionSnapshot(snapshot: {
    session_id: string;
    name: string;
    summary: string | null;
    forked_from: string | null;
    running: boolean;
    created_at: string | null;
    worktree: { branch: string | null; worktreePath: string | null; isWorktree: boolean } | null;
    current_model?: string;
    current_model_display?: string;
    agents: Array<Record<string, unknown>>;
    messages: Record<string, Array<Record<string, unknown>>>;
  }): void {
    const messagesMap = new Map<string, Array<any>>();
    for (const [agentId, msgs] of Object.entries(snapshot.messages || {})) {
      messagesMap.set(agentId, msgs as Array<any>);
    }

    this.store.restoreSession({
      id: snapshot.session_id,
      name: snapshot.name,
      summary: snapshot.summary ?? null,
      forkedFrom: snapshot.forked_from ?? null,
      running: snapshot.running,
      worktree: snapshot.worktree ?? null,
      createdAt: snapshot.created_at,
      currentModel: snapshot.current_model,
      currentModelDisplay: snapshot.current_model_display,
      agents: snapshot.agents as Array<{
        id: string;
        name: string;
        state: string;
        role: string;
        model: string;
        parent_id: string | null;
        children_ids: string[];
        prompt_preview: string;
        created_at?: string | null;
        completed_at?: string | null;
        last_active?: string | null;
      }>,
      messages: messagesMap,
    });
  }

  private async showDeliveryModePicker(): Promise<
    "interrupt" | "inject" | "queue" | undefined
  > {
    const pick = await vscode.window.showQuickPick(
      [
        {
          label: "$(debug-stop) Interrupt",
          description: "Cancel current task, replace with this prompt",
          value: "interrupt" as const,
        },
        {
          label: "$(debug-step-over) Inject After Tool Call",
          description: "Finish current tool call, then process this prompt",
          value: "inject" as const,
        },
        {
          label: "$(list-ordered) Queue After Task",
          description: "Run this prompt after current task completes",
          value: "queue" as const,
        },
      ],
      {
        placeHolder: "Agent is busy. How should this prompt be delivered?",
      }
    );
    return pick?.value;
  }

  // ── File completion ──

  private async handleFileComplete(
    panel: vscode.WebviewPanel,
    prefix: string,
    limit: number,
    requestId?: number,
  ): Promise<void> {
    const transport = this.getTransport();
    if (!transport?.isConnected) {
      console.warn("[prsm] File completion skipped: transport not connected");
      panel.webview.postMessage({
        type: "fileCompleteResult",
        completions: [],
        requestId,
      });
      return;
    }

    try {
      const result = await transport.fileComplete(prefix, limit);
      panel.webview.postMessage({
        type: "fileCompleteResult",
        completions: result.completions ?? [],
        requestId,
      });
    } catch (err) {
      console.error("[prsm] File completion error:", err);
      panel.webview.postMessage({
        type: "fileCompleteResult",
        completions: [],
        requestId,
      });
    }
  }

  /** Open a file in the VS Code editor when clicked in the webview. */
  private handleOpenFile(filePath: string): void {
    const uri = vscode.Uri.file(filePath);
    vscode.commands.executeCommand("vscode.open", uri).then(
      undefined,
      (err) => console.error("[prsm] Failed to open file:", filePath, err),
    );
  }

  /** Open model selection and switch the session model. */
  async handleModelSelection(sessionId: string): Promise<void> {
    const transport = this.getTransport();
    if (!transport?.isConnected) {
      vscode.window.showErrorMessage("Not connected to PRSM server.");
      return;
    }

      try {
        // Get available models
        const response = await transport.getAvailableModels(sessionId);
        const models = response.models;
        let runtimeAliases: Record<
          string,
          {
            model_id?: string;
            provider?: string;
            reasoning_effort?: string;
          }
        > = {};
        try {
          const configResponse = await transport.getConfig();
          runtimeAliases = (configResponse.runtime as {
            model_aliases?: Record<
              string,
              {
                model_id?: string;
                provider?: string;
                reasoning_effort?: string;
              }
            >;
          })?.model_aliases ?? {};
        } catch {
          runtimeAliases = {};
        }

        if (!models || models.length === 0) {
          vscode.window.showWarningMessage("No models available.");
          return;
        }

      // Group models by provider for better UX
      type AvailableModel = (typeof models)[number];
      interface QuickPickItem extends vscode.QuickPickItem {
        modelId?: string;
        legacyProvider?: string;
      }

      const providerGroups = new Map<string, typeof models>();

      // Group by provider
      for (const model of models) {
        if (!providerGroups.has(model.provider)) {
          providerGroups.set(model.provider, []);
        }
        providerGroups.get(model.provider)!.push(model);
      }

      const capitalizeProvider = (provider: string): string => {
        const trimmed = provider.trim().toLowerCase();
        if (!trimmed) return provider;
        return trimmed.charAt(0).toUpperCase() + trimmed.slice(1);
      };

      const normalizeProviderMenuLabel = (provider: string): string => {
        if (provider === "claude") return "Claude";
        if (provider === "codex") return "Codex";
        if (provider === "gemini") return "Gemini";
        return capitalizeProvider(provider);
      };

      const providerOrder = ["claude", "codex", "gemini", "prsm", "minimax"];
      const codeXReasoningLevels = ["low", "medium", "high"] as const;

      const normalizeModelId = (modelId: string): string => (
        modelId
          .toLowerCase()
          .replace(/_/g, "-")
          .split("::reasoning_effort=")[0]
          .replace(/\./g, "-")
          .trim()
      );

      const normalizeModelIdNoDate = (modelId: string): string => {
        const normalized = normalizeModelId(modelId);
        const tokens = normalized.split("-").filter(Boolean);
        if (tokens.length > 1 && /^\d{6,8}$/.test(tokens[tokens.length - 1] ?? "")) {
          tokens.pop();
        }
        return tokens.join("-");
      };

      const buildRuntimeAliasLookup = (): Map<string, string> => {
        const entries = new Map<string, string>();
        const isUsefulAlias = (alias: string): boolean => alias.trim().length > 0;
        for (const [alias, cfg] of Object.entries(runtimeAliases)) {
          if (!isUsefulAlias(alias) || !cfg || typeof cfg !== "object") continue;
          const modelId = String((cfg as { model_id?: string }).model_id || alias || "");
          const normalizedModelId = normalizeModelIdNoDate(modelId);
          if (!normalizedModelId) continue;
          if (!entries.has(normalizedModelId) || alias.length > entries.get(normalizedModelId)!.length) {
            entries.set(normalizedModelId, alias);
          }
        }
        return entries;
      };
      const runtimeAliasLookup = buildRuntimeAliasLookup();

      const resolveModelDisplayAlias = (modelId: string): string | undefined => {
        const baseModelId = modelId.split("::reasoning_effort=")[0];
        const normalized = normalizeModelIdNoDate(baseModelId);
        if (normalized) {
          const runtimeAlias = runtimeAliasLookup.get(normalized);
          if (runtimeAlias) return runtimeAlias;
        }
        return toModelAliasLabel(baseModelId);
      };

      const parseClaudeFamilyVersion = (
        modelId: string,
      ): { family: string; version: string } | null => {
        const normalized = normalizeModelId(modelId);
        const modelMatch = /^claude-(opus|sonnet|haiku)-(.+)$/i.exec(normalized);
        if (!modelMatch) return null;
        const family = modelMatch[1]?.toLowerCase();
        const rawVersion = modelMatch[2];
        if (!family || !rawVersion) return null;
        const parts = rawVersion.split("-").filter(Boolean);
        if (parts.length === 0) return null;
        if (parts.length > 1 && /^\d{6,8}$/.test(parts[parts.length - 1])) {
          parts.pop();
        }
        return { family, version: parts.join("-") };
      };

      const compareVersionParts = (a: string, b: string): number => {
        const aParts = a.split("-").filter(Boolean);
        const bParts = b.split("-").filter(Boolean);
        const maxLen = Math.max(aParts.length, bParts.length);
        for (let i = 0; i < maxLen; i++) {
          const aPart = aParts[i] ?? "";
          const bPart = bParts[i] ?? "";
          if (!aPart && bPart) return -1;
          if (aPart && !bPart) return 1;

          const aNum = /^\d+$/.test(aPart) ? Number(aPart) : NaN;
          const bNum = /^\d+$/.test(bPart) ? Number(bPart) : NaN;
          if (!Number.isNaN(aNum) && !Number.isNaN(bNum)) {
            if (aNum !== bNum) return aNum - bNum;
            continue;
          }

          if (aPart !== bPart) {
            return aPart.localeCompare(bPart);
          }
        }
        return 0;
      };

      const claudeLatestByFamily = new Map<string, string>();
      for (const claudeModel of providerGroups.get("claude") || []) {
        const parsed = parseClaudeFamilyVersion(claudeModel.model_id);
        if (!parsed) continue;
        const currentLatest = claudeLatestByFamily.get(parsed.family);
        if (!currentLatest) {
          claudeLatestByFamily.set(parsed.family, claudeModel.model_id);
          continue;
        }

        const currentParsed = parseClaudeFamilyVersion(currentLatest);
        if (!currentParsed) {
          claudeLatestByFamily.set(parsed.family, claudeModel.model_id);
          continue;
        }
        if (compareVersionParts(parsed.version, currentParsed.version) > 0) {
          claudeLatestByFamily.set(parsed.family, claudeModel.model_id);
        }
      }

      const isPrimaryCodexModel = (modelId: string): boolean => {
        const normalized = normalizeModelId(modelId);
        return /^(?:gpt-5-3-codex|gpt-5-3(?:-codex)?-spark)$/.test(normalized);
      };

      const isPrimaryGeminiModel = (modelId: string): boolean => {
        const normalized = normalizeModelId(modelId);
        return /^gemini-3(?:-|$)/.test(normalized);
      };

      const parseModelBaseId = (modelId: string): string => (
        modelId.split("::reasoning_effort=")[0].toLowerCase()
      );

      const hasTrailingDateSuffix = (modelId: string): boolean => {
        const base = parseModelBaseId(modelId);
        const tokens = base.split("-").filter(Boolean);
        return tokens.length > 1 && /^\d{6,8}$/.test(tokens[tokens.length - 1] ?? "");
      };

      const normalizeForDedupe = (modelId: string): string => {
        const tokens = parseModelBaseId(modelId)
          .split("-")
          .filter(Boolean);
        if (tokens.length > 1 && /^\d{6,8}$/.test(tokens[tokens.length - 1] ?? "")) {
          tokens.pop();
        }
        return tokens.join("-");
      };

      const extractDedupVersion = (modelId: string): string => {
        const tokens = normalizeForDedupe(modelId).split("-").filter(Boolean);
        const versionStart = tokens.findIndex((part) => /^\d/.test(part));
        if (versionStart === -1) {
          return "";
        }
        return tokens.slice(versionStart).join("-");
      };

      const isPreferredDuplicateModel = (candidate: AvailableModel, current: AvailableModel): boolean => {
        const candidateHasDate = hasTrailingDateSuffix(candidate.model_id);
        const currentHasDate = hasTrailingDateSuffix(current.model_id);
        if (candidateHasDate !== currentHasDate) {
          return !candidateHasDate;
        }

        const candidateVersion = extractDedupVersion(candidate.model_id);
        const currentVersion = extractDedupVersion(current.model_id);
        if (candidateVersion !== currentVersion) {
          const compare = compareVersionParts(candidateVersion, currentVersion);
          if (compare !== 0) {
            return compare > 0;
          }
        }
        return candidate.model_id.length < current.model_id.length;
      };

      const dedupeByDisplayLabel = (models: AvailableModel[]): AvailableModel[] => {
        const seen = new Map<string, AvailableModel>();
        for (const model of models) {
          const labelKey = `${model.provider}|${formatModelLabel(model).toLowerCase()}`;
          const existing = seen.get(labelKey);
          if (!existing || isPreferredDuplicateModel(model, existing)) {
            seen.set(labelKey, model);
          }
        }
        return Array.from(seen.values());
      };

      const isPrimaryModelForProvider = (
        provider: string,
        model: AvailableModel,
      ): boolean => {
        if (provider === "codex") {
          return isPrimaryCodexModel(model.model_id);
        }
        if (provider === "gemini") {
          return isPrimaryGeminiModel(model.model_id);
        }
        if (provider === "claude") {
          const parsed = parseClaudeFamilyVersion(model.model_id);
          if (!parsed) return !model.is_legacy;
          return claudeLatestByFamily.get(parsed.family) === model.model_id;
        }
        return !model.is_legacy;
      };

      const modelDisplayBase = (model: AvailableModel): string => {
        const normalizeDisplayLabel = (value: string): string => {
          return value
            .replace(/_/g, " ")
            .trim()
            .split(/[-\s]+/)
            .filter(Boolean)
            .map((part) => {
              if (/^\d/.test(part)) return part;
              const lower = part.toLowerCase();
              if (lower === "gpt") return "GPT";
              if (lower === "claude") return "Claude";
              if (lower === "gemini") return "Gemini";
              if (lower === "codex") return "Codex";
              return part.charAt(0).toUpperCase() + part.slice(1);
            })
            .join(" ");
        };

        const formatCodexModelLabel = (modelId: string): string | undefined => {
          const modelParts = modelId.split("::reasoning_effort=");
          const baseModelId = modelParts[0] ?? modelId;
          const effort = modelParts[1]?.trim();
          const normalized = normalizeModelIdNoDate(baseModelId);
          const isSpark = /^gpt-(?:[0-9]+(?:[.-][0-9]+)*)-(?:codex-)?spark\\b/.test(normalized);
          const suffix = effort && ["low", "medium", "high"].includes(effort)
            ? ` (${effort})`
            : "";

          if (isSpark) {
            const sparkMatch = /^gpt-([0-9]+(?:[.-][0-9]+)*)(?:-codex)?-spark(?:-|$)/.exec(normalized);
            if (sparkMatch) {
              return `GPT ${sparkMatch[1].replace(/-/g, ".")} Spark${suffix}`;
            }
          }

          const codexMatch = /^gpt-([0-9]+(?:[.-][0-9]+)*)-codex(?:-(.+))?(?:-|$)/.exec(normalized);
          if (codexMatch) {
            const version = codexMatch[1].replace(/-/g, ".");
            const suffixLabel = codexMatch[2]
              ? ` ${codexMatch[2].split("-").map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1)).join(" ")}`
              : "";
            return `GPT ${version}${suffixLabel}${suffix}`;
          }

          const aliasLabel = toModelAliasLabel(baseModelId);
          if (aliasLabel) {
            return `${aliasLabel}${suffix}`;
          }

          return undefined;
        };

        if (model.provider === "codex") {
          const alias = resolveModelDisplayAlias(model.model_id);
          const codexLabel = alias || formatCodexModelLabel(model.model_id);
          if (codexLabel) {
            return normalizeDisplayLabel(codexLabel);
          }
        }

        const aliasLabel = resolveModelDisplayAlias(model.model_id);
        if (aliasLabel) {
          return normalizeDisplayLabel(aliasLabel);
        }

        const cleaned = model.model_id
          .split("::reasoning_effort=")[0]
          .replace(/_/g, " ")
          .replace(/\b(\d{6,8})\b$/, "");

        return normalizeDisplayLabel(cleaned);
      };

      const formatModelLabel = (model: AvailableModel): string => {
        return modelDisplayBase(model);
      };

      const buildModelPickItem = (
        model: AvailableModel,
        options?: {
          modelId?: string;
          label?: string;
          isCurrent?: boolean;
        },
      ): QuickPickItem => {
        const tierBadge =
          model.tier === "frontier" ? "🔥" :
          model.tier === "strong" ? "💪" :
          model.tier === "fast" ? "⚡" :
          model.tier === "economy" ? "💰" : "";
        const isCurrent = options?.isCurrent ?? model.is_current;
        const selectedModelId = options?.modelId ?? model.model_id;
        const currentMark = isCurrent ? "$(check) " : "";
        const availableMark = model.available ? "" : " $(warning) unavailable";
        const label = options?.label ?? formatModelLabel(model);
        return {
          label: `${currentMark}${tierBadge} ${label}`,
          description: `${model.tier}${availableMark}`,
          modelId: selectedModelId,
        };
      };

      const buildCodexReasoningItems = (model: AvailableModel): QuickPickItem[] => (
        codeXReasoningLevels.map((effort) => buildModelPickItem(model, {
          modelId: `${model.model_id}::reasoning_effort=${effort}`,
          label: `${formatModelLabel(model)} (${effort})`,
        }))
      );

      const openLegacyModels = async (
        provider: string,
        legacyModels: AvailableModel[]
      ): Promise<QuickPickItem | undefined> => {
        const legacyItems = legacyModels.map((model) => buildModelPickItem(model));
        const providerLabel = normalizeProviderMenuLabel(provider);
        const selection = await vscode.window.showQuickPick(legacyItems, {
          placeHolder: `Select a legacy ${providerLabel} model`,
          matchOnDescription: true,
        });
        return selection;
      };

      for (const [provider, providerModels] of providerGroups.entries()) {
        providerGroups.set(provider, dedupeByDisplayLabel(providerModels));
      }

      const openModelPicker = async (): Promise<QuickPickItem | undefined> => {
        const modelItems: QuickPickItem[] = [];
        const orderedProviders = Array.from(providerGroups.entries()).sort(([providerA], [providerB]) => {
          const a = providerOrder.indexOf(providerA);
          const b = providerOrder.indexOf(providerB);
          if (a === -1 && b === -1) return providerA.localeCompare(providerB);
          if (a === -1) return 1;
          if (b === -1) return -1;
          return a - b;
        });

        for (const [provider, providerModels] of orderedProviders) {
          const dedupedProviderModels = dedupeByDisplayLabel(providerModels);
          const providerLabel = normalizeProviderMenuLabel(provider);
          modelItems.push({
            label: providerLabel.toUpperCase(),
            kind: vscode.QuickPickItemKind.Separator,
          });

          let primaryModels = dedupedProviderModels.filter((model) => (
            isPrimaryModelForProvider(provider, model)
          ));
          let legacyModels = dedupedProviderModels.filter((model) => !isPrimaryModelForProvider(provider, model));

          if (primaryModels.length === 0 && providerModels.length > 0) {
            primaryModels = providerModels;
            legacyModels = [];
          }

          for (const model of primaryModels) {
            if (provider === "codex" && isPrimaryCodexModel(model.model_id)) {
              const codingItems = buildCodexReasoningItems(model);
              modelItems.push(...codingItems);
              continue;
            }
            modelItems.push(buildModelPickItem(model));
          }

          if (legacyModels.length > 0) {
            modelItems.push({
              label: `More ${providerLabel} models...`,
              legacyProvider: provider,
            });
          }
        }

        return vscode.window.showQuickPick(modelItems, {
          placeHolder: "Select a model for this session",
          matchOnDescription: true,
        });
      };

      let selected: QuickPickItem | undefined;
      while (true) {
        selected = await openModelPicker();
        if (!selected) {
          return;
        }

        if (selected.modelId) {
          break;
        }

        if (!selected.legacyProvider) {
          return;
        }

        const legacyProvider = selected.legacyProvider;
        const providerModels = providerGroups.get(legacyProvider) || [];
        const legacyModels = providerModels.filter((model) => (
          !isPrimaryModelForProvider(legacyProvider, model)
        ));
        const legacySelection = await openLegacyModels(legacyProvider, legacyModels);
        if (!legacySelection) {
          continue;
        }
        selected = legacySelection;
        break;
      }

      if (!selected || !selected.modelId) {
        return;
      }

      // Set the model on the server (also broadcasts model_switched SSE event
      // which triggers the visual indicator in the conversation webview)
      const result = await transport.setModel(sessionId, selected.modelId);
      const resolvedId = result.model_id ?? selected.modelId;
      const resolvedDisplay = toModelAliasLabel(resolvedId)
        ?? toModelAliasLabel(selected.modelId)
        ?? resolvedId;

      // Update the store immediately for UI responsiveness
      // (SSE event will also update it, but may arrive slightly later)
      this.store.updateSessionModel(sessionId, resolvedId, resolvedDisplay);

      // Refresh the webview to show the new model in the header/button
      this.sendSessionState(sessionId);
    } catch (err) {
      vscode.window.showErrorMessage(`Failed to change model: ${(err as Error).message}`);
    }
  }

  // ── Queue management ──

  private normalizeMode(
    action: string,
  ): "interrupt" | "inject" | "queue" | undefined {
    if (action === "interrupt") return "interrupt";
    if (action === "inject") return "inject";
    if (action === "queue") return "queue";
    return undefined;
  }

  private async deliverPromptWithMode(
    sessionId: string,
    agentId: string,
    panel: vscode.WebviewPanel,
    prompt: string,
    mode: "interrupt" | "inject" | "queue",
  ): Promise<void> {
    const transport = this.getTransport();
    if (!transport?.isConnected) {
      panel.webview.postMessage({
        type: "errorMessage",
        content: "Not connected to PRSM server.",
      });
      return;
    }

    if (!agentId) {
      panel.webview.postMessage({
        type: "errorMessage",
        content: "No target agent available for this action.",
      });
      return;
    }

    panel.webview.postMessage({ type: "userMessage", content: prompt });

    try {
      await transport.injectPrompt(sessionId, agentId, prompt, mode);
      if (mode === "interrupt") {
        panel.webview.postMessage({ type: "setBusyState", busy: true });
      }
    } catch (err) {
      panel.webview.postMessage({
        type: "errorMessage",
        content: `Failed: ${(err as Error).message}`,
      });
    }
  }

  private async handleAgentPromptAction(
    sessionId: string,
    agentId: string,
    panel: vscode.WebviewPanel,
    prompt: string,
    action: string,
  ): Promise<void> {
    const mode = this.normalizeMode(action);
    if (!mode) {
      panel.webview.postMessage({
        type: "errorMessage",
        content: `Unknown prompt action: ${action}`,
      });
      return;
    }
    await this.deliverPromptWithMode(
      sessionId,
      agentId,
      panel,
      prompt,
      mode,
    );
  }

  private addToQueue(
    sessionId: string,
    prompt: string,
    mode: "queue" | "inject",
  ): void {
    let queue = this.messageQueues.get(sessionId);
    if (!queue) {
      queue = [];
      this.messageQueues.set(sessionId, queue);
    }
    queue.push({
      id: crypto.randomUUID(),
      prompt,
      mode,
    });
  }

  private async handleWebviewAction(
    sessionId: string,
    panel: vscode.WebviewPanel,
    msg: { prompt: string; action: string },
  ): Promise<void> {
    const mode = this.normalizeMode(msg.action);
    if (!mode) {
      panel.webview.postMessage({
        type: "errorMessage",
        content: `Unknown prompt action: ${msg.action}`,
      });
      return;
    }
    if (mode === "queue") {
      this.addToQueue(sessionId, msg.prompt, "queue");
      this.broadcastQueueState(sessionId);
      return;
    }
    const master = this.store.getMasterAgent(sessionId);
    await this.deliverPromptWithMode(
      sessionId,
      master?.id ?? "",
      panel,
      msg.prompt,
      mode,
    );
  }

  private setBusyForSession(sessionId: string, busy: boolean): void {
    const sessionPanel = this.sessionPanels.get(sessionId);
    if (sessionPanel) {
      sessionPanel.webview.postMessage({ type: "setBusyState", busy });
    }
    for (const [key, agentPanel] of this.agentPanels.entries()) {
      if (key.startsWith(`${sessionId}:`)) {
        agentPanel.webview.postMessage({ type: "setBusyState", busy });
      }
    }
  }

  private async handleStopRun(
    sessionId: string,
    panel: vscode.WebviewPanel,
  ): Promise<void> {
    const transport = this.getTransport();
    if (!transport?.isConnected) {
      panel.webview.postMessage({
        type: "errorMessage",
        content: "Not connected to PRSM server.",
      });
      return;
    }

    try {
      await transport.cancelLatestToolCall(sessionId);
      // Force immediate local UI stop even if SSE shutdown events arrive late.
      this.store.forceSessionStopped(sessionId);
      this.setBusyForSession(sessionId, false);
      this.sendSessionState(sessionId);
    } catch (err) {
      panel.webview.postMessage({
        type: "errorMessage",
        content: `Failed to stop run: ${(err as Error).message}`,
      });
      this.setBusyForSession(sessionId, false);
      this.sendSessionState(sessionId);
    }
  }

  private async confirmAndStopRun(
    sessionId: string,
    panel: vscode.WebviewPanel,
  ): Promise<void> {
    const selection = await vscode.window.showWarningMessage(
      "Stop the current run? This will halt active agent execution and keep the chat in its current state.",
      { modal: true },
      "Stop Run",
    );
    if (selection !== "Stop Run") return;
    await this.handleStopRun(sessionId, panel);
  }

  private handleCancelQueued(
    sessionId: string,
    queueId: string,
  ): void {
    const queue = this.messageQueues.get(sessionId);
    if (queue) {
      const idx = queue.findIndex((q) => q.id === queueId);
      if (idx >= 0) queue.splice(idx, 1);
      this.broadcastQueueState(sessionId);
    }
  }

  private broadcastQueueState(sessionId: string): void {
    const queue = this.messageQueues.get(sessionId) ?? [];
    const payload = {
      type: "queueUpdate",
      items: queue.map((q) => ({ id: q.id, prompt: q.prompt, mode: q.mode })),
    };
    const panel = this.sessionPanels.get(sessionId);
    panel?.webview.postMessage(payload);
    for (const [key, agentPanel] of this.agentPanels.entries()) {
      if (key.startsWith(`${sessionId}:`)) {
        agentPanel.webview.postMessage(payload);
      }
    }
  }

  private async processNextQueuedMessage(sessionId: string): Promise<void> {
    const queue = this.messageQueues.get(sessionId);
    if (!queue || queue.length === 0) return;
    const next = queue[0];
    if (!next) return;
    const transport = this.getTransport();
    if (!transport?.isConnected) return;
    const panel = this.sessionPanels.get(sessionId);
    const session = this.store.getSession(sessionId);
    const master = this.store.getMasterAgent(sessionId);

    // Show message when it starts executing, then remove it from queue.
    panel?.webview.postMessage({ type: "userMessage", content: next.prompt });
    this.setBusyForSession(sessionId, true);

    try {
      if (!this.isSessionEffectivelyRunning(sessionId) && master) {
        await transport.sendAgentMessage(sessionId, master.id, next.prompt);
      } else {
        await transport.runPrompt(sessionId, next.prompt);
      }
      queue.shift();
      this.broadcastQueueState(sessionId);
    } catch (err) {
      panel?.webview.postMessage({
        type: "errorMessage",
        content: `Failed: ${(err as Error).message}`,
      });
      this.setBusyForSession(sessionId, false);
    }
  }

  // ── Question handling ──

  /**
   * Route a question to the correct agent's chat panel (inline card).
   * Never forces a panel reveal (to avoid top-of-screen popup behavior).
   * If the target chat is inactive/not open, show a notification with an
   * explicit "Open Chat" action instead.
   */
  showQuestion(data: UserQuestionData): void {
    this.pendingQuestions.set(data.request_id, data);

    const agentName =
      this.store.getAgent(data.session_id, data.agent_id)?.name ??
      data.agent_name ??
      "Unknown Agent";

    const questionPayload = {
      type: "showQuestion" as const,
      requestId: data.request_id,
      agentName,
      question: data.question,
      options: data.options ?? [],
    };

    const master = this.store.getMasterAgent(data.session_id);
    const isMasterQuestion = !!master && master.id === data.agent_id;

    if (isMasterQuestion) {
      const sessionPanel = this.sessionPanels.get(data.session_id);
      if (sessionPanel) {
        sessionPanel.webview.postMessage(questionPayload);
      }
      if (!this.isChatActive(this.toSessionChatKey(data.session_id))) {
        this.showQuestionNotification(data, isMasterQuestion);
      }
      return;
    }

    // Child-agent question: only post to that specific agent panel if open.
    const agentKey = `${data.session_id}:${data.agent_id}`;
    const agentPanel = this.agentPanels.get(agentKey);
    if (agentPanel) {
      agentPanel.webview.postMessage(questionPayload);
    }
    if (!this.isChatActive(this.toAgentChatKey(data.session_id, data.agent_id))) {
      this.showQuestionNotification(data, isMasterQuestion);
    }
  }

  /**
   * Route a permission request to the inline question form UI.
   * This replaces the modal popup permission dialog.
   */
  showPermissionRequest(data: PermissionRequestData): void {
    this.pendingPermissionRequests.set(data.request_id, data);

    const agentName =
      this.store.getAgent(data.session_id, data.agent_id)?.name ??
      data.agent_name ??
      "Unknown Agent";
    const argPreview = data.arguments?.slice(0, 160) ?? "";
    const question = argPreview
      ? `${agentName} wants to use ${data.tool_name}.\n\nArgs:\n${argPreview}`
      : `${agentName} wants to use ${data.tool_name}.`;

    const questionData: UserQuestionData = {
      session_id: data.session_id,
      agent_id: data.agent_id,
      request_id: data.request_id,
      agent_name: agentName,
      question,
      options: [
        { label: "Allow" },
        { label: "Allow Always (Project)" },
        { label: "Allow Always (Global)" },
        { label: "Always Reject (Project)" },
        { label: "Deny" },
      ],
    };
    this.showQuestion(questionData);
  }

  private showQuestionNotification(
    data: UserQuestionData,
    isMasterQuestion: boolean,
  ): void {
    const agentName =
      this.store.getAgent(data.session_id, data.agent_id)?.name ??
      data.agent_name ??
      "Agent";
    const openLabel = "Open Chat";
    void vscode.window
      .showInformationMessage(`${agentName} is waiting for your input.`, openLabel)
      .then((selection) => {
        if (selection !== openLabel) return;
        if (isMasterQuestion) {
          this.showSession(data.session_id);
        } else {
          this.showAgent(data.session_id, data.agent_id);
        }
      });
  }

  /**
   * Handle a question answer from the webview, resolve it via transport.
   */
  private async handleQuestionAnswer(
    sessionId: string,
    requestId: string,
    answer: string,
  ): Promise<void> {
    const transport = this.getTransport();
    if (!transport?.isConnected) {
      this.outputChannel?.appendLine(
        `[question] Cannot resolve question ${requestId}: transport not connected`
      );
      return;
    }

    if (this.pendingPermissionRequests.has(requestId)) {
      const normalized = answer.trim();

      this.pendingQuestions.delete(requestId);
      this.pendingPermissionRequests.delete(requestId);

      const result =
        normalized === "Allow Always (Project)"
          ? "allow_project"
          : normalized === "Allow Always (Global)"
            ? "allow_global"
            : normalized === "Allow"
              ? "allow"
              : normalized === "Always Reject (Project)"
                ? "deny_project"
                : "deny";

      try {
        await transport.resolvePermission(sessionId, requestId, result);
        this.outputChannel?.appendLine(
          `[permission] Resolved ${requestId} with result=${result}`
        );
        this.dismissQuestionInPanels(sessionId, requestId);
      } catch (err) {
        this.outputChannel?.appendLine(
          `[permission] Failed to resolve ${requestId}: ${(err as Error).message}`
        );
        vscode.window.showErrorMessage(
          `Failed to resolve permission: ${(err as Error).message}`
        );
      }
      return;
    }

    // Remove normal question from pending
    this.pendingQuestions.delete(requestId);

    try {
      await transport.resolveQuestion(sessionId, requestId, answer);
      this.outputChannel?.appendLine(
        `[question] Resolved ${requestId} with answer: "${answer.slice(0, 60)}"`
      );

      // Dismiss the question card in all panels for this session
      // (in case the same question was shown in multiple places)
      this.dismissQuestionInPanels(sessionId, requestId);
    } catch (err) {
      this.outputChannel?.appendLine(
        `[question] Failed to resolve ${requestId}: ${(err as Error).message}`
      );
      vscode.window.showErrorMessage(
        `Failed to resolve question: ${(err as Error).message}`
      );
    }
  }

  /**
   * Send dismissQuestion to all relevant panels for a session.
   */
  private dismissQuestionInPanels(sessionId: string, requestId: string): void {
    const msg = { type: "dismissQuestion", requestId };

    const sessionPanel = this.sessionPanels.get(sessionId);
    if (sessionPanel) {
      sessionPanel.webview.postMessage(msg);
    }

    for (const [key, panel] of this.agentPanels) {
      if (key.startsWith(sessionId + ":")) {
        panel.webview.postMessage(msg);
      }
    }
  }

  private toAgentChatKey(sessionId: string, agentId: string): string {
    return `agent:${sessionId}:${agentId}`;
  }

  private toSessionChatKey(sessionId: string): string {
    return `session:${sessionId}`;
  }

  private isChatActive(chatKey: string): boolean {
    return this.activeChatKey === chatKey;
  }

  private setActiveChat(target: ActiveChatTarget | undefined): void {
    const nextKey =
      target?.kind === "agent"
        ? this.toAgentChatKey(target.sessionId, target.agentId)
        : target?.kind === "session"
          ? this.toSessionChatKey(target.sessionId)
          : undefined;
    if (nextKey === this.activeChatKey) return;
    this.activeChatKey = nextKey;
    this._onDidChangeActiveChat.fire(target);
  }

  private postPendingQuestionsForAgent(
    panel: vscode.WebviewPanel,
    sessionId: string,
    agentId: string,
  ): void {
    for (const q of this.pendingQuestions.values()) {
      if (q.session_id !== sessionId || q.agent_id !== agentId) continue;
      const agentName =
        this.store.getAgent(q.session_id, q.agent_id)?.name ??
        q.agent_name ??
        "Unknown Agent";
      panel.webview.postMessage({
        type: "showQuestion",
        requestId: q.request_id,
        agentName,
        question: q.question,
        options: q.options ?? [],
      });
    }

    for (const req of this.pendingPermissionRequests.values()) {
      if (req.session_id !== sessionId || req.agent_id !== agentId) continue;
      const agentName =
        this.store.getAgent(req.session_id, req.agent_id)?.name ??
        req.agent_name ??
        "Unknown Agent";
      const argPreview = req.arguments?.slice(0, 160) ?? "";
      const question = argPreview
        ? `${agentName} wants to use ${req.tool_name}.\n\nArgs:\n${argPreview}`
        : `${agentName} wants to use ${req.tool_name}.`;
      panel.webview.postMessage({
        type: "showQuestion",
        requestId: req.request_id,
        agentName,
        question,
        options: [
          { label: "Allow" },
          { label: "Allow Always (Project)" },
          { label: "Allow Always (Global)" },
          { label: "Always Reject (Project)" },
          { label: "Deny" },
        ],
      });
    }
  }

  private postPendingQuestionsForSession(
    panel: vscode.WebviewPanel,
    sessionId: string,
  ): void {
    const master = this.store.getMasterAgent(sessionId);
    if (!master) return;
    this.postPendingQuestionsForAgent(panel, sessionId, master.id);
  }

  // ── HTML ──

  private getWebviewHtml(webview: vscode.Webview): string {
    const cacheBust = Date.now().toString(36);
    const scriptUri = webview.asWebviewUri(
      vscode.Uri.joinPath(
        this.context.extensionUri,
        "media",
        "webview",
        "main.js"
      )
    );
    const styleUri = webview.asWebviewUri(
      vscode.Uri.joinPath(
        this.context.extensionUri,
        "media",
        "webview",
        "styles.css"
      )
    );
    const scriptSrc = `${scriptUri.toString()}?v=${cacheBust}`;
    const styleSrc = `${styleUri.toString()}?v=${cacheBust}`;
    const nonce = getNonce();

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none';
             style-src ${webview.cspSource} 'unsafe-inline';
             script-src 'nonce-${nonce}';">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="${styleSrc}">
  <title>Agent Conversation</title>
</head>
<body>
  <div id="debug-bar" style="background:#1a1a2e;color:#888;font-size:10px;padding:2px 8px;font-family:monospace;border-bottom:1px solid #333;display:none;"></div>
  <div id="agent-header"></div>
  <div id="conversation"></div>
  <div id="input-container" class="hidden">
    <div id="input-wrapper">
      <button id="model-selector-btn" class="model-selector-btn" data-tooltip="Select Model">
        <span class="model-name"></span>
      </button>
      <div id="context-usage-badge" class="hidden"></div>
      <textarea id="prompt-input" placeholder="Type a message..." rows="1"></textarea>
      <button id="send-btn" data-tooltip="Send (Enter)">
        <span class="send-icon">&#9654;</span>
      </button>
    </div>
  </div>
  <script nonce="${nonce}" src="${scriptSrc}"></script>
</body>
</html>`;
  }

  dispose(): void {
    for (const interval of this.pollingIntervals.values()) {
      clearInterval(interval);
    }
    this.pollingIntervals.clear();
    for (const panel of this.agentPanels.values()) {
      panel.dispose();
    }
    this.agentPanels.clear();
    for (const panel of this.sessionPanels.values()) {
      panel.dispose();
    }
    this.sessionPanels.clear();
    this._onDidChangeActiveChat.dispose();
  }
}

function getNonce(): string {
  const chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let result = "";
  for (let i = 0; i < 32; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}
