/**
 * Central state management for multiple PRSM sessions.
 * Each session has its own agent tree and message history.
 * Events from the server are processed here and fire VS Code events.
 */
import * as vscode from "vscode";
import {
  AgentNode,
  AgentState,
  AgentRole,
  Message,
  ToolCall,
  SessionInfo,
  AgentSpawnedData,
  AgentStateChangedData,
  AgentKilledData,
  StreamChunkData,
  ToolCallStartedData,
  ToolCallCompletedData,
  ToolCallDeltaData,
  PermissionRequestData,
  UserQuestionData,
  EngineStartedData,
  EngineFinishedData,
  SessionCreatedData,
  SessionRemovedData,
  AgentResultData,
  ThinkingData,
  UserPromptData,
  ContextWindowUsageData,
  FileChangedData,
  SnapshotCreatedData,
  SnapshotRestoredData,
  PlanFileUpdatedData,
  ModelSwitchedData,
} from "../protocol/types";

/** Worktree metadata attached to a session. */
export interface SessionWorktreeInfo {
  /** Branch name the session was created on, or null if detached. */
  branch: string | null;
  /** Absolute path of the worktree the session was created in. */
  worktreePath: string | null;
  /** Whether the session was created in a linked worktree (not main). */
  isWorktree: boolean;
}

/** State for a single session. */
export interface SessionState {
  id: string;
  name: string;
  summary?: string | null;
  forkedFrom: string | null;
  running: boolean;
  agents: Map<string, AgentNode>;
  messages: Map<string, Message[]>;
  streamBuffers: Map<string, string>;
  pendingToolCalls: Map<string, ToolCall>;
  contextUsage: Map<string, ContextWindowUsageData>;
  /** Worktree context at time of session creation. */
  worktree: SessionWorktreeInfo | null;
  /** ISO timestamp of when the session was created. */
  createdAt: string | null;
  /** ISO timestamp of last activity (agent spawn, state change, message, etc.). */
  lastActivity: string | null;
  /** Current model ID for the session. */
  currentModel?: string;
  /** User-facing model label for the session (alias-aware display name). */
  currentModelDisplay?: string;
}

export class SessionStore {
  private static readonly RUN_STOPPED_MARKER = "Run stopped by user.";
  private sessions = new Map<string, SessionState>();

  // Pending user messages queued before master agent exists
  private _pendingUserMessages = new Map<string, Array<{ text: string; snapshotId?: string }>>();

  // Agents currently in thinking state (for re-showing after fullUpdate)
  private _thinkingAgents = new Set<string>();
  // Sessions explicitly stopped by user: ignore stale thinking events
  // until the next engine_started for that session.
  private _thinkingSuppressedSessions = new Set<string>();

  // Optional output channel for diagnostic logging
  private _outputChannel?: vscode.OutputChannel;

  // Worktree info to stamp on new sessions
  private _worktreeInfo: SessionWorktreeInfo | null = null;

  // Whether to filter sessions by current worktree (default: true)
  private _filterByWorktree = true;

  // Collapse state: tracks which tree nodes the user has manually collapsed.
  // Keys are node IDs: session IDs for SessionTreeItems, agent IDs for AgentTreeItems.
  // A node in this set means it was explicitly collapsed by the user.
  // Nodes NOT in this set default to expanded (matching previous behavior).
  private _collapsedNodes = new Set<string>();

  // VS Code workspace state reference for persisting collapse state
  private _workspaceState?: vscode.Memento;

  // Event emitters
  private _onDidChangeTree = new vscode.EventEmitter<void>();
  readonly onDidChangeTree = this._onDidChangeTree.event;

  private _onDidChangeMessages = new vscode.EventEmitter<{
    sessionId: string;
    agentId: string;
  }>();
  readonly onDidChangeMessages = this._onDidChangeMessages.event;

  private _onStreamChunk = new vscode.EventEmitter<StreamChunkData>();
  readonly onStreamChunk = this._onStreamChunk.event;

  private _onPermissionRequest =
    new vscode.EventEmitter<PermissionRequestData>();
  readonly onPermissionRequest = this._onPermissionRequest.event;

  /** Manually emit a permission request (used for re-prompting after viewing agent context). */
  emitPermissionRequest(data: PermissionRequestData): void {
    this._onPermissionRequest.fire(data);
  }

  private _onUserQuestion = new vscode.EventEmitter<UserQuestionData>();
  readonly onUserQuestion = this._onUserQuestion.event;

  private _onEngineStarted = new vscode.EventEmitter<EngineStartedData>();
  readonly onEngineStarted = this._onEngineStarted.event;

  private _onEngineFinished = new vscode.EventEmitter<EngineFinishedData>();
  readonly onEngineFinished = this._onEngineFinished.event;

  private _onThinking = new vscode.EventEmitter<ThinkingData>();
  readonly onThinking = this._onThinking.event;

  private _onUserPrompt = new vscode.EventEmitter<UserPromptData>();
  readonly onUserPrompt = this._onUserPrompt.event;

  private _onSessionRenamed = new vscode.EventEmitter<{
    sessionId: string;
    newName: string;
  }>();
  readonly onSessionRenamed = this._onSessionRenamed.event;

  private _onFileChanged = new vscode.EventEmitter<FileChangedData>();
  readonly onFileChanged = this._onFileChanged.event;

  private _onSnapshotCreated = new vscode.EventEmitter<SnapshotCreatedData>();
  readonly onSnapshotCreated = this._onSnapshotCreated.event;

  private _onSnapshotRestored = new vscode.EventEmitter<SnapshotRestoredData>();
  readonly onSnapshotRestored = this._onSnapshotRestored.event;

  private _onPlanFileUpdated = new vscode.EventEmitter<PlanFileUpdatedData>();
  readonly onPlanFileUpdated = this._onPlanFileUpdated.event;

  private _onModelSwitched = new vscode.EventEmitter<ModelSwitchedData>();
  readonly onModelSwitched = this._onModelSwitched.event;

  /** Set output channel for diagnostic logging. */
  setOutputChannel(ch: vscode.OutputChannel): void {
    this._outputChannel = ch;
  }

  /** Set worktree context to stamp on new sessions. */
  setWorktreeInfo(info: SessionWorktreeInfo | null): void {
    this._worktreeInfo = info;
  }

  /** Get current worktree info. */
  getWorktreeInfo(): SessionWorktreeInfo | null {
    return this._worktreeInfo;
  }

  /** Set whether to filter sessions by the current worktree. */
  setFilterByWorktree(filter: boolean): void {
    this._filterByWorktree = filter;
    this._onDidChangeTree.fire();
  }

  /** Check whether sessions are being filtered by worktree. */
  isFilteringByWorktree(): boolean {
    return this._filterByWorktree;
  }

  // ── Collapse state ──

  /** Bind workspace state for persisting collapse state across restarts. */
  setWorkspaceState(state: vscode.Memento): void {
    this._workspaceState = state;
    // Restore saved collapse state
    const saved = state.get<string[]>("prsm.tree.collapsedNodes");
    if (saved) {
      this._collapsedNodes = new Set(saved);
    }
  }

  /** Mark a node as collapsed (user collapsed it). */
  setCollapsed(nodeId: string): void {
    this._collapsedNodes.add(nodeId);
    this._persistCollapseState();
  }

  /** Mark a node as expanded (user expanded it). */
  setExpanded(nodeId: string): void {
    this._collapsedNodes.delete(nodeId);
    this._persistCollapseState();
  }

  /** Check if a node has been explicitly collapsed by the user. */
  isCollapsed(nodeId: string): boolean {
    return this._collapsedNodes.has(nodeId);
  }

  /** Persist collapse state to workspace state. */
  private _persistCollapseState(): void {
    this._workspaceState?.update(
      "prsm.tree.collapsedNodes",
      Array.from(this._collapsedNodes)
    );
  }

  // ── Session accessors ──

  /**
   * Get sessions, optionally filtered to the current worktree.
   * When filtering is enabled and worktree info is available,
   * only returns sessions that belong to the current worktree
   * (by matching worktreePath).
   */
  getSessions(): SessionState[] {
    const all = Array.from(this.sessions.values());
    if (!this._filterByWorktree || !this._worktreeInfo) {
      return all;
    }
    const currentPath = this._worktreeInfo.worktreePath;
    if (!currentPath) {
      return all;
    }
    return all.filter((s) => {
      // Show sessions without stable worktree path information (legacy/migration
      // data) to avoid hiding everything on startup.
      if (!s.worktree || typeof s.worktree.worktreePath !== "string") {
        return true;
      }
      return s.worktree.worktreePath === currentPath;
    });
  }

  /** Get ALL sessions regardless of worktree filter. */
  getAllSessions(): SessionState[] {
    return Array.from(this.sessions.values());
  }

  getSession(sessionId: string): SessionState | undefined {
    return this.sessions.get(sessionId);
  }

  getAgent(sessionId: string, agentId: string): AgentNode | undefined {
    return this.sessions.get(sessionId)?.agents.get(agentId);
  }

  getRootAgents(sessionId: string): AgentNode[] {
    const session = this.sessions.get(sessionId);
    if (!session) return [];
    return Array.from(session.agents.values()).filter(
      (a) => a.parentId === null || a.parentId === undefined
    );
  }

  getChildren(sessionId: string, agentId: string): AgentNode[] {
    const session = this.sessions.get(sessionId);
    if (!session) return [];
    const agent = session.agents.get(agentId);
    if (!agent) return [];
    return agent.childrenIds
      .map((id) => session.agents.get(id))
      .filter((a): a is AgentNode => a !== undefined);
  }

  /** Get the master (orchestrator) agent for a session, if one exists.
   *  Returns the most recently spawned root orchestrator — important for
   *  follow-up prompts which create a new master agent. */
  getMasterAgent(sessionId: string): AgentNode | undefined {
    const roots = this.getRootAgents(sessionId);
    const orchestrators = roots.filter((a) => a.role === "orchestrator");
    return orchestrators[orchestrators.length - 1] ?? roots[roots.length - 1];
  }

  getMessages(sessionId: string, agentId: string): Message[] {
    return this.sessions.get(sessionId)?.messages.get(agentId) ?? [];
  }

  /** Get combined messages from all root orchestrators in spawn order.
   *  This preserves conversation history across follow-up prompts. */
  getAllMasterMessages(sessionId: string): Message[] {
    const roots = this.getRootAgents(sessionId);
    const orchestrators = roots.filter((a) => a.role === "orchestrator");
    const agents = orchestrators.length > 0 ? orchestrators : roots;
    const combined: Message[] = [];
    for (const agent of agents) {
      const msgs = this.getMessages(sessionId, agent.id);
      combined.push(...msgs);
    }
    return combined;
  }

  getStreamBuffer(sessionId: string, agentId: string): string {
    return (
      this.sessions.get(sessionId)?.streamBuffers.get(agentId) ?? ""
    );
  }

  /** Check if an agent is currently in a thinking state. */
  isAgentThinking(agentId: string): boolean {
    return this._thinkingAgents.has(agentId);
  }

  getContextUsage(
    sessionId: string,
    agentId: string
  ): ContextWindowUsageData | undefined {
    return this.sessions.get(sessionId)?.contextUsage.get(agentId);
  }

  // ── Event processing ──

  processEvent(eventType: string, data: Record<string, unknown>): void {
    try {
      this._processEventInner(eventType, data);
    } catch (err) {
      const msg = `[prsm] Error processing event '${eventType}': ${(err as Error).message} data: ${JSON.stringify(data).slice(0, 500)}`;
      console.error(msg);
      this._outputChannel?.appendLine(msg);
    }
  }

  private _processEventInner(
    eventType: string,
    data: Record<string, unknown>,
  ): void {
    switch (eventType) {
      case "session_created":
        this.handleSessionCreated(data as unknown as SessionCreatedData);
        break;
      case "session_removed":
        this.handleSessionRemoved(data as unknown as SessionRemovedData);
        break;
      case "agent_spawned": {
        const spawnData = data as unknown as AgentSpawnedData;
        this.handleAgentSpawned(spawnData);
        // Agent just spawned — show thinking indicator (matches TUI behavior)
        if (!this._thinkingSuppressedSessions.has(spawnData.session_id)) {
          this._thinkingAgents.add(spawnData.agent_id);
          this._onThinking.fire({
            session_id: spawnData.session_id,
            agent_id: spawnData.agent_id,
            text: "",
          });
        }
        break;
      }
      case "agent_restarted": {
        const restartData = data as unknown as AgentSpawnedData;
        this.handleAgentRestarted(restartData);
        // Agent restarted — show thinking indicator (matches TUI behavior)
        if (!this._thinkingSuppressedSessions.has(restartData.session_id)) {
          this._thinkingAgents.add(restartData.agent_id);
          this._onThinking.fire({
            session_id: restartData.session_id,
            agent_id: restartData.agent_id,
            text: "",
          });
        }
        break;
      }
      case "agent_state_changed": {
        const stateData = data as unknown as AgentStateChangedData;
        this.handleAgentStateChanged(stateData);
        // Clear thinking on terminal states (matches TUI behavior)
        const terminal = ["completed", "error", "failed", "killed"];
        if (terminal.includes(stateData.new_state)) {
          this._thinkingAgents.delete(stateData.agent_id);
        }
        break;
      }
      case "agent_killed": {
        const killData = data as unknown as AgentKilledData;
        this._thinkingAgents.delete(killData.agent_id);
        this.handleAgentKilled(killData);
        break;
      }
      case "stream_chunk":
        this._thinkingAgents.delete(
          (data as unknown as StreamChunkData).agent_id
        );
        this.handleStreamChunk(data as unknown as StreamChunkData);
        break;
      case "tool_call_started":
        this._thinkingAgents.delete(
          (data as unknown as ToolCallStartedData).agent_id
        );
        this.handleToolCallStarted(
          data as unknown as ToolCallStartedData
        );
        break;
      case "tool_call_completed": {
        const tcData = data as unknown as ToolCallCompletedData;
        this.handleToolCallCompleted(tcData);
        // Tool finished — agent resumes thinking (matches TUI behavior)
        if (!this._thinkingSuppressedSessions.has(tcData.session_id)) {
          this._thinkingAgents.add(tcData.agent_id);
          this._onThinking.fire({
            session_id: tcData.session_id,
            agent_id: tcData.agent_id,
            text: "",
          });
        }
        break;
      }
      case "tool_call_delta":
        this.handleToolCallDelta(data as unknown as ToolCallDeltaData);
        break;
      case "permission_request": {
        const permData = data as unknown as PermissionRequestData;
        // Clear thinking — agent is waiting for permission (matches TUI behavior)
        this._thinkingAgents.delete(permData.agent_id);
        this._onPermissionRequest.fire(permData);
        break;
      }
      case "user_question_request": {
        const questionData = data as unknown as UserQuestionData;
        // Clear thinking — agent is waiting for user answer (matches TUI behavior)
        this._thinkingAgents.delete(questionData.agent_id);
        this._onUserQuestion.fire(questionData);
        break;
      }
      case "engine_started":
        this.handleEngineStarted(data as unknown as EngineStartedData);
        break;
      case "engine_finished":
        this.handleEngineFinished(data as unknown as EngineFinishedData);
        break;
      case "agent_result":
        this.handleAgentResult(data as unknown as AgentResultData);
        break;
      case "thinking":
        {
          const thinkingData = data as unknown as ThinkingData;
          if (!this._thinkingSuppressedSessions.has(thinkingData.session_id)) {
            this._thinkingAgents.add(thinkingData.agent_id);
            this._onThinking.fire(thinkingData);
          }
        }
        break;
      case "user_prompt":
        this.handleUserPrompt(data as unknown as UserPromptData);
        break;
      case "context_window_usage":
        this.handleContextWindowUsage(
          data as unknown as ContextWindowUsageData
        );
        break;
      case "file_changed":
        this._onFileChanged.fire(data as unknown as FileChangedData);
        break;
      case "snapshot_created":
        this._onSnapshotCreated.fire(
          data as unknown as SnapshotCreatedData
        );
        break;
      case "snapshot_restored":
        this._onSnapshotRestored.fire(
          data as unknown as SnapshotRestoredData
        );
        break;
      case "plan_file_updated":
        this._onPlanFileUpdated.fire(data as unknown as PlanFileUpdatedData);
        break;
      case "session_renamed":
        this.handleSessionRenamed(data as unknown as { session_id: string; name: string });
        break;
      case "agent_message":
        this.handleAgentMessage(
          data as unknown as {
            session_id: string;
            agent_id: string;
            content: string;
            role: string;
            snapshot_id?: string;
          }
        );
        break;
      case "model_switched":
        this.handleModelSwitched(data as unknown as ModelSwitchedData);
        break;
      case "connected":
        // SSE connection established — populate store from session_list
        // if the server included session metadata in the connected event.
        this.handleConnected(data as Record<string, unknown>);
        break;
      default:
        // Unknown event, ignore for forward compatibility
        break;
    }
  }

  /** Add a session directly (e.g., from initial load). */
  addSession(info: {
    id: string;
    name: string;
    summary?: string | null;
    forkedFrom?: string | null;
    worktree?: SessionWorktreeInfo | null;
    createdAt?: string | null;
    currentModel?: string;
    currentModelDisplay?: string;
  }): void {
    if (this.sessions.has(info.id)) return;
    this.sessions.set(info.id, {
      id: info.id,
      name: info.name,
      summary: info.summary ?? null,
      forkedFrom: info.forkedFrom ?? null,
      running: false,
      agents: new Map(),
      messages: new Map(),
      streamBuffers: new Map(),
      pendingToolCalls: new Map(),
      contextUsage: new Map(),
      worktree: info.worktree ?? this._worktreeInfo,
      createdAt: info.createdAt ?? new Date().toISOString(),
      lastActivity: info.createdAt ?? new Date().toISOString(),
      currentModel: info.currentModel,
      currentModelDisplay: info.currentModelDisplay,
    });
    this._onDidChangeTree.fire();
  }

  clear(): void {
    this.sessions.clear();
    this._thinkingSuppressedSessions.clear();
    this._onDidChangeTree.fire();
  }

  /** Update the current model for a session. */
  updateSessionModel(sessionId: string, modelId: string, modelDisplay?: string): void {
    const session = this.sessions.get(sessionId);
    if (session) {
      session.currentModel = modelId;
      session.currentModelDisplay = modelDisplay ?? modelId;
      const master = this.getMasterAgent(sessionId);
      if (master) {
        master.model = modelId;
      }
      this.touchSession(sessionId);
      this._onDidChangeTree.fire();
      if (master) {
        this._onDidChangeMessages.fire({
          sessionId,
          agentId: master.id,
        });
      }
    }
  }

  /**
   * Restore sessions from server data (used after reconnecting to a running server).
   * Populates session, agent tree, and messages from REST API responses.
   */
  restoreSession(info: {
    id: string;
    name: string;
    summary?: string | null;
    forkedFrom?: string | null;
    running?: boolean;
    worktree?: SessionWorktreeInfo | null;
    createdAt?: string | null;
    lastActivity?: string | null;
    currentModel?: string;
    currentModelDisplay?: string;
    agents: Array<{
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
    }>;
    messages: Map<string, Array<{
      role: string;
      content: string;
      agent_id: string;
      timestamp: string | null;
      tool_calls: Array<{
        id: string;
        name: string;
        arguments: string;
        result: string | null;
        success: boolean;
      }>;
      streaming: boolean;
    }>>;
  }): void {
    const session: SessionState = {
      id: info.id,
      name: info.name,
      summary: info.summary ?? null,
      forkedFrom: info.forkedFrom ?? null,
      running: info.running ?? false,
      agents: new Map(),
      messages: new Map(),
      streamBuffers: new Map(),
      pendingToolCalls: new Map(),
      contextUsage: new Map(),
      worktree: info.worktree ?? null,
      createdAt: info.createdAt ?? null,
      lastActivity: null,
      currentModel: info.currentModel,
      currentModelDisplay: info.currentModelDisplay,
    };

    // Restore agents, tracking the most recent activity timestamp
    // Prefer server-provided lastActivity if available, otherwise compute from agents
    let latestActivity: string | null = info.lastActivity ?? info.createdAt ?? null;
    for (const a of info.agents) {
      session.agents.set(a.id, {
        id: a.id,
        name: a.name,
        state: this.mapState(a.state),
        rawState: a.state,
        role: this.mapRole(a.role),
        model: a.model,
        parentId: a.parent_id ?? null,
        childrenIds: a.children_ids ?? [],
        promptPreview: this.stripPromptPrefix(a.prompt_preview ?? ""),
        createdAt: a.created_at ?? null,
        completedAt: a.completed_at ?? null,
        lastActive: a.last_active ?? null,
      });
      // Track the most recent agent activity for session-level timestamp
      const agentTs = a.last_active ?? a.completed_at ?? a.created_at;
      if (agentTs && (!latestActivity || agentTs > latestActivity)) {
        latestActivity = agentTs;
      }
    }
    session.lastActivity = latestActivity;

    // Restore messages
    for (const [agentId, msgs] of info.messages) {
      const mapped: Message[] = msgs.map((m) => ({
        role: m.role as Message["role"],
        content: m.content,
        agentId: m.agent_id,
        timestamp: m.timestamp,
        snapshotId: (m as any).snapshot_id,
        toolCalls: (m.tool_calls ?? []).map((tc) => ({
          id: tc.id,
          name: tc.name,
          arguments: tc.arguments,
          result: tc.result,
          success: tc.success,
        })),
        streaming: m.streaming ?? false,
      }));
      session.messages.set(agentId, mapped);
    }

    this.sessions.set(info.id, session);
    this._onDidChangeTree.fire();
  }

  // ── Private helpers ──

  /** Update the session's last activity timestamp to now. */
  private touchSession(sessionId: string): void {
    const session = this.sessions.get(sessionId);
    if (session) {
      session.lastActivity = new Date().toISOString();
    }
  }

  // ── Private handlers ──

  private ensureSession(sessionId: string): SessionState {
    let session = this.sessions.get(sessionId);
    if (!session) {
      const now = new Date().toISOString();
      session = {
        id: sessionId,
        name: `Untitled Session (${sessionId.slice(0, 8)})`,
        summary: null,
        forkedFrom: null,
        running: false,
        agents: new Map(),
        messages: new Map(),
        streamBuffers: new Map(),
        pendingToolCalls: new Map(),
        contextUsage: new Map(),
        worktree: this._worktreeInfo,
        createdAt: now,
        lastActivity: now,
      };
      this.sessions.set(sessionId, session);
    }
    return session;
  }

  private handleSessionCreated(data: SessionCreatedData): void {
    const existing = this.sessions.get(data.session_id);
    if (existing) {
      existing.name = data.name;
      existing.forkedFrom = data.forked_from;
      if (data.current_model) {
        existing.currentModel = data.current_model;
      }
      if (data.current_model_display) {
        existing.currentModelDisplay = data.current_model_display;
      }
      this._onDidChangeTree.fire();
      return;
    }
    this.addSession({
      id: data.session_id,
      name: data.name,
      summary: data.summary ?? null,
      forkedFrom: data.forked_from,
      worktree: this._worktreeInfo,
      currentModel: data.current_model,
      currentModelDisplay: data.current_model_display,
    });
  }

  private handleSessionRemoved(data: SessionRemovedData): void {
    this.sessions.delete(data.session_id);
    this._onDidChangeTree.fire();
  }

  private handleSessionRenamed(data: { session_id: string; name: string; summary?: string | null }): void {
    const session = this.sessions.get(data.session_id);
    if (session) {
      session.name = data.name;
      if (typeof data.summary === "string") {
        session.summary = data.summary;
      }
      this._onDidChangeTree.fire();
      this._onSessionRenamed.fire({
        sessionId: data.session_id,
        newName: data.name,
      });
    }
  }

  private handleAgentMessage(data: {
    session_id: string;
    agent_id: string;
    content: string;
    role: string;
    snapshot_id?: string;
  }): void {
    const session = this.sessions.get(data.session_id);
    if (!session) return;
    this.touchSession(data.session_id);

    const messages = session.messages.get(data.agent_id) ?? [];
    messages.push({
      role: data.role as "user" | "assistant" | "system" | "tool",
      content: data.content,
      agentId: data.agent_id,
      timestamp: new Date().toISOString(),
      snapshotId: data.snapshot_id,
      toolCalls: [],
      streaming: false,
    });
    session.messages.set(data.agent_id, messages);
    this._onDidChangeMessages.fire({
      sessionId: data.session_id,
      agentId: data.agent_id,
    });
  }

  private handleAgentSpawned(data: AgentSpawnedData): void {
    const session = this.ensureSession(data.session_id);
    this.touchSession(data.session_id);

    const now = new Date().toISOString();
    // A just-spawned agent is always about to start running.  Default
    // to "pending" (maps to the "running" UI state) so the tree shows
    // the spinning icon immediately, rather than briefly flashing idle.
    const rawState = data.state ?? "pending";
    const agent: AgentNode = {
      id: data.agent_id,
      name: data.name ?? this.deriveName(data.role, data.prompt),
      state: this.mapState(rawState),
      rawState,
      role: this.mapRole(data.role),
      model: data.model,
      parentId: data.parent_id ?? null,
      childrenIds: [],
      promptPreview: this.stripPromptPrefix(data.prompt ?? "").slice(0, 100),
      createdAt: now,
      completedAt: null,
      lastActive: now,
    };

    session.agents.set(agent.id, agent);
    session.messages.set(agent.id, []);

    // Link to parent
    if (agent.parentId) {
      const parent = session.agents.get(agent.parentId);
      if (parent && !parent.childrenIds.includes(agent.id)) {
        parent.childrenIds.push(agent.id);
      }
    }

    // If root agent spawned, flush any pending user messages
    if (!agent.parentId) {
      this._outputChannel?.appendLine(
        `[store] ROOT agent_spawned session=${data.session_id.slice(0, 8)} agent=${agent.id.slice(0, 8)} role=${agent.role} — getMasterAgent will now return this agent`
      );
      const pending = this._pendingUserMessages.get(data.session_id);
      if (pending && pending.length > 0) {
        for (const entry of pending) {
          this.addMessage(session, agent.id, {
            role: "user",
            content: entry.text,
            agentId: agent.id,
            timestamp: new Date().toISOString(),
            snapshotId: entry.snapshotId,
            toolCalls: [],
            streaming: false,
          });
        }
        this._pendingUserMessages.delete(data.session_id);
      }
    }

    this._onDidChangeTree.fire();
  }

  private handleAgentRestarted(data: AgentSpawnedData): void {
    const session = this.sessions.get(data.session_id);
    if (!session) return;
    this.touchSession(data.session_id);

    const now = new Date().toISOString();
    const existing = session.agents.get(data.agent_id);
    if (existing) {
      // Mark as running — agent is starting a new run and will be active shortly
      existing.state = "running";
      existing.rawState = "running";
      existing.lastActive = now;
      existing.promptPreview = data.prompt ? this.stripPromptPrefix(data.prompt).slice(0, 100) : existing.promptPreview;
      if (data.parent_id) {
        const parent = session.agents.get(data.parent_id);
        if (parent && !parent.childrenIds.includes(data.agent_id)) {
          parent.childrenIds.push(data.agent_id);
        }
      }
    } else {
      // Agent was fully cleaned up — re-add it.
      // Default to "running" since a restarted agent is about to start immediately.
      const rawState = data.state ?? "running";
      const agent: AgentNode = {
        id: data.agent_id,
        name: data.name ?? this.deriveName(data.role, data.prompt),
        state: this.mapState(rawState),
        rawState,
        role: this.mapRole(data.role),
        model: data.model,
        parentId: data.parent_id ?? null,
        childrenIds: [],
        promptPreview: this.stripPromptPrefix(data.prompt ?? "").slice(0, 100),
        createdAt: now,
        completedAt: null,
        lastActive: now,
      };
      session.agents.set(agent.id, agent);
      session.messages.set(agent.id, []);
      if (agent.parentId) {
        const parent = session.agents.get(agent.parentId);
        if (parent && !parent.childrenIds.includes(agent.id)) {
          parent.childrenIds.push(agent.id);
        }
      }
    }

    this._onDidChangeTree.fire();
    // Also fire messages change so agent panels get a fullUpdate (busy state, etc.)
    this._onDidChangeMessages.fire({
      sessionId: data.session_id,
      agentId: data.agent_id,
    });
  }

  private handleAgentStateChanged(data: AgentStateChangedData): void {
    const session = this.sessions.get(data.session_id);
    if (!session) return;
    this.touchSession(data.session_id);

    const agent = session.agents.get(data.agent_id);
    if (agent) {
      const now = new Date().toISOString();
      agent.rawState = data.new_state;
      agent.state = this.mapState(data.new_state);
      agent.lastActive = now;
      if (data.new_state === "completed" || data.new_state === "failed" || data.new_state === "error") {
        agent.completedAt = now;
      }
      this._onDidChangeTree.fire();
      // Also fire messages change so agent panels get a fullUpdate (busy state, etc.)
      this._onDidChangeMessages.fire({
        sessionId: data.session_id,
        agentId: data.agent_id,
      });
    }
  }

  private handleAgentKilled(data: AgentKilledData): void {
    const session = this.sessions.get(data.session_id);
    if (!session) return;

    const agent = session.agents.get(data.agent_id);
    if (agent) {
      agent.state = "error";
      agent.rawState = "killed";
      agent.completedAt = new Date().toISOString();
    }
    // Preserve killed agent and message history so chat state is frozen in-place.
    // A restarted agent can reuse the same ID and continue on the existing thread.
    this._onDidChangeTree.fire();
    this._onDidChangeMessages.fire({
      sessionId: data.session_id,
      agentId: data.agent_id,
    });
  }

  private handleStreamChunk(data: StreamChunkData): void {
    const session = this.sessions.get(data.session_id);
    if (!session) return;

    // Accumulate stream text
    const current = session.streamBuffers.get(data.agent_id) ?? "";
    session.streamBuffers.set(data.agent_id, current + data.text);

    this._onStreamChunk.fire(data);
  }

  private handleToolCallStarted(data: ToolCallStartedData): void {
    const session = this.sessions.get(data.session_id);
    if (!session) {
      this._outputChannel?.appendLine(
        `[store] tool_call_started DROPPED: session ${data.session_id?.slice(0, 8)} not found (known: ${[...this.sessions.keys()].map(k => k.slice(0, 8)).join(",")})`
      );
      return;
    }

    const tc: ToolCall = {
      id: data.tool_id,
      name: data.tool_name,
      arguments: data.arguments,
      result: null,
      success: true,
      pending: true,
    };

    session.pendingToolCalls.set(data.tool_id, tc);

    // Flush any accumulated stream text as an assistant message first
    this.flushStreamBuffer(session, data.agent_id);

    this.addMessage(session, data.agent_id, {
      role: "tool",
      content: "",
      agentId: data.agent_id,
      timestamp: new Date().toISOString(),
      toolCalls: [tc],
      streaming: false,
    });
  }

  private handleToolCallCompleted(data: ToolCallCompletedData): void {
    const session = this.sessions.get(data.session_id);
    if (!session) return;

    // Update the pending tool call
    const tc = session.pendingToolCalls.get(data.tool_id);
    if (tc) {
      tc.result = this.pickCompletedToolResult(tc.result, data.result, tc.name);
      tc.success = !data.is_error;
      tc.pending = false;
      session.pendingToolCalls.delete(data.tool_id);
    }

    // Also update in message history
    const msgs = session.messages.get(data.agent_id);
    if (msgs) {
      for (let i = msgs.length - 1; i >= 0; i--) {
        for (const mtc of msgs[i].toolCalls) {
          if (mtc.id === data.tool_id) {
            mtc.result = this.pickCompletedToolResult(mtc.result, data.result, mtc.name);
            mtc.success = !data.is_error;
            mtc.pending = false;
            break;
          }
        }
      }
    }

    this._onDidChangeMessages.fire({
      sessionId: data.session_id,
      agentId: data.agent_id,
    });
  }

  private pickCompletedToolResult(
    existingResult: string | null,
    completionResult: string | null,
    toolName?: string
  ): string | null {
    const completedText = completionResult ?? "";
    const existingText = existingResult ?? "";
    const streamedSummary = completedText.includes("Output streamed previously.");
    const completionEnvelope = this.isCommandCompletionEnvelope(completedText);
    const cancellationSummary = this.isCancellationSummary(completedText);
    const isTerminalTool = this.isTerminalToolName(toolName);

    if (isTerminalTool && this.hasAnyToolOutput(existingText)) {
      if (streamedSummary || completionEnvelope) {
        return existingResult;
      }
      return this.appendUnique(existingText, completedText);
    }

    if ((streamedSummary || completionEnvelope) && this.hasAnyToolOutput(existingText)) {
      return existingResult;
    }
    if (cancellationSummary && this.hasAnyToolOutput(existingText)) {
      return this.appendUnique(existingText, completedText);
    }
    return completionResult;
  }

  private isCommandCompletionEnvelope(text: string): boolean {
    const trimmed = (text ?? "").trim();
    if (!trimmed) return false;
    const hasStatusLine =
      /Command completed successfully \(exit code \d+\)\./.test(trimmed)
      || /Command failed \(exit code \d+\)\./.test(trimmed);
    const hasEnvelopeFields =
      /(?:^|\n)Command:\s/.test(trimmed)
      && /(?:^|\n)CWD:\s/.test(trimmed)
      && /(?:^|\n)Captured stdout:\s/.test(trimmed);
    return hasStatusLine && hasEnvelopeFields;
  }

  private hasAnyToolOutput(text: string): boolean {
    return (text ?? "").trim().length > 0;
  }

  private isTerminalToolName(name?: string): boolean {
    return ["bash", "run_bash"].includes((name ?? "").toLowerCase());
  }

  private appendUnique(existing: string, addition: string): string {
    const trimmedExisting = (existing ?? "").replace(/\s+$/g, "");
    const trimmedAddition = (addition ?? "").trim();
    if (!trimmedAddition) return trimmedExisting;
    // Avoid duplicate completion summaries if the same completion payload is
    // processed more than once.
    if (trimmedExisting.endsWith(trimmedAddition)) return trimmedExisting;
    if (!trimmedExisting) return trimmedAddition;
    return `${trimmedExisting}\n\n${trimmedAddition}`;
  }

  private isCancellationSummary(text: string): boolean {
    const normalized = (text ?? "").toLowerCase();
    if (!normalized.trim()) return false;
    return (
      normalized.includes("bash command was cancelled") ||
      normalized.includes("was canceled") ||
      normalized.includes("was cancelled") ||
      normalized.includes("stopped by user")
    );
  }

  private handleToolCallDelta(data: ToolCallDeltaData): void {
    const session = this.sessions.get(data.session_id);
    if (!session || !data.delta) return;

    let tc = data.tool_id ? session.pendingToolCalls.get(data.tool_id) : undefined;
    if (!tc) {
      const agentMessages = session.messages.get(data.agent_id) || [];
      for (let i = agentMessages.length - 1; i >= 0; i -= 1) {
        const msg = agentMessages[i];
        for (const candidate of msg.toolCalls) {
          if (
            candidate.result === null
            && ["bash", "run_bash"].includes((candidate.name || "").toLowerCase())
          ) {
            tc = candidate;
            break;
          }
        }
        if (tc) break;
      }
    }
    if (!tc) return;

    tc.pending = true;
    if (tc.result === null) tc.result = "";
    if (data.stream === "stderr" && !tc.result.includes("STDERR:\n")) {
      tc.result += "\n\nSTDERR:\n";
    }
    tc.result += data.delta;

    this._onDidChangeMessages.fire({
      sessionId: data.session_id,
      agentId: data.agent_id,
    });
  }

  private handleEngineStarted(data: EngineStartedData): void {
    this._thinkingSuppressedSessions.delete(data.session_id);
    const session = this.ensureSession(data.session_id);
    session.running = true;
    this.touchSession(data.session_id);
    const agents = [...session.agents.values()];
    const roots = agents.filter(a => !a.parentId);
    this._outputChannel?.appendLine(
      `[store] engine_started session=${data.session_id.slice(0, 8)} agents=${agents.length} roots=${roots.map(r => r.id.slice(0, 8) + ":" + r.state).join(",")}`
    );
    this._onEngineStarted.fire(data);
    this._onDidChangeTree.fire();
  }

  private handleEngineFinished(data: EngineFinishedData): void {
    const session = this.sessions.get(data.session_id);
    if (session) {
      session.running = false;
      this.touchSession(data.session_id);

      // Flush all remaining stream buffers
      for (const agentId of session.streamBuffers.keys()) {
        this.flushStreamBuffer(session, agentId);
      }
    }
    // Clear all thinking state
    this._thinkingAgents.clear();
    this._onEngineFinished.fire(data);
    this._onDidChangeTree.fire();
  }

  /**
   * Force local UI/session state to non-running immediately.
   * Used for explicit user stop actions where server-side shutdown may
   * not emit engine_finished synchronously.
   */
  forceSessionStopped(sessionId: string): void {
    const session = this.sessions.get(sessionId);
    if (!session) return;
    this._thinkingSuppressedSessions.add(sessionId);
    session.running = false;
    const now = new Date().toISOString();
    for (const agent of session.agents.values()) {
      if (agent.state === "completed" || agent.state === "error") {
        continue;
      }
      agent.state = "error";
      agent.rawState = "error";
      agent.completedAt = now;
      agent.lastActive = now;
    }
    this.touchSession(sessionId);
    for (const agentId of session.streamBuffers.keys()) {
      this.flushStreamBuffer(session, agentId);
    }
    for (const agentId of session.agents.keys()) {
      this._thinkingAgents.delete(agentId);
    }
    const master = this.getMasterAgent(sessionId);
    if (master) {
      const msgs = session.messages.get(master.id) ?? [];
      const lastMsg = msgs[msgs.length - 1];
      const marker = SessionStore.RUN_STOPPED_MARKER;
      if (!lastMsg || lastMsg.role !== "system" || lastMsg.content !== marker) {
        msgs.push({
          role: "system",
          content: marker,
          agentId: master.id,
          timestamp: new Date().toISOString(),
          toolCalls: [],
          streaming: false,
        });
        session.messages.set(master.id, msgs);
      }
    }
    this._onDidChangeTree.fire();
    const masterId = master?.id;
    if (masterId) {
      this._onDidChangeMessages.fire({ sessionId, agentId: masterId });
    }
  }

  private handleAgentResult(data: AgentResultData): void {
    const session = this.sessions.get(data.session_id);
    if (!session) return;

    // Flush stream buffer as the final assistant message
    this.flushStreamBuffer(session, data.agent_id);
  }

  private handleUserPrompt(data: UserPromptData): void {
    const session = this.sessions.get(data.session_id);
    if (!session) {
      this._onUserPrompt.fire(data);
      return;
    }

    // If agent_id is specified, attach directly to that agent
    if (data.agent_id) {
      this.flushStreamBuffer(session, data.agent_id);
      this.addMessage(session, data.agent_id, {
        role: "user",
        content: data.text,
        agentId: data.agent_id,
        timestamp: new Date().toISOString(),
        snapshotId: data.snapshot_id,
        toolCalls: [],
        streaming: false,
      });
      this._onUserPrompt.fire(data);
      return;
    }

    // No agent_id — queue for the next root agent that spawns.
    // This handles both the initial prompt (no master yet) and follow-up
    // prompts (old master exists but a new one will spawn for this run).
    const pending =
      this._pendingUserMessages.get(data.session_id) || [];
    pending.push({ text: data.text, snapshotId: data.snapshot_id });
    this._pendingUserMessages.set(data.session_id, pending);
    this._onUserPrompt.fire(data);
  }

  private handleContextWindowUsage(data: ContextWindowUsageData): void {
    const session = this.ensureSession(data.session_id);
    session.contextUsage.set(data.agent_id, data);
    this.touchSession(data.session_id);
    this._onDidChangeMessages.fire({
      sessionId: data.session_id,
      agentId: data.agent_id,
    });
  }

  /**
   * Handle the SSE "connected" event, which may include either
   * full session objects or just session IDs.
   * This is used as an opportunistic bootstrap for tree labels.
   */
  private handleConnected(data: Record<string, unknown>): void {
    const rawSessionList = data.session_list ??
      data.sessions ??
      [];
    const sessionList = Array.isArray(rawSessionList)
      ? rawSessionList.map((entry) => {
        if (typeof entry === "string") {
          return { sessionId: entry, name: entry };
        }
        return entry as Record<string, unknown>;
      })
      : [];
    if (sessionList.length === 0) {
      return;
    }
    let added = 0;
    for (const s of sessionList) {
      const sessionId = (s.sessionId as string) ?? "";
      if (!sessionId || this.sessions.has(sessionId)) continue;
      this.sessions.set(sessionId, {
        id: sessionId,
        name: (s.name as string) || sessionId,
        summary: (s.summary as string) ?? null,
        forkedFrom: (s.forkedFrom as string) ?? null,
        running: (s.running as boolean) ?? false,
        agents: new Map(),
        messages: new Map(),
        streamBuffers: new Map(),
        pendingToolCalls: new Map(),
        contextUsage: new Map(),
        worktree: null,
        createdAt: (s.createdAt as string) ?? null,
        lastActivity: (s.lastActivity as string) ?? null,
        currentModel: s.currentModel as string | undefined,
        currentModelDisplay: s.currentModelDisplay as string | undefined,
      });
      added += 1;
    }
    if (added > 0) {
      this._outputChannel?.appendLine(
        `[store] connected event: added ${added} session(s) from server`
      );
      this._onDidChangeTree.fire();
    }
  }

  private handleModelSwitched(data: ModelSwitchedData): void {
    const session = this.sessions.get(data.session_id);
    if (session) {
      session.currentModel = data.new_model;
      session.currentModelDisplay = data.new_model_display ?? data.new_model;
      const master = this.getMasterAgent(data.session_id);
      if (master) {
        master.model = data.new_model;
      }
      this.touchSession(data.session_id);
      this._onDidChangeTree.fire();
      if (master) {
        this._onDidChangeMessages.fire({
          sessionId: data.session_id,
          agentId: master.id,
        });
      }
    }
    this._onModelSwitched.fire(data);
  }

  private flushStreamBuffer(
    session: SessionState,
    agentId: string
  ): void {
    const text = session.streamBuffers.get(agentId);
    if (text && text.length > 0) {
      this.addMessage(session, agentId, {
        role: "assistant",
        content: text,
        agentId,
        timestamp: new Date().toISOString(),
        toolCalls: [],
        streaming: false,
      });
      session.streamBuffers.set(agentId, "");
    }
  }

  private addMessage(
    session: SessionState,
    agentId: string,
    msg: Message
  ): void {
    let list = session.messages.get(agentId);
    if (!list) {
      list = [];
      session.messages.set(agentId, list);
    }
    const lastMsg = list[list.length - 1];
    const stopMarkerAtEnd =
      !!lastMsg
      && lastMsg.role === "system"
      && lastMsg.content === SessionStore.RUN_STOPPED_MARKER;

    // Keep the "Run stopped by user." marker pinned as the final item.
    // Late-arriving user/system events should appear before it.
    if (
      stopMarkerAtEnd
      && !(msg.role === "system" && msg.content === SessionStore.RUN_STOPPED_MARKER)
    ) {
      list.splice(Math.max(0, list.length - 1), 0, msg);
    } else {
      list.push(msg);
    }
    this._outputChannel?.appendLine(
      `[store.addMessage] session=${session.id.slice(0, 8)} agent=${agentId.slice(0, 8)} role=${msg.role} toolCalls=${msg.toolCalls.length} totalMsgs=${list.length}`
    );
    this._onDidChangeMessages.fire({
      sessionId: session.id,
      agentId,
    });
  }

  private mapState(state: string): AgentState {
    const map: Record<string, AgentState> = {
      pending: "running",
      starting: "running",
      idle: "idle",
      running: "running",
      waiting_for_parent: "waiting",
      waiting_for_child: "waiting_for_child",
      waiting_for_expert: "waiting",
      waiting: "waiting",
      completed: "completed",
      failed: "error",
      killed: "error",
      error: "error",
    };
    return map[state] ?? "idle";
  }

  hasWaitingForChildAgents(): boolean {
    for (const session of this.sessions.values()) {
      for (const agent of session.agents.values()) {
        if (
          agent.state === "waiting_for_child"
          || agent.rawState === "waiting_for_child"
        ) {
          return true;
        }
      }
    }
    return false;
  }

  private mapRole(role: string): AgentRole {
    const map: Record<string, AgentRole> = {
      master: "orchestrator",
      worker: "worker",
      expert: "expert",
      reviewer: "worker",
      orchestrator: "orchestrator",
    };
    return map[role] ?? "worker";
  }

  /**
   * Strip the injected docs-first / assumption-minimization prefix from a prompt.
   * The engine prepends system instructions delimited by a "TASK:\n" marker.
   * This returns only the user-facing task description.
   */
  private stripPromptPrefix(prompt: string): string {
    if (!prompt) return prompt;
    const marker = "TASK:\n";
    const idx = prompt.indexOf(marker);
    if (idx !== -1) return prompt.slice(idx + marker.length);
    if (prompt.startsWith("[DOCS-FIRST-ARCHITECTURE]")) return "";
    return prompt;
  }

  private deriveName(role: string, prompt: string): string {
    const clean = this.stripPromptPrefix(prompt);
    if (role === "master") return "Orchestrator";
    if (role === "expert") return clean?.slice(0, 40) || "Expert";
    const name = (clean ?? "").slice(0, 50).trim();
    return (name.length >= 50 ? name + "..." : name) || "Worker";
  }

  dispose(): void {
    this._onDidChangeTree.dispose();
    this._onDidChangeMessages.dispose();
    this._onStreamChunk.dispose();
    this._onPermissionRequest.dispose();
    this._onUserQuestion.dispose();
    this._onEngineStarted.dispose();
    this._onEngineFinished.dispose();
    this._onThinking.dispose();
    this._onUserPrompt.dispose();
    this._onFileChanged.dispose();
    this._onSnapshotCreated.dispose();
    this._onSnapshotRestored.dispose();
    this._onModelSwitched.dispose();
  }
}
