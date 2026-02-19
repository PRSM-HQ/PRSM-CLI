/**
 * PRSM VSCode Extension — main entry point.
 *
 * Spawns `prsm --server`, connects via HTTP+SSE, and renders the agent
 * tree in the Explorer with webview tabs for conversations.
 *
 * The server is spawned detached so it survives VSCode window reloads.
 * On reactivation the extension reconnects to the existing server and
 * restores all session state from the REST API.
 */
import * as vscode from "vscode";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import { PrsmTransport } from "./protocol/transport";
import { SessionInfo } from "./protocol/types";
import { SessionStore } from "./state/sessionStore";
import {
  AgentTreeProvider,
  AgentTreeItem,
  SessionTreeItem,
} from "./views/agentTreeProvider";
import {
  ActiveChatTarget,
  AgentWebviewManager,
} from "./views/agentWebviewManager";
import { AgentContextPanelManager } from "./views/agentContextPanel";
import { registerPermissionHandler } from "./commands/resolvePermission";
import { registerQuestionHandler } from "./commands/resolveQuestion";
import { generateConfigFile } from "./config/configGenerator";
import { checkWorkspaceSetup, performSetup } from "./commands/setupWorkspace";
import { registerExpertCommands } from "./commands/expertManager";
import { registerWorktreeCommands } from "./commands/worktreeCommands";
import { WorktreeManager } from "./git/worktreeManager";
import { FileChangeTracker, TrackedFileChange } from "./tracking/fileChangeTracker";
import { AgentEditDecorationProvider } from "./tracking/decorationProvider";
import { ChangeLensProvider } from "./tracking/changeLensProvider";
import { FileChangesTreeProvider } from "./views/fileChangesTreeProvider";
import { SnapshotTreeProvider } from "./views/snapshotTreeProvider";
import { OriginalContentProvider, ORIGINAL_SCHEME } from "./tracking/originalContentProvider";
import { AgentFileDecorationProvider } from "./tracking/fileDecorationProvider";
import { FileChangedData, SnapshotCreatedData } from "./protocol/types";
import { SettingsPanel } from "./views/settingsPanel";
import { toModelAliasLabel } from "./utils/modelLabel";
import { notifyIfUpdateAvailable } from "./update/updateChecker";

const DEBUG_LOG = path.join(os.homedir(), ".prsm", "logs", "webview-debug.log");
function debugLog(msg: string): void {
  try {
    fs.appendFileSync(DEBUG_LOG, `${new Date().toISOString()} ${msg}\n`);
  } catch { /* ignore */ }
}

let transport: PrsmTransport | undefined;
let store: SessionStore | undefined;
let statusItem: vscode.StatusBarItem | undefined;
let modelStatusItem: vscode.StatusBarItem | undefined;

/** Workspace state keys for persisting server info across reloads. */
const STATE_PORT = "prsm.server.port";
const STATE_PID = "prsm.server.pid";

export async function activate(
  context: vscode.ExtensionContext
): Promise<void> {
  const outputChannel =
    vscode.window.createOutputChannel("PRSM");
  context.subscriptions.push(outputChannel);
  outputChannel.appendLine("[startup] activate() begin");

  store = new SessionStore();
  store.setOutputChannel(outputChannel);
  store.setWorkspaceState(context.workspaceState);
  context.subscriptions.push({ dispose: () => store?.dispose() });

  // Tree view
  const treeProvider = new AgentTreeProvider(store);
  const treeView = vscode.window.createTreeView("prsmAgentTree", {
    treeDataProvider: treeProvider,
    showCollapseAll: true,
    canSelectMany: true,
  });
  context.subscriptions.push(treeView);
  context.subscriptions.push(treeProvider);

  // Track user collapse/expand actions to persist tree state across restarts
  context.subscriptions.push(
    treeView.onDidCollapseElement((e) => {
      if (e.element instanceof SessionTreeItem) {
        store!.setCollapsed(e.element.session.id);
      } else if (e.element instanceof AgentTreeItem) {
        store!.setCollapsed(e.element.agent.id);
      }
    }),
    treeView.onDidExpandElement((e) => {
      if (e.element instanceof SessionTreeItem) {
        store!.setExpanded(e.element.session.id);
      } else if (e.element instanceof AgentTreeItem) {
        store!.setExpanded(e.element.agent.id);
      }
    })
  );

  // Helper to get/ensure transport
  const getTransport = (): PrsmTransport | undefined => transport;

  const getPreferredWorkspaceRoot = (): string | undefined => {
    const folders = vscode.workspace.workspaceFolders;
    if (!folders || folders.length === 0) return undefined;

    for (const folder of folders) {
      const root = folder.uri.fsPath;
      if (
        fs.existsSync(path.join(root, ".prism", "prsm.yaml")) ||
        fs.existsSync(path.join(root, ".prism", "models.yaml"))
      ) {
        return root;
      }
    }

    // Legacy fallback
    for (const folder of folders) {
      const root = folder.uri.fsPath;
      if (fs.existsSync(path.join(root, "prsm.yaml"))) {
        return root;
      }
    }

    return folders[0].uri.fsPath;
  };

  // Webview manager (needs transport getter for prompt submission)
  const webviewManager = new AgentWebviewManager(
    context,
    store,
    getTransport,
    outputChannel,
  );
  context.subscriptions.push({
    dispose: () => webviewManager.dispose(),
  });

  let syncingTreeSelection = false;
  const syncTreeSelectionToFocusedChat = async (
    target: ActiveChatTarget | undefined,
  ) => {
    if (!target || syncingTreeSelection) return;

    const item =
      target.kind === "session"
        ? treeProvider.getSessionItem(target.sessionId)
        : treeProvider.getAgentItem(target.sessionId, target.agentId);
    if (!item) return;

    syncingTreeSelection = true;
    try {
      await treeView.reveal(item, {
        select: true,
        focus: false,
        expand: target.kind === "agent" ? 10 : false,
      });
    } catch {
      // Tree can be briefly unavailable during refresh/hydration.
    } finally {
      syncingTreeSelection = false;
    }
  };
  context.subscriptions.push(
    webviewManager.onDidChangeActiveChat((target) => {
      void syncTreeSelectionToFocusedChat(target);
    }),
  );

  // Agent context side panel
  const contextPanel = new AgentContextPanelManager(
    context,
    store,
    getTransport,
  );
  context.subscriptions.push({
    dispose: () => contextPanel.dispose(),
  });

  // File change tracking + decorations
  const fileTracker = new FileChangeTracker();
  context.subscriptions.push(fileTracker);

  const decorationProvider = new AgentEditDecorationProvider(fileTracker);
  context.subscriptions.push(decorationProvider);

  // CodeLens for accept/reject/go-to-agent
  const changeLensProvider = new ChangeLensProvider(fileTracker);
  context.subscriptions.push(
    vscode.languages.registerCodeLensProvider("*", changeLensProvider)
  );

  // File Changes tree view (store enables live agent name resolution)
  const fileChangesTreeProvider = new FileChangesTreeProvider(fileTracker, store);
  const fileChangesTree = vscode.window.createTreeView("prsmFileChanges", {
    treeDataProvider: fileChangesTreeProvider,
  });
  context.subscriptions.push(fileChangesTree);

  // Original content provider for diff editor
  const originalContentProvider = new OriginalContentProvider(fileTracker);
  context.subscriptions.push(
    vscode.workspace.registerTextDocumentContentProvider(
      ORIGINAL_SCHEME,
      originalContentProvider
    )
  );
  context.subscriptions.push(originalContentProvider);

  // File decoration provider (explorer badges)
  const fileDecorationProvider = new AgentFileDecorationProvider(fileTracker);
  context.subscriptions.push(
    vscode.window.registerFileDecorationProvider(fileDecorationProvider)
  );
  context.subscriptions.push(fileDecorationProvider);

  // Activity bar badge — update pending change count
  const updateBadge = () => {
    const count = fileTracker.getPendingCount();
    fileChangesTree.badge = count > 0
      ? { value: count, tooltip: `${count} pending agent change${count !== 1 ? "s" : ""}` }
      : undefined;
  };
  fileTracker.onDidChange(() => {
    updateBadge();
    const activeEditor = vscode.window.activeTextEditor;
    vscode.commands.executeCommand(
      "setContext",
      "prsm.fileHasPendingChanges",
      activeEditor
        ? fileTracker.hasPendingChanges(activeEditor.document.uri.fsPath)
        : false
    );
  });
  updateBadge();

  // Snapshot tree view
  const snapshotTreeProvider = new SnapshotTreeProvider();
  snapshotTreeProvider.setWorkspaceState(context.workspaceState);
  const snapshotTree = vscode.window.createTreeView("prsmSnapshots", {
    treeDataProvider: snapshotTreeProvider,
  });
  context.subscriptions.push(snapshotTree);
  context.subscriptions.push(
    snapshotTree.onDidCollapseElement((e) => {
      const node = e.element as { nodeId?: string };
      if (node?.nodeId) snapshotTreeProvider.setCollapsed(node.nodeId);
    }),
    snapshotTree.onDidExpandElement((e) => {
      const node = e.element as { nodeId?: string };
      if (node?.nodeId) snapshotTreeProvider.setExpanded(node.nodeId);
    })
  );

  // Wire file_changed events from store to tracker
  store.onFileChanged((data: FileChangedData) => {
    const agent = store!.getAgent(data.session_id, data.agent_id);
    fileTracker.trackChange(data, agent?.name);
    // Update context key for editor title bar actions
    const activeEditor = vscode.window.activeTextEditor;
    if (activeEditor) {
      vscode.commands.executeCommand(
        "setContext",
        "prsm.fileHasPendingChanges",
        fileTracker.hasPendingChanges(activeEditor.document.uri.fsPath)
      );
    }
  });

  // Update context key when active editor changes
  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor((editor) => {
      vscode.commands.executeCommand(
        "setContext",
        "prsm.fileHasPendingChanges",
        editor
          ? fileTracker.hasPendingChanges(editor.document.uri.fsPath)
          : false
      );
    })
  );

  // Wire snapshot events
  store.onSnapshotCreated((data: SnapshotCreatedData) => {
    const sessionName = store?.getSession(data.session_id)?.name ?? "";
    const agent = data.agent_id
      ? store?.getAgent(data.session_id, data.agent_id)
      : undefined;
    snapshotTreeProvider.addSnapshot(
      {
        snapshot_id: data.snapshot_id,
        session_id: data.session_id,
        session_name: sessionName,
        description: data.description,
        timestamp: data.timestamp ?? new Date().toISOString(),
        git_branch: data.git_branch ?? null,
        parent_snapshot_id: data.parent_snapshot_id ?? null,
        agent_id: data.agent_id ?? null,
        agent_name: data.agent_name ?? agent?.name ?? null,
        parent_agent_id: data.parent_agent_id ?? agent?.parentId ?? null,
      },
      data.session_id
    );
    if (sessionName) {
      snapshotTreeProvider.setSessionLabel(data.session_id, sessionName);
    }
  });

  // When a snapshot is restored, re-fetch file changes from the server
  // (the server replaces the file tracker state on restore)
  store.onSnapshotRestored(async (data) => {
    fileTracker.clear();
    if (transport?.isConnected) {
      try {
        const fcResp = await transport.getFileChanges(data.session_id);
        const fileChanges = fcResp.file_changes ?? {};
        const changeCount = Object.values(fileChanges).reduce(
          (n, arr) => n + arr.length, 0
        );
        if (changeCount > 0) {
          fileTracker.restoreFromServer(
            data.session_id,
            fileChanges,
            (agentId) => store!.getAgent(data.session_id, agentId)?.name
          );
          outputChannel.appendLine(
            `Restored ${changeCount} file change(s) after snapshot restore`
          );
        }
      } catch (err) {
        outputChannel.appendLine(
          `Failed to restore file changes after snapshot: ${(err as Error).message}`
        );
      }
    }
  });

  /** Save server port and PID to workspace state for reconnection. */
  const saveServerState = () => {
    if (transport?.currentPort) {
      context.workspaceState.update(STATE_PORT, transport.currentPort);
    }
    if (transport?.currentPid) {
      context.workspaceState.update(STATE_PID, transport.currentPid);
    }
  };

  /** Clear saved server state. */
  const clearServerState = () => {
    context.workspaceState.update(STATE_PORT, undefined);
    context.workspaceState.update(STATE_PID, undefined);
  };

  /** Best-effort local PID liveness check. */
  const isProcessAlive = (pid: number): boolean => {
    if (!Number.isFinite(pid) || pid <= 0) return false;
    try {
      process.kill(pid, 0);
      return true;
    } catch {
      return false;
    }
  };

  /** Best-effort stale process cleanup to prevent duplicate servers. */
  const tryTerminateProcess = (pid: number) => {
    if (!Number.isFinite(pid) || pid <= 0) return;
    try {
      process.kill(pid, "SIGTERM");
      outputChannel.appendLine(`Sent SIGTERM to stale PRSM pid=${pid}`);
    } catch (err) {
      outputChannel.appendLine(
        `Failed to terminate stale pid=${pid}: ${(err as Error).message}`
      );
    }
  };

  /**
   * Normalize persisted worktree metadata into the shape used by SessionStore.
   *
   * Persisted sessions may store { root, branch, common_dir }, while
   * SessionStore expects { worktreePath, branch, isWorktree }.
   */
  const normalizeWorktreeMetadata = (rawWorktree: unknown): {
    worktreePath: string | null;
    branch: string | null;
    isWorktree: boolean;
  } | null => {
    if (!rawWorktree || typeof rawWorktree !== "object") return null;
    const w = rawWorktree as {
      worktreePath?: unknown;
      root?: unknown;
      branch?: unknown;
      isWorktree?: unknown;
    };

    const worktreePath =
      typeof w.worktreePath === "string"
        ? w.worktreePath
        : typeof w.root === "string"
          ? w.root
          : null;
    const branch =
      typeof w.branch === "string" ? w.branch : null;
    const isWorktree = typeof w.isWorktree === "boolean"
      ? w.isWorktree
      : false;

    return { worktreePath, branch, isWorktree };
  };

  /**
   * Eagerly load sessions from disk (~/.prsm/sessions/{repo-or-workspace}/).
   * This runs synchronously at startup so the tree is populated instantly,
   * before the server even starts.
   */
  const loadSessionsFromDisk = (): number => {
    const workspaceRoot =
      vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    if (!workspaceRoot) return 0;
    outputChannel.appendLine(
      `[startup] loadSessionsFromDisk: workspaceRoot=${workspaceRoot}`
    );

    const sessionsBase = path.join(os.homedir(), ".prsm", "sessions");
    if (!fs.existsSync(sessionsBase)) {
      outputChannel.appendLine(
        "[startup] loadSessionsFromDisk: no session directory at ~/.prsm/sessions"
      );
      return 0;
    }

    const candidateDirs = new Set<string>();

    // Primary attempt: basename matches legacy/known workspace layout.
    candidateDirs.add(path.join(sessionsBase, path.basename(workspaceRoot)));

    // Compatibility: sessions are now stored by repo identity in many cases.
    // Scan all subfolders to avoid missing sessions when the basename
    // no longer matches the repository identity.
    for (const entry of fs.readdirSync(sessionsBase)) {
      const resolved = path.join(sessionsBase, entry);
      try {
        if (fs.statSync(resolved).isDirectory()) {
          candidateDirs.add(resolved);
        }
      } catch {
        continue;
      }
    }
    outputChannel.appendLine(
      `[startup] loadSessionsFromDisk: scanning ${candidateDirs.size} candidate dir(s)`
    );

    const sessionFiles: string[] = [];
    for (const sessionsDir of candidateDirs) {
      if (!fs.existsSync(sessionsDir)) continue;
      try {
        const filesInDir = fs.readdirSync(sessionsDir).filter((f) => f.endsWith(".json"));
        if (filesInDir.length > 0) {
          outputChannel.appendLine(
            `[startup] loadSessionsFromDisk: ${filesInDir.length} file(s) in ${sessionsDir}`
          );
        }
        for (const file of filesInDir) {
          sessionFiles.push(path.join(sessionsDir, file));
        }
      } catch {
        continue;
      }
    }

    const files = sessionFiles.sort((a, b) => {
      // Most recent first
      try {
        const sa = fs.statSync(a).mtimeMs;
        const sb = fs.statSync(b).mtimeMs;
        return sb - sa;
      } catch {
        return 0;
      }
    });

    let loaded = 0;
    for (const file of files) {
      try {
        const raw = fs.readFileSync(file, "utf-8");
        const data = JSON.parse(raw);
        const sessionId = data.session_id || path.basename(file).replace(".json", "");
        const sessionName = data.name || sessionId;
        const agents = data.agents || {};
        const messages = data.messages || {};

        // Convert agents from {id: {...}} to array format
        const agentArray = Object.values(agents) as Array<{
          id: string; name: string; state: string; role: string;
          model: string; parent_id: string | null;
          children_ids: string[]; prompt_preview: string;
        }>;

        // Convert messages from {agentId: [...]} to Map
        const messagesMap = new Map<string, Array<{
          role: string; content: string; agent_id: string;
          timestamp: string | null;
          tool_calls: Array<{
            id: string; name: string; arguments: string;
            result: string | null; success: boolean;
          }>;
          streaming: boolean;
        }>>();

        for (const [agentId, msgs] of Object.entries(messages)) {
          messagesMap.set(agentId, msgs as typeof messagesMap extends Map<string, infer V> ? V : never);
        }

        // Restore worktree metadata if present (normalize legacy disk schema)
        const worktree = normalizeWorktreeMetadata(data.worktree);

        store!.restoreSession({
          id: sessionId,
          name: sessionName,
          summary: (data.summary as string | null | undefined) ?? null,
          forkedFrom: data.forked_from ?? null,
          running: false,
          worktree,
          createdAt: data.created_at ?? data.saved_at ?? null,
          agents: agentArray,
          messages: messagesMap,
        });
        loaded++;
      } catch (err) {
        outputChannel.appendLine(
          `  Failed to load session from ${file}: ${(err as Error).message}`
        );
      }
    }

      if (loaded > 0) {
        outputChannel.appendLine(
          `Eagerly loaded ${loaded} session(s) from disk`
        );
      }
      if (loaded === 0) {
        outputChannel.appendLine("[startup] loadSessionsFromDisk: no sessions restored from disk cache");
      }
    return loaded;
  };

  /**
   * Restore sessions from the running server via REST API.
   * Restores sessions and eagerly hydrates per-session agents/messages,
   * file changes, and snapshots.
   */
  const hydratedSessions = new Set<string>();

  const hydrateSessionFromServer = async (
    t: PrsmTransport,
    sessionId: string
  ) => {
    if (hydratedSessions.has(sessionId)) {
      outputChannel.appendLine(
        `  Skipping hydration for ${sessionId}: already hydrated`
      );
      return;
    }
    const base = store!.getSession(sessionId);
    if (!base) {
      outputChannel.appendLine(
        `  Skipping hydration for ${sessionId}: not in local cache`
      );
      return;
    }
    outputChannel.appendLine(`  Hydrating session ${sessionId} (${base.name})`);

    // Fetch agents for this session
    outputChannel.appendLine(`    Fetching agents for ${sessionId}`);
    let agents: Array<Record<string, unknown>> = [];
    try {
      const agentResp = await t.getAgents(sessionId);
      agents = (agentResp.agents ?? []) as unknown as Array<Record<string, unknown>>;
      outputChannel.appendLine(
        `    Received ${agents.length} agent(s) for ${sessionId}`
      );
    } catch (err) {
      outputChannel.appendLine(
        `  Failed to fetch agents for ${sessionId}: ${(err as Error).message}`
      );
    }

    // Fetch messages for each agent
    const messagesMap = new Map<string, Array<Record<string, unknown>>>();
    let totalMessageCount = 0;
    for (const a of agents) {
      const agentId = a.id as string;
      if (!agentId) continue;
      try {
        const msgResp = await t.getMessages(sessionId, agentId);
        const msgCount = Array.isArray(msgResp.messages)
          ? msgResp.messages.length
          : 0;
        totalMessageCount += msgCount;
        outputChannel.appendLine(
          `    Loaded ${msgCount} message(s) for agent ${agentId} in ${sessionId}`
        );
        messagesMap.set(
          agentId,
          (msgResp.messages ?? []) as unknown as Array<Record<string, unknown>>
        );
      } catch {
        messagesMap.set(agentId, []);
      }
    }
    outputChannel.appendLine(
      `    Total messages for ${sessionId}: ${totalMessageCount}`
    );

    // Restore into store
    store!.restoreSession({
      id: sessionId,
      name: base.name,
      summary: base.summary ?? null,
      forkedFrom: base.forkedFrom,
      running: base.running ?? false,
      createdAt: base.createdAt ?? null,
      lastActivity: base.lastActivity ?? null,
      currentModel: base.currentModel,
      agents: agents as unknown as Array<{
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
      messages: messagesMap as unknown as Map<string, Array<{
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
      }>>,
    });

    // Restore file changes from server
    const agentMap = new Map<string, string>();
    for (const a of agents) {
      if (a.id && a.name) {
        agentMap.set(a.id as string, a.name as string);
      }
    }
    const fileChangeCount = await restoreSessionFileChangesFromServer(
      t,
      sessionId,
      (agentId) => agentMap.get(agentId)
    );
    outputChannel.appendLine(
      `    Restored ${fileChangeCount} file change(s) for ${sessionId}`
    );

    // Restore snapshots from server
    outputChannel.appendLine(`    Fetching snapshots for ${sessionId}`);
    try {
      const snapResp = await t.listSnapshots(sessionId);
      const snapshots = (snapResp.snapshots ?? []).map((snapshot) => {
        const agent = snapshot.agent_id
          ? store!.getAgent(sessionId, snapshot.agent_id)
          : undefined;
        return {
          ...snapshot,
          agent_name: snapshot.agent_name ?? agent?.name ?? null,
          parent_agent_id: snapshot.parent_agent_id ?? agent?.parentId ?? null,
        };
      });
      outputChannel.appendLine(
        `    Received ${snapshots.length} snapshot(s) for ${sessionId}`
      );
      if (snapshots.length > 0) {
        snapshotTreeProvider.setSnapshots(snapshots, sessionId);
      }
    } catch (err) {
      outputChannel.appendLine(
        `  Failed to restore snapshots for ${sessionId}: ${(err as Error).message}`
      );
    }

    hydratedSessions.add(sessionId);
    outputChannel.appendLine(
      `  Hydrated session ${sessionId} (${base.name}) with ${agents.length} agents`
    );
  };

  const hydrateSessionTreeFromServer = async (
    t: PrsmTransport,
    sessionId: string
  ) => {
    const base = store!.getSession(sessionId);
    if (!base) {
      outputChannel.appendLine(
        `  Skip agent-tree hydration for ${sessionId}: missing base session`
      );
      return;
    }
    // Preserve fully-hydrated sessions.
    if (base.messages.size > 0 || hydratedSessions.has(sessionId)) {
      outputChannel.appendLine(
        `  Skip agent-tree hydration for ${sessionId}: already hydrated`
      );
      return;
    }
    outputChannel.appendLine(`  Hydrating agent tree for ${sessionId}`);

    let agents: Array<Record<string, unknown>> = [];
    try {
      const agentResp = await t.getAgents(sessionId);
      agents = (agentResp.agents ?? []) as unknown as Array<Record<string, unknown>>;
      outputChannel.appendLine(
        `  Agent tree for ${sessionId}: ${agents.length} agent(s)`
      );
    } catch (err) {
      outputChannel.appendLine(
        `  Failed to fetch agent tree for ${sessionId}: ${(err as Error).message}`
      );
      return;
    }

    store!.restoreSession({
      id: sessionId,
      name: base.name,
      summary: base.summary ?? null,
      forkedFrom: base.forkedFrom,
      running: base.running ?? false,
      createdAt: base.createdAt ?? null,
      lastActivity: base.lastActivity ?? null,
      currentModel: base.currentModel,
      agents: agents as unknown as Array<{
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
      messages: new Map(),
    });
  };

  const restoreSessionFileChangesFromServer = async (
    t: PrsmTransport,
    sessionId: string,
    agentNameResolver?: (agentId: string) => string | undefined
  ): Promise<number> => {
    outputChannel.appendLine(
      `  Loading file changes from server for ${sessionId}`
    );
    try {
      const fcResp = await t.getFileChanges(sessionId);
      const fileChanges = fcResp.file_changes ?? {};
      const changeCount = Object.values(fileChanges).reduce(
        (n, arr) => n + arr.length, 0
      );
      if (changeCount > 0) {
        fileTracker.restoreFromServer(
          sessionId,
          fileChanges,
          agentNameResolver
        );
      }
      outputChannel.appendLine(
        `  Loaded ${changeCount} file change(s) for ${sessionId}`
      );
      return changeCount;
    } catch (err) {
      outputChannel.appendLine(
        `  Failed to restore file changes for ${sessionId}: ${(err as Error).message}`
      );
      return 0;
    }
  };

  const ensureSessionHydrated = async (sessionId: string) => {
    const t = transport ?? (await ensureTransport());
    await hydrateSessionFromServer(t, sessionId);
  };

  const restoreSessionsFromServer = async (t: PrsmTransport) => {
    let resp: { sessions: SessionInfo[] };
    outputChannel.appendLine("Restoring sessions from server");
    try {
      resp = await t.listSessions();
    } catch (err) {
      outputChannel.appendLine(
        `Failed to list sessions from server: ${(err as Error).message}`
      );
      return;
    }

    const sessions = resp.sessions ?? [];
    outputChannel.appendLine(
      `Restoring ${sessions.length} session(s) from server`
    );
    if (sessions.length === 0) {
      outputChannel.appendLine("No sessions returned from server");
    }

    // Clear store and trackers first so disk-loaded data doesn't duplicate
    store!.clear();
    fileTracker.clear();
    snapshotTreeProvider.clear();
    hydratedSessions.clear();
    outputChannel.appendLine(
      "Cleared local store, file tracker, and snapshot cache before server restore"
    );

    let indexedCount = 0;
    let skippedCount = 0;
    for (const s of sessions) {
      const sessionId = s.sessionId;
      const sessionName = s.name || sessionId || "Untitled";
      if (!sessionId) {
        skippedCount += 1;
        outputChannel.appendLine(`  Skipping session with no ID`);
        continue;
      }
      try {
        store!.restoreSession({
          id: sessionId,
          name: sessionName,
          summary: s.summary ?? null,
          forkedFrom: s.forkedFrom,
          running: s.running ?? false,
          createdAt: s.createdAt ?? null,
          lastActivity: s.lastActivity ?? null,
          currentModel: s.currentModel,
          currentModelDisplay: s.currentModelDisplay,
          agents: [],
          messages: new Map(),
        });
        indexedCount += 1;
        outputChannel.appendLine(`  Indexed session ${sessionId} (${sessionName})`);
      } catch (err) {
        outputChannel.appendLine(
          `  Failed to index session ${sessionId}: ${(err as Error).message}`
        );
      }
    }
    outputChannel.appendLine(
      `Indexed ${indexedCount} session(s) from server; skipped ${skippedCount} without ID`
    );

    snapshotTreeProvider.setSessionLabels(
      store!.getSessions().map((session) => ({
        sessionId: session.id,
        name: session.name,
      }))
    );

    // Eagerly hydrate agent trees so session/agent icons and expand arrows
    // are available without requiring the user to click each session.
    const batchSize = 6;
    for (let i = 0; i < sessions.length; i += batchSize) {
      const batch = sessions.slice(i, i + batchSize);
      outputChannel.appendLine(
        `  Hydrating agent trees for batch ${Math.floor(i / batchSize) + 1}`
      );
      await Promise.all(
        batch.map(async (s) => {
          if (!s.sessionId) return;
          await hydrateSessionTreeFromServer(t, s.sessionId);
        })
      );
    }
    outputChannel.appendLine(
      `Eagerly hydrated agent trees for ${sessions.length} session(s)`
    );

    // Eagerly hydrate full session history (agents + all messages) so
    // extension reload/restart restores complete chat content immediately.
    const historyBatchSize = 3;
    let historyHydrated = 0;
    for (let i = 0; i < sessions.length; i += historyBatchSize) {
      const batch = sessions.slice(i, i + historyBatchSize);
      outputChannel.appendLine(
        `  Hydrating full history for batch ${Math.floor(i / historyBatchSize) + 1}`
      );
      const hydratedBatch = await Promise.all<number>(
        batch.map(async (s) => {
          if (!s.sessionId) return 0;
          await hydrateSessionFromServer(t, s.sessionId);
          return 1;
        })
      );
      historyHydrated += hydratedBatch.reduce<number>((sum, n) => sum + n, 0);
    }
    outputChannel.appendLine(
      `Eagerly hydrated full history for ${historyHydrated} session(s)`
    );

    // Eagerly load file changes for every session so pending edits are visible
    // without requiring the user to click/open each session first.
    let sessionsWithFileChanges = 0;
    let totalFileChanges = 0;
    const fileChangeBatchSize = 6;
    for (let i = 0; i < sessions.length; i += fileChangeBatchSize) {
      const batch = sessions.slice(i, i + fileChangeBatchSize);
      outputChannel.appendLine(
        `  Restoring file changes for batch ${Math.floor(i / fileChangeBatchSize) + 1}`
      );
      const counts = await Promise.all(
        batch.map(async (s) => {
          if (!s.sessionId) return 0;
          return restoreSessionFileChangesFromServer(
            t,
            s.sessionId,
            (agentId) => store!.getAgent(s.sessionId!, agentId)?.name
          );
        })
      );
      for (const count of counts) {
        if (count > 0) {
          sessionsWithFileChanges += 1;
          totalFileChanges += count;
        }
      }
    }
    outputChannel.appendLine(
      `Eagerly restored ${totalFileChanges} file change(s) across ${sessionsWithFileChanges} session(s)`
    );
  };

  /** Create a new transport instance wired to the store. */
  const createTransport = (): PrsmTransport => {
    const config = vscode.workspace.getConfiguration("prsm");
    const executablePath =
      config.get<string>("executablePath") ?? "prsm";
    const sessionInactivityMinutes =
      config.get<number>("engine.sessionInactivityMinutes") ?? 15;
    const cwd = getPreferredWorkspaceRoot() ?? ".";
    const configPath = generateConfigFile();

    const t = new PrsmTransport({
      executablePath,
      cwd,
      outputChannel,
      configPath,
      sessionInactivityMinutes,
    });

    // Wire SSE events to store
    t.on(
      "event",
      (eventType: string, data: Record<string, unknown>) => {
        debugLog(`SSE ${eventType} session=${(data.session_id as string ?? "?").slice(0, 8)} agent=${(data.agent_id as string ?? "").slice(0, 8)}`);
        store!.processEvent(eventType, data);

        // When the server sends the "connected" event, update the snapshot
        // labels so they can be displayed before restoreSessionsFromServer()
        // completes.
        if (eventType === "connected") {
          outputChannel.appendLine(
            `[startup] SSE connected event payload keys=${Object.keys(data).join(",")}`
          );
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
          outputChannel.appendLine(
            `[startup] SSE connected event: received ${sessionList.length} item(s)` +
            ` from ${rawSessionList === data.session_list ? "session_list" : "sessions"}`
          );
          if (sessionList.length > 0) {
            snapshotTreeProvider.setSessionLabels(
              sessionList
                .filter((s) => s.sessionId)
                .map((s) => ({
                  sessionId: s.sessionId as string,
                  name: (s.name as string) || (s.sessionId as string),
                }))
            );
          }
        }
      }
    );

    t.on("terminated", () => {
      clearServerState();
      updateStatusBar("disconnected");
      vscode.window
        .showWarningMessage(
          "PRSM server stopped.",
          "Restart"
        )
        .then((choice) => {
          if (choice === "Restart") {
            vscode.commands.executeCommand("prsm.newSession");
          }
        });
    });

    t.on("error", (err: Error) => {
      outputChannel.appendLine(`Transport error: ${err.message}`);
    });

    return t;
  };

  store.onDidChangeTree(() => {
    fileChangesTreeProvider.refresh();
    snapshotTreeProvider.setSessionLabels(
      store!.getSessions().map((session) => ({
        sessionId: session.id,
        name: session.name,
      }))
    );
  });

  /**
   * Try to reconnect to a previously running server.
   * Returns true if reconnection succeeded, false otherwise.
   */
  const tryReconnect = async (): Promise<boolean> => {
    const savedPort = context.workspaceState.get<number>(STATE_PORT);
    const savedPid = context.workspaceState.get<number>(STATE_PID);
    outputChannel.appendLine(
      `[reconnect] begin: savedPort=${savedPort ?? "none"}, savedPid=${savedPid ?? "none"}`
    );
    if (!savedPort) return false;

    outputChannel.appendLine(
      `Found saved server port ${savedPort} pid=${savedPid ?? "?"}, attempting reconnect...`
    );
    if (savedPid && !isProcessAlive(savedPid)) {
      outputChannel.appendLine(
        `Saved PRSM pid ${savedPid} is not alive; skipping reconnect`
      );
      clearServerState();
      return false;
    }

    const t = createTransport();
    try {
      await t.reconnect(savedPort, savedPid);
      transport = t;
      fileTracker.setTransport(t);
      updateStatusBar("connected");
      outputChannel.appendLine("[reconnect] transport connected to existing server");
      outputChannel.appendLine("Reconnected to existing PRSM server");

      // Restore sessions from server
      await restoreSessionsFromServer(t);
      saveServerState();
      return true;
    } catch (err) {
      outputChannel.appendLine(
        `Reconnect failed: ${(err as Error).message}`
      );
      if (savedPid && isProcessAlive(savedPid)) {
        // Reconnect failure against a live remembered process usually
        // means stale/bad state; terminate before starting clean.
        tryTerminateProcess(savedPid);
      }
      clearServerState();
      return false;
    }
  };

  const ensureTransport = async (): Promise<PrsmTransport> => {
    outputChannel.appendLine("ensureTransport: begin");
    const savedPort = context.workspaceState.get<number>(STATE_PORT);
    const savedPid = context.workspaceState.get<number>(STATE_PID);
    outputChannel.appendLine(
      `[ensureTransport] saved port=${savedPort ?? "none"}, pid=${savedPid ?? "none"}`
    );
    if (transport?.isConnected) {
      const healthy = await transport.healthCheck();
      outputChannel.appendLine(
        `[ensureTransport] existing transport isConnected=${transport.isConnected} healthy=${healthy}`
      );
      if (healthy) return transport;

      outputChannel.appendLine(
        "Existing transport is stale (health check failed); restarting connection..."
      );
      clearServerState();
      transport.disconnect();
      transport = undefined;
      fileTracker.setTransport(null);
    }

    // Try reconnecting to existing server first
    outputChannel.appendLine("ensureTransport: trying reconnect path");
    if (await tryReconnect()) {
      outputChannel.appendLine("ensureTransport: reconnected existing server");
      return transport!;
    }

    // Guard against duplicate detached servers after failed reconnects.
    const stalePid = context.workspaceState.get<number>(STATE_PID);
    if (stalePid && isProcessAlive(stalePid)) {
      tryTerminateProcess(stalePid);
      clearServerState();
    }

    // Start a new server
    outputChannel.appendLine("ensureTransport: starting new server process");
    transport = createTransport();
    fileTracker.setTransport(transport);

    try {
      await transport.start();
      updateStatusBar("connected");
      outputChannel.appendLine(
        `[ensureTransport] new server started on port=${transport.currentPort}`
      );
      outputChannel.appendLine("Connected to PRSM server");
      saveServerState();

      // Restore sessions that the server loaded from disk
      await restoreSessionsFromServer(transport);
    } catch (err) {
      const msg = (err as Error).message;
      vscode.window.showErrorMessage(`Failed to start PRSM: ${msg}`);
      throw err;
    }

    return transport;
  };

  // ── Commands ──

  context.subscriptions.push(
    vscode.commands.registerCommand(
      "prsm.newSession",
      async () => {
        try {
          const t = await ensureTransport();
          const result = await t.createSession();
          outputChannel.appendLine(
            `Created session: ${result.session_id} (${result.name})`
          );
          // Ensure the session is in the store before opening the webview.
          // The SSE "session_created" event may not have arrived yet, so
          // addSession() here prevents showSession() from bailing out.
          store!.addSession({
            id: result.session_id,
            name: result.name,
            summary: result.summary ?? null,
            currentModel: result.current_model,
            currentModelDisplay: result.current_model_display,
          });
          // Auto-open the orchestrator webview
          webviewManager.showSession(result.session_id);
        } catch (err) {
          vscode.window.showErrorMessage(
            `Failed to create session: ${(err as Error).message}`
          );
        }
      }
    ),

    // Open orchestrator conversation (from tree click)
    vscode.commands.registerCommand(
      "prsm.showOrchestratorConversation",
      async (sessionId: string) => {
        try {
          await ensureSessionHydrated(sessionId);
        } catch (err) {
          outputChannel.appendLine(
            `Failed to hydrate session ${sessionId}: ${(err as Error).message}`
          );
        }
        webviewManager.showSession(sessionId);
      }
    ),

    vscode.commands.registerCommand(
      "prsm.copySessionId",
      async (sessionId: string) => {
        if (!sessionId) return;
        await vscode.env.clipboard.writeText(sessionId);
        vscode.window.showInformationMessage(`Copied session UUID: ${sessionId}`);
      }
    ),

    vscode.commands.registerCommand(
      "prsm.runPrompt",
      async (item?: SessionTreeItem) => {
        try {
          const t = await ensureTransport();

          // Determine which session to use
          let sessionId: string;
          if (item instanceof SessionTreeItem) {
            sessionId = item.session.id;
          } else {
            // Pick from existing sessions or create new
            const sessions = store!.getSessions();
            if (sessions.length === 0) {
              const result = await t.createSession();
              store!.addSession({
                id: result.session_id,
                name: result.name,
                summary: result.summary ?? null,
                currentModel: result.current_model,
                currentModelDisplay: result.current_model_display,
              });
              sessionId = result.session_id;
            } else if (sessions.length === 1) {
              sessionId = sessions[0].id;
            } else {
              const picks = sessions.map((s) => ({
                label: s.name,
                description: `${s.agents.size} agents`,
                sessionId: s.id,
              }));
              picks.push({
                label: "$(add) New Session",
                description: "",
                sessionId: "__new__",
              });

              const selected = await vscode.window.showQuickPick(
                picks,
                {
                  title: "Select Session",
                  placeHolder: "Which session to run in?",
                }
              );
              if (!selected) return;

              if (selected.sessionId === "__new__") {
                const result = await t.createSession();
                store!.addSession({
                  id: result.session_id,
                  name: result.name,
                  summary: result.summary ?? null,
                  currentModel: result.current_model,
                  currentModelDisplay: result.current_model_display,
                });
                sessionId = result.session_id;
              } else {
                sessionId = selected.sessionId;
                await ensureSessionHydrated(sessionId);
              }
            }
          }

          const existing = store!.getSession(sessionId);
          if (existing && existing.agents.size === 0) {
            await ensureSessionHydrated(sessionId);
          }

          // Open the webview and let the user type there
          webviewManager.showSession(sessionId);
        } catch (err) {
          vscode.window.showErrorMessage(
            `Failed: ${(err as Error).message}`
          );
        }
      }
    ),

    vscode.commands.registerCommand(
      "prsm.forkSession",
      async (item?: SessionTreeItem) => {
        if (!(item instanceof SessionTreeItem)) return;

        const name = await vscode.window.showInputBox({
          title: "Fork Session",
          prompt: "Name for the forked session",
          value: `Fork of ${item.session.name}`,
        });
        if (name === undefined) return;

        try {
          const t = await ensureTransport();
          const result = await t.forkSession(
            item.session.id,
            name
          );
          outputChannel.appendLine(
            `Forked session: ${result.session_id} (${result.name})`
          );
          store!.addSession({
            id: result.session_id,
            name: result.name,
            summary: result.summary ?? null,
            forkedFrom: result.forked_from,
          });
          webviewManager.showSession(result.session_id);
        } catch (err) {
          vscode.window.showErrorMessage(
            `Failed to fork session: ${(err as Error).message}`
          );
        }
      }
    ),

    vscode.commands.registerCommand(
      "prsm.renameSession",
      async (item?: SessionTreeItem) => {
        if (!(item instanceof SessionTreeItem)) return;

        const newName = await vscode.window.showInputBox({
          title: "Rename Session",
          prompt: "Enter a new name for this session",
          value: item.session.name,
        });
        if (newName === undefined || newName.trim() === "") return;

        try {
          const t = await ensureTransport();
          await t.renameSession(item.session.id, newName.trim());
        } catch (err) {
          vscode.window.showErrorMessage(
            `Failed to rename session: ${(err as Error).message}`
          );
        }
      }
    ),

    vscode.commands.registerCommand(
      "prsm.removeSession",
      async (item?: SessionTreeItem) => {
        if (!(item instanceof SessionTreeItem)) return;

        const confirm = await vscode.window.showWarningMessage(
          `Remove session "${item.session.name}"?`,
          { modal: true },
          "Remove"
        );
        if (confirm !== "Remove") return;

        try {
          const t = await ensureTransport();
          await t.removeSession(item.session.id);
        } catch (err) {
          vscode.window.showErrorMessage(
            `Failed to remove session: ${(err as Error).message}`
          );
        }
      }
    ),

    vscode.commands.registerCommand(
      "prsm.deleteSelected",
      async (
        item?: SessionTreeItem | AgentTreeItem,
        allItems?: (SessionTreeItem | AgentTreeItem)[],
      ) => {
        const items = allItems?.length ? allItems : item ? [item] : [];
        if (items.length === 0) return;

        const sessions = items.filter(
          (i): i is SessionTreeItem => i instanceof SessionTreeItem
        );
        const agents = items.filter(
          (i): i is AgentTreeItem => i instanceof AgentTreeItem
        );

        const parts: string[] = [];
        if (sessions.length === 1) {
          parts.push(`session "${sessions[0].session.name}"`);
        } else if (sessions.length > 1) {
          parts.push(`${sessions.length} sessions`);
        }
        if (agents.length === 1) {
          parts.push(`agent "${agents[0].agent.name}"`);
        } else if (agents.length > 1) {
          parts.push(`${agents.length} agents`);
        }

        const confirm = await vscode.window.showWarningMessage(
          `Delete ${parts.join(" and ")}?`,
          { modal: true },
          "Delete"
        );
        if (confirm !== "Delete") return;

        try {
          const t = await ensureTransport();
          for (const s of sessions) {
            await t.removeSession(s.session.id);
          }
          for (const a of agents) {
            await t.killAgent(a.sessionId, a.agent.id);
          }
        } catch (err) {
          vscode.window.showErrorMessage(
            `Failed to delete: ${(err as Error).message}`
          );
        }
      }
    ),

    vscode.commands.registerCommand(
      "prsm.showAgentConversation",
      async (sessionId: string, agentId: string) => {
        if (!store!.getAgent(sessionId, agentId)) {
          try {
            await ensureSessionHydrated(sessionId);
          } catch (err) {
            outputChannel.appendLine(
              `Failed to hydrate session ${sessionId}: ${(err as Error).message}`
            );
          }
        }
        webviewManager.showAgent(sessionId, agentId);
      }
    ),

    vscode.commands.registerCommand(
      "prsm.killAgent",
      async (item?: AgentTreeItem) => {
        if (!(item instanceof AgentTreeItem)) return;

        try {
          const t = await ensureTransport();
          await t.killAgent(
            item.sessionId,
            item.agent.id
          );
        } catch (err) {
          vscode.window.showErrorMessage(
            `Failed to kill agent: ${(err as Error).message}`
          );
        }
      }
    ),

    vscode.commands.registerCommand(
      "prsm.stopOrchestration",
      async (item?: SessionTreeItem) => {
        if (!(item instanceof SessionTreeItem)) return;

        try {
          const t = await ensureTransport();
          await t.shutdownSession(item.session.id);
        } catch (err) {
          vscode.window.showErrorMessage(
            `Failed to stop: ${(err as Error).message}`
          );
        }
      }
    ),

    vscode.commands.registerCommand("prsm.refreshTree", async () => {
      if (transport?.isConnected) {
        store!.clear();
        await restoreSessionsFromServer(transport);
      }
      treeProvider.refresh();
    }),

    vscode.commands.registerCommand("prsm.searchTree", () => {
      const inputBox = vscode.window.createInputBox();
      inputBox.title = "Search Agent Tree";
      inputBox.placeholder = "Type to filter agents…";
      inputBox.value = treeProvider.searchQuery;

      // Toggle button: switches between name-only and name+chat search
      const makeButton = (
        includeChat: boolean,
      ): vscode.QuickInputButton => ({
        iconPath: new vscode.ThemeIcon(
          includeChat ? "comment-discussion" : "person",
        ),
        tooltip: includeChat
          ? "Searching names + chat content (click to search names only)"
          : "Searching names only (click to include chat content)",
      });

      let includeChatContent = treeProvider.includeChatContent;
      inputBox.buttons = [makeButton(includeChatContent)];

      // Apply filter as user types
      inputBox.onDidChangeValue((value) => {
        treeProvider.setSearchQuery(value);
      });

      // Toggle mode on button click
      inputBox.onDidTriggerButton(() => {
        includeChatContent = !includeChatContent;
        inputBox.buttons = [makeButton(includeChatContent)];
        treeProvider.setIncludeChatContent(includeChatContent);
      });

      // Accept (Enter) — close input but keep filter active
      inputBox.onDidAccept(() => {
        inputBox.dispose();
      });

      // Hide — close input and keep current filter state
      inputBox.onDidHide(() => {
        inputBox.dispose();
      });

      inputBox.show();
    }),

    vscode.commands.registerCommand("prsm.clearTreeSearch", () => {
      treeProvider.clearSearch();
    }),

    // ── Accept/Reject/GoToAgent commands ──

    vscode.commands.registerCommand(
      "prsm.acceptChange",
      (filePathOrItem: string | { filePath?: string; changeIndex?: number; change?: TrackedFileChange }, changeIndex?: number) => {
        if (typeof filePathOrItem === "string" && changeIndex !== undefined) {
          fileTracker.acceptChange(filePathOrItem, changeIndex);
        } else if (typeof filePathOrItem === "object" && filePathOrItem.filePath !== undefined && filePathOrItem.changeIndex !== undefined) {
          fileTracker.acceptChange(filePathOrItem.filePath, filePathOrItem.changeIndex);
        }
      }
    ),

    vscode.commands.registerCommand(
      "prsm.rejectChange",
      async (filePathOrItem: string | { filePath?: string; changeIndex?: number; change?: TrackedFileChange }, changeIndex?: number) => {
        if (typeof filePathOrItem === "string" && changeIndex !== undefined) {
          await fileTracker.rejectChange(filePathOrItem, changeIndex);
        } else if (typeof filePathOrItem === "object" && filePathOrItem.filePath !== undefined && filePathOrItem.changeIndex !== undefined) {
          await fileTracker.rejectChange(filePathOrItem.filePath, filePathOrItem.changeIndex);
        }
      }
    ),

    vscode.commands.registerCommand(
      "prsm.acceptAllFileChanges",
      (filePathOrItem?: string | { filePath?: string }) => {
        let filePath: string | undefined;
        if (typeof filePathOrItem === "string") {
          filePath = filePathOrItem;
        } else if (filePathOrItem && "filePath" in filePathOrItem) {
          filePath = filePathOrItem.filePath;
        } else {
          // Use active editor
          filePath = vscode.window.activeTextEditor?.document.uri.fsPath;
        }
        if (filePath) {
          fileTracker.acceptAll(filePath);
        }
      }
    ),

    vscode.commands.registerCommand(
      "prsm.rejectAllFileChanges",
      async (filePathOrItem?: string | { filePath?: string }) => {
        let filePath: string | undefined;
        if (typeof filePathOrItem === "string") {
          filePath = filePathOrItem;
        } else if (filePathOrItem && "filePath" in filePathOrItem) {
          filePath = filePathOrItem.filePath;
        } else {
          filePath = vscode.window.activeTextEditor?.document.uri.fsPath;
        }
        if (filePath) {
          await fileTracker.rejectAll(filePath);
        }
      }
    ),

    vscode.commands.registerCommand(
      "prsm.acceptAllSessionChanges",
      (item?: { sessionId?: string }) => {
        if (item && item.sessionId) {
          fileTracker.acceptAllForSession(item.sessionId);
        }
      }
    ),

    vscode.commands.registerCommand(
      "prsm.rejectAllSessionChanges",
      async (item?: { sessionId?: string }) => {
        if (item && item.sessionId) {
          await fileTracker.rejectAllForSession(item.sessionId);
        }
      }
    ),

    vscode.commands.registerCommand("prsm.acceptAllChanges", () => {
      fileTracker.acceptAllGlobal();
    }),

    vscode.commands.registerCommand("prsm.rejectAllChanges", async () => {
      const confirm = await vscode.window.showWarningMessage(
        "Reject all agent changes? This will try to revert file modifications.",
        { modal: true },
        "Reject All"
      );
      if (confirm !== "Reject All") return;
      for (const f of fileTracker.getChangedFiles()) {
        await fileTracker.rejectAll(f);
      }
    }),

    // ── View Diff command ──

    vscode.commands.registerCommand(
      "prsm.viewFileDiff",
      async (filePathOrItem?: string | { filePath?: string }, toolCallId?: string) => {
        let filePath: string | undefined;
        let tcId: string | undefined = toolCallId;

        if (typeof filePathOrItem === "string") {
          filePath = filePathOrItem;
        } else if (filePathOrItem && "filePath" in filePathOrItem) {
          filePath = filePathOrItem.filePath;
        } else {
          filePath = vscode.window.activeTextEditor?.document.uri.fsPath;
        }

        if (!filePath) return;

        // Build the original content URI
        let originalUri: vscode.Uri;
        if (tcId) {
          originalUri = OriginalContentProvider.buildUri(filePath, tcId);
        } else {
          originalUri = OriginalContentProvider.buildFileUri(filePath);
        }

        const modifiedUri = vscode.Uri.file(filePath);
        const fileName = path.basename(filePath);
        const title = `${fileName} (before agent edit) ↔ ${fileName} (current)`;

        await vscode.commands.executeCommand("vscode.diff", originalUri, modifiedUri, title);
      }
    ),

    // ── Keyboard navigation between changes ──

    vscode.commands.registerCommand("prsm.nextChange", () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      const filePath = editor.document.uri.fsPath;
      const pending = fileTracker.getPendingChanges(filePath);
      if (pending.length === 0) return;

      const cursorLine = editor.selection.active.line;
      // Find the next change after cursor
      let nextLine: number | undefined;
      for (const change of pending) {
        const ranges = [...change.addedRanges, ...change.removedRanges];
        for (const range of ranges) {
          if (range.startLine > cursorLine) {
            if (nextLine === undefined || range.startLine < nextLine) {
              nextLine = range.startLine;
            }
          }
        }
      }
      // Wrap around to first change
      if (nextLine === undefined && pending.length > 0) {
        const firstRange = pending[0].addedRanges[0] ?? pending[0].removedRanges[0];
        if (firstRange) nextLine = firstRange.startLine;
      }
      if (nextLine !== undefined) {
        const pos = new vscode.Position(nextLine, 0);
        editor.selection = new vscode.Selection(pos, pos);
        editor.revealRange(new vscode.Range(pos, pos), vscode.TextEditorRevealType.InCenter);
      }
    }),

    vscode.commands.registerCommand("prsm.prevChange", () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      const filePath = editor.document.uri.fsPath;
      const pending = fileTracker.getPendingChanges(filePath);
      if (pending.length === 0) return;

      const cursorLine = editor.selection.active.line;
      // Find the previous change before cursor
      let prevLine: number | undefined;
      for (const change of pending) {
        const ranges = [...change.addedRanges, ...change.removedRanges];
        for (const range of ranges) {
          if (range.startLine < cursorLine) {
            if (prevLine === undefined || range.startLine > prevLine) {
              prevLine = range.startLine;
            }
          }
        }
      }
      // Wrap around to last change
      if (prevLine === undefined && pending.length > 0) {
        const lastChange = pending[pending.length - 1];
        const lastRange = lastChange.addedRanges[lastChange.addedRanges.length - 1] ?? lastChange.removedRanges[lastChange.removedRanges.length - 1];
        if (lastRange) prevLine = lastRange.startLine;
      }
      if (prevLine !== undefined) {
        const pos = new vscode.Position(prevLine, 0);
        editor.selection = new vscode.Selection(pos, pos);
        editor.revealRange(new vscode.Range(pos, pos), vscode.TextEditorRevealType.InCenter);
      }
    }),

    vscode.commands.registerCommand("prsm.acceptCurrentChange", () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      const filePath = editor.document.uri.fsPath;
      const allChanges = fileTracker.getAllChanges(filePath);
      const cursorLine = editor.selection.active.line;

      for (let i = 0; i < allChanges.length; i++) {
        const change = allChanges[i];
        if (change.status !== "pending") continue;
        const ranges = [...change.addedRanges, ...change.removedRanges];
        for (const range of ranges) {
          if (cursorLine >= range.startLine && cursorLine <= range.endLine) {
            fileTracker.acceptChange(filePath, i);
            return;
          }
        }
      }
    }),

    vscode.commands.registerCommand("prsm.rejectCurrentChange", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      const filePath = editor.document.uri.fsPath;
      const allChanges = fileTracker.getAllChanges(filePath);
      const cursorLine = editor.selection.active.line;

      for (let i = 0; i < allChanges.length; i++) {
        const change = allChanges[i];
        if (change.status !== "pending") continue;
        const ranges = [...change.addedRanges, ...change.removedRanges];
        for (const range of ranges) {
          if (cursorLine >= range.startLine && cursorLine <= range.endLine) {
            await fileTracker.rejectChange(filePath, i);
            return;
          }
        }
      }
    }),

    vscode.commands.registerCommand(
      "prsm.inspectChange",
      async (
        filePathOrItem:
          | string
          | {
              filePath?: string;
              changeIndex?: number;
              change?: TrackedFileChange;
              sessionId?: string;
              agentId?: string;
              toolCallId?: string;
              messageIndex?: number;
            },
        changeIndex?: number
      ) => {
        if (typeof filePathOrItem === "object" && filePathOrItem?.change) {
          await vscode.commands.executeCommand("prsm.goToAgent", {
            change: filePathOrItem.change,
          });
          return;
        }

        if (typeof filePathOrItem === "object" && filePathOrItem?.toolCallId) {
          await vscode.commands.executeCommand("prsm.goToAgent", {
            sessionId: filePathOrItem.sessionId,
            agentId: filePathOrItem.agentId,
            toolCallId: filePathOrItem.toolCallId,
            messageIndex: filePathOrItem.messageIndex,
          });
          return;
        }

        let filePath: string | undefined;
        let idx: number | undefined = changeIndex;

        if (typeof filePathOrItem === "string") {
          filePath = filePathOrItem;
        } else if (typeof filePathOrItem === "object") {
          filePath = filePathOrItem.filePath;
          idx = filePathOrItem.changeIndex ?? idx;
        }

        if (!filePath || idx === undefined) return;
        const change = fileTracker.getAllChanges(filePath)[idx];
        if (!change) return;

        await vscode.commands.executeCommand("prsm.goToAgent", {
          sessionId: change.sessionId,
          agentId: change.agentId,
          toolCallId: change.toolCallId,
          messageIndex: change.messageIndex,
        });
      }
    ),

    vscode.commands.registerCommand("prsm.inspectCurrentChange", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;
      const filePath = editor.document.uri.fsPath;
      const allChanges = fileTracker.getAllChanges(filePath);
      const cursorLine = editor.selection.active.line;

      for (let i = 0; i < allChanges.length; i++) {
        const change = allChanges[i];
        if (change.status !== "pending") continue;
        const ranges = [...change.addedRanges, ...change.removedRanges];
        for (const range of ranges) {
          if (cursorLine >= range.startLine && cursorLine <= range.endLine) {
            await vscode.commands.executeCommand("prsm.inspectChange", filePath, i);
            return;
          }
        }
      }
    }),

    vscode.commands.registerCommand(
      "prsm.goToAgent",
      async (args: {
        sessionId?: string;
        agentId?: string;
        toolCallId?: string;
        messageIndex?: number;
        change?: TrackedFileChange;
      }) => {
        // Handle both direct args and tree item with change property
        // (context menu passes the ChangeTreeItem which has a .change)
        let sessionId: string | undefined;
        let agentId: string | undefined;
        let toolCallId: string | undefined;
        let messageIndex: number | undefined;

        if (args.change) {
          sessionId = args.change.sessionId;
          agentId = args.change.agentId;
          toolCallId = args.change.toolCallId;
          messageIndex = args.change.messageIndex;
        } else {
          sessionId = args.sessionId;
          agentId = args.agentId;
          toolCallId = args.toolCallId;
          messageIndex = args.messageIndex;
        }

        if (!sessionId || !agentId) return;
        if (!store!.getAgent(sessionId, agentId)) {
          await ensureSessionHydrated(sessionId);
        }
        const targetSessionId = sessionId;
        const targetAgentId = agentId;

        const targetAgent = store!.getAgent(targetSessionId, targetAgentId);
        const masterAgent = store!.getMasterAgent(targetSessionId);
        const useSessionChat =
          targetAgent?.role === "orchestrator" &&
          masterAgent?.id === targetAgentId;

        if (useSessionChat) {
          webviewManager.showSession(targetSessionId);
        } else {
          webviewManager.showAgent(targetSessionId, targetAgentId);
        }

        // Post multiple jump attempts so new panels and freshly-hydrated content
        // reliably receive and apply the jump request.
        if (toolCallId || Number.isInteger(messageIndex)) {
          const jump = () => {
            webviewManager.scrollToToolCall(
              targetSessionId,
              targetAgentId,
              toolCallId ?? "",
              messageIndex,
              useSessionChat,
            );
          };
          jump();
          setTimeout(jump, 350);
          setTimeout(jump, 900);
        }
      }
    ),

    vscode.commands.registerCommand(
      "prsm.goToSnapshot",
      async (args: {
        sessionId: string;
        snapshotId: string;
        agentId?: string | null;
      }) => {
        const { sessionId, snapshotId, agentId } = args;
        if (!sessionId || !snapshotId) return;

        if (!store!.getSession(sessionId)) {
          await ensureSessionHydrated(sessionId);
        }

        const targetAgentId = agentId || store!.getMasterAgent(sessionId)?.id;
        if (!targetAgentId) return;

        const targetAgent = store!.getAgent(sessionId, targetAgentId);
        const masterAgent = store!.getMasterAgent(sessionId);
        const useSessionChat =
          targetAgent?.role === "orchestrator" &&
          masterAgent?.id === targetAgentId;

        if (useSessionChat) {
          webviewManager.showSession(sessionId);
        } else {
          webviewManager.showAgent(sessionId, targetAgentId);
        }

        // Post multiple jump attempts so new panels and freshly-hydrated content
        // reliably receive and apply the jump request.
        const jump = () => {
          webviewManager.scrollToSnapshot(
            sessionId,
            targetAgentId,
            snapshotId,
            useSessionChat,
          );
        };
        jump();
        setTimeout(jump, 350);
        setTimeout(jump, 900);
      }
    ),

    // ── Snapshot commands ──

    vscode.commands.registerCommand(
      "prsm.createSnapshot",
      async () => {
        if (!transport?.isConnected) return;
        const sessions = store!.getSessions();
        if (sessions.length === 0) return;

        const sessionId = sessions[0].id;
        const desc = await vscode.window.showInputBox({
          title: "Create Snapshot",
          prompt: "Description (optional)",
          placeHolder: "e.g., Before refactoring auth",
        });
        if (desc === undefined) return;

        try {
          const result = await transport.createSnapshot(sessionId, desc);
          vscode.window.showInformationMessage(
            `Snapshot created: ${result.snapshot_id}`
          );
        } catch (err) {
          vscode.window.showErrorMessage(
            `Failed to create snapshot: ${(err as Error).message}`
          );
        }
      }
    ),

    vscode.commands.registerCommand(
      "prsm.restoreSnapshot",
      async (item?: { snapshot?: { snapshot_id: string }; sessionId?: string }) => {
        if (!transport?.isConnected) return;

        const sessions = store!.getSessions();
        if (sessions.length === 0) return;
        const sessionId = item?.sessionId ?? sessions[0].id;
        const snapshotId = item?.snapshot?.snapshot_id;

        if (snapshotId) {
          const confirm = await vscode.window.showWarningMessage(
            "Restore this snapshot? Current changes will be reverted.",
            { modal: true },
            "Restore"
          );
          if (confirm !== "Restore") return;

          try {
            await transport.restoreSnapshot(sessionId, snapshotId);
            fileTracker.clear();
            vscode.window.showInformationMessage("Snapshot restored.");
          } catch (err) {
            vscode.window.showErrorMessage(
              `Failed to restore: ${(err as Error).message}`
            );
          }
        }
      }
    ),

    vscode.commands.registerCommand(
      "prsm.forkSnapshot",
      async (item?: { snapshot?: { snapshot_id: string; description?: string; session_name?: string }; sessionId?: string }) => {
        if (!transport?.isConnected) return;

        const sessions = store!.getSessions();
        if (sessions.length === 0) return;
        const sessionId = item?.sessionId ?? sessions[0].id;
        const snapshotId = item?.snapshot?.snapshot_id;
        if (!snapshotId) return;

        const baseName =
          item?.snapshot?.session_name ||
          store!.getSession(sessionId)?.name ||
          item?.snapshot?.description ||
          "Snapshot";
        const defaultName = baseName.startsWith("(Forked) ")
          ? baseName
          : `(Forked) ${baseName}`;

        const nameInput = await vscode.window.showInputBox({
          title: "Fork Snapshot",
          prompt: "Name for the forked session",
          value: defaultName,
        });
        if (nameInput === undefined) return;
        const trimmed = nameInput.trim();

        try {
          const result = await transport.forkSnapshot(
            sessionId,
            snapshotId,
            trimmed || undefined
          );
          vscode.window.showInformationMessage(
            `Forked snapshot into ${result.name}`
          );
        } catch (err) {
          vscode.window.showErrorMessage(
            `Failed to fork snapshot: ${(err as Error).message}`
          );
        }
      }
    ),

    vscode.commands.registerCommand(
      "prsm.deleteSnapshot",
      async (item?: { snapshot?: { snapshot_id: string }; sessionId?: string }) => {
        if (!transport?.isConnected) return;
        const sessionId = item?.sessionId;
        const snapshotId = item?.snapshot?.snapshot_id;
        if (!sessionId || !snapshotId) return;

        try {
          await transport.deleteSnapshot(sessionId, snapshotId);
        } catch (err) {
          vscode.window.showErrorMessage(
            `Failed to delete snapshot: ${(err as Error).message}`
          );
        }
      }
    ),

    vscode.commands.registerCommand(
      "prsm.showAgentContext",
      async (
        args:
          | {
              sessionId: string;
              agentId: string;
              toolCallId?: string;
              messageIndex?: number;
            }
          | { change?: TrackedFileChange }
      ) => {
        // Handle both direct args and tree item with change property
        let sessionId: string;
        let agentId: string;
        let toolCallId: string | undefined;
        let messageIndex: number | undefined;

        if ("change" in args && args.change) {
          // Called from tree view with ChangeTreeItem
          sessionId = args.change.sessionId;
          agentId = args.change.agentId;
          toolCallId = args.change.toolCallId;
          messageIndex = args.change.messageIndex;
        } else if ("sessionId" in args && "agentId" in args) {
          // Called directly with explicit args
          sessionId = args.sessionId;
          agentId = args.agentId;
          toolCallId = args.toolCallId;
          messageIndex = args.messageIndex;
        } else {
          vscode.window.showWarningMessage("Invalid arguments for showAgentContext.");
          return;
        }

        const agent = store!.getAgent(sessionId, agentId);
        if (!agent) {
          await ensureSessionHydrated(sessionId);
        }
        const hydratedAgent = store!.getAgent(sessionId, agentId);
        if (!hydratedAgent) {
          vscode.window.showWarningMessage("Agent not found.");
          return;
        }

        // Open the agent context side panel
        await contextPanel.show({
          sessionId,
          agentId,
          toolCallId,
          messageIndex,
        });
      }
    ),

    vscode.commands.registerCommand(
      "prsm.showAgentContextFromTree",
      async (item?: { change?: TrackedFileChange }) => {
        const change = item?.change;
        if (!change) return;

        const agent = store!.getAgent(change.sessionId, change.agentId);
        if (!agent) {
          await ensureSessionHydrated(change.sessionId);
        }
        const hydratedAgent = store!.getAgent(change.sessionId, change.agentId);
        if (!hydratedAgent) {
          vscode.window.showWarningMessage("Agent not found.");
          return;
        }

        // Open the agent context side panel
        await contextPanel.show({
          sessionId: change.sessionId,
          agentId: change.agentId,
          toolCallId: change.toolCallId,
          messageIndex: change.messageIndex,
        });
      }
    ),

    vscode.commands.registerCommand(
      "prsm.saveSession",
      async () => {
        if (!transport?.isConnected) return;
        const sessions = store!.getSessions();
        if (sessions.length === 0) return;
        try {
          await transport.saveSession(sessions[0].id);
          vscode.window.showInformationMessage("Session saved.");
        } catch (err) {
          vscode.window.showErrorMessage(
            `Failed to save: ${(err as Error).message}`
          );
        }
      }
    )
  );

  // Permission and question handlers
  context.subscriptions.push(
    registerPermissionHandler(store, webviewManager),
    registerQuestionHandler(store, webviewManager)
  );

  // Status bar
  statusItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Left,
    100
  );
  statusItem.command = "prsm.newSession";
  updateStatusBar("disconnected");
  statusItem.show();
  context.subscriptions.push(statusItem);

  // Model Status Bar Item
  modelStatusItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Left,
    98 // Lower priority than main status, higher than worktree
  );
  modelStatusItem.command = "prsm.selectModel";
  modelStatusItem.text = "$(light-bulb) Model: None";
  modelStatusItem.tooltip = "Click to select model for current session";
  modelStatusItem.hide(); // Hidden by default, shown when a session is active
  context.subscriptions.push(modelStatusItem);

  // Register command for model selection from status bar
  context.subscriptions.push(
    vscode.commands.registerCommand("prsm.selectModel", async () => {
      const activeChat = webviewManager.getActiveChatKey();
      if (!activeChat) {
        vscode.window.showInformationMessage("No active session to select a model for.");
        return;
      }
      const parts = activeChat.split(":");
      const sessionId = parts[1]; // activeChat is like "session:<sessionId>" or "agent:<sessionId>:<agentId>"
      if (sessionId) {
        await webviewManager.handleModelSelection(sessionId);
      }
    })
  );

  // Listen for active chat changes to update model status bar item
  let currentActiveSessionId: string | undefined;
  const updateModelStatusBar = () => {
    if (modelStatusItem) {
      if (currentActiveSessionId) {
        const session = store!.getSession(currentActiveSessionId);
        if (session?.currentModel) {
          const label = toModelAliasLabel(session.currentModel) || session.currentModel;
          modelStatusItem.text = `$(light-bulb) Model: ${label}`;
          modelStatusItem.tooltip = `Current model for ${session.name}. Click to change.`;
          modelStatusItem.show();
        } else {
          modelStatusItem.hide();
        }
      } else {
        modelStatusItem.hide();
      }
    }
  };

  context.subscriptions.push(
    webviewManager.onDidChangeActiveChat((target) => {
      if (target?.kind === "session") {
        currentActiveSessionId = target.sessionId;
      } else if (target?.kind === "agent") {
        currentActiveSessionId = target.sessionId;
      } else {
        currentActiveSessionId = undefined;
      }
      updateModelStatusBar();
    })
  );

  // Listen for model switched events to update model status bar item if it's the active session
  context.subscriptions.push(
    store.onModelSwitched((data) => {
      if (data.session_id === currentActiveSessionId) {
        updateModelStatusBar();
      }
    })
  );

  // Initial update
  updateModelStatusBar();

  // Engine lifecycle status updates
  store.onEngineStarted(() => updateStatusBar("running"));
  store.onEngineFinished((data) => {
    updateStatusBar(data.success ? "done" : "error");
    setTimeout(() => updateStatusBar("connected"), 5000);
  });

  // Workspace setup command
  context.subscriptions.push(
    vscode.commands.registerCommand("prsm.setupWorkspace", async () => {
      const workspaceRoot =
        vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
      if (!workspaceRoot) {
        vscode.window.showWarningMessage(
          "Open a workspace folder first."
        );
        return;
      }
      const wsConfig = vscode.workspace.getConfiguration("prsm");
      await performSetup(workspaceRoot, wsConfig);
      vscode.window.showInformationMessage(
        "PRSM workspace setup complete."
      );
    })
  );

  // Expert management commands
  registerExpertCommands(context);

  // ── Git Worktree support ──

  const workspaceRoot =
    vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  const worktreeManager = workspaceRoot
    ? new WorktreeManager(workspaceRoot, outputChannel)
    : null;

  if (worktreeManager) {
    context.subscriptions.push(worktreeManager);

    // Register worktree commands
    registerWorktreeCommands(context, worktreeManager);

    // Worktree status bar item (shows branch + worktree indicator)
    const worktreeStatusItem = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Left,
      99 // Just to the right of the main PRSM status bar item
    );
    worktreeStatusItem.command = "prsm.worktree.list";
    worktreeStatusItem.tooltip = "PRSM: Click to list worktrees";
    context.subscriptions.push(worktreeStatusItem);

    // Toggle worktree filter command
    context.subscriptions.push(
      vscode.commands.registerCommand(
        "prsm.worktree.toggleFilter",
        () => {
          const current = store!.isFilteringByWorktree();
          store!.setFilterByWorktree(!current);
          const label = !current
            ? "Showing sessions from current worktree"
            : "Showing sessions from all worktrees";
          vscode.window.showInformationMessage(label);
        }
      )
    );

    // Detect worktree and update UI
    const updateWorktreeUI = async () => {
      try {
        const ctx = await worktreeManager.refresh();
        if (ctx.isGitRepo) {
          const branchLabel = ctx.branch ?? "detached";
          if (ctx.isWorktree) {
            worktreeStatusItem.text = `$(git-branch) ${branchLabel} $(link)`;
            worktreeStatusItem.tooltip =
              `PRSM Worktree: ${branchLabel}\n` +
              `Path: ${ctx.worktreePath}\n` +
              `Main: ${ctx.mainWorktreePath}\n` +
              `Click to list worktrees`;
          } else {
            worktreeStatusItem.text = `$(git-branch) ${branchLabel}`;
            worktreeStatusItem.tooltip =
              `PRSM Branch: ${branchLabel}\nClick to list worktrees`;
          }
          worktreeStatusItem.show();

          // Stamp worktree info on session store
          store!.setWorktreeInfo({
            branch: ctx.branch,
            worktreePath: ctx.worktreePath,
            isWorktree: ctx.isWorktree,
          });

          // Set context for menu when clauses
          vscode.commands.executeCommand(
            "setContext",
            "prsm.isGitWorktree",
            ctx.isWorktree
          );
          vscode.commands.executeCommand(
            "setContext",
            "prsm.isGitRepo",
            true
          );
        } else {
          worktreeStatusItem.hide();
          store!.setWorktreeInfo(null);
          vscode.commands.executeCommand(
            "setContext",
            "prsm.isGitRepo",
            false
          );
          vscode.commands.executeCommand(
            "setContext",
            "prsm.isGitWorktree",
            false
          );
        }
      } catch (err) {
        outputChannel.appendLine(
          `Worktree detection failed: ${(err as Error).message}`
        );
      }
    };

    // Initial detection
    updateWorktreeUI();

    // Re-detect when workspace folders change (e.g., after switching worktree)
    context.subscriptions.push(
      vscode.workspace.onDidChangeWorkspaceFolders(() => {
        updateWorktreeUI();
      })
    );

    // Re-detect periodically to catch branch changes from external git ops
    const worktreeTimer = setInterval(() => updateWorktreeUI(), 30000);
    context.subscriptions.push({
      dispose: () => clearInterval(worktreeTimer),
    });
  }

  // Open settings command — shows rich webview settings panel
  context.subscriptions.push(
    vscode.commands.registerCommand("prsm.openSettings", async () => {
      try {
        await ensureTransport();
      } catch {
        // Settings panel will show "not connected" fallback
      }
      SettingsPanel.createOrShow(context, getTransport, outputChannel);
    })
  );

  // Archive import / export commands
  outputChannel.appendLine("[startup] registering archive command handlers");
  context.subscriptions.push(
    vscode.commands.registerCommand(
      "prsm.importSessionArchive",
      async () => {
        const uris = await vscode.window.showOpenDialog({
          canSelectFiles: true,
          canSelectFolders: false,
          canSelectMany: false,
          openLabel: "Import Archive",
          title: "Select PRSM Session Archive",
          filters: {
            "Session Archives": ["tar.gz", "tgz", "zip", "tar"],
            "All Files": ["*"],
          },
        });
        if (!uris || uris.length === 0) return;

        const archivePath = uris[0].fsPath;
        const conflictChoice = await vscode.window.showQuickPick(
          [
            { label: "Skip existing", description: "Do not overwrite sessions that already exist", value: "skip" as const },
            { label: "Overwrite", description: "Replace existing sessions with imported data", value: "overwrite" as const },
            { label: "Rename", description: "Import with a renamed suffix to avoid conflicts", value: "rename" as const },
          ],
          { placeHolder: "How should conflicts be handled?", title: "Import Conflict Mode" }
        );
        if (!conflictChoice) return;

        try {
          const t = await ensureTransport();
          const result = await t.importArchive(archivePath, conflictChoice.value);
          if (result.success) {
            const parts: string[] = [];
            if (result.sessions_imported) parts.push(`${result.sessions_imported} session(s)`);
            if (result.files_imported) parts.push(`${result.files_imported} file(s)`);
            let msg = `Imported ${parts.join(", ") || "no new files"}`;
            if (result.sessions_skipped) msg += ` (${result.sessions_skipped} already existed)`;
            vscode.window.showInformationMessage(msg);
            outputChannel.appendLine(`Archive import: ${msg}`);
            // Refresh session list
            if (store) {
              try {
                const sessResp = await t.listSessions();
                for (const s of sessResp.sessions) {
                  store.addSession({
                    id: s.sessionId,
                    name: s.name,
                    summary: s.summary ?? null,
                    forkedFrom: s.forkedFrom,
                  });
                }
              } catch {
                // Non-critical
              }
            }
          } else {
            vscode.window.showErrorMessage(`Import failed: ${result.error || "Unknown error"}`);
          }
        } catch (err) {
          vscode.window.showErrorMessage(`Import failed: ${(err as Error).message}`);
        }
      }
    ),

    vscode.commands.registerCommand(
      "prsm.exportSessions",
      async () => {
        const saveUri = await vscode.window.showSaveDialog({
          defaultUri: vscode.Uri.file(
            path.join(os.homedir(), "Downloads", `prsm_export_${Date.now()}.tar.gz`)
          ),
          filters: {
            "Tar Archive": ["tar.gz"],
            "ZIP Archive": ["zip"],
          },
          title: "Export Sessions to Archive",
        });
        if (!saveUri) return;

        const outputPath = saveUri.fsPath;
        const format = outputPath.endsWith(".zip") ? "zip" : "tar.gz";

        try {
          const t = await ensureTransport();
          const result = await t.exportArchive("", outputPath, format as "tar.gz" | "zip");
          if (result.success) {
            const manifest = result.manifest as Record<string, unknown> | null;
            const count = manifest?.session_count ?? "unknown";
            const msg = `Exported ${count} session(s) to ${path.basename(outputPath)}`;
            vscode.window.showInformationMessage(msg);
            outputChannel.appendLine(`Archive export: ${msg} → ${outputPath}`);
          } else {
            vscode.window.showErrorMessage(`Export failed: ${result.error || "Unknown error"}`);
          }
        } catch (err) {
          vscode.window.showErrorMessage(`Export failed: ${(err as Error).message}`);
        }
      }
    )
  );

  // Config change listener — prompt restart when settings change
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("prsm") && transport?.isConnected) {
        vscode.window
          .showInformationMessage(
            "PRSM settings changed. Restart server to apply?",
            "Restart",
            "Later"
          )
          .then(async (choice) => {
            if (choice === "Restart") {
              await transport?.stop();
              transport = undefined;
              fileTracker.setTransport(null);
              clearServerState();
              updateStatusBar("disconnected");
              await ensureTransport();
            }
          });
      }
    })
  );

  // Always connect to server in the background so commands work.
  // ensureTransport() tries reconnecting first, then starts a new server.
  const shouldAutoStart = vscode.workspace
    .getConfiguration("prsm")
    .get<boolean>("autoStart", true);

  // Populate UI from local cache first, so extension restores are visible immediately.
  const loadedSessionCount = loadSessionsFromDisk();
  outputChannel.appendLine(`[startup] loadSessionsFromDisk returned ${loadedSessionCount}`);

  const hasCachedSessions = loadedSessionCount > 0;
  const hasSavedServerState = Boolean(context.workspaceState.get<number>(STATE_PORT));
  const shouldHydrateOnStartup =
    shouldAutoStart || hasCachedSessions || hasSavedServerState;
  outputChannel.appendLine(
    `[startup] shouldAutoStart=${shouldAutoStart}, hasCachedSessions=${hasCachedSessions}, hasSavedServerState=${hasSavedServerState}, shouldHydrateOnStartup=${shouldHydrateOnStartup}`
  );
  if (shouldHydrateOnStartup) {
    ensureTransport().catch((err) => {
      outputChannel.appendLine(
        `Startup transport bootstrap failed: ${(err as Error).message}`
      );
      // Already showed error in ensureTransport
    });
  } else {
    outputChannel.appendLine(
      "Auto-start disabled and no cached sessions detected; start a session to connect."
    );
  }

  notifyIfUpdateAvailable(outputChannel).catch(() => {
    // Non-blocking: never fail activation.
  });

  // Check workspace setup (async, won't block activation)
  checkWorkspaceSetup(context).catch(() => {
    // Non-critical — log to output channel
  });
}

function updateStatusBar(
  state: "disconnected" | "connected" | "running" | "done" | "error"
): void {
  if (!statusItem) return;

  switch (state) {
    case "disconnected":
      statusItem.text = "$(circle-outline) PRSM";
      statusItem.backgroundColor = undefined;
      statusItem.tooltip = "Click to start PRSM";
      break;
    case "connected":
      statusItem.text = "$(check) PRSM";
      statusItem.backgroundColor = undefined;
      statusItem.tooltip = "PRSM server running";
      break;
    case "running":
      statusItem.text = "$(sync~spin) PRSM";
      statusItem.backgroundColor = new vscode.ThemeColor(
        "statusBarItem.warningBackground"
      );
      statusItem.tooltip = "Orchestration in progress";
      break;
    case "done":
      statusItem.text = "$(pass-filled) PRSM";
      statusItem.backgroundColor = undefined;
      statusItem.tooltip = "Orchestration complete";
      break;
    case "error":
      statusItem.text = "$(error) PRSM";
      statusItem.backgroundColor = new vscode.ThemeColor(
        "statusBarItem.errorBackground"
      );
      statusItem.tooltip = "Orchestration failed";
      break;
  }
}

export function deactivate(): Thenable<void> | undefined {
  // Disconnect SSE but DON'T kill the server — it's detached
  // and will be reconnected on next activation
  if (transport) {
    transport.disconnect();
    transport = undefined;
  }
  return undefined;
}
