/**
 * TreeDataProvider showing orchestrator sessions at the top level
 * and child agents nested within.
 *
 * Tree structure:
 *   Orchestrator "Refactor auth"  (session, styled as agent)
 *     > Code Explorer [worker]
 *     > Test Runner [worker]
 *   Orchestrator "Fix bugs"
 *     > ...
 */
import * as vscode from "vscode";
import { SessionStore, SessionState } from "../state/sessionStore";
import { AgentNode, AgentState } from "../protocol/types";
import { toModelAliasLabel } from "../utils/modelLabel";

const STATIC_STATE_ICONS: Record<
  AgentState,
  { icon: string; color: string }
> = {
  running: { icon: "sync~spin", color: "testing.iconPassed" },
  waiting: { icon: "watch", color: "list.warningForeground" },
  waiting_for_child: { icon: "circle-outline", color: "list.warningForeground" },
  completed: { icon: "pass-filled", color: "testing.iconPassed" },
  error: { icon: "error", color: "testing.iconFailed" },
  idle: { icon: "circle-outline", color: "disabledForeground" },
};

const ACTIVE_SPIN_ICON = "sync~spin";
const ACTIVE_SPIN_COLOR = "testing.iconPassed";
const WAITING_CHILD_BLINK_ICONS = ["primitive-dot", "circle-outline"] as const;
const WAITING_CHILD_COLOR = "list.warningForeground";

function isWaitingForChild(agent: AgentNode | undefined): boolean {
  if (!agent) return false;
  return agent.state === "waiting_for_child" || agent.rawState === "waiting_for_child";
}

function isActiveWorkState(agent: AgentNode | undefined): boolean {
  if (!agent || isWaitingForChild(agent)) return false;
  if (agent.state === "running" || agent.state === "waiting") {
    return true;
  }
  // Fall back to rawState for all engine states that represent active work.
  // This covers edge cases where mapState() hasn't been called (e.g. direct
  // REST restore) and ensures the spinning icon shows for all active agents
  // regardless of where they sit in the hierarchy.
  const raw = agent.rawState;
  return (
    raw === "pending"
    || raw === "starting"
    || raw === "running"
    || raw === "waiting_for_parent"
    || raw === "waiting_for_expert"
  );
}

/**
 * Format an ISO timestamp for display in the tree.
 * - If within the last 24 hours: show relative time like "3h 20m ago"
 * - Otherwise: show date and time like "Feb 14, 2:30 PM"
 */
function formatTimestamp(isoString: string | null | undefined): string {
  if (!isoString) return "";
  const date = new Date(isoString);
  if (isNaN(date.getTime())) return "";

  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);

  // Within last 24 hours: show relative time
  if (diffMs >= 0 && diffHours < 24) {
    if (diffMins < 1) return "just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    const remainMins = diffMins % 60;
    return remainMins > 0 ? `${diffHours}h ${remainMins}m ago` : `${diffHours}h ago`;
  }

  // Older: show date + time
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const month = months[date.getMonth()];
  const day = date.getDate();
  let hours = date.getHours();
  const mins = date.getMinutes().toString().padStart(2, "0");
  const ampm = hours >= 12 ? "PM" : "AM";
  hours = hours % 12 || 12;
  return `${month} ${day}, ${hours}:${mins} ${ampm}`;
}

function sessionIdCopyLink(sessionId: string): string {
  const args = encodeURIComponent(JSON.stringify([sessionId]));
  return `[${sessionId} $(copy)](command:prsm.copySessionId?${args})`;
}

/** Union type for tree items: either a session or an agent. */
export type TreeItemNode = SessionTreeItem | AgentTreeItem;

/**
 * Session node styled as an orchestrator agent.
 * Clicking opens the orchestrator conversation webview.
 */
export class SessionTreeItem extends vscode.TreeItem {
  constructor(
    public readonly session: SessionState,
    store?: SessionStore,
    waitingChildBlinkOn = true,
  ) {
    // Use the most recently spawned root orchestrator (last in insertion order)
    const rootOrchestrators = Array.from(session.agents.values()).filter(
      (a) => (a.role === "orchestrator") && (a.parentId === null || a.parentId === undefined)
    );
    const master = rootOrchestrators[rootOrchestrators.length - 1];
    const masterModel = toModelAliasLabel(session.currentModel || master?.model || "") || "";
    const label = `${session.name || session.id || "Untitled"}`;
    // Count children across ALL root orchestrators (follow-up runs add more)
    const childCount = rootOrchestrators.reduce(
      (sum, orch) => sum + orch.childrenIds.length, 0
    );
    // Determine collapse state: respect user's manual collapse/expand choice
    const collapsibleState = childCount > 0
      ? (store?.isCollapsed(session.id)
          ? vscode.TreeItemCollapsibleState.Collapsed
          : vscode.TreeItemCollapsibleState.Expanded)
      : vscode.TreeItemCollapsibleState.None;
    super(label, collapsibleState);
    this.id = `session:${session.id}`;
    this.contextValue = "prsmSession";

    // Branch suffix for description
    const branchSuffix = session.worktree?.branch
      ? ` \u00b7 ${session.worktree.branch}`
      : "";

    // Timestamp suffix showing last activity
    const tsText = formatTimestamp(session.lastActivity);
    const tsSuffix = tsText ? ` \u00b7 ${tsText}` : "";

    // Style based on session/master state
    if (isWaitingForChild(master)) {
      this.iconPath = new vscode.ThemeIcon(
        WAITING_CHILD_BLINK_ICONS[waitingChildBlinkOn ? 0 : 1],
        new vscode.ThemeColor(WAITING_CHILD_COLOR),
      );
      const modelSuffix = masterModel ? ` · ${masterModel}` : "";
      this.description = `waiting for child${modelSuffix}${branchSuffix}`;
    } else if (session.running || isActiveWorkState(master)) {
      this.iconPath = new vscode.ThemeIcon(
        ACTIVE_SPIN_ICON,
        new vscode.ThemeColor(ACTIVE_SPIN_COLOR)
      );
      const modelSuffix = masterModel ? ` · ${masterModel}` : "";
      this.description = `running${modelSuffix}${branchSuffix}`;
    } else if (master?.state === "completed") {
      this.iconPath = new vscode.ThemeIcon(
        "pass-filled",
        new vscode.ThemeColor("testing.iconPassed")
      );
      const modelSuffix = masterModel ? ` · ${masterModel}` : "";
      this.description = `completed${modelSuffix}${tsSuffix}${branchSuffix}`;
    } else if (master?.state === "error") {
      this.iconPath = new vscode.ThemeIcon(
        "error",
        new vscode.ThemeColor("testing.iconFailed")
      );
      const modelSuffix = masterModel ? ` · ${masterModel}` : "";
      this.description = `error${modelSuffix}${tsSuffix}${branchSuffix}`;
    } else {
      this.iconPath = new vscode.ThemeIcon(
        "circle-outline",
        new vscode.ThemeColor("disabledForeground")
      );
      const countText = childCount > 0 ? `${childCount} agents` : "";
      const modelText = masterModel ? ` · ${masterModel}` : "";
      const baseText = countText
        ? `${countText}${tsSuffix}`
        : tsText;
      const idleParts = [
        baseText,
        modelText ? modelText.slice(3) : "",
        branchSuffix ? branchSuffix.slice(3) : "",
      ].filter((part) => part.length > 0);
      this.description = idleParts.join(" · ");
    }

    // Build tooltip with worktree info
    let tooltipText =
      `**${session.name}**\n\n` +
      `- Agents: ${session.agents.size}\n` +
      `- Running: ${session.running}\n`;
    if (masterModel) {
      tooltipText += `- Model: \`${masterModel}\`\n`;
    }
    if (session.lastActivity) {
      tooltipText += `- Last active: ${formatTimestamp(session.lastActivity)}\n`;
    }
    if (session.createdAt) {
      tooltipText += `- Created: ${formatTimestamp(session.createdAt)}\n`;
    }
    if (session.forkedFrom) {
      tooltipText += `- Forked from: \`${session.forkedFrom}\`\n`;
    }
    if (session.worktree) {
      tooltipText += `- Branch: \`${session.worktree.branch ?? "detached"}\`\n`;
      if (session.worktree.worktreePath) {
        tooltipText += `- Worktree: \`${session.worktree.worktreePath}\`\n`;
      }
      if (session.worktree.isWorktree) {
        tooltipText += `- $(git-branch) Linked worktree\n`;
      }
    }
    tooltipText += `- Session ID: ${sessionIdCopyLink(session.id)}\n`;
    const tooltip = new vscode.MarkdownString(tooltipText);
    tooltip.isTrusted = true;
    tooltip.supportThemeIcons = true;
    this.tooltip = tooltip;

    // Click to open orchestrator conversation
    this.command = {
      command: "prsm.showOrchestratorConversation",
      title: "Show Conversation",
      arguments: [session.id],
    };
  }
}

export class AgentTreeItem extends vscode.TreeItem {
  constructor(
    public readonly agent: AgentNode,
    public readonly sessionId: string,
    store?: SessionStore,
    waitingChildBlinkOn = true,
  ) {
    // Determine collapse state: respect user's manual collapse/expand choice
    const collapsibleState = agent.childrenIds.length > 0
      ? (store?.isCollapsed(agent.id)
          ? vscode.TreeItemCollapsibleState.Collapsed
          : vscode.TreeItemCollapsibleState.Expanded)
      : vscode.TreeItemCollapsibleState.None;
    super(
      agent.name,
      collapsibleState
    );
    this.id = `agent:${sessionId}:${agent.id}`;

    this.contextValue = "prsmAgent";

    // Active work states use the spinning green arrow icon.
    if (isWaitingForChild(agent)) {
      this.iconPath = new vscode.ThemeIcon(
        WAITING_CHILD_BLINK_ICONS[waitingChildBlinkOn ? 0 : 1],
        new vscode.ThemeColor(WAITING_CHILD_COLOR)
      );
    } else if (isActiveWorkState(agent)) {
      this.iconPath = new vscode.ThemeIcon(
        ACTIVE_SPIN_ICON,
        new vscode.ThemeColor(ACTIVE_SPIN_COLOR)
      );
    } else {
      const stateInfo = STATIC_STATE_ICONS[agent.state] ?? STATIC_STATE_ICONS.idle;
      this.iconPath = new vscode.ThemeIcon(
        stateInfo.icon,
        new vscode.ThemeColor(stateInfo.color)
      );
    }

    // Show state in description when active
    const displayState = agent.rawState ?? agent.state;
    const modelLabel = toModelAliasLabel(agent.model) ?? agent.model;
    const stateLabel = (
      isActiveWorkState(agent) || isWaitingForChild(agent)
    ) ? ` · ${displayState}` : "";
    this.description = `[${agent.role}] ${modelLabel}${stateLabel}`;

    this.tooltip = new vscode.MarkdownString(
      `**${agent.name}**\n\n` +
        `- State: ${displayState}\n` +
        `- Role: ${agent.role}\n` +
        `- Model: ${modelLabel}\n` +
        (agent.promptPreview
          ? `\n\`\`\`\n${agent.promptPreview}\n\`\`\``
          : "")
    );

    // Click to open conversation
    this.command = {
      command: "prsm.showAgentConversation",
      title: "Show Conversation",
      arguments: [sessionId, agent.id],
    };
  }
}

export class AgentTreeProvider
  implements vscode.TreeDataProvider<TreeItemNode>
{
  private static readonly WAITING_CHILD_BLINK_INTERVAL_MS = 450;

  private _onDidChangeTreeData = new vscode.EventEmitter<
    TreeItemNode | undefined | void
  >();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  // ── Search state ──
  private _searchQuery = "";
  private _includeChatContent = false;
  private _waitingChildBlinkOn = true;
  private _waitingChildBlinkTimer: ReturnType<typeof setInterval> | undefined;

  constructor(private readonly store: SessionStore) {
    store.onDidChangeTree(() => {
      this._syncWaitingChildBlinkTimer();
      this._onDidChangeTreeData.fire();
    });
    this._syncWaitingChildBlinkTimer();
  }

  /** Get the current search query. */
  get searchQuery(): string {
    return this._searchQuery;
  }

  /** Get whether chat content is included in search. */
  get includeChatContent(): boolean {
    return this._includeChatContent;
  }

  /** Set the search query and refresh the tree. */
  setSearchQuery(query: string): void {
    this._searchQuery = query;
    vscode.commands.executeCommand(
      "setContext",
      "prsm.treeSearchActive",
      query.trim().length > 0,
    );
    this._onDidChangeTreeData.fire();
  }

  /** Set whether to include chat content in search and refresh. */
  setIncludeChatContent(include: boolean): void {
    this._includeChatContent = include;
    this._onDidChangeTreeData.fire();
  }

  /** Clear search and restore full tree. */
  clearSearch(): void {
    this._searchQuery = "";
    this._includeChatContent = false;
    vscode.commands.executeCommand(
      "setContext",
      "prsm.treeSearchActive",
      false,
    );
    this._onDidChangeTreeData.fire();
  }

  refresh(): void {
    this._syncWaitingChildBlinkTimer();
    this._onDidChangeTreeData.fire();
  }

  getSessionItem(sessionId: string): SessionTreeItem | undefined {
    const session = this.store.getSession(sessionId);
    if (!session) return undefined;
    return new SessionTreeItem(session, this.store, this._waitingChildBlinkOn);
  }

  getAgentItem(sessionId: string, agentId: string): AgentTreeItem | undefined {
    const agent = this.store.getAgent(sessionId, agentId);
    if (!agent) return undefined;
    return new AgentTreeItem(agent, sessionId, this.store, this._waitingChildBlinkOn);
  }

  getTreeItem(element: TreeItemNode): vscode.TreeItem {
    return element;
  }

  // ── Search helpers ──

  /**
   * Check whether a single agent matches the current search query.
   * Checks agent name always; optionally checks chat message content.
   */
  private _agentMatchesQuery(
    sessionId: string,
    agent: AgentNode,
    queryLower: string,
  ): boolean {
    // Check agent name
    if (agent.name.toLowerCase().includes(queryLower)) {
      return true;
    }
    // Check descriptor/task phrase captured from spawn prompt
    if (agent.promptPreview.toLowerCase().includes(queryLower)) {
      return true;
    }
    // Optionally check chat content
    if (this._includeChatContent) {
      const messages = this.store.getMessages(sessionId, agent.id);
      for (const msg of messages) {
        if (msg.content && msg.content.toLowerCase().includes(queryLower)) {
          return true;
        }
      }
    }
    return false;
  }

  /** Check whether the session descriptor/title matches the current query. */
  private _sessionMatchesQuery(sessionId: string, queryLower: string): boolean {
    const session = this.store.getSession(sessionId);
    if (!session) return false;
    return session.name.toLowerCase().includes(queryLower);
  }

  /**
   * Recursively collect the set of agent IDs that match or have a
   * descendant that matches the search query.  This lets us preserve
   * hierarchy context (parent nodes needed to reach a match).
   */
  private _collectVisibleAgentIds(
    sessionId: string,
    agentId: string,
    queryLower: string,
  ): Set<string> {
    const visible = new Set<string>();
    const agent = this.store.getAgent(sessionId, agentId);
    if (!agent) return visible;

    const selfMatches = this._agentMatchesQuery(sessionId, agent, queryLower);

    // Recurse into children first
    for (const childId of agent.childrenIds) {
      const childVisible = this._collectVisibleAgentIds(
        sessionId,
        childId,
        queryLower,
      );
      for (const id of childVisible) {
        visible.add(id);
      }
    }

    // If self matches or any descendant is visible, include this agent
    if (selfMatches || visible.size > 0) {
      visible.add(agent.id);
    }

    return visible;
  }

  /** Collect all descendants (and self) under an agent. */
  private _collectSubtreeAgentIds(
    sessionId: string,
    agentId: string,
  ): Set<string> {
    const ids = new Set<string>();
    const agent = this.store.getAgent(sessionId, agentId);
    if (!agent) return ids;

    ids.add(agent.id);
    for (const childId of agent.childrenIds) {
      const childIds = this._collectSubtreeAgentIds(sessionId, childId);
      for (const id of childIds) {
        ids.add(id);
      }
    }
    return ids;
  }

  /**
   * Build the full set of visible agent IDs for a session.
   * Walks from root orchestrators (or root agents) downward.
   */
  private _getVisibleAgentIds(sessionId: string, queryLower: string): Set<string> {
    const visible = new Set<string>();
    // If the session descriptor itself matches, keep the full session subtree visible.
    if (this._sessionMatchesQuery(sessionId, queryLower)) {
      const session = this.store.getSession(sessionId);
      if (session) {
        for (const id of session.agents.keys()) {
          visible.add(id);
        }
      }
      return visible;
    }
    const roots = this.store.getRootAgents(sessionId);

    for (const root of roots) {
      const rootMatches = this._agentMatchesQuery(sessionId, root, queryLower);
      const rootVisible = rootMatches
        ? this._collectSubtreeAgentIds(sessionId, root.id)
        : this._collectVisibleAgentIds(sessionId, root.id, queryLower);
      for (const id of rootVisible) {
        visible.add(id);
      }
    }

    return visible;
  }

  /** Check whether a session has any visible agents under the current search. */
  private _sessionHasVisibleAgents(sessionId: string, queryLower: string): boolean {
    return (
      this._sessionMatchesQuery(sessionId, queryLower) ||
      this._getVisibleAgentIds(sessionId, queryLower).size > 0
    );
  }

  getChildren(element?: TreeItemNode): TreeItemNode[] {
    const queryLower = this._searchQuery.trim().toLowerCase();
    const isSearching = queryLower.length > 0;

    if (!element) {
      // Root level: all sessions sorted by most recently active first
      let sessions = this.store.getSessions().slice().sort((a, b) => {
        // Running sessions always come first
        if (a.running && !b.running) return -1;
        if (!a.running && b.running) return 1;
        // Then sort by lastActivity descending (most recent first)
        const ta = a.lastActivity ? new Date(a.lastActivity).getTime() : 0;
        const tb = b.lastActivity ? new Date(b.lastActivity).getTime() : 0;
        return tb - ta;
      });

      // Filter sessions when searching: only show sessions with matching agents
      if (isSearching) {
        sessions = sessions.filter((s) =>
          this._sessionHasVisibleAgents(s.id, queryLower)
        );
      }

      return sessions.map((s) => new SessionTreeItem(
        s,
        this.store,
        this._waitingChildBlinkOn,
      ));
    }

    if (element instanceof SessionTreeItem) {
      // Children of a session: merge children from ALL root orchestrators.
      const roots = this.store.getRootAgents(element.session.id);
      const orchestratorRoots = roots.filter((a) => a.role === "orchestrator");

      let children: AgentTreeItem[];
      if (orchestratorRoots.length > 0) {
        children = [];
        for (const orch of orchestratorRoots) {
          for (const child of this.store.getChildren(element.session.id, orch.id)) {
            children.push(new AgentTreeItem(
              child,
              element.session.id,
              this.store,
              this._waitingChildBlinkOn,
            ));
          }
        }
      } else {
        // No orchestrator roots — show all root agents
        children = roots.map((a) => new AgentTreeItem(
          a,
          element.session.id,
          this.store,
          this._waitingChildBlinkOn,
        ));
      }

      // Filter when searching
      if (isSearching) {
        const visibleIds = this._getVisibleAgentIds(element.session.id, queryLower);
        children = children.filter((c) => visibleIds.has(c.agent.id));
      }

      return children;
    }

    if (element instanceof AgentTreeItem) {
      // Children of an agent
      let children = this.store
        .getChildren(element.sessionId, element.agent.id)
        .map(
          (a) => new AgentTreeItem(
            a,
            element.sessionId,
            this.store,
            this._waitingChildBlinkOn,
          )
        );

      // Filter when searching
      if (isSearching) {
        const visibleIds = this._getVisibleAgentIds(element.sessionId, queryLower);
        children = children.filter((c) => visibleIds.has(c.agent.id));
      }

      return children;
    }

    return [];
  }

  getParent(
    element: TreeItemNode
  ): vscode.ProviderResult<TreeItemNode> {
    if (element instanceof AgentTreeItem) {
      const agent = element.agent;
      if (agent.parentId) {
        // Check if parent is any root orchestrator — if so, parent is the session
        const roots = this.store.getRootAgents(element.sessionId);
        const isParentRoot = roots.some((r) => r.id === agent.parentId);
        if (isParentRoot) {
          const session = this.store.getSession(element.sessionId);
          if (session) {
            return new SessionTreeItem(
              session,
              this.store,
              this._waitingChildBlinkOn,
            );
          }
        }
        const parent = this.store.getAgent(
          element.sessionId,
          agent.parentId
        );
        if (parent) {
          return new AgentTreeItem(
            parent,
            element.sessionId,
            this.store,
            this._waitingChildBlinkOn,
          );
        }
      }
      // Root agents: parent is the session
      const session = this.store.getSession(element.sessionId);
      if (session) {
        return new SessionTreeItem(
          session,
          this.store,
          this._waitingChildBlinkOn,
        );
      }
    }
    // Sessions are root-level
    return undefined;
  }

  private _syncWaitingChildBlinkTimer(): void {
    if (!this.store.hasWaitingForChildAgents()) {
      this._stopWaitingChildBlinkTimer();
      return;
    }
    if (this._waitingChildBlinkTimer) {
      return;
    }
    this._waitingChildBlinkTimer = setInterval(() => {
      if (!this.store.hasWaitingForChildAgents()) {
        this._stopWaitingChildBlinkTimer();
        this._onDidChangeTreeData.fire();
        return;
      }
      this._waitingChildBlinkOn = !this._waitingChildBlinkOn;
      this._onDidChangeTreeData.fire();
    }, AgentTreeProvider.WAITING_CHILD_BLINK_INTERVAL_MS);
  }

  private _stopWaitingChildBlinkTimer(): void {
    if (!this._waitingChildBlinkTimer) {
      return;
    }
    clearInterval(this._waitingChildBlinkTimer);
    this._waitingChildBlinkTimer = undefined;
    this._waitingChildBlinkOn = true;
  }

  dispose(): void {
    this._stopWaitingChildBlinkTimer();
    this._onDidChangeTreeData.dispose();
  }
}
