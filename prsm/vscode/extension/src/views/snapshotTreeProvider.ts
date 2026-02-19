/**
 * TreeDataProvider for snapshots grouped by session, then agent tree, then snapshot list.
 *
 * Structure:
 *   Session
 *     Agent
 *       Child Agent
 *       Snapshot
 */
import * as vscode from "vscode";
import { SnapshotMeta } from "../protocol/types";

type SnapshotNode = SnapshotSessionItem | SnapshotAgentItem | SnapshotTreeItem;

class SnapshotTreeItem extends vscode.TreeItem {
  readonly nodeId: string;

  constructor(
    public readonly snapshot: SnapshotMeta,
    public readonly sessionId: string,
    public readonly hasChildren: boolean,
    isExpanded: boolean,
    sessionLabel?: string,
    agentLabel?: string,
  ) {
    const ts = new Date(snapshot.timestamp);
    const timeStr = ts.toLocaleString();
    const desc = snapshot.description || "No description";
    const sessionName = sessionLabel || snapshot.session_name || sessionId;

    super(
      desc,
      hasChildren
        ? (isExpanded
            ? vscode.TreeItemCollapsibleState.Expanded
            : vscode.TreeItemCollapsibleState.Collapsed)
        : vscode.TreeItemCollapsibleState.None,
    );

    this.nodeId = `snapshot:${snapshot.snapshot_id}`;
    this.contextValue = "prsmSnapshot";
    this.description = timeStr;

    this.iconPath = new vscode.ThemeIcon(
      "history",
      new vscode.ThemeColor("charts.blue"),
    );

    const branch = snapshot.git_branch
      ? `\n- Branch: \`${snapshot.git_branch}\``
      : "";
    const parent = snapshot.parent_snapshot_id
      ? `\n- Parent snapshot: \`${snapshot.parent_snapshot_id}\``
      : "";
    const agent = agentLabel ? `\n- Agent: ${agentLabel}` : "";
    this.tooltip = new vscode.MarkdownString(
      `**${desc}**\n\n` +
        `- Time: ${timeStr}\n` +
        `- Session: ${sessionName}${agent}\n` +
        `- ID: \`${snapshot.snapshot_id}\`` +
        parent +
        branch,
    );

    this.command = {
      command: "prsm.goToSnapshot",
      title: "Go to Snapshot",
      arguments: [
        {
          sessionId: this.sessionId,
          snapshotId: snapshot.snapshot_id,
          agentId: snapshot.agent_id,
        },
      ],
    };
  }
}

class SnapshotAgentItem extends vscode.TreeItem {
  readonly nodeId: string;

  constructor(
    public readonly sessionId: string,
    public readonly agentId: string,
    public readonly label: string,
    public readonly snapshotCount: number,
    public readonly hasAgentChildren: boolean,
    isExpanded: boolean,
  ) {
    super(
      label,
      (hasAgentChildren || snapshotCount > 0)
        ? (isExpanded
            ? vscode.TreeItemCollapsibleState.Expanded
            : vscode.TreeItemCollapsibleState.Collapsed)
        : vscode.TreeItemCollapsibleState.None,
    );

    this.nodeId = `agent:${sessionId}:${agentId}`;
    this.contextValue = "prsmSnapshotAgent";
    this.description = `${snapshotCount} snapshot${snapshotCount === 1 ? "" : "s"}`;
    this.iconPath = new vscode.ThemeIcon(
      "account",
      new vscode.ThemeColor("charts.blue"),
    );
  }
}

class SnapshotSessionItem extends vscode.TreeItem {
  readonly nodeId: string;

  constructor(
    public readonly sessionId: string,
    label: string,
    count: number,
    isExpanded: boolean,
  ) {
    super(
      label,
      isExpanded
        ? vscode.TreeItemCollapsibleState.Expanded
        : vscode.TreeItemCollapsibleState.Collapsed,
    );
    this.nodeId = `session:${sessionId}`;
    this.contextValue = "prsmSnapshotSession";
    this.description = `${count} snapshot${count === 1 ? "" : "s"}`;
    this.iconPath = new vscode.ThemeIcon(
      "folder",
      new vscode.ThemeColor("charts.blue"),
    );
  }
}

interface AgentInfo {
  id: string;
  name: string;
  parentId: string | null;
}

export class SnapshotTreeProvider
  implements vscode.TreeDataProvider<SnapshotNode>
{
  private _onDidChangeTreeData = new vscode.EventEmitter<
    SnapshotNode | undefined | void
  >();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private snapshotsBySession = new Map<string, SnapshotMeta[]>();
  private sessionLabels = new Map<string, string>();
  private expandedNodes = new Set<string>();
  private workspaceState?: vscode.Memento;

  refresh(): void {
    this._onDidChangeTreeData.fire();
  }

  setWorkspaceState(state: vscode.Memento): void {
    this.workspaceState = state;
    const saved = state.get<string[]>("prsm.snapshots.expandedNodes");
    this.expandedNodes = new Set(saved ?? []);
  }

  setExpanded(nodeId: string): void {
    this.expandedNodes.add(nodeId);
    this.persistExpandedState();
  }

  setCollapsed(nodeId: string): void {
    this.expandedNodes.delete(nodeId);
    this.persistExpandedState();
  }

  private isExpanded(nodeId: string): boolean {
    return this.expandedNodes.has(nodeId);
  }

  private persistExpandedState(): void {
    this.workspaceState?.update(
      "prsm.snapshots.expandedNodes",
      Array.from(this.expandedNodes),
    );
  }

  setSnapshots(snapshots: SnapshotMeta[], sessionId: string): void {
    this.snapshotsBySession.set(sessionId, snapshots);
    this._onDidChangeTreeData.fire();
  }

  addSnapshot(snapshot: SnapshotMeta, sessionId: string): void {
    const existing = this.snapshotsBySession.get(sessionId) ?? [];
    existing.push(snapshot);
    this.snapshotsBySession.set(sessionId, existing);
    this._onDidChangeTreeData.fire();
  }

  clear(): void {
    this.snapshotsBySession.clear();
    this.sessionLabels.clear();
    this._onDidChangeTreeData.fire();
  }

  setSessionLabels(labels: Array<{ sessionId: string; name: string }>): void {
    this.sessionLabels.clear();
    for (const { sessionId, name } of labels) {
      if (sessionId && name) {
        this.sessionLabels.set(sessionId, name);
      }
    }
    this._onDidChangeTreeData.fire();
  }

  setSessionLabel(sessionId: string, name: string): void {
    if (sessionId && name) {
      this.sessionLabels.set(sessionId, name);
      this._onDidChangeTreeData.fire();
    }
  }

  getTreeItem(element: SnapshotNode): vscode.TreeItem {
    return element;
  }

  getChildren(element?: SnapshotNode): SnapshotNode[] {
    if (!element) {
      return this.getSessionNodes();
    }

    if (element instanceof SnapshotSessionItem) {
      return this.getSessionAgentRoots(element.sessionId);
    }

    if (element instanceof SnapshotAgentItem) {
      return this.getAgentChildren(element.sessionId, element.agentId);
    }

    if (element instanceof SnapshotTreeItem) {
      return [];
    }

    return [];
  }

  private getSessionNodes(): SnapshotSessionItem[] {
    const sessionIds = Array.from(this.snapshotsBySession.keys());
    return sessionIds
      .sort((a, b) => {
        const nameA = this.sessionLabels.get(a) ?? a;
        const nameB = this.sessionLabels.get(b) ?? b;
        return nameA.localeCompare(nameB);
      })
      .map((sessionId) => {
        const label = this.sessionLabels.get(sessionId) ?? sessionId;
        const count = this.snapshotsBySession.get(sessionId)?.length ?? 0;
        return new SnapshotSessionItem(
          sessionId,
          label,
          count,
          this.isExpanded(`session:${sessionId}`),
        );
      });
  }

  private getSessionAgentRoots(sessionId: string): SnapshotAgentItem[] {
    const agentIndex = this.getAgentIndex(sessionId);
    return Array.from(agentIndex.values())
      .filter((agent) => !agent.parentId || !agentIndex.has(agent.parentId))
      .sort((a, b) => a.name.localeCompare(b.name))
      .map((agent) => this.makeAgentItem(sessionId, agent, agentIndex));
  }

  private getAgentChildren(sessionId: string, agentId: string): SnapshotNode[] {
    const agentIndex = this.getAgentIndex(sessionId);
    const childAgents = Array.from(agentIndex.values())
      .filter((agent) => agent.parentId === agentId)
      .sort((a, b) => a.name.localeCompare(b.name))
      .map((agent) => this.makeAgentItem(sessionId, agent, agentIndex));

    const snapshots = this.getAgentSnapshotList(sessionId, agentId);
    return [...childAgents, ...snapshots];
  }

  private makeAgentItem(
    sessionId: string,
    agent: AgentInfo,
    agentIndex: Map<string, AgentInfo>,
  ): SnapshotAgentItem {
    const hasAgentChildren = Array.from(agentIndex.values()).some(
      (candidate) => candidate.parentId === agent.id,
    );
    const snapshotCount = this.getAgentSnapshots(sessionId, agent.id).length;
    return new SnapshotAgentItem(
      sessionId,
      agent.id,
      this.formatAgentLabel(agent.name, agent.id),
      snapshotCount,
      hasAgentChildren,
      this.isExpanded(`agent:${sessionId}:${agent.id}`),
    );
  }

  private getAgentIndex(sessionId: string): Map<string, AgentInfo> {
    const snapshots = this.snapshotsBySession.get(sessionId) ?? [];
    const index = new Map<string, AgentInfo>();

    for (const snap of snapshots) {
      const agentId = snap.agent_id;
      if (!agentId) {
        continue;
      }
      const existing = index.get(agentId);
      const name = snap.agent_name?.trim() || existing?.name || `Agent ${agentId.slice(0, 8)}`;
      const parentId = snap.parent_agent_id ?? existing?.parentId ?? null;
      index.set(agentId, { id: agentId, name, parentId });
    }

    if (snapshots.some((snap) => !snap.agent_id)) {
      const orchestratorId = this.resolveOrchestratorAgentId(sessionId);
      if (!index.has(orchestratorId)) {
        index.set(orchestratorId, {
          id: orchestratorId,
          name: "Orchestrator",
          parentId: null,
        });
      }
    }

    return index;
  }

  private getAgentSnapshots(sessionId: string, agentId: string): SnapshotMeta[] {
    const snapshots = this.snapshotsBySession.get(sessionId) ?? [];
    const orchestratorId = this.resolveOrchestratorAgentId(sessionId);
    return snapshots.filter(
      (snap) => this.getAgentBucketId(snap.agent_id, orchestratorId) === agentId,
    );
  }

  private getAgentSnapshotList(
    sessionId: string,
    agentId: string,
  ): SnapshotTreeItem[] {
    const snapshots = this.getAgentSnapshots(sessionId, agentId);
    return snapshots
      .sort((a, b) => (a.timestamp < b.timestamp ? 1 : -1))
      .map((snap) =>
        new SnapshotTreeItem(
          snap,
          sessionId,
          false,
          this.isExpanded(`snapshot:${snap.snapshot_id}`),
          this.sessionLabels.get(sessionId),
          this.getAgentLabel(sessionId, agentId),
        ),
      );
  }

  private getAgentLabel(sessionId: string, agentId: string): string {
    const index = this.getAgentIndex(sessionId);
    const agent = index.get(agentId);
    return this.formatAgentLabel(agent?.name ?? agentId, agentId);
  }

  private formatAgentLabel(name: string, agentId: string): string {
    return `${name} (${agentId})`;
  }

  private getAgentBucketId(
    agentId: string | null | undefined,
    orchestratorId: string,
  ): string {
    return agentId || orchestratorId;
  }

  private resolveOrchestratorAgentId(sessionId: string): string {
    const snapshots = this.snapshotsBySession.get(sessionId) ?? [];
    const rootCandidates = new Map<string, { latestTs: string }>();
    const seenAgents = new Map<string, { latestTs: string }>();

    for (const snap of snapshots) {
      if (!snap.agent_id) {
        continue;
      }
      const seen = seenAgents.get(snap.agent_id);
      if (!seen || snap.timestamp > seen.latestTs) {
        seenAgents.set(snap.agent_id, { latestTs: snap.timestamp });
      }
      if (!snap.parent_agent_id) {
        const root = rootCandidates.get(snap.agent_id);
        if (!root || snap.timestamp > root.latestTs) {
          rootCandidates.set(snap.agent_id, { latestTs: snap.timestamp });
        }
      }
    }

    if (rootCandidates.size > 0) {
      const [bestRootId] = Array.from(rootCandidates.entries())
        .sort((a, b) => (a[1].latestTs < b[1].latestTs ? 1 : -1))[0];
      return bestRootId;
    }

    if (seenAgents.size > 0) {
      const [bestAgentId] = Array.from(seenAgents.entries())
        .sort((a, b) => (a[1].latestTs < b[1].latestTs ? 1 : -1))[0];
      return bestAgentId;
    }

    return "__orchestrator__";
  }
}
