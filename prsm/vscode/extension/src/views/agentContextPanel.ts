/**
 * Agent Context Side Panel — shows comprehensive agent context
 * in a webview panel on the side (ViewColumn.Two).
 *
 * Displays:
 * - Agent metadata (role, state, model, task prompt)
 * - Agent hierarchy tree (orchestrator → worker → expert chain)
 * - Full conversation history (text, tool calls, tool results, thinking)
 * - Tool rationale for a specific tool call (if provided)
 */
import * as vscode from "vscode";
import { SessionStore } from "../state/sessionStore";
import { PrsmTransport } from "../protocol/transport";
import { AgentNode } from "../protocol/types";

/** Arguments for opening the context panel. */
export interface AgentContextArgs {
  sessionId: string;
  agentId: string;
  toolCallId?: string;
  messageIndex?: number;
}

export class AgentContextPanelManager {
  /** Active context panels keyed by `sessionId:agentId`. */
  private panels = new Map<string, vscode.WebviewPanel>();

  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly store: SessionStore,
    private readonly getTransport: () => PrsmTransport | undefined
  ) {}

  /**
   * Show the agent context side panel.
   * Re-uses an existing panel if one is already open for this agent.
   */
  async show(args: AgentContextArgs): Promise<void> {
    const { sessionId, agentId, toolCallId } = args;
    const key = `${sessionId}:${agentId}`;

    // Re-use existing panel
    const existing = this.panels.get(key);
    if (existing) {
      existing.reveal(vscode.ViewColumn.Two);
      // Refresh data in case context changed
      await this.updatePanel(existing, args);
      return;
    }

    const agent = this.store.getAgent(sessionId, agentId);
    const agentName = agent?.name ?? agentId.slice(0, 8);

    const panel = vscode.window.createWebviewPanel(
      "prsmAgentContext",
      `Context: ${agentName}`,
      { viewColumn: vscode.ViewColumn.Two, preserveFocus: true },
      {
        enableScripts: true,
        retainContextWhenHidden: true,
      }
    );

    this.panels.set(key, panel);

    panel.onDidDispose(() => {
      this.panels.delete(key);
    });

    // Handle messages from webview
    panel.webview.onDidReceiveMessage((msg) => {
      if (msg.type === "openAgent") {
        // Open conversation webview for clicked agent
        vscode.commands.executeCommand("prsm.openAgent", {
          sessionId: msg.sessionId,
          agentId: msg.agentId,
        });
      } else if (msg.type === "refresh") {
        this.updatePanel(panel, args);
      }
    });

    await this.updatePanel(panel, args);
  }

  /** Refresh a panel's HTML with the latest data. */
  private async updatePanel(
    panel: vscode.WebviewPanel,
    args: AgentContextArgs
  ): Promise<void> {
    const { sessionId, agentId, toolCallId } = args;
    const transport = this.getTransport();

    // 1. Gather agent metadata
    const agent = this.store.getAgent(sessionId, agentId);

    // 2. Build agent hierarchy chain (walk up to root, then down)
    const hierarchy = this.buildHierarchy(sessionId, agentId);

    // 3. Fetch rationale if toolCallId provided
    let rationale: { tool_name: string; rationale: string } | null = null;
    if (toolCallId && transport?.isConnected) {
      try {
        const r = await transport.getToolRationale(
          sessionId,
          agentId,
          toolCallId
        );
        rationale = { tool_name: r.tool_name, rationale: r.rationale };
      } catch {
        // Rationale not available — that's fine
      }
    }

    // 4. Fetch full conversation history from server
    let history: Array<{
      type: string;
      content?: string;
      timestamp: number;
      tool_name?: string;
      tool_id?: string;
      tool_args?: string;
      is_error?: boolean;
    }> = [];

    if (transport?.isConnected) {
      try {
        const resp = await transport.getAgentHistory(
          sessionId,
          agentId,
          "full"
        );
        history = resp.history;
      } catch {
        // Fall back to store messages if server history unavailable
      }
    }

    // 5. If server history is empty, fall back to store messages
    if (history.length === 0) {
      const msgs = this.store.getMessages(sessionId, agentId);
      for (const msg of msgs) {
        if (msg.role === "user") {
          history.push({
            type: "user_message",
            content: msg.content,
            timestamp: msg.timestamp
              ? new Date(msg.timestamp).getTime() / 1000
              : 0,
          });
        } else if (msg.role === "assistant") {
          history.push({
            type: "text",
            content: msg.content,
            timestamp: msg.timestamp
              ? new Date(msg.timestamp).getTime() / 1000
              : 0,
          });
        }
        for (const tc of msg.toolCalls) {
          history.push({
            type: "tool_call",
            tool_name: tc.name,
            tool_id: tc.id,
            tool_args: tc.arguments,
            timestamp: msg.timestamp
              ? new Date(msg.timestamp).getTime() / 1000
              : 0,
          });
          if (tc.result !== null) {
            history.push({
              type: "tool_result",
              content: tc.result,
              tool_name: tc.name,
              tool_id: tc.id,
              is_error: !tc.success,
              timestamp: msg.timestamp
                ? new Date(msg.timestamp).getTime() / 1000
                : 0,
            });
          }
        }
      }
    }

    // 6. Generate HTML
    panel.webview.html = this.buildHtml(
      panel.webview,
      agent ?? null,
      hierarchy,
      history,
      rationale,
      toolCallId ?? null,
      sessionId
    );
  }

  /** Walk up from the target agent to root, building the ancestor chain. */
  private buildHierarchy(
    sessionId: string,
    agentId: string
  ): AgentNode[] {
    const chain: AgentNode[] = [];
    let currentId: string | null = agentId;

    while (currentId) {
      const node = this.store.getAgent(sessionId, currentId);
      if (!node) break;
      chain.unshift(node); // prepend so root is first
      currentId = node.parentId;
    }

    // Also append direct children of the target agent
    const target = this.store.getAgent(sessionId, agentId);
    if (target) {
      for (const childId of target.childrenIds) {
        const child = this.store.getAgent(sessionId, childId);
        if (child) {
          chain.push(child);
        }
      }
    }

    return chain;
  }

  private buildHtml(
    webview: vscode.Webview,
    agent: AgentNode | null,
    hierarchy: AgentNode[],
    history: Array<{
      type: string;
      content?: string;
      timestamp: number;
      tool_name?: string;
      tool_id?: string;
      tool_args?: string;
      is_error?: boolean;
    }>,
    rationale: { tool_name: string; rationale: string } | null,
    highlightToolCallId: string | null,
    sessionId: string
  ): string {
    const nonce = getNonce();

    // --- Build sections ---

    // Agent metadata
    const metaHtml = agent
      ? `<div class="section metadata">
          <h2>Agent Info</h2>
          <table class="meta-table">
            <tr><td class="label">Name</td><td>${esc(agent.name)}</td></tr>
            <tr><td class="label">Role</td><td><span class="badge badge-${esc(agent.role)}">${esc(agent.role)}</span></td></tr>
            <tr><td class="label">State</td><td><span class="state state-${esc(agent.state)}">${esc(agent.state)}</span></td></tr>
            <tr><td class="label">Model</td><td><code>${esc(agent.model || "—")}</code></td></tr>
            <tr><td class="label">ID</td><td><code class="id">${esc(agent.id)}</code></td></tr>
            ${agent.parentId ? `<tr><td class="label">Parent</td><td><code class="id">${esc(agent.parentId)}</code></td></tr>` : ""}
            ${agent.promptPreview ? `<tr><td class="label">Task</td><td class="task-preview">${esc(agent.promptPreview)}</td></tr>` : ""}
          </table>
        </div>`
      : `<div class="section metadata"><h2>Agent Info</h2><p class="muted">Agent not found</p></div>`;

    // Rationale section (only if present)
    const rationaleHtml = rationale
      ? `<div class="section rationale">
          <h2>Why <code>${esc(rationale.tool_name)}</code>?</h2>
          <div class="rationale-text">${esc(rationale.rationale)}</div>
        </div>`
      : "";

    // Hierarchy
    let hierarchyHtml = "";
    if (hierarchy.length > 0) {
      hierarchyHtml = `<div class="section hierarchy">
        <h2>Agent Hierarchy</h2>
        <div class="tree">`;

      const targetId = agent?.id;
      // Find the index of the target agent in the chain
      const targetIndex = hierarchy.findIndex((h) => h.id === targetId);

      for (let i = 0; i < hierarchy.length; i++) {
        const node = hierarchy[i];
        const isCurrent = node.id === targetId;
        const isChild = i > targetIndex && targetIndex >= 0;
        const indent = isChild ? targetIndex + 1 : i;
        const connector =
          i === 0 ? "" : isChild ? "├─ " : "└─ ";
        const cls = isCurrent ? "tree-node current" : "tree-node";

        hierarchyHtml += `<div class="${cls}" style="padding-left:${indent * 20}px" data-agent-id="${esc(node.id)}" data-session-id="${esc(sessionId)}">
          <span class="connector">${connector}</span>
          <span class="badge badge-${esc(node.role)}">${esc(node.role)}</span>
          <span class="node-name">${esc(node.name)}</span>
          <span class="state state-${esc(node.state)}">${esc(node.state)}</span>
        </div>`;
      }

      hierarchyHtml += `</div></div>`;
    }

    // Conversation history
    let historyHtml = `<div class="section conversation">
      <h2>Conversation History <span class="count">(${history.length} entries)</span></h2>`;

    if (history.length === 0) {
      historyHtml += `<p class="muted">No conversation history available</p>`;
    } else {
      historyHtml += `<div class="history">`;

      for (const entry of history) {
        const ts = entry.timestamp
          ? new Date(entry.timestamp * 1000).toLocaleTimeString()
          : "";
        const isHighlighted =
          highlightToolCallId &&
          entry.tool_id === highlightToolCallId;

        switch (entry.type) {
          case "user_message":
            historyHtml += `<div class="entry entry-user">
              <div class="entry-header"><span class="entry-type type-user">User</span><span class="ts">${esc(ts)}</span></div>
              <div class="entry-content">${escPreWithCodeBlocks(entry.content ?? "")}</div>
            </div>`;
            break;

          case "text":
            historyHtml += `<div class="entry entry-assistant">
              <div class="entry-header"><span class="entry-type type-assistant">Assistant</span><span class="ts">${esc(ts)}</span></div>
              <div class="entry-content">${escPre(entry.content ?? "")}</div>
            </div>`;
            break;

          case "thinking":
            historyHtml += `<div class="entry entry-thinking">
              <div class="entry-header"><span class="entry-type type-thinking">Thinking</span><span class="ts">${esc(ts)}</span></div>
              <div class="entry-content thinking-content">${escPre(entry.content ?? "")}</div>
            </div>`;
            break;

          case "tool_call": {
            const highlightCls = isHighlighted ? " highlighted" : "";
            const argsPreview = truncate(entry.tool_args ?? "", 500);
            historyHtml += `<div class="entry entry-tool-call${highlightCls}" ${entry.tool_id ? `id="tc-${esc(entry.tool_id)}"` : ""}>
              <div class="entry-header">
                <span class="entry-type type-tool">Tool Call</span>
                <code class="tool-name">${esc(entry.tool_name ?? "unknown")}</code>
                <span class="ts">${esc(ts)}</span>
              </div>
              ${argsPreview ? `<details class="tool-args"><summary>Arguments</summary><pre>${esc(argsPreview)}</pre></details>` : ""}
            </div>`;
            break;
          }

          case "tool_result": {
            const resultPreview = truncate(entry.content ?? "", 800);
            const errorCls = entry.is_error ? " error" : "";
            historyHtml += `<div class="entry entry-tool-result${errorCls}">
              <div class="entry-header">
                <span class="entry-type type-result${errorCls}">Result</span>
                <code class="tool-name">${esc(entry.tool_name ?? "")}</code>
                ${entry.is_error ? '<span class="error-badge">ERROR</span>' : ""}
                <span class="ts">${esc(ts)}</span>
              </div>
              ${resultPreview ? `<details class="tool-result-content"><summary>Output (${(entry.content ?? "").length} chars)</summary><pre>${esc(resultPreview)}</pre></details>` : ""}
            </div>`;
            break;
          }

          default:
            historyHtml += `<div class="entry entry-unknown">
              <div class="entry-header"><span class="entry-type">${esc(entry.type)}</span><span class="ts">${esc(ts)}</span></div>
              <div class="entry-content">${escPre(entry.content ?? "")}</div>
            </div>`;
        }
      }

      historyHtml += `</div>`;
    }

    historyHtml += `</div>`;

    // --- Assemble full HTML ---
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Agent Context</title>
  <style>
    :root {
      --bg: var(--vscode-editor-background);
      --fg: var(--vscode-editor-foreground);
      --border: var(--vscode-panel-border, var(--vscode-widget-border, #444));
      --muted: var(--vscode-descriptionForeground, #888);
      --accent: var(--vscode-textLink-foreground, #4fc1ff);
      --badge-orch: #b48ead;
      --badge-worker: #88c0d0;
      --badge-expert: #ebcb8b;
      --state-running: #a3be8c;
      --state-waiting: #ebcb8b;
      --state-completed: #a3be8c;
      --state-error: #bf616a;
      --state-idle: var(--muted);
      --highlight-bg: rgba(79, 193, 255, 0.12);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: var(--vscode-font-family, system-ui, sans-serif);
      font-size: var(--vscode-font-size, 13px);
      color: var(--fg);
      background: var(--bg);
      line-height: 1.5;
      padding: 12px 16px;
      overflow-y: auto;
    }
    h2 {
      font-size: 13px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--muted);
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .count {
      font-weight: 400;
      font-size: 11px;
      text-transform: none;
      letter-spacing: 0;
    }
    .section {
      margin-bottom: 20px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
    }
    .section:last-child { border-bottom: none; }

    /* Top header with refresh button */
    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }
    .panel-header h1 {
      font-size: 15px;
      font-weight: 600;
    }
    .refresh-btn {
      background: transparent;
      border: 1px solid var(--border);
      color: var(--fg);
      padding: 3px 8px;
      border-radius: 3px;
      cursor: pointer;
      font-size: 12px;
    }
    .refresh-btn:hover {
      background: var(--vscode-toolbar-hoverBackground, rgba(255,255,255,0.1));
    }

    /* Metadata table */
    .meta-table {
      width: 100%;
      border-collapse: collapse;
    }
    .meta-table td {
      padding: 3px 8px 3px 0;
      vertical-align: top;
    }
    .meta-table .label {
      font-weight: 600;
      white-space: nowrap;
      width: 70px;
      color: var(--muted);
    }
    code {
      font-family: var(--vscode-editor-font-family, "Fira Code", monospace);
      font-size: 12px;
      background: var(--vscode-textCodeBlock-background, rgba(255,255,255,0.06));
      padding: 1px 4px;
      border-radius: 3px;
    }
    code.id {
      font-size: 11px;
      color: var(--muted);
    }
    .task-preview {
      font-style: italic;
      max-width: 400px;
      word-break: break-word;
    }

    /* Badges */
    .badge {
      display: inline-block;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      padding: 1px 6px;
      border-radius: 3px;
      background: rgba(255,255,255,0.08);
    }
    .badge-orchestrator { color: var(--badge-orch); border: 1px solid var(--badge-orch); }
    .badge-worker { color: var(--badge-worker); border: 1px solid var(--badge-worker); }
    .badge-expert { color: var(--badge-expert); border: 1px solid var(--badge-expert); }

    /* State indicators */
    .state {
      font-size: 11px;
      font-weight: 500;
    }
    .state-running { color: var(--state-running); }
    .state-waiting { color: var(--state-waiting); }
    .state-waiting_for_child { color: var(--state-waiting); }
    .state-completed { color: var(--state-completed); }
    .state-error { color: var(--state-error); }
    .state-idle { color: var(--state-idle); }

    /* Rationale */
    .rationale {
      background: var(--highlight-bg);
      padding: 12px;
      border-radius: 6px;
      border: 1px solid var(--accent);
      margin-bottom: 20px;
    }
    .rationale h2 {
      color: var(--accent);
    }
    .rationale-text {
      margin-top: 6px;
      white-space: pre-wrap;
      line-height: 1.6;
    }

    /* Hierarchy tree */
    .tree { font-family: var(--vscode-editor-font-family, monospace); }
    .tree-node {
      padding: 3px 4px;
      display: flex;
      align-items: center;
      gap: 6px;
      border-radius: 3px;
      cursor: pointer;
    }
    .tree-node:hover {
      background: var(--vscode-list-hoverBackground, rgba(255,255,255,0.04));
    }
    .tree-node.current {
      background: var(--highlight-bg);
      font-weight: 600;
    }
    .connector { color: var(--muted); font-size: 12px; white-space: pre; }
    .node-name { flex: 1; }

    /* Conversation history */
    .history { display: flex; flex-direction: column; gap: 6px; }
    .entry {
      border-radius: 6px;
      padding: 8px 10px;
      border: 1px solid var(--border);
    }
    .entry-header {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 4px;
    }
    .entry-type {
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.3px;
      padding: 1px 5px;
      border-radius: 3px;
    }
    .type-user { background: #2e6b4f; color: #a3d9c0; }
    .type-assistant { background: #2d5a8e; color: #a0c8f0; }
    .type-thinking { background: #5c4a6e; color: #c0a8d8; }
    .type-tool { background: #4a5568; color: #cbd5e0; }
    .type-result { background: #3d4f3d; color: #a8d8a8; }
    .type-result.error { background: #5f2d2d; color: #f0a0a0; }
    .ts { color: var(--muted); font-size: 11px; margin-left: auto; }
    .tool-name { font-size: 12px; }
    .error-badge {
      font-size: 9px;
      font-weight: 700;
      background: var(--state-error);
      color: #fff;
      padding: 0 4px;
      border-radius: 2px;
    }

    .entry-content {
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 300px;
      overflow-y: auto;
      font-size: 12px;
      line-height: 1.5;
    }
    .acp-code-block {
      margin: 6px 0;
      padding: 8px 10px;
      background: var(--vscode-textCodeBlock-background, rgba(0, 0, 0, 0.15));
      border-radius: 4px;
      overflow-x: auto;
      font-family: var(--vscode-editor-font-family, monospace);
      font-size: 11px;
      line-height: 1.4;
      white-space: pre;
    }
    .acp-code-block code {
      font-family: inherit;
      background: transparent;
    }
    .thinking-content {
      color: var(--muted);
      font-style: italic;
    }
    .entry-user { border-left: 3px solid #2e6b4f; }
    .entry-assistant { border-left: 3px solid #2d5a8e; }
    .entry-thinking { border-left: 3px solid #5c4a6e; }
    .entry-tool-call { border-left: 3px solid #4a5568; }
    .entry-tool-result { border-left: 3px solid #3d4f3d; }
    .entry-tool-result.error { border-left: 3px solid var(--state-error); }
    .entry-tool-call.highlighted {
      border: 2px solid var(--accent);
      background: var(--highlight-bg);
    }

    details { margin-top: 4px; }
    details summary {
      cursor: pointer;
      font-size: 11px;
      color: var(--muted);
      user-select: none;
    }
    details summary:hover { color: var(--fg); }
    details pre {
      margin-top: 4px;
      font-size: 11px;
      line-height: 1.4;
      white-space: pre-wrap;
      word-break: break-all;
      max-height: 400px;
      overflow-y: auto;
      background: var(--vscode-textCodeBlock-background, rgba(255,255,255,0.04));
      padding: 8px;
      border-radius: 4px;
    }

    .muted { color: var(--muted); font-style: italic; }

    /* Scroll-to-highlight animation */
    @keyframes flash {
      0% { background: var(--accent); }
      100% { background: var(--highlight-bg); }
    }
    .flash { animation: flash 1s ease-out; }
  </style>
</head>
<body>
  <div class="panel-header">
    <h1>${agent ? esc(agent.name) : "Agent Context"}</h1>
    <button class="refresh-btn" id="refreshBtn">↻ Refresh</button>
  </div>
  ${rationaleHtml}
  ${metaHtml}
  ${hierarchyHtml}
  ${historyHtml}
  <script nonce="${nonce}">
    (function() {
      const vscode = acquireVsCodeApi();

      // Refresh button
      document.getElementById('refreshBtn')?.addEventListener('click', () => {
        vscode.postMessage({ type: 'refresh' });
      });

      // Click on tree nodes to open agent
      document.querySelectorAll('.tree-node').forEach(node => {
        node.addEventListener('click', () => {
          const agentId = node.getAttribute('data-agent-id');
          const sessionId = node.getAttribute('data-session-id');
          if (agentId && sessionId) {
            vscode.postMessage({ type: 'openAgent', agentId, sessionId });
          }
        });
      });

      // Auto-scroll to highlighted tool call
      const highlighted = document.querySelector('.highlighted');
      if (highlighted) {
        setTimeout(() => {
          highlighted.scrollIntoView({ behavior: 'smooth', block: 'center' });
          highlighted.classList.add('flash');
        }, 100);
      }
    })();
  </script>
</body>
</html>`;
  }

  dispose(): void {
    for (const panel of this.panels.values()) {
      panel.dispose();
    }
    this.panels.clear();
  }
}

// ── Helpers ──

function getNonce(): string {
  const chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let result = "";
  for (let i = 0; i < 32; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}

/** HTML-escape a string. */
function esc(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** HTML-escape and preserve whitespace. */
function escPre(s: string): string {
  return esc(s);
}

/**
 * Render text with fenced code blocks as HTML.
 * Splits on triple-backtick fences and wraps code in <pre><code> blocks.
 * Non-code segments are HTML-escaped with whitespace preserved.
 */
function escPreWithCodeBlocks(s: string): string {
  const segments = s.split(/(```[\s\S]*?```)/g);
  return segments
    .map((seg) => {
      if (seg.startsWith("```") && seg.endsWith("```")) {
        const inner = seg.slice(3, -3);
        const firstNewline = inner.indexOf("\n");
        const lang = firstNewline >= 0 ? inner.slice(0, firstNewline).trim() : "";
        const code = firstNewline >= 0 ? inner.slice(firstNewline + 1) : inner;
        const langAttr = lang ? ` class="language-${esc(lang)}"` : "";
        return `<pre class="acp-code-block"><code${langAttr}>${esc(code)}</code></pre>`;
      }
      return esc(seg);
    })
    .join("");
}

/** Truncate a string to maxLen characters. */
function truncate(s: string, maxLen: number): string {
  if (s.length <= maxLen) return s;
  return s.slice(0, maxLen) + `\n... (${s.length - maxLen} more chars)`;
}
