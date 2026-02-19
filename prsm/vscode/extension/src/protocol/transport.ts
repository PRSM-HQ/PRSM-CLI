/**
 * Transport layer: spawns `prsm --server`, reads the port from stdout,
 * connects via SSE for events and HTTP for commands.
 *
 * The server is spawned detached so it survives VSCode window reloads.
 * On reactivation the extension reconnects to the existing server.
 */
import * as vscode from "vscode";
import { ChildProcess, spawn, execFileSync } from "child_process";
import { EventEmitter } from "events";
import * as http from "http";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import {
  SessionInfo,
  AgentNode,
  Message,
  SSEEventData,
} from "./types";

/**
 * Resolve the prsm executable to a full path.
 * VSCode often spawns processes with a minimal PATH that doesn't include
 * ~/.local/bin or other user-installed locations. This function tries:
 * 1. The configured path as-is (if absolute)
 * 2. `which` to find it on the current PATH
 * 3. Common install locations: ~/.local/bin, project .venv/bin
 */
function resolveExecutable(configured: string, cwd: string): string {
  // If it's already an absolute path, use it directly
  if (path.isAbsolute(configured)) {
    return configured;
  }

  // Prefer the project's virtualenv first — this ensures consistent resolution
  // across VSCode reloads (prevents spawning duplicate servers from different paths)
  const home = os.homedir();
  const priorityCandidates = [
    path.join(cwd, ".venv", "bin", configured),
  ];

  for (const candidate of priorityCandidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  // Try `which` (may return ~/.local/bin or other PATH entries)
  try {
    const resolved = execFileSync("which", [configured], {
      encoding: "utf-8",
      timeout: 3000,
    }).trim();
    if (resolved && fs.existsSync(resolved)) {
      return resolved;
    }
  } catch {
    // which failed — try common locations
  }

  // Check remaining common install locations
  const fallbackCandidates = [
    path.join(home, ".local", "bin", configured),
    path.join(home, ".venv", "bin", configured),
  ];

  for (const candidate of fallbackCandidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  // Fall back to the configured name (will produce a clear ENOENT error)
  return configured;
}

export interface TransportOptions {
  executablePath: string;
  cwd: string;
  outputChannel: vscode.OutputChannel;
  configPath?: string;
  sessionInactivityMinutes?: number;
}

export class PrsmTransport extends EventEmitter {
  private process: ChildProcess | null = null;
  private port: number | null = null;
  private serverPid: number | null = null;
  private sseAbort: AbortController | null = null;
  private _connected = false;
  private readonly outputChannel: vscode.OutputChannel;
  private readonly executablePath: string;
  private readonly cwd: string;
  private readonly configPath?: string;
  private readonly sessionInactivityMinutes?: number;
  private requestSeq = 0;

  constructor(options: TransportOptions) {
    super();
    this.executablePath = options.executablePath;
    this.cwd = options.cwd;
    this.outputChannel = options.outputChannel;
    this.configPath = options.configPath;
    this.sessionInactivityMinutes = options.sessionInactivityMinutes;
  }

  get isConnected(): boolean {
    return this._connected;
  }

  get currentPort(): number | null {
    return this.port;
  }

  get currentPid(): number | null {
    return this.serverPid;
  }

  /**
   * Load environment variables from the user's shell (~/.zshrc, ~/.bashrc).
   * This ensures API keys like MINIMAX_API_KEY, ANTHROPIC_API_KEY, etc.
   * are available to the PRSM server even though VS Code doesn't load them.
   */
  private loadShellEnvironment(): Record<string, string> {
    const home = os.homedir();
    const shell = process.env.SHELL || "/bin/bash";
    const shellName = path.basename(shell);

    // Detect shell config file
    let rcFile = "";
    if (shellName === "zsh") {
      rcFile = path.join(home, ".zshrc");
    } else if (shellName === "bash") {
      rcFile = path.join(home, ".bashrc");
    } else {
      return {};  // Unsupported shell
    }

    if (!fs.existsSync(rcFile)) {
      return {};
    }

    try {
      // Run shell command to export environment variables
      // Use -i (interactive) to load rc files, -c to run a command
      const result = execFileSync(shell, ["-i", "-c", "env"], {
        encoding: "utf-8",
        timeout: 5000,
        maxBuffer: 1024 * 1024,  // 1MB max
      });

      // Parse env output (KEY=VALUE per line)
      const env: Record<string, string> = {};
      for (const line of result.split("\n")) {
        const match = line.match(/^([^=]+)=(.*)$/);
        if (match) {
          const [, key, value] = match;
          // Only load API keys and other safe env vars, skip shell-specific vars
          if (
            key.endsWith("_API_KEY") ||
            key.endsWith("_KEY") ||
            key === "ANTHROPIC_API_KEY" ||
            key === "OPENAI_API_KEY" ||
            key === "MINIMAX_API_KEY" ||
            key === "GOOGLE_API_KEY"
          ) {
            env[key] = value;
          }
        }
      }

      this.outputChannel.appendLine(
        `Loaded ${Object.keys(env).length} API keys from ${rcFile}`
      );
      return env;
    } catch (err) {
      this.outputChannel.appendLine(
        `Failed to load shell environment: ${(err as Error).message}`
      );
      return {};
    }
  }

  /** Lightweight liveness check against the server health endpoint. */
  async healthCheck(): Promise<boolean> {
    if (!this.port) return false;
    try {
      const health = await this.requestRaw<{ status: string }>(
        "GET",
        "/health",
        undefined,
        this.port
      );
      return health?.status === "ok";
    } catch {
      return false;
    }
  }

  /** Start the prsm --server process and connect. */
  async start(): Promise<void> {
    if (this.process || this._connected) {
      throw new Error("Transport already started");
    }

    // Resolve the executable to a full path
    const resolvedPath = resolveExecutable(this.executablePath, this.cwd);

    const args = ["--server"];
    if (this.configPath) {
      args.push("--config", this.configPath);
    }

    this.outputChannel.appendLine(
      `Starting: ${resolvedPath} ${args.join(" ")}`
    );
    if (resolvedPath !== this.executablePath) {
      this.outputChannel.appendLine(
        `  (resolved from "${this.executablePath}")`
      );
    }
    this.outputChannel.appendLine(`  cwd: ${this.cwd}`);
    this.outputChannel.appendLine(
      `  config: ${this.configPath ?? "<none>"}`
    );

    // Augment PATH to include common user bin directories
    const home = os.homedir();
    const extraPaths = [
      path.join(home, ".local", "bin"),
      path.join(this.cwd, ".venv", "bin"),
    ];
    const currentPath = process.env.PATH ?? "";
    const augmentedPath = [...extraPaths, currentPath].join(path.delimiter);

    // Load shell environment to get API keys and other user env vars
    const shellEnv = this.loadShellEnvironment();

    // Spawn detached so the server survives extension host restarts
    this.process = spawn(resolvedPath, args, {
      cwd: this.cwd,
      env: {
        ...process.env,
        ...shellEnv,  // Merge shell env (API keys, etc.)
        PRSM_SESSION_INACTIVITY_MINUTES: String(
          this.sessionInactivityMinutes ?? 15
        ),
        PATH: augmentedPath,
        PYTHONUNBUFFERED: "1"
      },
      stdio: ["pipe", "pipe", "pipe"],
      detached: true,
    });

    // Unref so the extension host can exit without waiting for the server
    this.process.unref();

    // Save PID for later reconnection
    this.serverPid = this.process.pid ?? null;
    this.outputChannel.appendLine(`Spawned PRSM pid=${this.serverPid ?? "?"}`);

    // Pipe stderr to output channel IMMEDIATELY (before readPort)
    // so we can see crash errors during startup
    this.process.stderr?.on("data", (data: Buffer) => {
      this.outputChannel.appendLine(`[prsm] ${data.toString().trim()}`);
    });

    this.process.on("error", (err) => {
      this.outputChannel.appendLine(`prsm process error: ${err.message}`);
      this.emit("error", err);
    });

    // Read port from first stdout line (with early-exit detection)
    const port = await this.readPort();
    this.port = port;
    this.outputChannel.appendLine(`Server started on port ${port}`);

    this.process.on("exit", (code, signal) => {
      this.outputChannel.appendLine(
        `prsm process exited (code=${code}, signal=${signal})`
      );
      this._connected = false;
      this.process = null;
      this.port = null;
      this.serverPid = null;
      this.emit("terminated", code, signal);
    });

    // Connect SSE
    await this.connectSSE();
    this._connected = true;
  }

  /**
   * Reconnect to an already-running server on a known port.
   * Used after VSCode reload to pick up where we left off.
   */
  async reconnect(port: number, serverPid?: number): Promise<void> {
    if (this._connected) {
      throw new Error("Transport already connected");
    }

    this.outputChannel.appendLine(
      `Reconnecting to existing server on port ${port}...`
    );

    // Health check first
    try {
      const health = await this.requestRaw<{ status: string }>(
        "GET",
        "/health",
        undefined,
        port
      );
      if (health.status !== "ok") {
        throw new Error(`Unhealthy: ${JSON.stringify(health)}`);
      }
    } catch (err) {
      throw new Error(
        `Cannot reach server on port ${port}: ${(err as Error).message}`
      );
    }

    this.port = port;
    this.serverPid = serverPid ?? null;
    this.outputChannel.appendLine(
      `Health check passed, reconnecting SSE...`
    );

    // Connect SSE
    await this.connectSSE();
    this._connected = true;

    this.outputChannel.appendLine("Reconnected to existing PRSM server");
  }

  /** Stop the server process. */
  async stop(): Promise<void> {
    this.sseAbort?.abort();
    this.sseAbort = null;

    if (this.process) {
      this.process.kill("SIGTERM");
      // Force kill after 3 seconds
      await new Promise<void>((resolve) => {
        const timer = setTimeout(() => {
          if (this.process) {
            this.process.kill("SIGKILL");
          }
          resolve();
        }, 3000);
        this.process!.on("exit", () => {
          clearTimeout(timer);
          resolve();
        });
      });
      this.process = null;
    } else if (this.serverPid) {
      // Detached server — kill by PID
      try {
        process.kill(this.serverPid, "SIGTERM");
      } catch {
        // Already dead
      }
    }
    this.port = null;
    this.serverPid = null;
    this._connected = false;
  }

  /** Disconnect SSE without killing the server (for reload). */
  disconnect(): void {
    this.sseAbort?.abort();
    this.sseAbort = null;
    this._connected = false;
    // Don't kill the process — let it keep running
    if (this.process) {
      this.process.unref();
      // Detach all stdio so the extension host can exit cleanly
      this.process.stdout?.removeAllListeners();
      this.process.stderr?.removeAllListeners();
      this.process.removeAllListeners();
      this.process = null;
    }
  }

  // ── REST API methods ──

  async createSession(name?: string): Promise<{ session_id: string; name: string; summary?: string | null; current_model?: string; current_model_display?: string }> {
    return this.post("/sessions", { name });
  }

  async forkSession(
    sessionId: string,
    name?: string
  ): Promise<{ session_id: string; name: string; summary?: string | null; forked_from: string; current_model?: string; current_model_display?: string }> {
    return this.post(`/sessions/${sessionId}/fork`, { name });
  }

  async removeSession(sessionId: string): Promise<void> {
    await this.del(`/sessions/${sessionId}`);
  }

  async renameSession(
    sessionId: string,
    name: string
  ): Promise<{ session_id: string; name: string }> {
    return this.request("PATCH", `/sessions/${sessionId}`, { name });
  }

  async listSessions(): Promise<{ sessions: SessionInfo[] }> {
    return this.get("/sessions");
  }

  async runPrompt(
    sessionId: string,
    prompt: string
  ): Promise<{ status: string }> {
    return this.post(`/sessions/${sessionId}/run`, { prompt });
  }

  async runSlashCommand(
    sessionId: string,
    command: string
  ): Promise<{ status: string; kind?: string; lines?: string[]; sessions?: Array<Record<string, unknown>>; message?: string }> {
    return this.post(`/sessions/${sessionId}/command`, { command });
  }

  async resolvePermission(
    sessionId: string,
    requestId: string,
    result: string
  ): Promise<void> {
    await this.post(`/sessions/${sessionId}/resolve-permission`, {
      request_id: requestId,
      result,
    });
  }

  async resolveQuestion(
    sessionId: string,
    requestId: string,
    answer: string
  ): Promise<void> {
    await this.post(`/sessions/${sessionId}/resolve-question`, {
      request_id: requestId,
      answer,
    });
  }

  async killAgent(
    sessionId: string,
    agentId: string
  ): Promise<void> {
    await this.post(`/sessions/${sessionId}/kill-agent`, {
      agent_id: agentId,
    });
  }

  async sendAgentMessage(
    sessionId: string,
    agentId: string,
    prompt: string
  ): Promise<{ status: string }> {
    return this.post(`/sessions/${sessionId}/agents/${agentId}/message`, {
      prompt,
    });
  }

  async getAgents(
    sessionId: string
  ): Promise<{ agents: AgentNode[] }> {
    return this.get(`/sessions/${sessionId}/agents`);
  }

  async getMessages(
    sessionId: string,
    agentId: string
  ): Promise<{ messages: Message[] }> {
    return this.get(`/sessions/${sessionId}/agents/${agentId}/messages`);
  }

  async listImportSessions(
    provider: "all" | "codex" | "claude" | "prsm" = "all",
    limit = 25
  ): Promise<{
    sessions: Array<{
      provider: string;
      source_id: string;
      title: string | null;
      turn_count: number;
      source_path: string;
      started_at: string | null;
      updated_at: string | null;
    }>;
  }> {
    return this.get(
      `/import/sessions?provider=${encodeURIComponent(provider)}&limit=${limit}`
    );
  }

  async previewImportSession(
    provider: "codex" | "claude" | "prsm",
    sourceId: string
  ): Promise<{
    summary: {
      provider: string;
      source_id: string;
      title: string | null;
      turn_count: number;
      started_at: string | null;
      updated_at: string | null;
      source_path: string;
    };
    preview_turns: Array<{
      role: string;
      content: string;
      timestamp: string | null;
      tool_call_count: number;
    }>;
    warnings: string[];
  }> {
    return this.get(
      `/import/preview?provider=${encodeURIComponent(provider)}&source_id=${encodeURIComponent(sourceId)}`
    );
  }

  async runImport(
    sessionId: string,
    provider: "codex" | "claude" | "prsm",
    sourceId: string,
    options?: { sessionName?: string; maxTurns?: number | null }
  ): Promise<{
    status: string;
    message: string;
    warnings: string[];
    session: {
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
    };
  }> {
    return this.post(`/sessions/${sessionId}/import`, {
      provider,
      source_id: sourceId,
      session_name: options?.sessionName,
      max_turns: options?.maxTurns ?? null,
    });
  }

  async runImportAll(
    sessionId: string,
    provider: "codex" | "claude" | "prsm",
    options?: { maxTurns?: number | null }
  ): Promise<{
    status: string;
    message: string;
    imported_count: number;
    sessions: Array<{ session_id: string; name: string; turn_count: number }>;
  }> {
    return this.post(`/sessions/${sessionId}/import-all`, {
      provider,
      max_turns: options?.maxTurns ?? null,
    });
  }

  async importArchive(
    archivePath: string,
    conflictMode: "skip" | "overwrite" | "rename" = "skip"
  ): Promise<{
    success: boolean;
    sessions_imported: number;
    sessions_skipped: number;
    files_imported: number;
    files_skipped: number;
    warnings: string[];
    error: string | null;
    manifest: Record<string, unknown> | null;
  }> {
    return this.post("/archive/import", {
      archive_path: archivePath,
      conflict_mode: conflictMode,
    });
  }

  async exportArchive(
    repoIdentity: string,
    outputPath: string,
    format: "tar.gz" | "zip" = "tar.gz"
  ): Promise<{
    success: boolean;
    archive_path: string;
    manifest: Record<string, unknown> | null;
    error: string | null;
  }> {
    return this.post("/archive/export", {
      repo_identity: repoIdentity,
      output_path: outputPath,
      format,
    });
  }

  async shutdownSession(sessionId: string): Promise<void> {
    await this.post(`/sessions/${sessionId}/shutdown`, {});
  }

  async cancelLatestToolCall(sessionId: string): Promise<void> {
    await this.post(`/sessions/${sessionId}/cancel-latest-tool-call`, {});
  }

  async stopAfterTool(sessionId: string): Promise<void> {
    await this.post(`/sessions/${sessionId}/stop-after-tool`, {});
  }

  async injectPrompt(
    sessionId: string,
    agentId: string,
    prompt: string,
    mode: "interrupt" | "inject" | "queue"
  ): Promise<{ status: string; mode: string }> {
    return this.post(
      `/sessions/${sessionId}/agents/${agentId}/inject-prompt`,
      { prompt, mode }
    );
  }

  // ── Persistence + Snapshot methods ──

  async saveSession(sessionId: string): Promise<void> {
    await this.post(`/sessions/${sessionId}/save`, {});
  }

  async listRestorable(): Promise<{
    sessions: Array<{
      name: string;
      saved_at: string;
      agent_count: number;
      message_count: number;
    }>;
  }> {
    return this.get("/sessions/restore");
  }

  async restoreSessionFromDisk(
    name: string
  ): Promise<{ session_id: string; name: string }> {
    return this.post(`/sessions/restore/${encodeURIComponent(name)}`, {});
  }

  async listSnapshots(
    sessionId: string
  ): Promise<{
    snapshots: Array<{
      snapshot_id: string;
      session_id: string | null;
      session_name: string;
      description: string;
      timestamp: string;
      git_branch: string | null;
      parent_snapshot_id?: string | null;
      agent_id?: string | null;
      agent_name?: string | null;
      parent_agent_id?: string | null;
    }>;
  }> {
    return this.get(`/sessions/${sessionId}/snapshots`);
  }

  async forkSnapshot(
    sessionId: string,
    snapshotId: string,
    name?: string
  ): Promise<{ session_id: string; name: string; forked_from: string }> {
    return this.post(`/sessions/${sessionId}/snapshots/${snapshotId}/fork`, { name });
  }

  async createSnapshot(
    sessionId: string,
    description?: string
  ): Promise<{ snapshot_id: string }> {
    return this.post(`/sessions/${sessionId}/snapshots`, { description });
  }

  async restoreSnapshot(
    sessionId: string,
    snapshotId: string
  ): Promise<void> {
    await this.post(
      `/sessions/${sessionId}/snapshots/${snapshotId}/restore`,
      {}
    );
  }

  async deleteSnapshot(
    sessionId: string,
    snapshotId: string
  ): Promise<void> {
    await this.del(`/sessions/${sessionId}/snapshots/${snapshotId}`);
  }

  async fileComplete(
    prefix: string,
    limit: number = 10
  ): Promise<{ completions: Array<{ path: string; is_directory: boolean; size: number | null }> }> {
    return this.get(
      `/files/complete?prefix=${encodeURIComponent(prefix)}&limit=${limit}`
    );
  }

  async getFileChanges(
    sessionId: string
  ): Promise<{
    file_changes: Record<
      string,
      Array<{
        file_path: string;
        agent_id: string;
        change_type: string;
        tool_call_id: string;
        tool_name: string;
        message_index: number;
        old_content: string | null;
        new_content: string | null;
        pre_tool_content: string | null;
        added_ranges: Array<{ startLine: number; endLine: number }>;
        removed_ranges: Array<{ startLine: number; endLine: number }>;
        timestamp: string;
        status: string;
      }>
    >;
  }> {
    return this.get(`/sessions/${sessionId}/file-changes`);
  }

  async acceptFileChange(
    sessionId: string,
    toolCallId: string
  ): Promise<{ status: string }> {
    return this.post(
      `/sessions/${sessionId}/file-changes/${toolCallId}/accept`,
      {}
    );
  }

  async rejectFileChange(
    sessionId: string,
    toolCallId: string
  ): Promise<{
    status: string;
    old_content: string | null;
    new_content: string | null;
    pre_tool_content: string | null;
  }> {
    return this.post(
      `/sessions/${sessionId}/file-changes/${toolCallId}/reject`,
      {}
    );
  }

  async acceptAllFileChanges(
    sessionId: string
  ): Promise<{ status: string; count: number }> {
    return this.post(
      `/sessions/${sessionId}/file-changes/accept-all`,
      {}
    );
  }

  async rejectAllFileChanges(
    sessionId: string
  ): Promise<{
    status: string;
    count: number;
    changes: Array<{
      tool_call_id: string;
      file_path: string;
      old_content: string | null;
      new_content: string | null;
      pre_tool_content: string | null;
    }>;
  }> {
    return this.post(
      `/sessions/${sessionId}/file-changes/reject-all`,
      {}
    );
  }

  async getToolRationale(
    sessionId: string,
    agentId: string,
    toolCallId: string
  ): Promise<{ rationale: string; tool_name: string; agent_name: string }> {
    return this.get(
      `/api/sessions/${sessionId}/agents/${agentId}/tool-rationale/${toolCallId}`
    );
  }

  async getAgentHistory(
    sessionId: string,
    agentId: string,
    detailLevel: "full" | "summary" = "full"
  ): Promise<{
    agent_id: string;
    session_id: string;
    detail_level: string;
    history: Array<{
      type: string;
      content?: string;
      timestamp: number;
      tool_name?: string;
      tool_id?: string;
      tool_args?: string;
      is_error?: boolean;
    }>;
  }> {
    return this.get(
      `/api/sessions/${sessionId}/agents/${agentId}/history?detail_level=${detailLevel}`
    );
  }

  // ── Config methods ──

  /** Fetch current PRSM configuration (raw YAML as JSON). */
  async getConfig(): Promise<{
    config: Record<string, unknown>;
    config_path: string;
    exists: boolean;
    runtime?: {
      providers?: Record<
        string,
        {
          type?: string;
          available?: boolean;
          supports_master?: boolean;
          api_key_env?: string;
        }
      >;
      models?: Record<
        string,
        {
          model_id?: string;
          provider?: string;
          available?: boolean;
          tier?: string;
          cost_factor?: number;
          speed_factor?: number;
          affinities?: Record<string, number>;
        }
      >;
      model_aliases?: Record<
        string,
        {
          model_id?: string;
          provider?: string;
          reasoning_effort?: string;
        }
      >;
    };
  }> {
    return this.request("GET", "/config");
  }

  /** Update PRSM configuration: writes .prism/prsm.yaml + ~/.prsm/models.yaml and reloads. */
  async updateConfig(config: Record<string, unknown>): Promise<{ ok: boolean; config_path: string }> {
    return this.request("PUT", "/config", { config });
  }

  /** Fetch thinking verb lists loaded from prsm/shared_ui/*.txt on the server. */
  async getThinkingVerbs(): Promise<{ safe: string[]; nsfw: string[] }> {
    return this.request("GET", "/config/thinking-verbs");
  }

  /** Auto-detect available providers and models on the server. */
  async detectProviders(): Promise<{
    providers: Record<string, { type: string; api_key_env?: string }>;
    models: Record<string, { provider: string; model_id: string }>;
  }> {
    return this.request("GET", "/config/detect-providers");
  }

  /** Set the model for a session. */
  async setModel(
    sessionId: string,
    modelId: string
  ): Promise<{ status: string; model_id: string }> {
    return this.request("POST", `/sessions/${sessionId}/model`, { model_id: modelId });
  }

  /** Get available models for a session. */
  async getAvailableModels(
    sessionId: string
  ): Promise<{
    models: Array<{
      model_id: string;
      display_name?: string;
      provider: string;
      tier: string;
      available: boolean;
      is_current: boolean;
      is_legacy?: boolean;
    }>;
  }> {
    return this.request("GET", `/sessions/${sessionId}/models`);
  }

  // ── Private helpers ──

  private readPort(): Promise<number> {
    return new Promise((resolve, reject) => {
      let settled = false;
      let stderrBuffer = "";

      const cleanup = () => {
        settled = true;
        clearTimeout(timer);
      };

      const timer = setTimeout(() => {
        if (settled) return;
        cleanup();
        const hint = stderrBuffer
          ? `\nServer stderr: ${stderrBuffer.slice(0, 500)}`
          : "\nCheck PRSM output channel for details.";
        reject(new Error(`Timed out waiting for server port.${hint}`));
      }, 15000);

      // Capture stderr during startup for better error messages
      const onStderr = (data: Buffer) => {
        stderrBuffer += data.toString();
      };
      this.process?.stderr?.on("data", onStderr);

      // Detect early process exit
      const onExit = (code: number | null) => {
        if (settled) return;
        cleanup();
        const hint = stderrBuffer
          ? `\n${stderrBuffer.slice(0, 500)}`
          : "";
        reject(
          new Error(
            `prsm server exited with code ${code} before printing port.${hint}`
          )
        );
      };
      this.process?.on("exit", onExit);

      // Detect spawn error (e.g., executable not found)
      const onError = (err: Error) => {
        if (settled) return;
        cleanup();
        reject(
          new Error(`Failed to spawn prsm: ${err.message}`)
        );
      };
      this.process?.on("error", onError);

      let buffer = "";
      const onData = (data: Buffer) => {
        buffer += data.toString();
        const newline = buffer.indexOf("\n");
        if (newline >= 0) {
          const line = buffer.slice(0, newline).trim();
          this.process?.stdout?.removeListener("data", onData);
          this.process?.stderr?.removeListener("data", onStderr);
          this.process?.removeListener("exit", onExit);
          this.process?.removeListener("error", onError);
          cleanup();

          try {
            const parsed = JSON.parse(line);
            if (typeof parsed.port === "number") {
              resolve(parsed.port);
            } else {
              reject(new Error(`Invalid port response: ${line}`));
            }
          } catch {
            reject(new Error(`Failed to parse port: ${line}`));
          }

          // Continue piping remaining stdout to output channel
          if (buffer.length > newline + 1) {
            this.outputChannel.appendLine(
              `[prsm] ${buffer.slice(newline + 1).trim()}`
            );
          }
          this.process?.stdout?.on("data", (d: Buffer) => {
            this.outputChannel.appendLine(`[prsm] ${d.toString().trim()}`);
          });
        }
      };
      this.process?.stdout?.on("data", onData);
    });
  }

  private async connectSSE(): Promise<void> {
    this.sseAbort = new AbortController();
    const url = `http://127.0.0.1:${this.port}/events`;
    this.outputChannel.appendLine(`Connecting SSE: ${url}`);

    // Use Node's http to create a long-lived SSE connection
    this.startSSEConnection(url);
  }

  private startSSEConnection(url: string): void {
    const req = http.get(url, (res) => {
      let buffer = "";

      res.on("data", (chunk: Buffer) => {
        buffer += chunk.toString();

        // Parse SSE format: "event: <type>\ndata: <json>\n\n"
        const parts = buffer.split("\n\n");
        // Keep last incomplete part in buffer
        buffer = parts.pop() ?? "";

        for (const part of parts) {
          if (!part.trim()) continue;

          let eventType = "";
          let data = "";

          for (const line of part.split("\n")) {
            if (line.startsWith("event: ")) {
              eventType = line.slice(7).trim();
            } else if (line.startsWith("data: ")) {
              data = line.slice(6);
            } else if (line.startsWith(": ")) {
              // Comment (keepalive), ignore
            }
          }

          if (eventType && data) {
            try {
              const parsed: SSEEventData = JSON.parse(data);
              this.emit("event", eventType, parsed);
            } catch {
              this.outputChannel.appendLine(
                `Failed to parse SSE data: ${data.slice(0, 200)}`
              );
            }
          }
        }
      });

      res.on("end", () => {
        this.outputChannel.appendLine("SSE connection closed");
        // Reconnect after 2 seconds if still connected
        if (this._connected && this.port) {
          setTimeout(() => {
            if (this._connected && this.port) {
              this.outputChannel.appendLine("Reconnecting SSE...");
              this.startSSEConnection(url);
            }
          }, 2000);
        }
      });

      res.on("error", (err) => {
        this.outputChannel.appendLine(`SSE error: ${err.message}`);
      });
    });

    req.on("error", (err) => {
      this.outputChannel.appendLine(`SSE request error: ${err.message}`);
    });

    // Store reference for cleanup
    if (this.sseAbort) {
      this.sseAbort.signal.addEventListener("abort", () => {
        req.destroy();
      });
    }
  }

  private baseUrl(): string {
    if (!this.port) throw new Error("Not connected");
    return `http://127.0.0.1:${this.port}`;
  }

  private async get<T>(path: string): Promise<T> {
    return this.request<T>("GET", path);
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    return this.request<T>("POST", path, body);
  }

  private async del<T>(path: string): Promise<T> {
    return this.request<T>("DELETE", path);
  }

  private request<T>(
    method: string,
    reqPath: string,
    body?: unknown
  ): Promise<T> {
    return this.requestRaw<T>(method, reqPath, body, this.port!);
  }

  private requestRaw<T>(
    method: string,
    reqPath: string,
    body?: unknown,
    port?: number
  ): Promise<T> {
    const actualPort = port ?? this.port;
    if (!actualPort) throw new Error("Not connected");

    return new Promise((resolve, reject) => {
      this.requestSeq += 1;
      const reqId = `req-${this.requestSeq}`;
      const url = new URL(reqPath, `http://127.0.0.1:${actualPort}`);
      const started = Date.now();
      this.outputChannel.appendLine(
        `[http ${reqId}] -> ${method} ${url.pathname}${url.search} port=${actualPort}`
      );

      const options: http.RequestOptions = {
        method,
        hostname: url.hostname,
        port: url.port,
        path: url.pathname + url.search,
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
          "x-prsm-request-id": reqId,
        },
        timeout: 30000,
      };

      const req = http.request(options, (res) => {
        let data = "";
        res.on("data", (chunk) => (data += chunk));
        res.on("end", () => {
          const durationMs = Date.now() - started;
          this.outputChannel.appendLine(
            `[http ${reqId}] <- status=${res.statusCode ?? "?"} duration_ms=${durationMs}`
          );
          if (res.statusCode && res.statusCode >= 400) {
            reject(new Error(`HTTP ${res.statusCode}: ${data.slice(0, 200)}`));
            return;
          }
          try {
            resolve(JSON.parse(data) as T);
          } catch {
            resolve(data as unknown as T);
          }
        });
      });

      req.on("error", (err: NodeJS.ErrnoException) => {
        const durationMs = Date.now() - started;
        this.outputChannel.appendLine(
          `[http ${reqId}] !! ${err.code ?? "ERR"} ${err.message} duration_ms=${durationMs}`
        );
        // Mark disconnected on connection-level failures so callers can
        // restart/reconnect instead of repeatedly using a dead port.
        if (
          err.code === "ECONNREFUSED" ||
          err.code === "ECONNRESET" ||
          err.code === "EPIPE"
        ) {
          this._connected = false;
        }
        reject(err);
      });
      req.on("timeout", () => {
        this.outputChannel.appendLine(`[http ${reqId}] !! timeout`);
        req.destroy(new Error("Request timed out"));
      });

      if (body !== undefined) {
        req.write(JSON.stringify(body));
      }
      req.end();
    });
  }
}
