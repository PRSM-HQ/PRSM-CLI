/**
 * Settings panel webview for configuring .prism/prsm.yaml and
 * global ~/.prsm/models.yaml via the server API.
 *
 * Displays engine, provider, model, defaults, and model_registry settings
 * in a form-based UI.
 */
import * as vscode from "vscode";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { parse as parseYaml } from "yaml";
import { PrsmTransport } from "../protocol/transport";

export class SettingsPanel {
  public static readonly viewType = "prsmSettings";
  private static instance: SettingsPanel | undefined;
  private readonly panel: vscode.WebviewPanel;
  private readonly getTransport: () => PrsmTransport | undefined;
  private readonly outputChannel: vscode.OutputChannel;
  private disposed = false;
  private retryTimer: NodeJS.Timeout | undefined;
  private retryAttempts = 0;
  private static readonly MAX_RETRY_ATTEMPTS = 15;
  private static readonly RETRY_DELAY_MS = 1000;

  private hasSectionData(config: Record<string, unknown>): boolean {
    const keys = ["defaults", "providers", "models", "model_registry"];
    return keys.some((key) => {
      const value = config[key];
      return !!value && typeof value === "object" && Object.keys(value as object).length > 0;
    });
  }

  private deriveModelRegistryFromModels(
    models: Record<string, unknown>,
  ): Record<string, Record<string, unknown>> {
    const derived: Record<string, Record<string, unknown>> = {};
    for (const [alias, rawCfg] of Object.entries(models || {})) {
      if (!rawCfg || typeof rawCfg !== "object") continue;
      const cfg = rawCfg as Record<string, unknown>;
      const hasCaps =
        "tier" in cfg || "cost_factor" in cfg || "speed_factor" in cfg || "affinities" in cfg;
      if (!hasCaps) continue;

      const modelId = String(cfg.model_id ?? alias);
      const effort = cfg.reasoning_effort;
      const registryId = effort ? `${modelId}::reasoning_effort=${String(effort)}` : modelId;
      derived[registryId] = {
        provider: cfg.provider,
        tier: cfg.tier,
        cost_factor: cfg.cost_factor,
        speed_factor: cfg.speed_factor,
      };
      if (cfg.affinities && typeof cfg.affinities === "object") {
        (derived[registryId] as Record<string, unknown>).affinities = cfg.affinities;
      }
    }
    return derived;
  }

  private loadWorkspaceConfigFallback(): {
    config: Record<string, unknown>;
    configPath: string;
    exists: boolean;
  } | undefined {
    const folders = vscode.workspace.workspaceFolders ?? [];
    if (folders.length === 0) return undefined;

    const preferredRoot =
      folders.find((f) =>
        fs.existsSync(path.join(f.uri.fsPath, ".prism", "prsm.yaml"))
      )?.uri.fsPath ??
      folders.find((f) => fs.existsSync(path.join(f.uri.fsPath, "prsm.yaml")))?.uri.fsPath ??
      folders[0]?.uri.fsPath;

    if (!preferredRoot) return undefined;

    const prismPrsm = path.join(preferredRoot, ".prism", "prsm.yaml");
    const legacyPrsm = path.join(preferredRoot, "prsm.yaml");
    const prsmPath = fs.existsSync(prismPrsm) ? prismPrsm : legacyPrsm;
    if (!fs.existsSync(prsmPath)) return undefined;

    try {
      const prsmRaw = (parseYaml(fs.readFileSync(prsmPath, "utf-8")) || {}) as Record<string, unknown>;

      const merged: Record<string, unknown> = { ...prsmRaw };
      const mergedModels = {
        ...((prsmRaw.models as Record<string, unknown>) || {}),
      };
      if (Object.keys(mergedModels).length > 0) {
        merged.models = mergedModels;
      }

      let mergedRegistry = {
        ...((prsmRaw.model_registry as Record<string, unknown>) || {}),
      };
      if (Object.keys(mergedRegistry).length === 0 && Object.keys(mergedModels).length > 0) {
        mergedRegistry = this.deriveModelRegistryFromModels(mergedModels);
      }
      if (Object.keys(mergedRegistry).length > 0) {
        merged.model_registry = mergedRegistry;
      }

      return {
        config: merged,
        configPath: prsmPath,
        exists: true,
      };
    } catch (err) {
      this.outputChannel.appendLine(
        `[Settings] Local fallback parse failed: ${(err as Error).message}`
      );
      return undefined;
    }
  }

  private buildFallbackRuntime(config: Record<string, unknown>): {
    providers: Record<string, { type: string; api_key_env?: string }>;
    models: Record<string, { provider: string; model_id: string; available: boolean }>;
    model_aliases: Record<string, { provider: string; model_id: string }>;
  } {
    const providers = (config.providers as Record<string, unknown>) || {};
    const runtimeProviders: Record<string, { type: string; api_key_env?: string }> = {};
    for (const [name, rawCfg] of Object.entries(providers)) {
      const cfg = rawCfg as Record<string, unknown>;
      if (!cfg || typeof cfg !== "object") continue;
      const type = cfg.type;
      if (typeof type !== "string" || !type) continue;
      runtimeProviders[name] = { type };
      if (typeof cfg.api_key_env === "string") {
        runtimeProviders[name].api_key_env = cfg.api_key_env;
      }
    }

    const models = (config.models as Record<string, unknown>) || {};
    const runtimeModels: Record<string, { provider: string; model_id: string; available: boolean }> = {};
    const runtimeAliases: Record<string, { provider: string; model_id: string }> = {};

    for (const [alias, rawCfg] of Object.entries(models)) {
      const cfg = rawCfg as Record<string, unknown>;
      if (!cfg || typeof cfg !== "object") continue;
      const provider =
        typeof cfg.provider === "string" && cfg.provider.trim().length > 0
          ? cfg.provider
          : "unknown";
      const modelId =
        typeof cfg.model_id === "string" && cfg.model_id.trim().length > 0
          ? cfg.model_id
          : alias;
      runtimeAliases[alias] = {
        provider,
        model_id: modelId,
      };
      if (!runtimeModels[modelId]) {
        runtimeModels[modelId] = {
          provider,
          model_id: modelId,
          available: true,
        };
      }
    }

    return {
      providers: runtimeProviders,
      models: runtimeModels,
      model_aliases: runtimeAliases,
    };
  }

  public static createOrShow(
    context: vscode.ExtensionContext,
    getTransport: () => PrsmTransport | undefined,
    outputChannel: vscode.OutputChannel,
  ): SettingsPanel {
    if (SettingsPanel.instance && !SettingsPanel.instance.disposed) {
      SettingsPanel.instance.panel.reveal(vscode.ViewColumn.One);
      SettingsPanel.instance.loadConfig();
      return SettingsPanel.instance;
    }

    const panel = vscode.window.createWebviewPanel(
      SettingsPanel.viewType,
      "PRSM Settings",
      vscode.ViewColumn.One,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [],
      },
    );

    const instance = new SettingsPanel(panel, context, getTransport, outputChannel);
    SettingsPanel.instance = instance;
    return instance;
  }

  private constructor(
    panel: vscode.WebviewPanel,
    context: vscode.ExtensionContext,
    getTransport: () => PrsmTransport | undefined,
    outputChannel: vscode.OutputChannel,
  ) {
    this.panel = panel;
    this.getTransport = getTransport;
    this.outputChannel = outputChannel;

    // IMPORTANT: Register message handler BEFORE setting HTML to avoid
    // race condition where the webview posts 'ready' before the handler
    // is attached.
    this.panel.onDidDispose(() => {
      this.disposed = true;
      if (this.retryTimer) {
        clearTimeout(this.retryTimer);
        this.retryTimer = undefined;
      }
      SettingsPanel.instance = undefined;
    });

    this.panel.webview.onDidReceiveMessage(async (msg) => {
      switch (msg.type) {
        case "ready":
          this.outputChannel.appendLine("[Settings] Received 'ready' from webview");
          await this.loadConfig();
          break;
        case "save":
          await this.saveConfig(msg.config, msg.extensionSettings);
          break;
        case "reload":
          await this.loadConfig();
          break;
        case "importFromProvider":
          await this.handleImportFromProvider(msg.provider);
          break;
        case "importFile":
          await this.handleImportFile();
          break;
        case "exportArchive":
          await this.handleExportArchive();
          break;
        // detectProviders no longer needed — runtime info is included
        // in the configLoaded message automatically
      }
    });

    // Set HTML after handler is registered
    this.panel.webview.html = this.getHtml();
    this.panel.iconPath = vscode.Uri.file(
      path.join(context.extensionPath, "media", "icons", "prsm.svg")
    );
    // Also kick off a load from the extension side immediately.
    // If the webview-side "ready" message is delayed or missed, this
    // still hydrates settings once transport is available.
    void this.loadConfig();
  }

  private async loadConfig(): Promise<void> {
    this.outputChannel.appendLine("[Settings] loadConfig() called");

    // Quick retry if transport isn't connected yet — try 3 times with short
    // 500ms delays (1.5s max) instead of the old 5×2s (10s) loop.
    let transport = this.getTransport();
    if (!transport || !transport.isConnected) {
      this.outputChannel.appendLine(
        `[Settings] Transport not ready (exists=${!!transport}, connected=${transport?.isConnected}), retrying...`
      );
      for (let i = 0; i < 3; i++) {
        await new Promise((r) => setTimeout(r, 500));
        if (this.disposed) return;
        transport = this.getTransport();
        if (transport?.isConnected) {
          this.outputChannel.appendLine(`[Settings] Transport connected after ${i + 1} retries`);
          break;
        }
        this.outputChannel.appendLine(
          `[Settings] Retry ${i + 1}/3: transport=${!!transport}, connected=${transport?.isConnected}`
        );
      }
    }

    if (!transport || !transport.isConnected) {
      const fallback = this.loadWorkspaceConfigFallback();
      const fallbackRuntime = fallback?.config
        ? this.buildFallbackRuntime(fallback.config)
        : { providers: {}, models: {}, model_aliases: {} };
      this.outputChannel.appendLine(
        `[Settings] Transport unavailable after all retries (transport=${!!transport}, connected=${transport?.isConnected})`
      );
      this.panel.webview.postMessage({
        type: "configLoaded",
        config: fallback?.config ?? {},
        configPath: fallback?.configPath ?? "",
        exists: fallback?.exists ?? false,
        error: "Server not connected. Start a session first, then reopen settings.",
        runtime: fallbackRuntime,
        extensionSettings: this.getExtensionSettings(),
      });
      this.scheduleRetry("transport not connected");
      return;
    }

    try {
      this.outputChannel.appendLine("[Settings] Calling transport.getConfig()...");
      // Race the actual fetch against a 15-second timeout so the webview
      // never stays in a loading state if the server is slow/hung.
      const timeout = new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error("Config fetch timed out after 15s")), 15000)
      );
      const resp = await Promise.race([transport.getConfig(), timeout]);
      const respConfig = ((resp as { config?: Record<string, unknown> }).config || {}) as Record<string, unknown>;
      const fallback = this.loadWorkspaceConfigFallback();
      const fallbackRuntime = fallback?.config
        ? this.buildFallbackRuntime(fallback.config)
        : { providers: {}, models: {}, model_aliases: {} };
      const mergedConfig: Record<string, unknown> = { ...respConfig };
      if (fallback?.config) {
        const fallbackConfig = fallback.config;
        const sections = ["engine", "defaults", "providers", "models", "model_registry", "experts"];
        for (const section of sections) {
          const current = mergedConfig[section];
          const replacement = fallbackConfig[section];
          const currentEmpty =
            !current || (typeof current === "object" && Object.keys(current as object).length === 0);
          if (currentEmpty && replacement !== undefined) {
            mergedConfig[section] = replacement;
          }
        }
      }
      const useFallbackPath =
        (!respConfig || !this.hasSectionData(respConfig)) && !!fallback?.configPath;
      const hasRuntime = (() => {
        const runtime = (resp as { runtime?: Record<string, Record<string, unknown>> }).runtime;
        if (!runtime || typeof runtime !== "object") return false;
        return (
          Object.keys(runtime.providers || {}).length > 0 ||
          Object.keys(runtime.models || {}).length > 0 ||
          Object.keys(runtime.model_aliases || {}).length > 0
        );
      })();
      this.outputChannel.appendLine(
        `[Settings] Config loaded: exists=${resp.exists}, path=${resp.config_path}, ` +
        `runtime_providers=${Object.keys((resp as any).runtime?.providers || {}).length}, ` +
        `runtime_models=${Object.keys((resp as any).runtime?.models || {}).length}`
      );
      this.panel.webview.postMessage({
        type: "configLoaded",
        config: mergedConfig,
        configPath: useFallbackPath ? fallback!.configPath : resp.config_path,
        exists: useFallbackPath ? true : resp.exists,
        runtime: hasRuntime ? ((resp as any).runtime || {}) : fallbackRuntime,
        extensionSettings: this.getExtensionSettings(),
      });
      this.retryAttempts = 0;
      if (this.retryTimer) {
        clearTimeout(this.retryTimer);
        this.retryTimer = undefined;
      }
    } catch (err) {
      const fallback = this.loadWorkspaceConfigFallback();
      const fallbackRuntime = fallback?.config
        ? this.buildFallbackRuntime(fallback.config)
        : { providers: {}, models: {}, model_aliases: {} };
      this.outputChannel.appendLine(`[Settings] getConfig() failed: ${(err as Error).message}`);
      this.panel.webview.postMessage({
        type: "configLoaded",
        config: fallback?.config ?? {},
        configPath: fallback?.configPath ?? "",
        exists: fallback?.exists ?? false,
        error: (err as Error).message,
        runtime: fallbackRuntime,
        extensionSettings: this.getExtensionSettings(),
      });
      this.scheduleRetry("getConfig failed");
    }
  }

  private scheduleRetry(reason: string): void {
    if (this.disposed || this.retryTimer) return;
    if (this.retryAttempts >= SettingsPanel.MAX_RETRY_ATTEMPTS) {
      this.outputChannel.appendLine(
        `[Settings] Reached max retries (${SettingsPanel.MAX_RETRY_ATTEMPTS}) after ${reason}`
      );
      return;
    }
    this.retryAttempts += 1;
    this.outputChannel.appendLine(
      `[Settings] Scheduling retry ${this.retryAttempts}/${SettingsPanel.MAX_RETRY_ATTEMPTS} after ${reason}`
    );
    this.retryTimer = setTimeout(() => {
      this.retryTimer = undefined;
      void this.loadConfig();
    }, SettingsPanel.RETRY_DELAY_MS);
  }

  private getExtensionSettings(): {
    sessionInactivityMinutes: number;
    enableNsfwThinkingVerbs: boolean;
    customThinkingVerbs: string[];
    commandWhitelist: string[];
    commandBlacklist: string[];
  } {
    const cfg = vscode.workspace.getConfiguration("prsm");
    return {
      sessionInactivityMinutes: cfg.get<number>("engine.sessionInactivityMinutes", 15),
      enableNsfwThinkingVerbs: cfg.get<boolean>("thinkingVerbs.enableNsfw", true),
      customThinkingVerbs: cfg.get<string[]>("thinkingVerbs.custom", []),
      commandWhitelist: cfg.get<string[]>("security.commandWhitelist", []),
      commandBlacklist: cfg.get<string[]>("security.commandBlacklist", []),
    };
  }

  private async saveConfig(
    config: Record<string, unknown>,
    extensionSettings?: Record<string, unknown>,
  ): Promise<void> {
    const transport = this.getTransport();
    if (!transport) {
      vscode.window.showErrorMessage("Cannot save: server not connected");
      return;
    }

    try {
      if (extensionSettings && typeof extensionSettings.sessionInactivityMinutes === "number") {
        const minutes = Math.max(0, extensionSettings.sessionInactivityMinutes);
        const cfg = vscode.workspace.getConfiguration("prsm");
        await cfg.update(
          "engine.sessionInactivityMinutes",
          minutes,
          vscode.ConfigurationTarget.Workspace,
        );
        await cfg.update(
          "thinkingVerbs.enableNsfw",
          extensionSettings.enableNsfwThinkingVerbs !== false,
          vscode.ConfigurationTarget.Workspace,
        );
        await cfg.update(
          "thinkingVerbs.custom",
          Array.isArray(extensionSettings.customThinkingVerbs)
            ? extensionSettings.customThinkingVerbs
            : [],
          vscode.ConfigurationTarget.Workspace,
        );

        // Persist command policy lists
        const whitelist = Array.isArray(extensionSettings.commandWhitelist)
          ? extensionSettings.commandWhitelist
          : [];
        const blacklist = Array.isArray(extensionSettings.commandBlacklist)
          ? extensionSettings.commandBlacklist
          : [];
        await cfg.update(
          "security.commandWhitelist",
          whitelist,
          vscode.ConfigurationTarget.Workspace,
        );
        await cfg.update(
          "security.commandBlacklist",
          blacklist,
          vscode.ConfigurationTarget.Workspace,
        );

        // Sync to .prism/ text files so the engine picks them up
        const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
        if (workspaceRoot) {
          const prismDir = path.join(workspaceRoot, ".prism");
          fs.mkdirSync(prismDir, { recursive: true });
          fs.writeFileSync(
            path.join(prismDir, "command_whitelist.txt"),
            whitelist.length > 0 ? whitelist.join("\n") + "\n" : "",
            "utf-8",
          );
          fs.writeFileSync(
            path.join(prismDir, "command_blacklist.txt"),
            blacklist.length > 0 ? blacklist.join("\n") + "\n" : "",
            "utf-8",
          );
        }
      }

      const resp = await transport.updateConfig(config);
      this.panel.webview.postMessage({ type: "saved", configPath: resp.config_path });
      vscode.window.showInformationMessage(
        `PRSM settings saved to ${path.basename(resp.config_path)} and reloaded.`
      );
      this.outputChannel.appendLine(
        `Settings saved to ${resp.config_path} and configuration reloaded`
      );
    } catch (err) {
      vscode.window.showErrorMessage(`Failed to save settings: ${(err as Error).message}`);
      this.panel.webview.postMessage({ type: "saveError", error: (err as Error).message });
    }
  }

  private async handleImportFromProvider(provider: string): Promise<void> {
    const transport = this.getTransport();
    if (!transport || !transport.isConnected) {
      vscode.window.showErrorMessage("Cannot import: server not connected. Start a session first.");
      return;
    }
    try {
      const sessions = await transport.listImportSessions(provider as "all" | "codex" | "claude" | "prsm");
      if (!sessions.sessions || sessions.sessions.length === 0) {
        vscode.window.showInformationMessage(`No importable sessions found for ${provider}.`);
        return;
      }
      const items = sessions.sessions.map((s: any) => ({
        label: s.title || "(untitled)",
        description: `${s.provider} — ${s.turn_count} turns`,
        detail: s.source_session_id,
      }));
      const pick = await vscode.window.showQuickPick(items, {
        title: `Import from ${provider}`,
        placeHolder: "Select a session to import",
      });
      if (!pick) return;
      vscode.window.showInformationMessage(
        `Use /import run ${provider} ${pick.detail} in the chat to complete the import.`
      );
    } catch (err) {
      vscode.window.showErrorMessage(`Import listing failed: ${(err as Error).message}`);
    }
  }

  private async handleImportFile(): Promise<void> {
    const uris = await vscode.window.showOpenDialog({
      canSelectMany: false,
      filters: {
        "Session Archives": ["zip", "tar.gz", "tgz", "tar", "prsm"],
        "All Files": ["*"],
      },
      title: "Select session archive to import",
    });
    if (!uris || uris.length === 0) return;
    const filePath = uris[0].fsPath;
    const transport = this.getTransport();
    if (!transport || !transport.isConnected) {
      vscode.window.showErrorMessage("Cannot import: server not connected. Start a session first.");
      return;
    }

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
      const result = await transport.importArchive(filePath, conflictChoice.value);
      if (result.success) {
        const parts: string[] = [];
        if (result.sessions_imported) parts.push(`${result.sessions_imported} session(s)`);
        if (result.files_imported) parts.push(`${result.files_imported} file(s)`);
        let msg = `Imported ${parts.join(", ") || "no new files"}`;
        if (result.sessions_skipped) msg += ` (${result.sessions_skipped} already existed)`;
        vscode.window.showInformationMessage(msg);
      } else {
        vscode.window.showErrorMessage(`Import failed: ${result.error || "Unknown error"}`);
      }
    } catch (err) {
      vscode.window.showErrorMessage(`Import failed: ${(err as Error).message}`);
    }
  }

  private async handleExportArchive(): Promise<void> {
    const transport = this.getTransport();
    if (!transport || !transport.isConnected) {
      vscode.window.showErrorMessage("Cannot export: server not connected. Start a session first.");
      return;
    }

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
      const result = await transport.exportArchive("", outputPath, format as "tar.gz" | "zip");
      if (result.success) {
        const manifest = result.manifest as Record<string, unknown> | null;
        const count = manifest?.session_count ?? "unknown";
        vscode.window.showInformationMessage(`Exported ${count} session(s) to ${path.basename(outputPath)}`);
      } else {
        vscode.window.showErrorMessage(`Export failed: ${result.error || "Unknown error"}`);
      }
    } catch (err) {
      vscode.window.showErrorMessage(`Export failed: ${(err as Error).message}`);
    }
  }

  private getHtml(): string {
    return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline';">
<title>PRSM Settings</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  padding: 20px;
  color: var(--vscode-foreground);
  background: var(--vscode-editor-background);
  font-family: var(--vscode-font-family);
  font-size: var(--vscode-font-size);
  line-height: 1.5;
}
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--vscode-panel-border);
}
.header h1 {
  font-size: 18px;
  font-weight: 600;
}
.config-path {
  font-size: 12px;
  color: var(--vscode-descriptionForeground);
  font-family: var(--vscode-editor-font-family, monospace);
}
.actions {
  display: flex;
  gap: 8px;
}
.btn {
  padding: 6px 14px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
  font-family: var(--vscode-font-family);
}
.btn-primary {
  background: var(--vscode-button-background);
  color: var(--vscode-button-foreground);
}
.btn-primary:hover { background: var(--vscode-button-hoverBackground); }
.btn-secondary {
  background: var(--vscode-button-secondaryBackground);
  color: var(--vscode-button-secondaryForeground);
}
.btn-secondary:hover { background: var(--vscode-button-secondaryHoverBackground); }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.section {
  margin-bottom: 24px;
  border: 1px solid var(--vscode-panel-border);
  border-radius: 6px;
  overflow: hidden;
}
.section-header {
  padding: 10px 16px;
  background: var(--vscode-sideBar-background);
  border-bottom: 1px solid var(--vscode-panel-border);
  font-weight: 600;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  user-select: none;
}
.section-header .chevron {
  font-size: 10px;
  transition: transform 0.15s;
}
.section-header.collapsed .chevron { transform: rotate(-90deg); }
.section-body { padding: 16px; }
.section-header.collapsed + .section-body { display: none; }
.field {
  margin-bottom: 14px;
}
.field:last-child { margin-bottom: 0; }
.field label {
  display: block;
  font-size: 12px;
  font-weight: 600;
  margin-bottom: 4px;
  color: var(--vscode-foreground);
}
.field .description {
  font-size: 11px;
  color: var(--vscode-descriptionForeground);
  margin-bottom: 4px;
}
input[type="text"], input[type="number"], select, textarea {
  width: 100%;
  padding: 6px 10px;
  background: var(--vscode-input-background);
  color: var(--vscode-input-foreground);
  border: 1px solid var(--vscode-input-border, var(--vscode-panel-border));
  border-radius: 4px;
  font-family: var(--vscode-editor-font-family, monospace);
  font-size: 13px;
  outline: none;
}
input:focus, select:focus, textarea:focus {
  border-color: var(--vscode-focusBorder);
}
textarea { resize: vertical; min-height: 80px; }
.inline-fields {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
.model-row {
  display: grid;
  grid-template-columns: 120px 120px 1fr 110px 32px;
  gap: 8px;
  align-items: center;
  margin-bottom: 8px;
  padding: 6px 8px;
  background: var(--vscode-editor-inactiveSelectionBackground);
  border-radius: 4px;
}
.model-row input { font-size: 12px; padding: 4px 6px; }
.provider-row {
  display: grid;
  grid-template-columns: 16px 120px 100px 1fr 1fr 32px;
  gap: 8px;
  align-items: center;
  margin-bottom: 8px;
  padding: 6px 8px;
  background: var(--vscode-editor-inactiveSelectionBackground);
  border-radius: 4px;
}
.provider-row input, .provider-row select { font-size: 12px; padding: 4px 6px; }
.registry-row {
  margin-bottom: 12px;
  padding: 10px;
  background: var(--vscode-editor-inactiveSelectionBackground);
  border-radius: 4px;
}
.registry-row .registry-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.registry-row .registry-header strong {
  font-size: 13px;
}
.affinity-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 6px;
}
.affinity-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
}
.affinity-item label { font-size: 12px; font-weight: normal; margin: 0; flex: 1; }
.affinity-item input { width: 60px; text-align: center; padding: 2px 4px; }
.remove-btn {
  width: 24px;
  height: 24px;
  border: none;
  border-radius: 3px;
  background: transparent;
  color: var(--vscode-descriptionForeground);
  cursor: pointer;
  font-size: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.remove-btn:hover {
  background: var(--vscode-toolbar-hoverBackground);
  color: var(--vscode-errorForeground);
}
.add-btn {
  padding: 4px 12px;
  border: 1px dashed var(--vscode-panel-border);
  border-radius: 4px;
  background: transparent;
  color: var(--vscode-textLink-foreground);
  cursor: pointer;
  font-size: 12px;
  margin-top: 8px;
}
.add-btn:hover { background: var(--vscode-list-hoverBackground); }
.status-bar {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 8px 20px;
  background: var(--vscode-sideBar-background);
  border-top: 1px solid var(--vscode-panel-border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  z-index: 100;
}
.status-msg {
  font-size: 12px;
  color: var(--vscode-descriptionForeground);
}
.status-msg.error { color: var(--vscode-errorForeground); }
.status-msg.success { color: var(--vscode-testing-iconPassed); }
.dirty-indicator {
  display: none;
  font-size: 11px;
  color: var(--vscode-list-warningForeground);
  font-weight: 600;
}
.dirty-indicator.visible { display: inline; }
.avail-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-right: 4px;
}
.avail-dot.available { background: var(--vscode-testing-iconPassed); }
.avail-dot.unavailable { background: var(--vscode-errorForeground); }
.avail-label {
  font-size: 10px;
  color: var(--vscode-descriptionForeground);
}
.cmd-pattern-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 3px 6px;
  margin-bottom: 4px;
  background: var(--vscode-editor-inactiveSelectionBackground);
  border-radius: 3px;
  font-family: var(--vscode-editor-font-family, monospace);
  font-size: 12px;
}
.cmd-pattern-item span { flex: 1; word-break: break-all; }
body { padding-bottom: 60px; }
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>\u2699 PRSM Settings</h1>
    <div class="config-path" id="config-path"></div>
  </div>
  <div class="actions">
    <span class="dirty-indicator" id="dirty">\u25cf Unsaved changes</span>
    <button class="btn btn-secondary" id="reload-btn" title="Reload from file">\u21bb Reload</button>
    <button class="btn btn-primary" id="save-btn" disabled>Save &amp; Apply</button>
  </div>
</div>

<div id="content">
  <!-- Engine Settings -->
  <div class="section">
    <div class="section-header"><span class="chevron">\u25b6</span> Engine</div>
    <div class="section-body">
      <div class="inline-fields">
        <div class="field">
          <label>Max Agent Depth</label>
          <div class="description">Maximum nesting depth for spawned agents</div>
          <input type="number" id="engine.max_agent_depth" min="1" max="20" value="5">
        </div>
        <div class="field">
          <label>Max Concurrent Agents</label>
          <div class="description">Maximum number of agents running simultaneously</div>
          <input type="number" id="engine.max_concurrent_agents" min="1" max="100" value="10">
        </div>
        <div class="field">
          <label>Agent Timeout (seconds)</label>
          <div class="description">Per-agent cumulative reasoning timeout (0 disables timeout)</div>
          <input type="number" id="engine.agent_timeout_seconds" min="0" value="7200">
        </div>
        <div class="field">
          <label>Tool Call Timeout (seconds)</label>
          <div class="description">Per-tool-call execution timeout (0 disables timeout)</div>
          <input type="number" id="engine.tool_call_timeout_seconds" min="0" value="7200">
        </div>
        <div class="field">
          <label>User Question Timeout (seconds)</label>
          <div class="description">How long to wait for user responses to agent questions (0 disables timeout)</div>
          <input type="number" id="engine.user_question_timeout_seconds" min="0" value="0">
        </div>
        <div class="field">
          <label>Session Inactivity Kill (minutes)</label>
          <div class="description">Unload inactive agent runtime after this many minutes. Set 0 to disable. (Not recommended)</div>
          <input type="number" id="extension.sessionInactivityMinutes" min="0" step="1" value="15">
        </div>
        <div class="field">
          <label>Enable NSFW Thinking Verbs</label>
          <div class="description">When enabled, NSFW verbs are included in thinking indicator rotation.</div>
          <input type="checkbox" id="extension.enableNsfwThinkingVerbs" checked>
        </div>
        <div class="field" style="grid-column: 1 / -1;">
          <label>Custom Thinking Verbs</label>
          <div class="description">Add extra verbs (one per line).</div>
          <textarea id="extension.customThinkingVerbs" rows="4" placeholder="e.g.&#10;Debugging&#10;Refactoring"></textarea>
        </div>
      </div>
    </div>
  </div>

  <!-- Security / Command Policy -->
  <div class="section">
    <div class="section-header"><span class="chevron">\u25b6</span> Security</div>
    <div class="section-body">
      <div class="description" style="margin-bottom:12px;">Regex patterns controlling bash command auto-allow (whitelist) and always-deny (blacklist). Stored in <code>.prism/command_whitelist.txt</code> and <code>command_blacklist.txt</code>.</div>
      <div class="inline-fields" style="grid-template-columns:1fr 1fr;">
        <div class="field">
          <label style="color:var(--vscode-testing-iconPassed);">Whitelist (auto-allow)</label>
          <div class="description">Commands matching these patterns skip the permission prompt.</div>
          <div id="cmd-whitelist-list"></div>
          <div style="display:flex;gap:6px;margin-top:6px;">
            <input type="text" id="cmd-whitelist-input" placeholder="e.g. npm\\s+test" style="flex:1;">
            <button class="add-btn" id="add-whitelist-btn" style="margin-top:0;">+ Add</button>
          </div>
        </div>
        <div class="field">
          <label style="color:var(--vscode-errorForeground);">Blacklist (always deny)</label>
          <div class="description">Commands matching these patterns are always denied.</div>
          <div id="cmd-blacklist-list"></div>
          <div style="display:flex;gap:6px;margin-top:6px;">
            <input type="text" id="cmd-blacklist-input" placeholder="e.g. rm\\s+-rf\\s+/" style="flex:1;">
            <button class="add-btn" id="add-blacklist-btn" style="margin-top:0;">+ Add</button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Default Settings -->
  <div class="section">
    <div class="section-header"><span class="chevron">\u25b6</span> Defaults</div>
    <div class="section-body">
      <div class="inline-fields">
      <div class="field">
        <label>Default Model</label>
        <div class="description">Model alias used for new agents</div>
        <select id="defaults.model"></select>
      </div>
      <div class="field">
        <label>Peer Models</label>
        <div class="description">Select model aliases for peer consultations (Ctrl/Cmd-click for multiple)</div>
        <select id="defaults.peer_models" multiple size="4"></select>
      </div>
    </div>
  </div>
</div>

  <!-- Providers -->
  <div class="section">
    <div class="section-header"><span class="chevron">\u25b6</span> Providers</div>
    <div class="section-body">
      <div class="description" style="margin-bottom:8px;">Providers detected from your system. Green = available, red = CLI not found.</div>
      <div id="providers-list"></div>
      <button class="add-btn" id="add-provider-btn">+ Add Provider</button>
    </div>
  </div>

  <!-- Models -->
  <div class="section">
    <div class="section-header"><span class="chevron">\u25b6</span> Models</div>
    <div class="section-body">
      <div class="description" style="margin-bottom:8px;">Model aliases map short names to provider + model ID, with optional GPT/Codex reasoning effort.</div>
      <div style="font-size:11px;color:var(--vscode-descriptionForeground);margin-bottom:8px;">
        <span style="width:120px;display:inline-block">Alias</span>
        <span style="width:120px;display:inline-block">Provider</span>
        <span style="display:inline-block;min-width:220px;">Model ID</span>
        <span style="width:110px;display:inline-block">Reasoning</span>
      </div>
      <div id="models-list"></div>
      <button class="add-btn" id="add-model-btn">+ Add Model</button>
    </div>
  </div>

  <!-- Model Registry -->
  <div class="section">
    <div class="section-header collapsed"><span class="chevron">\u25b6</span> Model Registry</div>
    <div class="section-body">
      <div class="description" style="margin-bottom:12px;">Capability metadata for intelligent model selection. The orchestrator uses this to auto-select models based on task type.</div>
      <div id="registry-list"></div>
      <button class="add-btn" id="add-registry-btn">+ Add Registry Entry</button>
    </div>
  </div>
</div>

  <!-- Import / Export -->
  <div class="section">
    <div class="section-header"><span class="chevron">\u25b6</span> Import / Export</div>
    <div class="section-body">
      <div class="description" style="margin-bottom:12px;">Import conversations from external CLI tools or archive files, or export all sessions to a portable archive.</div>
      <div class="inline-fields">
        <div class="field">
          <label>Import from Provider</label>
          <div class="description">Select a transcript source to browse available sessions.</div>
          <select id="import-provider-select">
            <option value="codex">Codex</option>
            <option value="claude">Claude</option>
            <option value="gemini">Gemini</option>
          </select>
          <button class="btn btn-secondary" id="import-provider-btn" style="margin-top:8px;">Browse Sessions</button>
        </div>
        <div class="field">
          <label>Import Archive</label>
          <div class="description">Import sessions from a .prsm, .zip, or .tar.gz archive file.</div>
          <button class="btn btn-secondary" id="import-file-btn" style="margin-top:8px;">Choose File...</button>
        </div>
        <div class="field">
          <label>Export Sessions</label>
          <div class="description">Export all sessions to a portable archive (.tar.gz or .zip).</div>
          <button class="btn btn-secondary" id="export-archive-btn" style="margin-top:8px;">Export Archive...</button>
        </div>
      </div>
    </div>
  </div>

<div class="status-bar">
  <span class="status-msg" id="status"></span>
  <div class="actions">
    <button class="btn btn-primary" id="save-btn-bottom" disabled>Save &amp; Apply</button>
  </div>
</div>

<script>
const vscode = acquireVsCodeApi();
let currentConfig = {};
let isDirty = false;

// -- Section toggle --
function toggleSection(header) {
  header.classList.toggle('collapsed');
}

// -- Dirty tracking --
function markDirty() {
  isDirty = true;
  document.getElementById('dirty').classList.add('visible');
  document.getElementById('save-btn').disabled = false;
  document.getElementById('save-btn-bottom').disabled = false;
}

function markClean() {
  isDirty = false;
  document.getElementById('dirty').classList.remove('visible');
  document.getElementById('save-btn').disabled = true;
  document.getElementById('save-btn-bottom').disabled = true;
}

// -- Build config from form --
function buildConfig() {
  const config = {};

  // Engine
  const agentTimeout = parseInt(document.getElementById('engine.agent_timeout_seconds').value, 10);
  const toolTimeout = parseInt(document.getElementById('engine.tool_call_timeout_seconds').value, 10);
  const questionTimeout = parseInt(document.getElementById('engine.user_question_timeout_seconds').value, 10);

  config.engine = {
    max_agent_depth: parseInt(document.getElementById('engine.max_agent_depth').value) || 5,
    max_concurrent_agents: parseInt(document.getElementById('engine.max_concurrent_agents').value) || 10,
    agent_timeout_seconds: Number.isFinite(agentTimeout) ? agentTimeout : 7200,
    tool_call_timeout_seconds: Number.isFinite(toolTimeout) ? toolTimeout : 7200,
    user_question_timeout_seconds: Number.isFinite(questionTimeout) ? questionTimeout : 0,
  };
  // Include command policy in engine config for YAML round-trip
  const wl = getCmdPatterns('cmd-whitelist-list');
  const bl = getCmdPatterns('cmd-blacklist-list');
  if (wl.length > 0) config.engine.command_whitelist = wl;
  if (bl.length > 0) config.engine.command_blacklist = bl;

  // Defaults
  const model = document.getElementById('defaults.model').value.trim();
  const peerSelect = document.getElementById('defaults.peer_models');
  const peerModels = Array.from(peerSelect.selectedOptions)
    .map(option => option.value.trim())
    .filter(Boolean);
  config.defaults = {};
  if (model) config.defaults.model = model;
  if (peerModels.length > 0) config.defaults.peer_models = peerModels;

  // Providers
  config.providers = {};
  document.querySelectorAll('.provider-row').forEach(row => {
    const name = row.querySelector('.prov-name').value.trim();
    if (!name) return;
    const entry = { type: row.querySelector('.prov-type').value };
    const cmd = row.querySelector('.prov-command').value.trim();
    const apiKey = row.querySelector('.prov-apikey').value.trim();
    if (cmd) entry.command = cmd;
    if (apiKey) entry.api_key_env = apiKey;
    config.providers[name] = entry;
  });

  // Models
  config.models = {};
  document.querySelectorAll('.model-row').forEach(row => {
    const alias = row.querySelector('.model-alias').value.trim();
    if (!alias) return;
    const entry = {
      provider: row.querySelector('.model-provider').value.trim(),
      model_id: row.querySelector('.model-id').value.trim(),
    };
    const reasoning = row.querySelector('.model-reasoning').value.trim();
    if (reasoning) {
      entry.reasoning_effort = reasoning;
    }
    config.models[alias] = entry;
  });

  // Model Registry
  const registry = {};
  document.querySelectorAll('.registry-row').forEach(row => {
    const alias = (row.dataset.registryAlias || '').trim();
    const modelId = row.querySelector('.reg-model-id').value.trim();
    const registryAlias = alias || modelId;
    if (!registryAlias || !modelId) return;
    const entry = {
      tier: row.querySelector('.reg-tier').value,
      provider: row.querySelector('.reg-provider').value.trim(),
      cost_factor: parseFloat(row.querySelector('.reg-cost').value) || 1.0,
      speed_factor: parseFloat(row.querySelector('.reg-speed').value) || 1.0,
      model_id: modelId,
    };
    const affinities = {};
    row.querySelectorAll('.affinity-item').forEach(item => {
      const key = item.querySelector('.aff-key').value.trim();
      const val = parseFloat(item.querySelector('.aff-val').value);
      if (key && !isNaN(val)) affinities[key] = val;
    });
    if (Object.keys(affinities).length > 0) entry.affinities = affinities;
    registry[registryAlias] = entry;
  });
  if (Object.keys(registry).length > 0) config.model_registry = registry;

  // Preserve experts section from current config (not edited here)
  if (currentConfig.experts) config.experts = currentConfig.experts;
  if (currentConfig.plugins) config.plugins = currentConfig.plugins;

  return config;
}

function buildExtensionSettings() {
  const customRaw = document.getElementById('extension.customThinkingVerbs').value || '';
  const customThinkingVerbs = customRaw
    .split('\\n')
    .map((verb) => verb.trim())
    .filter((verb, idx, arr) => verb && arr.indexOf(verb) === idx);
  return {
    sessionInactivityMinutes: Math.max(
      0,
      parseInt(document.getElementById('extension.sessionInactivityMinutes').value) || 0
    ),
    enableNsfwThinkingVerbs: document.getElementById('extension.enableNsfwThinkingVerbs').checked,
    customThinkingVerbs,
    commandWhitelist: getCmdPatterns('cmd-whitelist-list'),
    commandBlacklist: getCmdPatterns('cmd-blacklist-list'),
  };
}

// -- Render functions --

// Tracks known provider types discovered from config + runtime.
// Populated by populateForm() before any provider rows are rendered.
let knownProviderTypes = ['claude', 'codex', 'gemini', 'minimax', 'alibaba'];
let availableModelAliases = [];

function getKnownProviderTypes() {
  return [...knownProviderTypes];
}

function renderProviders(providers, runtimeProviders) {
  const container = document.getElementById('providers-list');
  container.innerHTML = '';
  const entries = Object.entries(providers || {}).sort(([a], [b]) => a.localeCompare(b));
  for (const [name, cfg] of entries) {
    const ri = runtimeProviders ? runtimeProviders[name] : undefined;
    addProviderRow(container, name, cfg, ri);
  }
}

function addProviderRow(container, name, cfg, runtimeInfo) {
  const row = document.createElement('div');
  row.className = 'provider-row';
  const isAvail = runtimeInfo?.available;
  const statusHtml = (typeof isAvail === 'boolean')
    ? '<span class="avail-dot ' + (isAvail ? 'available' : 'unavailable') + '" title="' + (isAvail ? 'Available' : 'Not available') + '"></span>'
    : '';
  // Build provider type options dynamically from known providers
  const providerTypes = getKnownProviderTypes();
  const runtimeType = runtimeInfo?.type || '';
  const currentType = cfg?.type || runtimeType || '';
  // Ensure current type is included even if not in the known set
  if (currentType && !providerTypes.includes(currentType)) {
    providerTypes.push(currentType);
  }
  if (runtimeType && !providerTypes.includes(runtimeType)) {
    providerTypes.push(runtimeType);
  }
  providerTypes.sort((a, b) => a.localeCompare(b));
  const typeOptionsHtml = providerTypes.map(t =>
    '<option value="' + esc(t) + '"' + (currentType === t ? ' selected' : '') + '>' + esc(t) + '</option>'
  ).join('');
  row.innerHTML =
    statusHtml +
    '<input class="prov-name" type="text" value="' + esc(name || '') + '" placeholder="name">' +
    '<select class="prov-type">' + typeOptionsHtml + '</select>' +
    '<input class="prov-command" type="text" value="' + esc(cfg?.command || '') + '" placeholder="command (optional)">' +
    '<input class="prov-apikey" type="text" value="' + esc(cfg?.api_key_env || '') + '" placeholder="API key env var">' +
    '<button class="remove-btn" title="Remove">\u00d7</button>';
  row.querySelector('.remove-btn').addEventListener('click', () => { row.remove(); markDirty(); });
  row.querySelectorAll('input, select').forEach(el => {
    el.addEventListener('input', markDirty);
    el.addEventListener('change', markDirty);
  });
  container.appendChild(row);
}

function addProvider() {
  addProviderRow(document.getElementById('providers-list'), '', { type: 'claude' }, undefined);
  markDirty();
}

function renderModels(models) {
  const container = document.getElementById('models-list');
  container.innerHTML = '';
  for (const [alias, cfg] of Object.entries(models || {})) {
    addModelRow(container, alias, cfg);
  }
}

function addModelRow(container, alias, cfg) {
  const row = document.createElement('div');
  row.className = 'model-row';
  const reasoning = (cfg?.reasoning_effort || '').toString().trim();
  row.innerHTML =
    '<input class="model-alias" type="text" value="' + esc(alias || '') + '" placeholder="alias">' +
    '<input class="model-provider" type="text" value="' + esc(cfg?.provider || '') + '" placeholder="provider">' +
    '<input class="model-id" type="text" value="' + esc(cfg?.model_id || '') + '" placeholder="model_id">' +
    '<select class="model-reasoning">' +
      '<option value=""' + (!reasoning ? ' selected' : '') + '>default</option>' +
      '<option value="low"' + (reasoning === 'low' ? ' selected' : '') + '>low</option>' +
      '<option value="medium"' + (reasoning === 'medium' ? ' selected' : '') + '>medium</option>' +
      '<option value="high"' + (reasoning === 'high' ? ' selected' : '') + '>high</option>' +
    '</select>' +
    '<button class="remove-btn" title="Remove">\u00d7</button>';
  row.querySelector('.remove-btn').addEventListener('click', () => {
    row.remove();
    markDirty();
    refreshModelSelectsFromUI();
  });
  row.querySelectorAll('input').forEach(el => {
    el.addEventListener('input', () => {
      markDirty();
      refreshModelSelectsFromUI();
    });
    el.addEventListener('change', () => {
      markDirty();
      refreshModelSelectsFromUI();
    });
  });
  container.appendChild(row);
}

function addModel() {
  addModelRow(document.getElementById('models-list'), '', {});
  markDirty();
}

function renderRegistry(registry) {
  const container = document.getElementById('registry-list');
  container.innerHTML = '';
  for (const [modelId, cfg] of Object.entries(registry || {})) {
    addRegistryRow(container, modelId, cfg);
  }
}

function addRegistryRow(container, modelId, cfg) {
  const row = document.createElement('div');
  const alias = String(modelId || '');
  const registryModelId = String(
    (cfg && Object.prototype.hasOwnProperty.call(cfg, 'model_id')
      && cfg.model_id != null
      && String(cfg.model_id).trim())
    || alias,
  );
  row.className = 'registry-row';
  row.dataset.registryAlias = alias;

  let affinitiesHtml = '';
  if (cfg?.affinities) {
    for (const [key, val] of Object.entries(cfg.affinities)) {
      affinitiesHtml +=
        '<div class="affinity-item">' +
          '<input class="aff-key" type="text" value="' + esc(key) + '" placeholder="skill" style="flex:1">' +
          '<input class="aff-val" type="text" value="' + val + '" placeholder="0.0-1.0" style="width:60px">' +
          '<button class="remove-btn" style="width:20px;height:20px;font-size:14px">\u00d7</button>' +
        '</div>';
    }
  }

  row.innerHTML =
    '<div class="registry-header">' +
      '<strong>' + esc(alias || 'New Model') + '</strong>' +
      '<button class="remove-btn" title="Remove">\u00d7</button>' +
    '</div>' +
    '<div class="inline-fields">' +
      '<div class="field">' +
        '<label>Model ID</label>' +
        '<input class="reg-model-id" type="text" value="' + esc(registryModelId) + '">' +
      '</div>' +
      '<div class="field">' +
        '<label>Tier</label>' +
        '<select class="reg-tier">' +
          '<option value="frontier"' + (cfg?.tier==='frontier'?' selected':'') + '>frontier</option>' +
          '<option value="strong"' + (cfg?.tier==='strong'?' selected':'') + '>strong</option>' +
          '<option value="fast"' + (cfg?.tier==='fast'?' selected':'') + '>fast</option>' +
        '</select>' +
      '</div>' +
      '<div class="field">' +
        '<label>Provider</label>' +
        '<input class="reg-provider" type="text" value="' + esc(cfg?.provider || '') + '">' +
      '</div>' +
      '<div class="field">' +
        '<label>Cost Factor</label>' +
        '<input class="reg-cost" type="text" value="' + (cfg?.cost_factor ?? '1.0') + '">' +
      '</div>' +
      '<div class="field">' +
        '<label>Speed Factor</label>' +
        '<input class="reg-speed" type="text" value="' + (cfg?.speed_factor ?? '1.0') + '">' +
      '</div>' +
    '</div>' +
    '<div style="margin-top:8px">' +
      '<label style="font-size:12px;font-weight:600">Affinities</label>' +
      '<div class="affinity-grid" data-affinities>' +
        affinitiesHtml +
      '</div>' +
      '<button class="add-btn add-affinity-btn" style="margin-top:4px">+ Add Affinity</button>' +
    '</div>';
  // Registry header remove button
  row.querySelector('.registry-header .remove-btn').addEventListener('click', () => { row.remove(); markDirty(); });
  // Affinity item remove buttons
  row.querySelectorAll('.affinity-item .remove-btn').forEach(btn => {
    const item = btn.parentElement;
    btn.addEventListener('click', () => { item.remove(); markDirty(); });
  });
  // Add Affinity button
  const addAffBtn = row.querySelector('.add-affinity-btn');
  addAffBtn.addEventListener('click', () => { addAffinity(addAffBtn.previousElementSibling); });
  row.querySelectorAll('input, select').forEach(el => {
    el.addEventListener('input', markDirty);
    el.addEventListener('change', markDirty);
  });
  container.appendChild(row);
}

function addRegistryEntry() {
  addRegistryRow(document.getElementById('registry-list'), '', { tier: 'strong' });
  markDirty();
}

function addAffinity(grid) {
  const item = document.createElement('div');
  item.className = 'affinity-item';
  item.innerHTML =
    '<input class="aff-key" type="text" placeholder="skill" style="flex:1">' +
    '<input class="aff-val" type="text" value="0.85" placeholder="0.0-1.0" style="width:60px">' +
    '<button class="remove-btn" style="width:20px;height:20px;font-size:14px">\u00d7</button>';
  item.querySelector('.remove-btn').addEventListener('click', () => { item.remove(); markDirty(); });
  item.querySelectorAll('input').forEach(el => {
    el.addEventListener('input', markDirty);
    el.addEventListener('change', markDirty);
  });
  grid.appendChild(item);
  markDirty();
}

// -- Command policy (whitelist / blacklist) --

function renderCmdPatterns(containerId, patterns) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  if (!patterns || patterns.length === 0) {
    container.innerHTML = '<div style="font-size:11px;color:var(--vscode-descriptionForeground);padding:4px 0;">No patterns configured</div>';
    return;
  }
  patterns.forEach((pat, idx) => {
    const item = document.createElement('div');
    item.className = 'cmd-pattern-item';
    item.innerHTML = '<span>' + esc(pat) + '</span>' +
      '<button class="remove-btn" title="Remove" data-list="' + containerId + '" data-idx="' + idx + '">\u00d7</button>';
    item.querySelector('.remove-btn').addEventListener('click', (e) => {
      removeCmdPattern(e.target.dataset.list, parseInt(e.target.dataset.idx));
    });
    container.appendChild(item);
  });
}

function addCmdPattern(listId, inputId) {
  const input = document.getElementById(inputId);
  const pattern = input.value.trim();
  if (!pattern) return;
  const container = document.getElementById(listId);
  // Gather existing patterns
  const existing = getCmdPatterns(listId);
  if (existing.includes(pattern)) {
    input.value = '';
    return;
  }
  existing.push(pattern);
  renderCmdPatterns(listId, existing);
  input.value = '';
  markDirty();
}

function removeCmdPattern(listId, idx) {
  const patterns = getCmdPatterns(listId);
  patterns.splice(idx, 1);
  renderCmdPatterns(listId, patterns);
  markDirty();
}

function getCmdPatterns(listId) {
  const container = document.getElementById(listId);
  const items = container.querySelectorAll('.cmd-pattern-item span');
  return Array.from(items).map(el => el.textContent);
}

function normalizeModelIdForAvailability(value) {
  const raw = String(value || "").trim().toLowerCase();
  if (!raw) return "";
  const baseModel = raw.split("::reasoning_effort=")[0].trim();
  const normalized = baseModel
    .replaceAll("_", "-")
    .replaceAll(".", "-");
  const parts = normalized.split("-").filter(Boolean);
  const last = parts[parts.length - 1] ?? "";
  const hasOnlyDigits = last.length >= 6
    && last.length <= 8
    && last.split("").every((char) => char >= "0" && char <= "9");
  if (parts.length > 1 && hasOnlyDigits) {
    parts.pop();
  }
  return parts.join("-");
}

function getRuntimeModelIds(runtimeModels) {
  if (!runtimeModels) return [];

  if (Array.isArray(runtimeModels)) {
    return runtimeModels
      .map((model) => {
        if (typeof model === "string") return model;
        if (model && typeof model === "object") return model.model_id || model.model || model.id || "";
        return "";
      })
      .filter(Boolean);
  }

  if (typeof runtimeModels !== "object") return [];

  return Object.entries(runtimeModels)
    .map((entry) => {
      const value = entry[1];
      if (typeof value === "string") return value;
      if (value && typeof value === "object") {
        return value.model_id || value.model || value.id || "";
      }
      return entry[0];
    })
    .filter(Boolean);
}

const TECHNICAL_ALIAS_PREFIXES = [
  "opus",
  "sonnet",
  "haiku",
  "claude",
  "gpt",
  "gemini",
  "minimax",
  "spark",
  "codex",
];

function hasDigitChar(value) {
  const text = String(value || "");
  for (let i = 0; i < text.length; i += 1) {
    const code = text.charCodeAt(i);
    if (code >= 48 && code <= 57) return true;
  }
  return false;
}

function hasWhitespace(value) {
  const text = String(value || "");
  for (let i = 0; i < text.length; i += 1) {
    const code = text.charCodeAt(i);
    if (code <= 32) return true;
  }
  return false;
}

function getAliasDisplayScore(alias) {
  const text = String(alias || "");
  let score = 0;
  if (text.indexOf(" ") >= 0) score += 10;
  if (/[A-Z]/.test(text)) score += 5;
  if (!text.includes("-")) score += 3;
  if (text.includes("(")) score += 2;
  if (!text.includes("::")) score += 1;
  return score;
}

function isTechnicalModelAlias(alias, modelId) {
  const normalized = String(alias || "").trim().toLowerCase();
  if (!normalized) return false;
  if (normalized.includes("::reasoning_effort=")) return true;

  const normalizedTechnical = normalized
    .replaceAll("_", "-")
    .replaceAll(".", "-");
  const hasUpper = /[A-Z]/.test(alias || "");
  const hasSpace = hasWhitespace(alias || "");
  if (hasUpper || hasSpace) return false;

  if (!/^[a-z0-9-]+$/.test(normalizedTechnical)) return false;

  const normalizedModelId = normalizeModelIdForAvailability(modelId || alias);
  if (!normalizedModelId) return false;
  if (TECHNICAL_ALIAS_PREFIXES.includes(normalizedTechnical)) return true;

  if (normalized === normalizedModelId) return true;
  if (normalized.startsWith(normalizedModelId + "-") && /-(?:low|medium|high)$/.test(normalized)) {
    return true;
  }

  const hasDigit = hasDigitChar(normalized);
  if (!hasDigit) {
    return TECHNICAL_ALIAS_PREFIXES.some((prefix) => (
      normalizedTechnical.startsWith(prefix + "-")
    ));
  }

  return TECHNICAL_ALIAS_PREFIXES.some((prefix) => (
    normalizedTechnical.startsWith(prefix + "-")
  ));
}

function getAvailableModelAliases(
  modelAliases,
  runtimeModels,
) {
  const filterTechnical = (entries) => Object.entries(entries || {}).filter(([alias, cfg]) => (
    typeof alias === "string" &&
    alias.trim().length > 0 &&
    !isTechnicalModelAlias(alias, cfg?.model_id)
  ));
  const dedupeAliasesByModelId = (entries) => {
    const byModelId = new Map();
    for (const [alias, cfg] of entries) {
      const normalizedModelId = normalizeModelIdForAvailability(cfg?.model_id || alias);
      if (!normalizedModelId) continue;
      const score = getAliasDisplayScore(alias);
      const existing = byModelId.get(normalizedModelId);
      if (!existing || score > getAliasDisplayScore(existing.alias)) {
        byModelId.set(normalizedModelId, {
          alias: String(alias),
          modelId: normalizedModelId,
        });
      }
    }
    return Array.from(byModelId.values());
  };
  const fallbackModelAliases = (entries) => dedupeAliasesByModelId(entries).map((entry) => entry.alias);
  const fallbackFromEntries = (entries, maybeRawEntries = null) => {
    const technicalFiltered = filterTechnical(entries);
    if (technicalFiltered.length > 0) {
      return fallbackModelAliases(technicalFiltered);
    }
    if (maybeRawEntries) {
      return fallbackModelAliases(Object.entries(maybeRawEntries));
    }
    return fallbackModelAliases(entries);
  };

  const runtimeModelIds = getRuntimeModelIds(runtimeModels).map((id) => normalizeModelIdForAvailability(id));
  if (runtimeModelIds.length === 0) {
    return fallbackModelAliases(filterTechnical(modelAliases || {}));
  }

  const collectRuntimeAliasEntries = () => {
    const runtimeAliasEntries = [];
    if (Array.isArray(runtimeModels)) {
      for (const model of runtimeModels) {
        if (typeof model === "string") {
          const modelId = normalizeModelIdForAvailability(model);
          if (!modelId) continue;
          runtimeAliasEntries.push([model, { model_id: modelId }]);
        } else if (model && typeof model === "object") {
          const modelId = normalizeModelIdForAvailability(model.model_id || model.model || model.id || "");
          if (!modelId) continue;
          const alias = normalizeModelIdForAvailability(model.model_id || model.model || model.id) || modelId;
          runtimeAliasEntries.push([alias, { model_id: modelId }]);
        }
      }
    } else if (runtimeModels && typeof runtimeModels === "object") {
      for (const [key, value] of Object.entries(runtimeModels)) {
        if (typeof value === "string") {
          const modelId = normalizeModelIdForAvailability(value);
          if (!modelId) continue;
          runtimeAliasEntries.push([key, { model_id: modelId }]);
        } else if (value && typeof value === "object") {
          const modelId = normalizeModelIdForAvailability(value.model_id || value.model || value.id || key);
          if (!modelId) continue;
          const alias = normalizeModelIdForAvailability(value.model_id || value.model || value.id) || key;
          runtimeAliasEntries.push([alias, { model_id: modelId }]);
        }
      }
    }

    return runtimeAliasEntries;
  };

  const normalizedAliases = filterTechnical(modelAliases || {});

  if (!modelAliases || Object.keys(modelAliases).length === 0) {
    const runtimeAliasEntries = collectRuntimeAliasEntries();
    if (runtimeAliasEntries.length > 0) {
      const runtimeAliasLookup = Object.fromEntries(runtimeAliasEntries);
      return fallbackFromEntries(runtimeAliasEntries, runtimeAliasLookup);
    }
  }

  if (normalizedAliases.length === 0) {
    const runtimeAliasEntries = collectRuntimeAliasEntries();
    if (runtimeAliasEntries.length > 0) {
      const runtimeAliasLookup = Object.fromEntries(runtimeAliasEntries);
      return fallbackFromEntries(runtimeAliasEntries, runtimeAliasLookup);
    }
    return [];
  }

  const runtimeSet = new Set(runtimeModelIds.filter(Boolean));
  const dedupeByModel = new Map();
  const unmatched = new Map();

  for (const [alias, cfg] of normalizedAliases) {
    const normalizedModelId = normalizeModelIdForAvailability(cfg?.model_id || alias);
    if (!normalizedModelId) continue;
    const bucket = {
      alias: String(alias),
      modelId: normalizedModelId,
    };
    const score = getAliasDisplayScore(bucket.alias);
    const existing = dedupeByModel.get(bucket.modelId);
    if (!existing || score > getAliasDisplayScore(existing.alias)) {
      dedupeByModel.set(bucket.modelId, bucket);
    }
    const existingUnmatched = unmatched.get(bucket.modelId);
    if (!existingUnmatched || score > getAliasDisplayScore(existingUnmatched.alias)) {
      unmatched.set(bucket.modelId, bucket);
    }
    if (!runtimeSet.has(normalizedModelId)) {
      unmatched.set(normalizedModelId, bucket);
    }
  }

  const accessible = Array.from(dedupeByModel.values())
    .filter((entry) => runtimeSet.has(entry.modelId))
    .map((entry) => entry.alias);

  if (accessible.length > 0) {
    return accessible;
  }

  const runtimeAliasEntries = collectRuntimeAliasEntries();
  if (runtimeAliasEntries.length > 0) {
    const runtimeAliasLookup = Object.fromEntries(runtimeAliasEntries);
    return fallbackFromEntries(runtimeAliasEntries, runtimeAliasLookup);
  }

  // If no alias can be resolved against runtime models (legacy edge case),
  // keep the filtered alias list to avoid surfacing raw runtime IDs.
  return Array.from(unmatched.values()).map((entry) => entry.alias);
}

function esc(str) {
  if (!str) return '';
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function updateModelSelectOptions(modelAliases, selectedModel, selectedPeers, availableAliases) {
  const aliasesList = Array.from(new Set((modelAliases || []).map(a => String(a || '').trim()).filter(Boolean)));
  const shouldFallback = !Array.isArray(availableAliases);
  const availableSet = new Set(
    (shouldFallback ? aliasesList : availableAliases)
      .map((alias) => String(alias || '').trim())
      .filter(Boolean),
  );
  const resolvedModelSet = new Set(availableSet);

  if (selectedModel && availableSet.has(selectedModel)) {
    resolvedModelSet.add(selectedModel);
  }
  if (selectedPeers?.length) {
    for (const peer of selectedPeers) {
      const value = String(peer || '').trim();
      if (value && availableSet.has(value)) resolvedModelSet.add(value);
    }
  }

  const defaultSelect = document.getElementById('defaults.model');
  const peerSelect = document.getElementById('defaults.peer_models');
  const aliases = Array.from(resolvedModelSet).sort((a, b) => a.localeCompare(b));
  const optionLabel = (alias) => alias;

  defaultSelect.innerHTML = '';
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = aliases.length > 0 ? 'Select a model' : 'No models available';
  if (aliases.length === 0) placeholder.disabled = true;
  defaultSelect.appendChild(placeholder);

  peerSelect.innerHTML = '';
  if (aliases.length === 0) {
    const emptyOption = document.createElement('option');
    emptyOption.value = '';
    emptyOption.textContent = 'No models available';
    emptyOption.disabled = true;
    peerSelect.appendChild(emptyOption);
  }

  for (const alias of aliases) {
    const option = document.createElement('option');
    option.value = alias;
    option.textContent = optionLabel(alias);
    if (selectedModel === alias) option.selected = true;
    defaultSelect.appendChild(option);

    const peerOption = document.createElement('option');
    peerOption.value = alias;
    peerOption.textContent = optionLabel(alias);
    if (selectedPeers?.includes(alias)) peerOption.selected = true;
    peerSelect.appendChild(peerOption);
  }

  if (selectedModel) defaultSelect.value = selectedModel;
}

function refreshModelSelectsFromUI() {
  const aliases = availableModelAliases;
  const defaultValue = document.getElementById('defaults.model').value.trim();
  const peerSelect = document.getElementById('defaults.peer_models');
  const peerValues = Array.from(peerSelect.selectedOptions).map(option => option.value);
  updateModelSelectOptions(
    aliases,
    defaultValue,
    peerValues,
    availableModelAliases,
  );
}

function resolveCanonicalModelSelectionValue(rawValue, modelAliasEntries) {
  const normalized = String(rawValue || "").trim();
  if (!normalized) return "";
  const aliases = modelAliasEntries || {};
  if (Object.prototype.hasOwnProperty.call(aliases, normalized)) {
    return normalized;
  }

  const normalizedLookup = normalizeModelIdForAvailability(normalized);
  if (!normalizedLookup) return "";

  const aliasEntries = Object.entries(aliases);
  for (let i = 0; i < aliasEntries.length; i += 1) {
    const [alias, cfg] = aliasEntries[i] || [];
    if (!alias) continue;
    const modelId = normalizeModelIdForAvailability(cfg?.model_id || alias);
    if (modelId && modelId === normalizedLookup) {
      return alias;
    }
  }

  return "";
}

// -- Populate form from config + runtime info --
function populateForm(config, runtime, extensionSettings) {
  currentConfig = config || {};
  const rt = runtime || {};
  const rtProviders = rt.providers || {};
  const rtModels = rt.models || {};
  const rtAliases = rt.model_aliases || {};

  // Engine
  const eng = config.engine || {};
  document.getElementById('engine.max_agent_depth').value = eng.max_agent_depth ?? 5;
  document.getElementById('engine.max_concurrent_agents').value = eng.max_concurrent_agents ?? 10;
  document.getElementById('engine.agent_timeout_seconds').value = eng.agent_timeout_seconds ?? 7200;
  document.getElementById('engine.tool_call_timeout_seconds').value = eng.tool_call_timeout_seconds ?? 7200;
  document.getElementById('engine.user_question_timeout_seconds').value = eng.user_question_timeout_seconds ?? 0;
  document.getElementById('extension.sessionInactivityMinutes').value =
    extensionSettings?.sessionInactivityMinutes ?? 15;
  document.getElementById('extension.enableNsfwThinkingVerbs').checked =
    extensionSettings?.enableNsfwThinkingVerbs !== false;
  document.getElementById('extension.customThinkingVerbs').value =
    Array.isArray(extensionSettings?.customThinkingVerbs)
      ? extensionSettings.customThinkingVerbs.join('\\n')
      : '';

  // Command policy — prefer extension settings, fall back to engine config
  const cmdWl = Array.isArray(extensionSettings?.commandWhitelist) && extensionSettings.commandWhitelist.length > 0
    ? extensionSettings.commandWhitelist
    : (eng.command_whitelist || []);
  const cmdBl = Array.isArray(extensionSettings?.commandBlacklist) && extensionSettings.commandBlacklist.length > 0
    ? extensionSettings.commandBlacklist
    : (eng.command_blacklist || []);
  renderCmdPatterns('cmd-whitelist-list', cmdWl);
  renderCmdPatterns('cmd-blacklist-list', cmdBl);

  // Defaults
  const def = config.defaults || {};
  const peers = def.peer_models || (def.peer_model ? [def.peer_model] : []);

  // Derive known provider types from config + runtime (no hardcoding)
  const provTypeSet = new Set(knownProviderTypes);
  for (const cfg of Object.values(config.providers || {})) {
    if (cfg?.type) provTypeSet.add(cfg.type);
  }
  for (const info of Object.values(rtProviders)) {
    if (info?.type) provTypeSet.add(info.type);
  }
  knownProviderTypes = Array.from(provTypeSet).sort();

  // Merge providers: start with config providers, then add any detected
  // providers not already in config
  const mergedProviders = Object.assign({}, config.providers || {});
  for (const [name, info] of Object.entries(rtProviders)) {
    if (!mergedProviders[name]) {
      mergedProviders[name] = { type: info.type };
    }
  }
  renderProviders(mergedProviders, rtProviders);

  // Merge models: config models are the source of truth (from YAML),
  // supplemented by any runtime aliases not already in the config.
  // No hardcoded model names — everything comes from ~/.prsm/models.yaml
  // and .prism/prsm.yaml via the server.
  const mergedModels = Object.assign({}, config.models || {});
  // Add aliases from runtime config (server-resolved from YAML)
  for (const [alias, acfg] of Object.entries(rtAliases)) {
    if (!mergedModels[alias]) {
      mergedModels[alias] = acfg;
    }
  }
  const runtimeAliasSource = Object.keys(rtAliases || {});
  const aliasSource = runtimeAliasSource.length > 0 ? runtimeAliasSource : Object.keys(mergedModels);
  renderModels(mergedModels);
  const mergedAliases = {
    ...mergedModels,
    ...rtAliases,
  };
  const selectionAliases = {
    ...(Object.keys(rtAliases || {}).length > 0 ? rtAliases : mergedAliases),
  };
  availableModelAliases = getAvailableModelAliases(selectionAliases, rtModels);
  const availableAliasSet = new Set(availableModelAliases);
  const canonicalDefault = resolveCanonicalModelSelectionValue(
    def.model || "",
    selectionAliases,
  );
  const canonicalPeers = (peers || [])
    .map((peer) => resolveCanonicalModelSelectionValue(peer, selectionAliases))
    .filter(Boolean)
    .filter((peer) => availableAliasSet.has(peer))
    .filter((peer, idx, arr) => arr.indexOf(peer) === idx);
  updateModelSelectOptions(
    aliasSource,
    canonicalDefault && availableAliasSet.has(canonicalDefault) ? canonicalDefault : "",
    canonicalPeers,
    availableModelAliases,
  );

  // Registry
  renderRegistry(config.model_registry);

  // Attach input+change listeners to scalar inputs for immediate dirty tracking
  document.querySelectorAll('#content input, #content select, #content textarea').forEach(el => {
    el.addEventListener('input', markDirty);
    el.addEventListener('change', markDirty);
  });

  // Check if we added anything beyond what was in the file
  const configProvKeys = Object.keys(config.providers || {});
  const configModelKeys = Object.keys(config.models || {});
  const mergedProvKeys = Object.keys(mergedProviders);
  const mergedModelKeys = Object.keys(mergedModels);
  if (mergedProvKeys.length > configProvKeys.length || mergedModelKeys.length > configModelKeys.length) {
    // Auto-detected items were added — mark dirty so user can save
    markDirty();
    const addedP = mergedProvKeys.length - configProvKeys.length;
    const addedM = mergedModelKeys.length - configModelKeys.length;
    setStatus('Auto-detected ' + addedP + ' provider(s) and ' + addedM + ' model(s)', 'success');
    setTimeout(() => setStatus(''), 5000);
  } else {
    markClean();
  }
}

// -- Message handling --
window.addEventListener('message', event => {
  const msg = event.data;
  switch (msg.type) {
    case 'configLoaded':
      try {
        populateForm(msg.config || {}, msg.runtime || {}, msg.extensionSettings);
      } catch (err) {
        setStatus('populateForm error: ' + (err.message || err), 'error');
        console.error('populateForm error:', err);
      }
      const pathEl = document.getElementById('config-path');
      pathEl.textContent = msg.configPath
        ? (msg.exists ? msg.configPath : msg.configPath + ' (will be created on save)')
        : 'Not connected';
      if (msg.error) {
        setStatus(msg.error, 'error');
      } else if (!isDirty) {
        // Status is set by populateForm if auto-detected items were added
        setStatus('Configuration loaded', 'success');
        setTimeout(() => setStatus(''), 3000);
      }
      break;
    case 'saved':
      markClean();
      setStatus('Settings saved and applied \u2713', 'success');
      setTimeout(() => setStatus(''), 5000);
      break;
    case 'saveError':
      setStatus('Save failed: ' + msg.error, 'error');
      break;
  }
});

function setStatus(text, cls) {
  const el = document.getElementById('status');
  el.textContent = text;
  el.className = 'status-msg' + (cls ? ' ' + cls : '');
}

// -- Button handlers --
document.getElementById('save-btn').addEventListener('click', () => {
  vscode.postMessage({
    type: 'save',
    config: buildConfig(),
    extensionSettings: buildExtensionSettings(),
  });
});
document.getElementById('save-btn-bottom').addEventListener('click', () => {
  vscode.postMessage({
    type: 'save',
    config: buildConfig(),
    extensionSettings: buildExtensionSettings(),
  });
});
document.getElementById('reload-btn').addEventListener('click', () => {
  if (isDirty && !confirm('Discard unsaved changes and reload?')) return;
  vscode.postMessage({ type: 'reload' });
});

// Static button handlers (replacing inline onclick which CSP blocks)
document.querySelectorAll('.section-header').forEach(h => {
  h.addEventListener('click', () => toggleSection(h));
});
document.getElementById('add-provider-btn').addEventListener('click', addProvider);
document.getElementById('add-model-btn').addEventListener('click', addModel);
document.getElementById('add-registry-btn').addEventListener('click', addRegistryEntry);
document.getElementById('add-whitelist-btn').addEventListener('click', () => {
  addCmdPattern('cmd-whitelist-list', 'cmd-whitelist-input');
});
document.getElementById('add-blacklist-btn').addEventListener('click', () => {
  addCmdPattern('cmd-blacklist-list', 'cmd-blacklist-input');
});
document.getElementById('cmd-whitelist-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); addCmdPattern('cmd-whitelist-list', 'cmd-whitelist-input'); }
});
document.getElementById('cmd-blacklist-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); addCmdPattern('cmd-blacklist-list', 'cmd-blacklist-input'); }
});
document.getElementById('import-provider-btn').addEventListener('click', () => {
  const provider = document.getElementById('import-provider-select').value;
  vscode.postMessage({ type: 'importFromProvider', provider });
});
document.getElementById('import-file-btn').addEventListener('click', () => {
  vscode.postMessage({ type: 'importFile' });
});
document.getElementById('export-archive-btn').addEventListener('click', () => {
  vscode.postMessage({ type: 'exportArchive' });
});

// Request initial config.  Send 'ready' once; the extension handler is
// registered before the HTML is set, so the first message should arrive
// quickly.  If nothing comes back within a few seconds, retry a couple
// of times with short delays.  A hard 10-second deadline guarantees we
// never stay in a loading state forever.
let configReceived = false;

window.addEventListener('message', function onFirstConfig(event) {
  if (event.data && event.data.type === 'configLoaded') {
    configReceived = true;
    window.removeEventListener('message', onFirstConfig);
  }
});

// Fire the initial request immediately.
vscode.postMessage({ type: 'ready' });

// A few short-interval retries in case of timing edge-cases.
let retryCount = 0;
const retryTimer = setInterval(() => {
  if (configReceived) { clearInterval(retryTimer); return; }
  retryCount++;
  if (retryCount >= 3) {
    clearInterval(retryTimer);
    // Hard deadline: show defaults, stop waiting.
    if (!configReceived) {
      document.getElementById('config-path').textContent =
        'Using defaults \u2014 server not connected';
      setStatus('Could not reach server. Showing defaults.', 'error');
    }
    return;
  }
  vscode.postMessage({ type: 'ready' });
}, 1500);
</script>
</body>
</html>`;
  }
}
