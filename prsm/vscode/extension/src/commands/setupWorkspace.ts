/**
 * Workspace auto-setup: creates .prism workspace files, ensures global
 * ~/.prsm/models.yaml exists, and repairs .claude/ settings.
 *
 * Runs when opening a workspace for the first time.
 *
 * Resolves the prsm executable symlink to find the install root,
 * then generates config files pointing to the correct paths.
 */
import * as vscode from "vscode";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import { execFileSync } from "child_process";

function ensureClaudeSettings(workspaceRoot: string): void {
  const claudeDir = path.join(workspaceRoot, ".claude");
  if (!fs.existsSync(claudeDir)) {
    fs.mkdirSync(claudeDir, { recursive: true });
  }

  const settingsPath = path.join(claudeDir, "settings.local.json");
  let settings: Record<string, unknown> = {};
  if (fs.existsSync(settingsPath)) {
    try {
      settings = JSON.parse(fs.readFileSync(settingsPath, "utf-8")) as Record<string, unknown>;
    } catch {
      settings = {};
    }
  }

  const permissions =
    (settings.permissions as Record<string, unknown> | undefined) ?? {};
  const allowRaw = permissions.allow;
  const allow = Array.isArray(allowRaw)
    ? allowRaw.filter((v): v is string => typeof v === "string")
    : [];

  if (!allow.includes("mcp__orchestrator__*")) {
    allow.push("mcp__orchestrator__*");
  }

  permissions.allow = allow;
  settings.permissions = permissions;
  settings.enableAllProjectMcpServers = true;

  const enabledRaw = settings.enabledMcpjsonServers;
  const enabled = Array.isArray(enabledRaw)
    ? enabledRaw.filter((v): v is string => typeof v === "string")
    : [];
  if (!enabled.includes("orchestrator")) {
    enabled.push("orchestrator");
  }
  settings.enabledMcpjsonServers = enabled;

  fs.writeFileSync(
    settingsPath,
    JSON.stringify(settings, null, 2) + "\n",
    "utf-8"
  );
}

function mergeLines(existingText: string, additions: string[]): string {
  const existing = existingText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !line.startsWith("#"));
  const merged = new Set(existing);
  for (const entry of additions) {
    const cleaned = entry.trim();
    if (cleaned) merged.add(cleaned);
  }
  if (merged.size === 0) return "";
  return Array.from(merged).sort().join("\n") + "\n";
}

function ensurePrismWorkspaceFiles(
  workspaceRoot: string,
  config: vscode.WorkspaceConfiguration,
  installRoot?: string,
): void {
  const prismDir = path.join(workspaceRoot, ".prism");
  fs.mkdirSync(prismDir, { recursive: true });

  const venvPython = installRoot
    ? path.join(installRoot, ".venv", "bin", "python")
    : ".venv/bin/python";
  const configYaml = ".prism/prsm.yaml";

  const mcpJsonPath = path.join(prismDir, ".mcp.json");
  if (!fs.existsSync(mcpJsonPath)) {
    const mcpConfig = {
      mcpServers: {
        orchestrator: {
          type: "stdio",
          command: venvPython,
          args: [
            "-m",
            "prsm.engine.mcp_server.stdio_server",
            "--config",
            configYaml,
          ],
        },
      },
    };
    fs.writeFileSync(
      mcpJsonPath,
      JSON.stringify(mcpConfig, null, 2) + "\n",
      "utf-8"
    );
  }

  const prsmYamlPath = path.join(prismDir, "prsm.yaml");
  const homeTemplatePrsm = path.join(
    process.env.HOME ?? "",
    ".prsm",
    "templates",
    "prsm.yaml"
  );
  const installTemplatePrsm = installRoot
    ? path.join(installRoot, ".prism", "prsm.yaml")
    : "";
  const candidatePrsmTemplates = [installTemplatePrsm, homeTemplatePrsm].filter(
    (p) => p && fs.existsSync(p)
  );
  if (!fs.existsSync(prsmYamlPath)) {
    if (candidatePrsmTemplates.length > 0) {
      fs.copyFileSync(candidatePrsmTemplates[0], prsmYamlPath);
    } else {
      fs.writeFileSync(
        prsmYamlPath,
        "engine:\n  max_agent_depth: 5\n  max_concurrent_agents: 10\n  agent_timeout_seconds: 7200\n  tool_call_timeout_seconds: 7200\n  user_question_timeout_seconds: 0\n",
        "utf-8"
      );
    }
  }

  const globalPrsmDir = path.join(os.homedir(), ".prsm");
  const modelsYamlPath = path.join(globalPrsmDir, "models.yaml");
  const homeTemplateModels = path.join(
    globalPrsmDir,
    "templates",
    "models.yaml",
  );
  const installTemplateModels = installRoot
    ? path.join(installRoot, ".prism", "models.yaml")
    : "";
  const candidateModelTemplates = [installTemplateModels, homeTemplateModels].filter(
    (p) => p && fs.existsSync(p)
  );
  if (!fs.existsSync(modelsYamlPath)) {
    fs.mkdirSync(globalPrsmDir, { recursive: true });
    if (candidateModelTemplates.length > 0) {
      fs.copyFileSync(candidateModelTemplates[0], modelsYamlPath);
    } else {
      fs.writeFileSync(modelsYamlPath, "models: {}\n", "utf-8");
    }
  }

  const whitelistPath = path.join(prismDir, "command_whitelist.txt");
  const blacklistPath = path.join(prismDir, "command_blacklist.txt");
  if (!fs.existsSync(whitelistPath)) fs.writeFileSync(whitelistPath, "", "utf-8");
  if (!fs.existsSync(blacklistPath)) fs.writeFileSync(blacklistPath, "", "utf-8");

  const whitelistSettings = config.get<string[]>("security.commandWhitelist", []);
  const blacklistSettings = config.get<string[]>("security.commandBlacklist", []);

  const whitelistText = fs.existsSync(whitelistPath)
    ? fs.readFileSync(whitelistPath, "utf-8")
    : "";
  const blacklistText = fs.existsSync(blacklistPath)
    ? fs.readFileSync(blacklistPath, "utf-8")
    : "";
  fs.writeFileSync(
    whitelistPath,
    mergeLines(whitelistText, whitelistSettings),
    "utf-8"
  );
  fs.writeFileSync(
    blacklistPath,
    mergeLines(blacklistText, blacklistSettings),
    "utf-8"
  );
}

/**
 * Check if workspace needs setup and repair local Claude settings.
 * Called on extension activation.
 */
export async function checkWorkspaceSetup(
  context: vscode.ExtensionContext
): Promise<void> {
  const config = vscode.workspace.getConfiguration("prsm");
  if (!config.get<boolean>("autoSetupWorkspace", true)) {
    return;
  }

  const workspaceRoot =
    vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  if (!workspaceRoot) return;

  // Always re-validate Claude settings on activation for configured workspaces.
  // This keeps permissions repaired even if settings were edited manually.
  const prismMcpJsonPath = path.join(workspaceRoot, ".prism", ".mcp.json");
  const installRoot = resolveInstallRoot(config.get<string>("executablePath") ?? "prsm");
  if (fs.existsSync(prismMcpJsonPath)) {
    ensurePrismWorkspaceFiles(workspaceRoot, config, installRoot);
    ensureClaudeSettings(workspaceRoot);
    await context.workspaceState.update("prsm.workspaceSetup", true);
    return;
  }

  // Only prompt once per workspace for initial setup
  if (context.workspaceState.get<boolean>("prsm.workspaceSetup")) {
    return;
  }

  const choice = await vscode.window.showInformationMessage(
    "Set up this workspace for PRSM orchestration?",
    "Setup",
    "Not now"
  );

  if (choice === "Setup") {
    await performSetup(workspaceRoot, config);
    await context.workspaceState.update("prsm.workspaceSetup", true);
    vscode.window.showInformationMessage(
      "PRSM workspace setup complete."
    );
  } else {
    // Don't prompt again this session
    await context.workspaceState.update("prsm.workspaceSetup", true);
  }
}

/**
 * Perform the actual workspace setup.
 * Can be triggered manually via the prsm.setupWorkspace command.
 */
export async function performSetup(
  workspaceRoot: string,
  config: vscode.WorkspaceConfiguration
): Promise<void> {
  const executablePath = config.get<string>("executablePath") ?? "prsm";
  const installRoot = resolveInstallRoot(executablePath);

  // 1. Ensure .prism workspace files exist
  ensurePrismWorkspaceFiles(workspaceRoot, config, installRoot);

  // 2. Ensure .claude/settings.local.json has orchestrator auto-allow entries
  ensureClaudeSettings(workspaceRoot);
}

/**
 * Resolve the prsm executable symlink back to the repo root.
 *
 * prsm → ~/.local/bin/prsm → /path/to/prsm-cli/.venv/bin/prsm
 * Walk up from .venv/bin/prsm → prsm-cli root
 */
function resolveInstallRoot(executablePath: string): string | undefined {
  try {
    // Find the executable on PATH using `which` (safe: execFileSync
    // does not invoke a shell, so no injection risk)
    const which = execFileSync("which", [executablePath], {
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
    }).trim();

    if (!which) return undefined;

    // Resolve all symlinks
    const realPath = fs.realpathSync(which);

    // Expected: /path/to/prsm-cli/.venv/bin/prsm
    // Walk up 3 levels: bin → .venv → prsm-cli
    const installRoot = path.resolve(path.dirname(realPath), "..", "..");

    // Verify it looks like a prsm-cli install
    if (
      fs.existsSync(path.join(installRoot, ".prism", "prsm.yaml")) ||
      fs.existsSync(path.join(installRoot, "prsm.yaml")) ||
      fs.existsSync(path.join(installRoot, "pyproject.toml"))
    ) {
      return installRoot;
    }
  } catch {
    // Failed to resolve — not critical
  }
  return undefined;
}
