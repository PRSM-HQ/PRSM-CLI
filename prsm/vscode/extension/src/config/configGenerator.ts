/**
 * Generates a YAML config file from VSCode settings.
 *
 * Reads all `prsm.*` settings, builds a YAML string matching the format
 * expected by `load_yaml_config()`, writes to a temp file, returns the path.
 *
 * If `prsm.yamlConfigPath` is set and the file exists, returns that path
 * directly (no generation).
 */
import * as vscode from "vscode";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";

export interface ExpertConfig {
  name: string;
  description: string;
  system_prompt: string;
  tools?: string[];
  model?: string;
  permission_mode?: string;
  max_concurrent_consultations?: number;
  cwd?: string;
}

/**
 * Generate a YAML config file from VSCode settings and return its path.
 * If prsm.yamlConfigPath is set and exists, returns that directly.
 */
export function generateConfigFile(): string | undefined {
  const config = vscode.workspace.getConfiguration("prsm");

  // Check if user has customized any settings beyond defaults.
  const inspect = (key: string) => {
    const i = config.inspect(key);
    return (
      i?.workspaceValue !== undefined ||
      i?.workspaceFolderValue !== undefined ||
      i?.globalValue !== undefined
    );
  };

  const folders = vscode.workspace.workspaceFolders ?? [];
  const preferredWorkspaceRoot =
    folders.find((f) =>
      fs.existsSync(path.join(f.uri.fsPath, ".prism", "prsm.yaml"))
    )?.uri.fsPath ??
    folders.find((f) => fs.existsSync(path.join(f.uri.fsPath, "prsm.yaml")))?.uri.fsPath ??
    folders[0]?.uri.fsPath;

  // If the workspace already has a .prism/prsm.yaml, always use it.
  // This prevents VS Code settings-derived generation from overriding
  // user-managed project config.
  const workspaceRootPath = preferredWorkspaceRoot;
  if (workspaceRootPath) {
    const workspaceYaml = path.join(workspaceRootPath, ".prism", "prsm.yaml");
    if (fs.existsSync(workspaceYaml)) {
      return workspaceYaml;
    }
  }

  // Check for explicit YAML path override if workspace config doesn't exist.
  const yamlOverride = config.get<string>("yamlConfigPath");
  if (yamlOverride && fs.existsSync(yamlOverride)) {
    return yamlOverride;
  }

  const hasCustom =
    inspect("engine.maxAgentDepth") ||
    inspect("engine.maxConcurrentAgents") ||
    inspect("engine.agentTimeoutSeconds") ||
    inspect("engine.toolCallTimeoutSeconds") ||
    inspect("engine.userQuestionTimeoutSeconds") ||
    inspect("security.commandWhitelist") ||
    inspect("security.commandBlacklist") ||
    inspect("providers") ||
    inspect("models") ||
    inspect("defaults.model") ||
    inspect("defaults.peerModel") ||
    inspect("experts");

  if (!hasCustom) {
    // Let the server resolve project/default config naturally.
    return undefined;
  }

  // Build YAML from individual settings
  const lines: string[] = [];

  // Engine settings
  lines.push("engine:");
  lines.push(
    `  max_agent_depth: ${config.get<number>("engine.maxAgentDepth", 5)}`
  );
  lines.push(
    `  max_concurrent_agents: ${config.get<number>(
      "engine.maxConcurrentAgents",
      10
    )}`
  );
  lines.push(
    `  agent_timeout_seconds: ${config.get<number>(
      "engine.agentTimeoutSeconds",
      7200
    )}`
  );
  lines.push(
    `  tool_call_timeout_seconds: ${config.get<number>(
      "engine.toolCallTimeoutSeconds",
      7200
    )}`
  );
  lines.push(
    `  user_question_timeout_seconds: ${config.get<number>(
      "engine.userQuestionTimeoutSeconds",
      0
    )}`
  );
  const commandWhitelist = config.get<string[]>("security.commandWhitelist", []);
  const commandBlacklist = config.get<string[]>("security.commandBlacklist", []);
  if (commandWhitelist.length > 0) {
    lines.push(`  command_whitelist: [${commandWhitelist.map(yamlValue).join(", ")}]`);
  }
  if (commandBlacklist.length > 0) {
    lines.push(`  command_blacklist: [${commandBlacklist.map(yamlValue).join(", ")}]`);
  }
  lines.push("");

  // Providers: include only when explicitly customized in settings.
  if (inspect("providers")) {
    const providers = config.get<Record<string, Record<string, string>>>(
      "providers",
      {}
    );
    lines.push("providers:");
    for (const [id, providerConfig] of Object.entries(providers)) {
      lines.push(`  ${id}:`);
      for (const [key, value] of Object.entries(providerConfig)) {
        lines.push(`    ${key}: ${yamlValue(value)}`);
      }
    }
    lines.push("");
  }

  // Defaults: include only explicitly customized values.
  const hasCustomDefaultModel = inspect("defaults.model");
  const hasCustomDefaultPeerModel = inspect("defaults.peerModel");

  // Models: include when customized OR when defaults reference aliases.
  // This ensures alias resolution in YAML even if users only change
  // defaults.model/defaults.peerModel.
  if (inspect("models") || hasCustomDefaultModel || hasCustomDefaultPeerModel) {
    const models = config.get<
      Record<string, { provider: string; model_id: string; reasoning_effort?: string }>
    >("models", {});
    lines.push("models:");
    for (const [alias, modelConfig] of Object.entries(models)) {
      lines.push(`  ${alias}:`);
      lines.push(`    provider: ${modelConfig.provider}`);
      lines.push(`    model_id: ${modelConfig.model_id}`);
      if (modelConfig.reasoning_effort) {
        lines.push(`    reasoning_effort: ${modelConfig.reasoning_effort}`);
      }
    }
    lines.push("");
  }

  if (hasCustomDefaultModel || hasCustomDefaultPeerModel) {
    lines.push("defaults:");
    if (hasCustomDefaultModel) {
      lines.push(
        `  model: ${config.get<string>("defaults.model", "opus-4-6")}`
      );
    }
    if (hasCustomDefaultPeerModel) {
      const peerModel = config.get<string>("defaults.peerModel", "gpt-5-3-medium");
      if (peerModel) {
        lines.push(`  peer_model: ${peerModel}`);
      }
    }
    lines.push("");
  }

  // Experts
  const experts = config.get<Record<string, ExpertConfig>>("experts", {});
  if (Object.keys(experts).length > 0) {
    lines.push("experts:");
    for (const [expertId, expert] of Object.entries(experts)) {
      lines.push(`  ${expertId}:`);
      lines.push(`    name: ${yamlValue(expert.name)}`);
      lines.push(`    description: >`);
      for (const line of wordWrap(expert.description, 70)) {
        lines.push(`      ${line}`);
      }
      lines.push(`    system_prompt: |`);
      for (const line of expert.system_prompt.split("\n")) {
        lines.push(`      ${line}`);
      }
      if (expert.tools && expert.tools.length > 0) {
        lines.push(`    tools: [${expert.tools.join(", ")}]`);
      }
      if (expert.model) {
        lines.push(`    model: ${expert.model}`);
      }
      if (expert.permission_mode) {
        lines.push(`    permission_mode: ${expert.permission_mode}`);
      }
      if (expert.cwd) {
        lines.push(`    cwd: ${expert.cwd}`);
      }
    }
  }

  const yamlContent = lines.join("\n") + "\n";

  // Write generated settings config to temp so workspace-managed
  // .prism/prsm.yaml is never overwritten.
  const tmpDir = os.tmpdir();
  const tmpPath = path.join(tmpDir, "prsm-config.yaml");
  fs.writeFileSync(tmpPath, yamlContent, "utf-8");
  return tmpPath;
}

/** Escape a YAML string value if needed. */
function yamlValue(value: string): string {
  if (
    value.includes(":") ||
    value.includes("#") ||
    value.includes("'") ||
    value.includes('"') ||
    value.includes("\n") ||
    value.startsWith(" ") ||
    value.startsWith("{") ||
    value.startsWith("[")
  ) {
    return `"${value.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
  }
  return value;
}

/** Word-wrap text to a max width. */
function wordWrap(text: string, maxWidth: number): string[] {
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let current = "";

  for (const word of words) {
    if (current.length + word.length + 1 > maxWidth && current.length > 0) {
      lines.push(current);
      current = word;
    } else {
      current = current ? current + " " + word : word;
    }
  }
  if (current) lines.push(current);
  return lines;
}
