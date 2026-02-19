/**
 * Expert management commands for adding, editing, removing,
 * and importing expert profiles from YAML config files.
 */
import * as vscode from "vscode";
import * as fs from "fs";
import { ExpertConfig } from "../config/configGenerator";

/**
 * Register all expert management commands.
 */
export function registerExpertCommands(
  context: vscode.ExtensionContext
): void {
  context.subscriptions.push(
    vscode.commands.registerCommand(
      "prsm.addExpert",
      addExpertWizard
    ),
    vscode.commands.registerCommand(
      "prsm.removeExpert",
      removeExpert
    ),
    vscode.commands.registerCommand(
      "prsm.importExpertsFromYaml",
      importExpertsFromYaml
    )
  );
}

async function addExpertWizard(): Promise<void> {
  // Step 1: Expert ID
  const expertId = await vscode.window.showInputBox({
    title: "Add Expert (1/4): Expert ID",
    prompt: "Unique identifier for this expert (kebab-case)",
    placeHolder: "e.g., react-frontend",
    validateInput: (value) => {
      if (!value.match(/^[a-z0-9-]+$/)) {
        return "Use lowercase letters, numbers, and hyphens only";
      }
      return null;
    },
  });
  if (!expertId) return;

  // Step 2: Name and description
  const name = await vscode.window.showInputBox({
    title: "Add Expert (2/4): Display Name",
    prompt: "Human-readable name for this expert",
    placeHolder: "e.g., React Frontend Expert",
  });
  if (!name) return;

  const description = await vscode.window.showInputBox({
    title: "Add Expert (3/4): Description",
    prompt: "What does this expert specialize in?",
    placeHolder: "e.g., React components, hooks, state management, styling",
  });
  if (!description) return;

  // Step 3: System prompt
  const systemPrompt = await vscode.window.showInputBox({
    title: "Add Expert (4/4): System Prompt",
    prompt:
      "Domain knowledge and instructions for this expert (can be edited later in settings)",
    placeHolder: "You are a React frontend expert...",
  });
  if (!systemPrompt) return;

  // Step 4: Optional settings via quick picks
  const toolPicks = await vscode.window.showQuickPick(
    [
      { label: "Read", picked: true },
      { label: "Write", picked: true },
      { label: "Edit", picked: true },
      { label: "Glob", picked: true },
      { label: "Grep", picked: true },
      { label: "Bash", picked: false },
    ],
    {
      title: "Select Tools",
      placeHolder: "Which tools should this expert have access to?",
      canPickMany: true,
    }
  );
  const tools = toolPicks?.map((p) => p.label) ?? [
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
  ];

  const modelPick = await vscode.window.showQuickPick(
    [
      { label: "opus", description: "Most capable" },
      { label: "sonnet", description: "Fast and capable" },
    ],
    { title: "Select Model", placeHolder: "Model for this expert" }
  );
  const model = modelPick?.label ?? "opus";

  // Save to settings
  const config = vscode.workspace.getConfiguration("prsm");
  const experts = config.get<Record<string, ExpertConfig>>(
    "experts",
    {}
  );

  experts[expertId] = {
    name,
    description,
    system_prompt: systemPrompt,
    tools,
    model,
    permission_mode: "acceptEdits",
  };

  await config.update(
    "experts",
    experts,
    vscode.ConfigurationTarget.Workspace
  );

  vscode.window.showInformationMessage(
    `Expert "${name}" added. Edit in Settings for full customization.`
  );
}

async function removeExpert(): Promise<void> {
  const config = vscode.workspace.getConfiguration("prsm");
  const experts = config.get<Record<string, ExpertConfig>>(
    "experts",
    {}
  );

  const ids = Object.keys(experts);
  if (ids.length === 0) {
    vscode.window.showInformationMessage("No experts configured.");
    return;
  }

  const picks = ids.map((id) => ({
    label: experts[id].name,
    description: id,
    detail: experts[id].description,
    expertId: id,
  }));

  const selected = await vscode.window.showQuickPick(picks, {
    title: "Remove Expert",
    placeHolder: "Select an expert to remove",
  });
  if (!selected) return;

  const confirm = await vscode.window.showWarningMessage(
    `Remove expert "${selected.label}"?`,
    { modal: true },
    "Remove"
  );
  if (confirm !== "Remove") return;

  delete experts[selected.expertId];
  await config.update(
    "experts",
    experts,
    vscode.ConfigurationTarget.Workspace
  );

  vscode.window.showInformationMessage(
    `Expert "${selected.label}" removed.`
  );
}

async function importExpertsFromYaml(): Promise<void> {
  const fileUri = await vscode.window.showOpenDialog({
    title: "Import Experts from YAML",
    filters: { "YAML files": ["yaml", "yml"] },
    canSelectMany: false,
  });
  if (!fileUri || fileUri.length === 0) return;

  const filePath = fileUri[0].fsPath;
  let content: string;
  try {
    content = fs.readFileSync(filePath, "utf-8");
  } catch (err) {
    vscode.window.showErrorMessage(
      `Failed to read file: ${(err as Error).message}`
    );
    return;
  }

  // Simple YAML parsing for experts section
  // This handles the common case — for complex YAML, users should
  // use the yamlConfigPath setting directly
  const experts = parseExpertsFromYaml(content);
  if (Object.keys(experts).length === 0) {
    vscode.window.showWarningMessage(
      "No experts found in the YAML file."
    );
    return;
  }

  const config = vscode.workspace.getConfiguration("prsm");
  const existing = config.get<Record<string, ExpertConfig>>(
    "experts",
    {}
  );

  const merged = { ...existing, ...experts };
  const newCount = Object.keys(experts).length;

  await config.update(
    "experts",
    merged,
    vscode.ConfigurationTarget.Workspace
  );

  vscode.window.showInformationMessage(
    `Imported ${newCount} expert(s) from YAML.`
  );
}

/**
 * Parse expert profiles from a YAML config string.
 * Uses a simple line-based parser — not a full YAML parser.
 */
function parseExpertsFromYaml(
  content: string
): Record<string, ExpertConfig> {
  const experts: Record<string, ExpertConfig> = {};
  const lines = content.split("\n");

  let inExperts = false;
  let currentId = "";
  let currentField = "";
  let multilineValue = "";
  let multilineIndent = 0;

  for (const line of lines) {
    const trimmed = line.trimEnd();
    const indent = line.length - line.trimStart().length;

    // Detect experts: section at top level
    if (trimmed === "experts:" && indent === 0) {
      inExperts = true;
      continue;
    }

    // Exit experts section when we hit another top-level key
    if (inExperts && indent === 0 && trimmed && !trimmed.startsWith("#")) {
      inExperts = false;
      // Flush any pending multiline
      if (currentId && currentField && multilineValue) {
        setField(experts, currentId, currentField, multilineValue.trim());
        multilineValue = "";
        currentField = "";
      }
      continue;
    }

    if (!inExperts) continue;

    // Expert ID at indent 2
    if (indent === 2 && trimmed.endsWith(":") && !trimmed.includes(" ")) {
      // Flush previous multiline
      if (currentId && currentField && multilineValue) {
        setField(experts, currentId, currentField, multilineValue.trim());
        multilineValue = "";
        currentField = "";
      }
      currentId = trimmed.slice(0, -1);
      experts[currentId] = {
        name: currentId,
        description: "",
        system_prompt: "",
      };
      continue;
    }

    if (!currentId) continue;

    // Field at indent 4
    if (indent === 4 && trimmed.includes(":")) {
      // Flush previous multiline
      if (currentField && multilineValue) {
        setField(experts, currentId, currentField, multilineValue.trim());
        multilineValue = "";
      }

      const colonIdx = trimmed.indexOf(":");
      const key = trimmed.slice(0, colonIdx).trim();
      let value = trimmed.slice(colonIdx + 1).trim();

      if (value === ">" || value === "|") {
        // Start multiline
        currentField = key;
        multilineIndent = 6;
        multilineValue = "";
      } else {
        currentField = "";
        // Handle inline arrays: [Read, Write, Edit]
        if (value.startsWith("[") && value.endsWith("]")) {
          const items = value
            .slice(1, -1)
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean);
          setField(experts, currentId, key, items);
        } else {
          setField(experts, currentId, key, value);
        }
      }
      continue;
    }

    // Multiline continuation
    if (currentField && indent >= multilineIndent) {
      multilineValue += (multilineValue ? "\n" : "") + trimmed;
    }
  }

  // Flush final multiline
  if (currentId && currentField && multilineValue) {
    setField(experts, currentId, currentField, multilineValue.trim());
  }

  return experts;
}

function setField(
  experts: Record<string, ExpertConfig>,
  id: string,
  field: string,
  value: string | string[]
): void {
  if (!experts[id]) return;
  switch (field) {
    case "name":
      experts[id].name = value as string;
      break;
    case "description":
      experts[id].description = value as string;
      break;
    case "system_prompt":
      experts[id].system_prompt = value as string;
      break;
    case "tools":
      experts[id].tools = Array.isArray(value) ? value : [value as string];
      break;
    case "model":
      experts[id].model = value as string;
      break;
    case "permission_mode":
      experts[id].permission_mode = value as string;
      break;
    case "cwd":
      experts[id].cwd = value as string;
      break;
  }
}
