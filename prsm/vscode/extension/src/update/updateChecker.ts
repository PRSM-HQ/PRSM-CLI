import * as fs from "fs";
import * as https from "https";
import * as os from "os";
import * as path from "path";
import * as vscode from "vscode";
import { execFile, execSync } from "child_process";

interface InstallState {
  installed_version?: string;
  last_update_check_at?: string;
  last_skipped_update_version?: string;
}

interface GitHubRelease {
  tag_name?: string;
  name?: string;
  html_url?: string;
  published_at?: string;
}

interface UpdateCandidate {
  version: string;
  name: string;
  htmlUrl: string;
  publishedAt: string;
}

const RELEASE_API_URL = "https://api.github.com/repos/prsm/prism-CLI/releases/latest";
const INSTALL_SCRIPT_URL = "https://raw.githubusercontent.com/prsm/prism-CLI/main/install.sh";
const INSTALL_BAT_URL = "https://raw.githubusercontent.com/prsm/prism-CLI/main/install.bat";
const UPDATE_INTERVAL_MS = 6 * 60 * 60 * 1000;
const STATE_FILE = path.join(os.homedir(), ".prsm", "install-state.json");

function normalizeVersion(input: string): string {
  return input.trim().replace(/^v/i, "").replace(/[^0-9.].*$/, "");
}

function parseVersion(input: string): number[] {
  const normalized = normalizeVersion(input);
  if (!normalized) return [];
  return normalized
    .split(".")
    .map((segment) => parseInt(segment, 10))
    .map((value) => (Number.isFinite(value) ? value : 0));
}

function compareVersions(a: string, b: string): number {
  const left = parseVersion(a);
  const right = parseVersion(b);
  const maxLength = Math.max(left.length, right.length);
  for (let i = 0; i < maxLength; i += 1) {
    const leftValue = left[i] ?? 0;
    const rightValue = right[i] ?? 0;
    if (leftValue !== rightValue) {
      return leftValue > rightValue ? 1 : -1;
    }
  }
  return 0;
}

function readInstallState(): InstallState {
  if (!fs.existsSync(STATE_FILE)) {
    return {};
  }

  try {
    const raw = fs.readFileSync(STATE_FILE, "utf-8");
    const parsed = JSON.parse(raw);
    if (typeof parsed === "object" && parsed !== null) {
      return parsed as InstallState;
    }
  } catch (error) {
    console.error("Failed to read install-state.json", error);
  }

  return {};
}

function saveInstallState(state: InstallState): void {
  try {
    fs.mkdirSync(path.dirname(STATE_FILE), { recursive: true });
    fs.writeFileSync(STATE_FILE, `${JSON.stringify(state, null, 2)}\n`, "utf-8");
  } catch (error) {
    console.error("Failed to save install-state.json", error);
  }
}

function shouldPollNow(state: InstallState): boolean {
  if (!state.last_update_check_at) {
    return true;
  }

  const lastChecked = Date.parse(state.last_update_check_at);
  if (!Number.isFinite(lastChecked)) {
    return true;
  }

  const elapsed = Date.now() - lastChecked;
  return elapsed >= UPDATE_INTERVAL_MS;
}

async function fetchLatestRelease(): Promise<UpdateCandidate> {
  const body = await new Promise<string>((resolve, reject) => {
    const request = https.get(
      RELEASE_API_URL,
      {
        headers: {
          "Accept": "application/vnd.github+json",
          "User-Agent": "prsm-vscode-update-checker",
        },
      },
      (response) => {
        if (response.statusCode === 403) {
          reject(new Error("GitHub API rate-limited. Try again later."));
          return;
        }
        if (!response.statusCode || response.statusCode < 200 || response.statusCode >= 300) {
          reject(new Error(`GitHub API returned ${response.statusCode ?? "unknown"}`));
          return;
        }

        let body = "";
        response.setEncoding("utf-8");
        response.on("data", (chunk) => {
          body += chunk;
        });
        response.on("end", () => resolve(body));
      },
    );
    request.on("error", reject);
    request.setTimeout(10_000, () => {
      request.destroy(new Error("GitHub API request timed out"));
    });
  });

  const release = JSON.parse(body) as GitHubRelease;
  const tag = normalizeVersion(release.tag_name ?? "");
  if (!tag) {
    throw new Error("Release did not include a tag_name");
  }

  return {
    version: tag,
    name: release.name ?? release.tag_name ?? tag,
    htmlUrl: release.html_url ?? "",
    publishedAt: release.published_at ?? "",
  };
}

function detectInstalledVersionFromCli(): string {
  try {
    const output = execSync("prsm --version", {
      encoding: "utf-8",
      timeout: 4_000,
    });
    const match = output.match(/\d+\.\d+(?:\.\d+)?(?:\.\d+)?/);
    if (match) {
      return match[0];
    }
  } catch {
    return "";
  }
  return "";
}

async function runInstallerCommand(): Promise<void> {
  if (process.platform === "win32") {
    const script = [
      "$ProgressPreference = \"SilentlyContinue\"",
      "$ErrorActionPreference = \"Stop\"",
      `$url = '${INSTALL_BAT_URL}'`,
      `$tmp = Join-Path $env:TEMP \"prsm-install-update.bat\"`,
      "Invoke-WebRequest -Uri $url -OutFile $tmp",
      "& $tmp --update --start-server",
    ].join(";");
    await runCommand("powershell", [
      "-NoProfile",
      "-ExecutionPolicy",
      "Bypass",
      "-Command",
      script,
    ]);
    return;
  }

  await runCommand("bash", [
    "-lc",
    `curl -fsSL ${INSTALL_SCRIPT_URL} | bash -s -- --update --start-server`,
  ]);
}

function runCommand(command: string, args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = execFile(command, args, {
      shell: false,
      env: process.env,
      timeout: 20 * 60 * 1000,
    });

    child.stdout?.on("data", (chunk) => {
      const message = chunk.toString().trim();
      if (message) {
        console.log(`[prsm-update] ${message}`);
      }
    });
    child.stderr?.on("data", (chunk) => {
      const message = chunk.toString().trim();
      if (message) {
        console.error(`[prsm-update] ${message}`);
      }
    });

    child.on("error", (error) => {
      reject(error);
    });
    child.on("close", (code) => {
      if (code === 0 || code === null) {
        resolve();
      } else {
        reject(new Error(`Update command exited with code ${code}`));
      }
    });
  });
}

export async function notifyIfUpdateAvailable(
  outputChannel: vscode.OutputChannel,
): Promise<void> {
  const state = readInstallState();
  if (!shouldPollNow(state)) {
    return;
  }

  let updateInfo: UpdateCandidate | null = null;
  try {
    updateInfo = await fetchLatestRelease();
    state.last_update_check_at = new Date().toISOString();
  } catch (error) {
    outputChannel.appendLine(`Update check failed: ${(error as Error).message}`);
    saveInstallState(state);
    return;
  }

  const installedVersion = normalizeVersion(
    state.installed_version ?? detectInstalledVersionFromCli(),
  );
  const ignoreVersion = normalizeVersion(state.last_skipped_update_version ?? "");
  state.last_update_check_at = new Date().toISOString();
  saveInstallState(state);

  if (!installedVersion || !updateInfo) {
    return;
  }

  if (!updateInfo.version || compareVersions(updateInfo.version, installedVersion) <= 0) {
    return;
  }
  if (ignoreVersion && compareVersions(updateInfo.version, ignoreVersion) === 0) {
    return;
  }

  const message = `PRSM update available: ${installedVersion} → ${updateInfo.version}`;
  const selection = await vscode.window.showInformationMessage(
    message,
    "Update now",
    "Not now",
    "Ignore this release",
  );

  if (selection === "Update now") {
    try {
      await runInstallerCommand();
      outputChannel.appendLine("Update installed. Reload VS Code to apply extension updates.");
      const reload = await vscode.window.showInformationMessage(
        "PRSM update installed. Reload window now?",
        "Reload",
        "Later",
      );
      if (reload === "Reload") {
        await vscode.commands.executeCommand("workbench.action.reloadWindow");
      }
    } catch (error) {
      outputChannel.appendLine(`Update failed: ${(error as Error).message}`);
      vscode.window.showErrorMessage(
        "PRSM update failed. Run install script manually from a terminal.",
      );
    }
    return;
  }

  if (selection === "Ignore this release") {
    state.last_skipped_update_version = updateInfo.version;
    saveInstallState(state);
  }
}
