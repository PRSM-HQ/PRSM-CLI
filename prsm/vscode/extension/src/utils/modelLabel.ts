import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { parse as parseYaml } from "yaml";

import type { AgentNode } from "../protocol/types";
import type { SessionState } from "../state/sessionStore";

const UNKNOWN_MODEL_LABEL = "Unknown model";
const REASONING_EFFORT_PATTERN = /^(.+?)::reasoning_effort=(low|medium|high)$/;
const GPT_CODEX_REASONING_PATTERN = /^gpt-([0-9]+(?:\.[0-9]+)*)-codex$/;
const CLAUDE_FAMILY_NAMES = new Set(["opus", "sonnet", "haiku"]);
const MODEL_VERSION_FAMILIES = new Set(["opus", "sonnet", "haiku", "spark"]);
const PROVIDER_TOKEN_SPLIT_RE = /[.\-_]+/;
const DATE_TOKEN_RE = /^\d{6,8}$/;
const CLAUDE_LEGACY_VERSIONED_FAMILY_RE =
  /^claude-(\d+(?:[.-]\d+)*)-(opus|sonnet|haiku)(?:-|$)/i;

function _stripTrailingDateVersion(version: string): string {
  const parts = version.split(PROVIDER_TOKEN_SPLIT_RE).filter(Boolean);
  if (parts.length > 1 && DATE_TOKEN_RE.test(parts[parts.length - 1])) {
    parts.pop();
  }
  return parts.join(".");
}

function _normalizeModelIdForAliasLookup(modelId: string): string {
  const modelWithoutEffort = modelId
    .split("::reasoning_effort=")[0]
    .trim()
    .toLowerCase();
  if (!modelWithoutEffort) return "";

  const parts = modelWithoutEffort
    .replace(/_/g, "-")
    .split(PROVIDER_TOKEN_SPLIT_RE)
    .filter(Boolean);
  if (parts.length > 1 && DATE_TOKEN_RE.test(parts[parts.length - 1])) {
    parts.pop();
  }

  return parts.join("-");
}

function _titleizeWord(word: string): string {
  return word.charAt(0).toUpperCase() + word.slice(1);
}

function _humanizeProviderAlias(
  alias: string,
  provider: "gemini" | "gpt",
): string {
  const normalized = alias.toLowerCase();
  const providerPrefix = `${provider}-`;
  const body = normalized.startsWith(providerPrefix)
    ? normalized.slice(providerPrefix.length)
    : normalized;

  const parts = body.split(PROVIDER_TOKEN_SPLIT_RE).filter(Boolean);
  const numericParts: string[] = [];
  while (parts.length > 0 && /^\d+$/.test(parts[0] ?? "")) {
    numericParts.push(parts.shift() ?? "");
  }

  let version = numericParts.join(".");
  if (version) {
    version = _stripTrailingDateVersion(version);
  }

  const suffixParts = provider === "gpt"
    ? parts.filter((p) => p !== "codex")
    : parts;
  const suffix = suffixParts.map(_titleizeWord).join(" ");

  const providerLabel = provider === "gpt" ? "GPT" : "Gemini";
  if (!version && !suffix) return providerLabel;
  if (!version) return `${providerLabel} ${suffix}`.trim();
  if (!suffix) return `${providerLabel} ${version}`.trim();
  return `${providerLabel} ${version} ${suffix}`.trim();
}

type AliasCache = {
  filePath?: string;
  mtimeMs?: number;
  aliasByRuntimeId: Map<string, string>;
};

let aliasCache: AliasCache = {
  aliasByRuntimeId: new Map(),
};

function _findModelsYamlPath(): string | undefined {
  const candidate = path.join(os.homedir(), ".prsm", "models.yaml");
  if (fs.existsSync(candidate)) {
    return candidate;
  }
  return undefined;
}

function _readAliasIndexFromModelsYaml(filePath: string): Map<string, string> {
  const next = new Map<string, string>();
  const raw = fs.readFileSync(filePath, "utf-8");
  const parsed = (parseYaml(raw) || {}) as Record<string, unknown>;
  const models = (parsed.models || {}) as Record<string, unknown>;
  const bareModelAliases = new Map<string, string[]>();

  for (const [alias, rawCfg] of Object.entries(models)) {
    if (!alias || typeof alias !== "string") continue;
    const cfg = (rawCfg || {}) as Record<string, unknown>;
    const modelId = String(cfg.model_id ?? alias).trim();
    if (!modelId) continue;
    const effort = String(cfg.reasoning_effort ?? "").trim();
    const runtimeId = effort
      ? `${modelId}::reasoning_effort=${effort}`
      : modelId;
    next.set(runtimeId, alias);
    const normalizedRuntimeId = _normalizeModelIdForAliasLookup(runtimeId);
    if (normalizedRuntimeId && !next.has(normalizedRuntimeId)) {
      next.set(normalizedRuntimeId, alias);
    }
    const normalizedModelId = _normalizeModelIdForAliasLookup(modelId);
    if (
      normalizedModelId &&
      !next.has(normalizedModelId) &&
      normalizedModelId !== normalizedRuntimeId
    ) {
      next.set(normalizedModelId, alias);
    }
    const list = bareModelAliases.get(modelId) ?? [];
    list.push(alias);
    bareModelAliases.set(modelId, list);
  }

  // Add unambiguous bare-ID mappings.
  for (const [modelId, aliases] of bareModelAliases.entries()) {
    if (aliases.length === 1 && !next.has(modelId)) {
      next.set(modelId, aliases[0]);
    }
  }

  return next;
}

function _extractVersionFromModelId(family: string, modelId: string): string | undefined {
  const normalized = modelId
    .split("::reasoning_effort=")[0]
    .trim()
    .toLowerCase()
    .replace(/_/g, "-");

  if (family === "spark") {
    const sparkMatch = /^gpt-([0-9]+(?:[.-][0-9]+)*)-spark(?:-|$)/.exec(normalized);
    if (!sparkMatch) {
      return undefined;
    }
    return sparkMatch[1].replace(/-/g, ".");
  }

  const pattern = new RegExp(`^claude-${family}-(.+)$`, "i");
  const modelMatch = pattern.exec(normalized);
  if (!modelMatch) {
    return undefined;
  }
  const rawVersion = (modelMatch[1] ?? "").trim();
  if (!rawVersion) {
    return undefined;
  }
  return _stripTrailingDateVersion(rawVersion.replace(/-/g, "."));
}

function _humanizeAlias(alias: string, modelId?: string): string {
  const trimmed = alias.trim();
  const lower = trimmed.toLowerCase();
  const normalizedAlias = lower.replace(/_/g, "-").split("::reasoning_effort=")[0];

  const legacyMatch = CLAUDE_LEGACY_VERSIONED_FAMILY_RE.exec(normalizedAlias);
  if (legacyMatch) {
    const version = legacyMatch[1]?.replace(/-/g, ".") || "";
    const family = legacyMatch[2]?.toLowerCase() || "";
    if (version && family) {
      const familyLabel = family.charAt(0).toUpperCase() + family.slice(1);
      return `${familyLabel} ${version}`;
    }
  }

  const familyMatch = /^(?:claude-)?(opus|sonnet|haiku|spark)(?:-(.+))?$/i.exec(trimmed);
  if (!familyMatch) {
    if (lower === "opus") return "Opus";
    if (lower === "sonnet") return "Sonnet";
    if (lower === "haiku") return "Haiku";
    if (lower === "codex" && modelId) {
      const normalized = modelId.toLowerCase().split("::reasoning_effort=")[0];
      return _humanizeProviderAlias(normalized, "gpt");
    }
    if (lower === "gemini" && modelId) {
      const normalized = modelId.toLowerCase().split("::reasoning_effort=")[0];
      return _humanizeProviderAlias(normalized, "gemini");
    }
    if (lower === "claude" && modelId) {
      const claudeMatch = /^(?:claude-)?(opus|sonnet|haiku)(?:-(.+))?$/i.exec(modelId);
      if (claudeMatch) {
        return _humanizeAlias(claudeMatch[0] ?? modelId, modelId);
      }
      return modelId;
    }
    if (lower.startsWith("gemini-")) {
      return _humanizeProviderAlias(trimmed, "gemini");
    }
    if (lower.startsWith("gpt-")) {
      return _humanizeProviderAlias(trimmed, "gpt");
    }
    if (lower.startsWith("opus-") || lower.startsWith("sonnet-") || lower.startsWith("haiku-")) {
      const fallbackMatch = /^(?:claude-)?(opus|sonnet|haiku)(?:-(.+))?$/i.exec(trimmed);
      if (fallbackMatch) {
        const family = fallbackMatch[1].toLowerCase();
        const familyLabel = family.charAt(0).toUpperCase() + family.slice(1);
        const suffix = fallbackMatch[2];
        if (!suffix) {
          return familyLabel;
        }
        return `${familyLabel} ${_stripTrailingDateVersion(suffix.replace(/-/g, "."))}`;
      }
    }
    return trimmed;
  }

  const family = familyMatch[1].toLowerCase();
  if (!MODEL_VERSION_FAMILIES.has(family)) {
    return trimmed;
  }

  const familyLabel = family.charAt(0).toUpperCase() + family.slice(1);
  const suffix = familyMatch[2];
  if (!suffix) {
    if (modelId) {
      const version = _extractVersionFromModelId(family, modelId);
      if (version) {
        if (CLAUDE_FAMILY_NAMES.has(family) || family === "spark") {
          return `${familyLabel} ${version}`;
        }
      }
    }
    return familyLabel;
  }
  return `${familyLabel} ${_stripTrailingDateVersion(suffix.replace(/-/g, "."))}`;
}

function _getAliasIndex(): Map<string, string> {
  const modelsPath = _findModelsYamlPath();
  if (!modelsPath) return aliasCache.aliasByRuntimeId;

  let mtimeMs = 0;
  try {
    mtimeMs = fs.statSync(modelsPath).mtimeMs;
  } catch {
    return aliasCache.aliasByRuntimeId;
  }

  if (
    aliasCache.filePath === modelsPath &&
    aliasCache.mtimeMs === mtimeMs &&
    aliasCache.aliasByRuntimeId.size > 0
  ) {
    return aliasCache.aliasByRuntimeId;
  }

  try {
    aliasCache = {
      filePath: modelsPath,
      mtimeMs,
      aliasByRuntimeId: _readAliasIndexFromModelsYaml(modelsPath),
    };
    return aliasCache.aliasByRuntimeId;
  } catch {
    return aliasCache.aliasByRuntimeId;
  }
}

/** Convert runtime model descriptors into user-facing alias labels. */
export function toModelAliasLabel(modelId?: string | null): string | undefined {
  if (!modelId) return undefined;
  const trimmed = modelId.trim();
  if (!trimmed) return undefined;

  const aliasByRuntimeId = _getAliasIndex();
  const normalizedLookup = _normalizeModelIdForAliasLookup(trimmed);
  const directAlias = aliasByRuntimeId.get(trimmed)
    || (normalizedLookup ? aliasByRuntimeId.get(normalizedLookup) : undefined);
  if (directAlias) {
    const directLabel = _humanizeAlias(directAlias, trimmed);
    const normalizedDirect = directLabel.toLowerCase().trim();
    const hasModelVersion = /\d/.test(trimmed);
    const isGenericAliasLabel =
      normalizedDirect === "opus" ||
      normalizedDirect === "sonnet" ||
      normalizedDirect === "haiku" ||
      normalizedDirect === "claude";
    if (isGenericAliasLabel && hasModelVersion) {
      return _humanizeAlias(trimmed) || directLabel;
    }
    return directLabel;
  }

  // If caller already gave an alias, preserve it.
  if (
    aliasByRuntimeId.size > 0 &&
    Array.from(aliasByRuntimeId.values()).includes(trimmed)
  ) {
    return _humanizeAlias(trimmed, trimmed);
  }

  // Backward-compatible fallback when no alias mapping exists.
  const match = trimmed.match(REASONING_EFFORT_PATTERN);
  if (!match) return trimmed;
  const base = match[1];
  const effort = match[2];
  if (!base || !effort) return trimmed;
  const gptMatch = base.match(GPT_CODEX_REASONING_PATTERN);
  if (gptMatch) {
    const version = gptMatch[1].replace(/\./g, "-");
    return `gpt-${version}-${effort}`;
  }
  const sparkMatch = /^gpt-([0-9]+(?:\.[0-9]+)*)-spark$/i.exec(base);
  if (sparkMatch) {
    return `Spark ${sparkMatch[1]}`;
  }
  if (CLAUDE_FAMILY_NAMES.has(base.toLowerCase())) {
    return `${base.charAt(0).toUpperCase()}${base.slice(1).toLowerCase()}-${effort}`;
  }
  return `${base}-${effort}`;
}

/**
 * Resolve a display label for a session model using canonical precedence:
 * 1) alias-normalized session.currentModel
 * 2) alias-normalized master agent model
 * 3) Unknown model
 */
export function resolveSessionModelLabel(
  session: SessionState | undefined,
  masterAgent?: AgentNode
): string {
  const sessionModel = toModelAliasLabel(session?.currentModel);
  if (sessionModel) {
    return sessionModel;
  }
  const masterModel = toModelAliasLabel(masterAgent?.model);
  if (masterModel) {
    return masterModel;
  }
  return UNKNOWN_MODEL_LABEL;
}

export function isUnknownModelLabel(label: string | undefined): boolean {
  return !label || label === UNKNOWN_MODEL_LABEL;
}

export { UNKNOWN_MODEL_LABEL };
