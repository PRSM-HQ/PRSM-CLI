// @ts-check
/**
 * Webview client-side script for rendering agent conversations.
 *
 * Security: All user-facing content is escaped via textContent before
 * being placed in the DOM. The only HTML construction uses static
 * templates with escaped data insertions.
 */

// Acquire VS Code API (can only be called once)
const vscode = acquireVsCodeApi();

const conversationEl = document.getElementById("conversation");
const headerEl = document.getElementById("agent-header");
const inputContainer = document.getElementById("input-container");
const promptInput = document.getElementById("prompt-input");
const sendBtn = document.getElementById("send-btn");
const contextUsageEl = document.getElementById("context-usage-badge");
const modelSelectorBtn = document.getElementById("model-selector-btn");
const debugBar = document.getElementById("debug-bar");
let debugBarHidden = true;

function updateDebugBar(agentId, msgCount, busy) {
  if (!debugBar) return;
  if (!debugBarHidden) debugBar.style.display = "block";
  debugBar.textContent = `[dbg] updates=${fullUpdateCount} agent=${agentId || "?"} msgs=${msgCount} busy=${busy} t=${new Date().toLocaleTimeString()}`;
}

function updateContextUsage(usage) {
  if (!contextUsageEl) return;
  if (!usage || typeof usage.percent_used !== "number") {
    contextUsageEl.classList.add("hidden");
    contextUsageEl.textContent = "";
    contextUsageEl.classList.remove("warn", "danger");
    return;
  }

  const percent = Math.max(0, usage.percent_used);
  contextUsageEl.textContent = `Context ${percent.toFixed(1)}%`;
  contextUsageEl.classList.remove("hidden", "warn", "danger");

  if (percent >= 90) {
    contextUsageEl.classList.add("danger");
  } else if (percent >= 75) {
    contextUsageEl.classList.add("warn");
  }
}

function updateModelSelector(currentModel) {
  if (!modelSelectorBtn) return;
  const nameEl = modelSelectorBtn.querySelector(".model-name");
  if (!nameEl) return;

  if (currentModel) {
    // Truncate long model names for display
    const displayName = currentModel.length > 20
      ? currentModel.substring(0, 17) + "..."
      : currentModel;
    nameEl.textContent = displayName;
    modelSelectorBtn.setAttribute("data-tooltip", `Model: ${currentModel}\nClick to change`);
  } else {
    nameEl.textContent = "";
    modelSelectorBtn.setAttribute("data-tooltip", "Select Model");
  }
}

function updateHeaderModel(currentModel) {
  if (!headerEl || !currentModel) return;
  const metaEl = headerEl.querySelector(".agent-meta");
  if (!metaEl) return;
  const role = metaEl.getAttribute("data-role") || "orchestrator";
  metaEl.textContent = `${role} \u00B7 ${currentModel}`;
}

function _canAutofocus() {
  return document.visibilityState === "visible" && document.hasFocus();
}

// Ctrl+D toggles debug bar visibility
document.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.key === "d") {
    e.preventDefault();
    if (!debugBar) return;
    debugBarHidden = !debugBarHidden;
    debugBar.style.display = debugBarHidden ? "none" : "block";
    return;
  }
  if (actionPickerState?.root?.isConnected) {
    if (e.key === "ArrowDown" || e.key === "ArrowRight") {
      e.preventDefault();
      stepActionPickerSelection(1);
      return;
    }
    if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
      e.preventDefault();
      stepActionPickerSelection(-1);
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      submitActionPickerSelection();
      return;
    }
    if (e.key === "Escape") {
      e.preventDefault();
      removeActionPicker();
      return;
    }
  }
  if (e.key === "Escape" && editResendIndex >= 0) {
    e.preventDefault();
    cancelEditResendMode();
  }
});

let activeStreamEl = null;
let thinkingEl = null;
let autoScroll = true;
let currentPlanFilePath = null;
const expandedToolCallIds = new Set();
let activeTranscriptAgentId = null;
let modelSwitchMarkers = [];
let actionPickerState = null;
/** @type {{ toolCallId: string; messageIndex: number | null } | null} */
let pendingToolCallJump = null;
/** @type {string | null} */
let pendingSnapshotJump = null;

// ── Persisted UI state (survives webview reload) ──
const _savedWebviewState = vscode.getState() || {};
const _savedUi = _savedWebviewState.uiState || {};
let promptDraft = typeof _savedUi.promptDraft === "string" ? _savedUi.promptDraft : "";
let questionDrafts = _savedUi.questionDrafts && typeof _savedUi.questionDrafts === "object"
  ? { ..._savedUi.questionDrafts }
  : {};
let pendingFocusState = _savedUi.focus && typeof _savedUi.focus === "object" ? _savedUi.focus : null;
let pendingScrollTop = Number.isFinite(_savedUi.scrollTop) ? Number(_savedUi.scrollTop) : null;
if (typeof _savedUi.autoScroll === "boolean") {
  autoScroll = _savedUi.autoScroll;
}
let _persistScheduled = false;

function schedulePersistUiState() {
  if (_persistScheduled) return;
  _persistScheduled = true;
  requestAnimationFrame(() => {
    _persistScheduled = false;
    const nextState = {
      ..._savedWebviewState,
      uiState: {
        promptDraft,
        questionDrafts,
        focus: pendingFocusState,
        scrollTop: conversationEl ? conversationEl.scrollTop : 0,
        autoScroll,
      },
    };
    vscode.setState(nextState);
  });
}

function setQuestionDraft(requestId, value) {
  if (!requestId) return;
  if (value) {
    questionDrafts[requestId] = value;
  } else {
    delete questionDrafts[requestId];
  }
  schedulePersistUiState();
}

function _captureFocusStateFromElement(target) {
  if (!target) return null;
  if (target === promptInput) {
    return {
      kind: "prompt",
      selectionStart: promptInput.selectionStart || 0,
      selectionEnd: promptInput.selectionEnd || 0,
    };
  }
  if (target.classList && target.classList.contains("question-card-input")) {
    const card = target.closest(".question-card");
    const requestId = card && card.dataset ? card.dataset.requestId : "";
    if (!requestId) return null;
    return {
      kind: "question-input",
      requestId,
      selectionStart: target.selectionStart || 0,
      selectionEnd: target.selectionEnd || 0,
    };
  }
  return null;
}

function _applyPendingFocusState() {
  if (!_canAutofocus()) return false;
  if (!pendingFocusState) return false;
  if (pendingFocusState.kind === "prompt" && promptInput) {
    promptInput.focus();
    if (typeof promptInput.setSelectionRange === "function") {
      promptInput.setSelectionRange(
        pendingFocusState.selectionStart || 0,
        pendingFocusState.selectionEnd || pendingFocusState.selectionStart || 0,
      );
    }
    return true;
  }
  if (pendingFocusState.kind === "question-input" && pendingFocusState.requestId) {
    const input = conversationEl
      ? conversationEl.querySelector(`.question-card[data-request-id="${pendingFocusState.requestId}"] .question-card-input`)
      : null;
    if (input) {
      input.focus();
      if (typeof input.setSelectionRange === "function") {
        input.setSelectionRange(
          pendingFocusState.selectionStart || 0,
          pendingFocusState.selectionEnd || pendingFocusState.selectionStart || 0,
        );
      }
      return true;
    }
  }
  return false;
}

// ── Edit-resend state ──
// When a user clicks a historical user message, we enter "edit-resend" mode.
// Messages below the selected prompt are greyed out, and a banner appears
// above the input with a cancel button and a "revert files" checkbox.
let editResendIndex = -1; // index in the messages array of the selected user msg
let editResendRevertFiles = false;
let editResendPromptPreview = "";

// ── File autocomplete state ──
let completerEl = null;
let completerItems = [];
let completerSelectedIndex = -1;
let completerActive = false;
let completerAnchor = -1; // cursor position of the @ character
let completerDebounce = null;
let completerRequestId = 0; // track request ordering to discard stale results

/**
 * Format a file size for display (matches CLI _format_size).
 * @param {number} size
 * @returns {string}
 */
function formatFileSize(size) {
  if (size < 1024) return `${size}B`;
  if (size < 1024 * 1024) return `${Math.floor(size / 1024)}KB`;
  return `${Math.floor(size / (1024 * 1024))}MB`;
}

// Track pending user message so it survives renderFull() clears
// (between engine_started and agent_spawned, fullUpdate may have empty messages)
let pendingUserMessage = null;

// Busy state: when true, sending a prompt shows the action picker
let isBusy = false;
let isLaunching = false;
let launchingTimeout = null;
let isSessionRunning = false;

// ── Webview self-refresh: request periodic state updates while busy ──
let refreshInterval = null;
let lastFullUpdateTime = 0;
let fullUpdateCount = 0;

function startSelfRefresh() {
  stopSelfRefresh();
  refreshInterval = setInterval(() => {
    const elapsed = Date.now() - lastFullUpdateTime;
    // If no fullUpdate received in 3+ seconds while busy, request one
    if (isBusy && elapsed > 3000) {
      vscode.postMessage({ type: "requestRefresh" });
    }
  }, 2000);
}

function stopSelfRefresh() {
  if (refreshInterval) {
    clearInterval(refreshInterval);
    refreshInterval = null;
  }
}

// Thinking verbs that cycle while the agent is thinking.
// Loaded from the server (prsm/shared_ui/thinking_verbs.txt and
// prsm/shared_ui/nsfw_thinking_verbs.txt). The hardcoded fallbacks below are
// only used when the server hasn't provided the lists yet.
const FALLBACK_SAFE_THINKING_VERBS = [
  "Thinking", "Pondering", "Analyzing", "Reasoning", "Contemplating",
  "Examining", "Processing", "Formulating", "Deliberating", "Evaluating",
  "Considering", "Synthesizing", "Deciphering", "Untangling", "Assembling",
  "Connecting dots", "Mewing", "Dabbing", "Vibing", "Chilling", "Grooving",
  "Ruminating", "Procrastinating", "Loafing", "Lazing", "Daydreaming",
  "Brooding", "Mulling over", "Yawning", "Exploring", "Speculating",
  "Loitering", "Sleeping", "Waking up", "Falling asleep",
  "Flibbertigibbeting", "Auto-defenestrating", "Gesticulating",
  "Extrapolating", "Congealing", "Elucidating", "Expectorating",
  "Gentrifying", "Seasoning", "Panicking",
];
const FALLBACK_NSFW_THINKING_VERBS = [
  "Shitting", "Fucking", "Jerking", "Jelking", "Sharting", "Farting",
  "Gooning", "Pissing", "Wiping", "Drinking", "Blacking out", "Twerking",
  "Quivering", "Self-fellating", "Murdering", "Hiding bodies",
  "Destroying evidence", "Lying to police", "Threatening witnesses",
  "Taking it to the grave",
];
let thinkingVerbs = [...FALLBACK_SAFE_THINKING_VERBS, ...FALLBACK_NSFW_THINKING_VERBS];
let thinkingInterval = null;

function applyThinkingVerbSettings(settings) {
  const includeNsfw = settings?.enableNsfw !== false;
  const customVerbs = Array.isArray(settings?.customVerbs)
    ? settings.customVerbs
      .filter((verb) => typeof verb === "string")
      .map((verb) => verb.trim())
      .filter((verb, idx, arr) => verb.length > 0 && arr.indexOf(verb) === idx)
    : [];

  // Use server-provided verb lists when available, otherwise fall back to hardcoded
  const safeVerbs = Array.isArray(settings?.safeVerbs) && settings.safeVerbs.length > 0
    ? settings.safeVerbs
    : FALLBACK_SAFE_THINKING_VERBS;
  const nsfwVerbs = Array.isArray(settings?.nsfwVerbs) && settings.nsfwVerbs.length > 0
    ? settings.nsfwVerbs
    : FALLBACK_NSFW_THINKING_VERBS;

  const merged = [...safeVerbs];
  if (includeNsfw) {
    for (const verb of nsfwVerbs) {
      if (!merged.includes(verb)) merged.push(verb);
    }
  }
  for (const verb of customVerbs) {
    if (!merged.includes(verb)) merged.push(verb);
  }
  thinkingVerbs = merged.length > 0 ? merged : ["Thinking", "Processing", "Analyzing"];
}

// ── Message listener ──

window.addEventListener("message", (event) => {
  const msg = event.data;
  switch (msg.type) {
    case "fullUpdate":
      applyThinkingVerbSettings(msg.thinkingVerbSettings);
      isSessionRunning = !!msg.sessionRunning;
      fullUpdateCount++;
      lastFullUpdateTime = Date.now();
      // Debug: send ack back so extension can log webview receives
      vscode.postMessage({
        type: "debugFullUpdate",
        agentId: msg.agent?.id?.slice(0, 8) ?? "none",
        agentState: msg.agent?.state ?? "?",
        msgCount: (msg.messages || []).length,
        busy: !!msg.busy,
        updateNum: fullUpdateCount,
      });
      renderFull(msg.agent, msg.messages, msg.sessionName, msg.currentModelLabel || msg.currentModel, msg.sessionSummary);
      updateDebugBar(msg.agent?.id?.slice(0, 8), (msg.messages || []).length, !!msg.busy);
      updateContextUsage(msg.contextUsage || null);
      updateModelSelector(msg.currentModelLabel || msg.currentModel);
      if (msg.showInput) {
        showInputBar();
        // Ignore stale non-busy updates while we're still waiting for launch confirmation.
        if (isLaunching && !msg.busy) {
          // Keep local launching UI until we observe a busy update.
        } else {
          const agent = msg.agent || {};
          const messages = Array.isArray(msg.messages) ? msg.messages : [];
          const state = String(agent.state || "");
          const terminalState =
            state === "completed" ||
            state === "error" ||
            state === "failed" ||
            state === "killed";
          const pendingPromptStillUnattached =
            !!pendingUserMessage &&
            !messages.some(
              (m) => m.role === "user" && m.content === pendingUserMessage
            );
          const keepLaunching =
            isLaunching &&
            !!msg.busy &&
            (!agent.id || (terminalState && pendingPromptStillUnattached));
          setBusyState(!!msg.busy, keepLaunching);
        }
      } else {
        hideInputBar();
      }
      // Re-show thinking indicator if agent is currently thinking
      if (msg.isThinking) {
        showThinking();
      }
      // Start/stop self-refresh based on busy state
      if (msg.busy) {
        startSelfRefresh();
      } else {
        stopSelfRefresh();
      }
      break;
    case "streamChunk":
      removeThinking();
      appendStreamChunk(msg.text);
      break;
    case "userMessage":
      pendingUserMessage = msg.content;
      appendUserMessage(msg.content);
      break;
    case "errorMessage":
      setLaunchingState(false);
      appendErrorMessage(msg.content);
      break;
    case "systemMessage":
      setLaunchingState(false);
      appendSystemMessage(msg.content);
      break;
    case "setBusyState":
      // Host marks busy immediately after prompt submission, often before
      // the root agent is spawned. Preserve launch UI during that gap.
      setBusyState(msg.busy, isLaunching && !!msg.busy);
      if (!msg.busy) removeThinking();
      break;
    case "thinking":
      showThinking();
      break;
    case "scrollToToolCall":
      requestToolCallJump(msg.toolCallId, msg.messageIndex);
      break;
    case "scrollToSnapshot":
      requestSnapshotJump(msg.snapshotId);
      break;
    case "queueUpdate":
      renderQueueDisplay(msg.items);
      break;
    case "focusInput":
      if (promptInput && _canAutofocus()) {
        if (pendingFocusState && pendingFocusState.kind === "question-input") {
          _applyPendingFocusState();
          break;
        }
        // Immediate focus attempt
        /** @type {HTMLTextAreaElement} */ (promptInput).focus();
        pendingFocusState = _captureFocusStateFromElement(promptInput);
        schedulePersistUiState();
        // Retry after a frame — VS Code webview focus can be flaky
        // if the panel hasn't fully received OS-level focus yet
        requestAnimationFrame(() => {
          /** @type {HTMLTextAreaElement} */ (promptInput).focus();
          pendingFocusState = _captureFocusStateFromElement(promptInput);
          schedulePersistUiState();
        });
      }
      break;
    case "showQuestion":
      appendQuestionCard(msg);
      break;
    case "showPlanLink":
      if (typeof msg.filePath === "string" && msg.filePath.length > 0) {
        currentPlanFilePath = msg.filePath;
        renderPlanLink();
      }
      break;
    case "dismissQuestion":
      dismissQuestionCard(msg.requestId);
      break;
    case "fileCompleteResult":
      console.log("[prsm] Received fileCompleteResult:", msg.completions?.length ?? 0, "items, requestId:", msg.requestId, "current:", completerRequestId);
      // Discard stale results from older requests
      if (msg.requestId !== undefined && msg.requestId !== completerRequestId) break;
      renderFileCompletions(msg.completions || []);
      break;
    case "modelSwitched":
      {
        const fallbackIndex = getRenderedMessageCount();
        const atMessageIndex = Number.isInteger(msg.atMessageIndex)
          ? Math.max(0, msg.atMessageIndex)
          : fallbackIndex;
        const marker = {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          atMessageIndex,
          oldModel: msg.oldModelLabel || msg.oldModel || "unknown",
          newModel: msg.newModelLabel || msg.newModel || "unknown",
        };
        const lastMarker = modelSwitchMarkers[modelSwitchMarkers.length - 1];
        const isDuplicate =
          !!lastMarker &&
          lastMarker.atMessageIndex === marker.atMessageIndex &&
          lastMarker.oldModel === marker.oldModel &&
          lastMarker.newModel === marker.newModel;
        if (!isDuplicate) {
          modelSwitchMarkers.push(marker);
        }
        appendModelSwitchIndicator(
          msg.oldModelLabel || msg.oldModel,
          msg.newModelLabel || msg.newModel,
        );
      }
      updateModelSelector(msg.newModelLabel || msg.newModel);
      updateHeaderModel(msg.newModelLabel || msg.newModel);
      break;
  }
});

// ── Input bar ──

function showInputBar() {
  if (inputContainer) {
    inputContainer.classList.remove("hidden");
    renderLaunchingStatus();
  }
}

function hideInputBar() {
  if (inputContainer) {
    inputContainer.classList.add("hidden");
    renderLaunchingStatus();
  }
}

function updateSendButton() {
  if (!sendBtn) return;
  const icon = sendBtn.querySelector(".send-icon");
  if (isLaunching) {
    sendBtn.classList.remove("stop-mode");
    sendBtn.classList.add("launching-mode");
    sendBtn.setAttribute("data-tooltip", "Launching agent\u2026");
    if (icon) icon.textContent = "\u25D4";
    return;
  }
  sendBtn.classList.remove("launching-mode");
  if (isBusy) {
    sendBtn.classList.add("stop-mode");
    sendBtn.setAttribute("data-tooltip", "Stop (interrupt current run)");
    if (icon) icon.textContent = "\u25A0";
    return;
  }
  sendBtn.classList.remove("stop-mode");
  sendBtn.setAttribute("data-tooltip", "Send (Enter)");
  if (icon) icon.textContent = "\u25B6";
}

function updateInputPlaceholder() {
  if (!promptInput) return;
  /** @type {HTMLTextAreaElement} */ (promptInput).placeholder = isLaunching
    ? "Launching agent\u2026"
    : isBusy
      ? "Agent working\u2026 type to queue, inject, or interrupt"
      : "Type a message...";
}

function setLaunchingState(launching) {
  isLaunching = launching;
  if (launching) {
    if (launchingTimeout) clearTimeout(launchingTimeout);
    launchingTimeout = setTimeout(() => {
      if (!isLaunching) return;
      setLaunchingState(false);
    }, 15000);
  } else if (launchingTimeout) {
    clearTimeout(launchingTimeout);
    launchingTimeout = null;
  }
  updateSendButton();
  updateInputPlaceholder();
  renderLaunchingStatus();
}

function renderLaunchingStatus() {
  if (!inputContainer) return;
  const existing = document.getElementById("launching-status");
  if (!isLaunching) {
    if (existing) existing.remove();
    return;
  }
  if (existing) return;
  const status = el("div", "launching-status");
  status.id = "launching-status";
  status.appendChild(el("span", "launching-spinner", ""));
  status.appendChild(el("span", "launching-label", "Launching agent\u2026"));
  const wrapper = document.getElementById("input-wrapper");
  if (wrapper) {
    inputContainer.insertBefore(status, wrapper);
  } else {
    inputContainer.insertBefore(status, inputContainer.firstChild);
  }
}

/** Update busy state — input is never disabled, only placeholder changes. */
function setBusyState(busy, preserveLaunching = false) {
  if (!preserveLaunching) {
    if (launchingTimeout) {
      clearTimeout(launchingTimeout);
      launchingTimeout = null;
    }
    isLaunching = false;
  }
  isBusy = busy;
  updateSendButton();
  updateInputPlaceholder();
  renderLaunchingStatus();
  if (!busy) {
    removeActionPicker();
  }
}

function sendPrompt() {
  if (!promptInput) return;
  if (isLaunching) return;
  const text = /** @type {HTMLTextAreaElement} */ (promptInput).value.trim();
  if (!text) return;

  dismissCompleter();

  if (isBusy || isSessionRunning) {
    if (actionPickerState?.root?.isConnected) {
      submitActionPickerSelection();
      return;
    }
    showActionPicker(text);
    return;
  }

  /** @type {any} */
  const msg = { type: "sendPrompt", prompt: text };

  // If in edit-resend mode, include metadata for the extension host
  if (editResendIndex >= 0) {
    msg.resendFromIndex = editResendIndex;
    msg.revertFiles = editResendRevertFiles;
  }

  setLaunchingState(true);
  vscode.postMessage(msg);
  /** @type {HTMLTextAreaElement} */ (promptInput).value = "";
  promptDraft = "";
  pendingFocusState = _captureFocusStateFromElement(promptInput);
  schedulePersistUiState();
  autoResizeInput();
  exitEditResendMode();
}

/**
 * Pre-fill the prompt input with historical user text for editing.
 * When msgIndex >= 0, activates edit-resend mode with visual invalidation.
 * @param {string} text
 * @param {number} [msgIndex=-1]
 */
function preloadPromptForEdit(text, msgIndex) {
  if (!promptInput) return;
  const ta = /** @type {HTMLTextAreaElement} */ (promptInput);
  ta.value = text || "";
  autoResizeInput();
  ta.focus();
  ta.selectionStart = ta.selectionEnd = ta.value.length;

  if (typeof msgIndex === "number" && msgIndex >= 0) {
    enterEditResendMode(msgIndex);
  }
}

// ── Edit-Resend Mode ──

/**
 * Enter edit-resend mode: grey out messages below the selected prompt,
 * show a banner above the input with cancel and "revert files" checkbox.
 * @param {number} msgIndex
 */
function enterEditResendMode(msgIndex) {
  editResendIndex = msgIndex;
  editResendRevertFiles = false;
  editResendPromptPreview = "";

  if (conversationEl && msgIndex >= 0 && msgIndex < conversationEl.children.length) {
    const selected = conversationEl.children[msgIndex];
    const body = selected.querySelector(".message-body");
    editResendPromptPreview = (body?.textContent || "").trim();
  }

  // Grey out conversation messages below the selected index
  applyInvalidation();

  // Show/update the banner above the input
  showEditResendBanner();
}

/** Exit edit-resend mode: reset state, remove banner, un-grey messages. */
function exitEditResendMode() {
  editResendIndex = -1;
  editResendRevertFiles = false;
  editResendPromptPreview = "";
  removeEditResendBanner();
  clearInvalidation();
}

/** Apply .message-invalidated to all conversation children past editResendIndex. */
function applyInvalidation() {
  if (!conversationEl || editResendIndex < 0) return;
  const children = conversationEl.children;
  for (let i = 0; i < children.length; i++) {
    children[i].classList.toggle("message-resend-origin", i === editResendIndex);
    if (i > editResendIndex) {
      children[i].classList.add("message-invalidated");
    } else {
      children[i].classList.remove("message-invalidated");
    }
  }
}

/** Remove .message-invalidated from all conversation children. */
function clearInvalidation() {
  if (!conversationEl) return;
  const styled = conversationEl.querySelectorAll(
    ".message-invalidated, .message-resend-origin"
  );
  styled.forEach((el) => {
    el.classList.remove("message-invalidated");
    el.classList.remove("message-resend-origin");
  });
}

/**
 * Cancel edit-resend mode using the same behavior as clicking "Cancel":
 * clear composer text and exit edit mode.
 */
function cancelEditResendMode() {
  if (editResendIndex < 0) return;
  /** @type {HTMLTextAreaElement} */ (promptInput).value = "";
  autoResizeInput();
  dismissCompleter();
  exitEditResendMode();
}

/** Show the edit-resend banner above the input wrapper. */
function showEditResendBanner() {
  removeEditResendBanner();
  if (!inputContainer) return;

  const banner = el("div", "edit-resend-banner");

  const infoRow = el("div", "edit-resend-info");
  const infoIcon = el("span", "edit-resend-icon", "\u270f\ufe0f");
  const infoText = el("span", "edit-resend-text",
    "Resending will revert the conversation to this prompt. Messages below will be discarded.");
  infoRow.appendChild(infoIcon);
  infoRow.appendChild(infoText);
  banner.appendChild(infoRow);

  if (editResendPromptPreview) {
    const preview = el("div", "edit-resend-preview", editResendPromptPreview);
    banner.appendChild(preview);
  }

  const controlsRow = el("div", "edit-resend-controls");

  // Checkbox: revert file changes
  const label = document.createElement("label");
  label.className = "edit-resend-checkbox-label";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.className = "edit-resend-checkbox";
  checkbox.checked = editResendRevertFiles;
  checkbox.addEventListener("change", () => {
    editResendRevertFiles = checkbox.checked;
  });
  label.appendChild(checkbox);
  label.appendChild(document.createTextNode(" Revert file changes to this prompt snapshot"));
  controlsRow.appendChild(label);

  // Cancel button
  const cancelBtn = el("button", "edit-resend-cancel", "Cancel");
  cancelBtn.addEventListener("click", cancelEditResendMode);
  controlsRow.appendChild(cancelBtn);

  banner.appendChild(controlsRow);

  // Insert before input wrapper
  const wrapper = document.getElementById("input-wrapper");
  if (wrapper) {
    inputContainer.insertBefore(banner, wrapper);
  } else {
    inputContainer.insertBefore(banner, inputContainer.firstChild);
  }
}

/** Remove the edit-resend banner. */
function removeEditResendBanner() {
  const existing = document.querySelector(".edit-resend-banner");
  if (existing) existing.remove();
}

function handlePrimaryButtonClick() {
  if (isBusy) {
    dismissCompleter();
    // Ask extension host to show a native VS Code confirmation modal.
    // Webview window.confirm() can be unreliable in some VS Code contexts.
    vscode.postMessage({ type: "confirmStopRun" });
    return;
  }
  sendPrompt();
}

// Handle Enter to send (Shift+Enter for newline) + autocomplete keys
if (promptInput) {
  if (promptDraft) {
    /** @type {HTMLTextAreaElement} */ (promptInput).value = promptDraft;
    autoResizeInput();
  }
  promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && editResendIndex >= 0) {
      e.preventDefault();
      cancelEditResendMode();
      return;
    }
    if (completerActive) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        completerSelectedIndex = Math.min(completerSelectedIndex + 1, completerItems.length - 1);
        highlightCompleterItem();
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        completerSelectedIndex = Math.max(completerSelectedIndex - 1, 0);
        highlightCompleterItem();
        return;
      }
      if (e.key === "Tab" || e.key === "Enter") {
        e.preventDefault();
        if (completerSelectedIndex >= 0 && completerSelectedIndex < completerItems.length) {
          acceptCompletion(completerItems[completerSelectedIndex]);
        } else if (completerItems.length > 0) {
          acceptCompletion(completerItems[0]);
        }
        // If no items, Enter/Tab is swallowed (matches CLI behavior)
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        dismissCompleter();
        return;
      }
      // Space ends completion mode (matches CLI behavior)
      // Let the space character through to the textarea
      if (e.key === " ") {
        dismissCompleter();
        // Fall through to normal handling — space is typed
      }
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      // Prevent the same Enter key event from bubbling to the global
      // action-picker key handler. Without this, pressing Enter while busy
      // can open the picker and immediately auto-submit the default option.
      e.stopPropagation();
      sendPrompt();
    }
  });
  // Auto-resize textarea + autocomplete trigger
  promptInput.addEventListener("input", () => {
    autoResizeInput();
    handleInputForCompleter();
    promptDraft = /** @type {HTMLTextAreaElement} */ (promptInput).value;
    pendingFocusState = _captureFocusStateFromElement(promptInput);
    schedulePersistUiState();
  });
}

if (sendBtn) {
  sendBtn.addEventListener("click", handlePrimaryButtonClick);
}
updateSendButton();

// Model selector button
if (modelSelectorBtn) {
  modelSelectorBtn.addEventListener("click", () => {
    vscode.postMessage({ type: "selectModel" });
  });
}

function autoResizeInput() {
  if (!promptInput) return;
  const ta = /** @type {HTMLTextAreaElement} */ (promptInput);
  ta.style.height = "auto";
  ta.style.height = Math.min(ta.scrollHeight, 150) + "px";
}

// ── File Autocomplete ──

/**
 * Called on every input event — detect @ and manage completer state.
 */
function handleInputForCompleter() {
  if (!promptInput) return;
  const ta = /** @type {HTMLTextAreaElement} */ (promptInput);
  const text = ta.value;
  const cursor = ta.selectionStart;

  // Only look at text before the cursor
  const before = text.slice(0, cursor);

  // Find the last @ before cursor that's not inside backticks
  const atIdx = findActiveAt(before);

  if (atIdx < 0) {
    dismissCompleter();
    return;
  }

  // Extract the prefix typed after @
  const prefix = before.slice(atIdx + 1);

  // If there's a space in the prefix, the @ reference is done
  if (/\s/.test(prefix)) {
    dismissCompleter();
    return;
  }

  completerAnchor = atIdx;
  completerActive = true;
  console.log("[prsm] @ detected, requesting completions for prefix:", JSON.stringify(prefix));
  requestFileCompletions(prefix);
}

/**
 * Find the last '@' in text that's not inside backticks.
 * Returns the index of '@', or -1 if none found.
 * @param {string} text
 * @returns {number}
 */
function findActiveAt(text) {
  // Mask backtick regions
  const masked = new Set();
  // Code blocks ```...```
  const blockRe = /```[\s\S]*?```/g;
  let m;
  while ((m = blockRe.exec(text)) !== null) {
    for (let i = m.index; i < m.index + m[0].length; i++) masked.add(i);
  }
  // Inline code `...`
  const inlineRe = /`[^`]*`/g;
  while ((m = inlineRe.exec(text)) !== null) {
    for (let i = m.index; i < m.index + m[0].length; i++) masked.add(i);
  }

  // Search backwards for the last unmasked @
  for (let i = text.length - 1; i >= 0; i--) {
    if (text[i] === "@" && !masked.has(i)) {
      // Make sure it's at start of line or preceded by whitespace
      if (i === 0 || /\s/.test(text[i - 1])) {
        return i;
      }
    }
  }
  return -1;
}

/**
 * Request file completions from the extension host (debounced).
 * @param {string} prefix
 */
function requestFileCompletions(prefix) {
  if (completerDebounce) clearTimeout(completerDebounce);
  completerDebounce = setTimeout(() => {
    completerRequestId++;
    console.log("[prsm] Sending fileComplete request:", JSON.stringify(prefix), "requestId:", completerRequestId);
    vscode.postMessage({ type: "fileComplete", prefix, limit: 10, requestId: completerRequestId });
  }, 100);
}

/**
 * Render the completion results into the dropdown.
 * @param {Array<{path: string, is_directory: boolean}>} completions
 */
function renderFileCompletions(completions) {
  if (!completerActive) return;
  console.log("[prsm] Rendering", completions.length, "file completions, active:", completerActive);

  ensureCompleterEl();
  completerItems = completions;
  completerSelectedIndex = completions.length > 0 ? 0 : -1;

  completerEl.textContent = "";

  if (completions.length === 0) {
    // No matches — hide completer entirely (matches CLI behavior)
    dismissCompleter();
    return;
  }

  for (let i = 0; i < completions.length; i++) {
    const item = completions[i];
    const row = el("div", "file-completer-item");
    if (i === 0) row.classList.add("selected");

    const icon = el("span", "file-completer-icon", item.is_directory ? "\uD83D\uDCC1" : "\uD83D\uDCC4");
    const pathSpan = el("span", "file-completer-path", item.path);

    row.appendChild(icon);
    row.appendChild(pathSpan);

    // Show file size (matches CLI behavior)
    if (item.size != null && !item.is_directory) {
      const sizeSpan = el("span", "file-completer-size", formatFileSize(item.size));
      row.appendChild(sizeSpan);
    }

    row.addEventListener("mousedown", (e) => {
      e.preventDefault(); // Don't blur the textarea
      acceptCompletion(item);
    });

    row.addEventListener("mouseenter", () => {
      completerSelectedIndex = i;
      highlightCompleterItem();
    });

    completerEl.appendChild(row);
  }

  completerEl.classList.remove("hidden");
}

/**
 * Highlight the currently selected item in the dropdown.
 */
function highlightCompleterItem() {
  if (!completerEl) return;
  const items = completerEl.querySelectorAll(".file-completer-item");
  items.forEach((item, i) => {
    item.classList.toggle("selected", i === completerSelectedIndex);
  });

  // Scroll selected item into view
  if (completerSelectedIndex >= 0 && items[completerSelectedIndex]) {
    items[completerSelectedIndex].scrollIntoView({ block: "nearest" });
  }
}

/**
 * Accept a completion — insert the path into the textarea.
 * @param {{path: string, is_directory: boolean}} item
 */
function acceptCompletion(item) {
  if (!promptInput) return;
  const ta = /** @type {HTMLTextAreaElement} */ (promptInput);
  const text = ta.value;
  const cursor = ta.selectionStart;

  // Replace text between the @ anchor and the current cursor position
  // "before" is everything up to and including the @
  const before = text.slice(0, completerAnchor + 1); // includes the @
  // "after" is everything after the cursor (text the user typed after the prefix)
  const after = text.slice(cursor);

  // For directories, the server returns paths with trailing "/" already (e.g. "src/")
  // so we don't need to add one. For files, add a trailing space.
  const suffix = item.is_directory ? "" : " ";
  const newValue = before + item.path + suffix + after;

  ta.value = newValue;

  // Position cursor right after the inserted path (+ suffix)
  const newCursor = completerAnchor + 1 + item.path.length + suffix.length;
  ta.selectionStart = ta.selectionEnd = newCursor;

  autoResizeInput();

  if (item.is_directory) {
    // Directory selected — keep completer open for drilling deeper.
    // Update anchor to stay at the same @ position; the prefix for the
    // next search is item.path (e.g. "src/") which lets FileIndex drill in.
    completerActive = true;
    requestFileCompletions(item.path);
  } else {
    dismissCompleter();
  }

  ta.focus();
}

/**
 * Create the completer dropdown element if it doesn't exist.
 */
function ensureCompleterEl() {
  if (completerEl) return;

  completerEl = el("div", "file-completer hidden");

  // Insert inside input-container, positioned relative to input-wrapper
  const container = document.getElementById("input-container");
  const wrapper = document.getElementById("input-wrapper");
  if (container && wrapper) {
    // Make the input container position:relative for absolute positioning
    container.style.position = "relative";
    container.style.overflow = "visible";
    container.insertBefore(completerEl, wrapper);
    console.log("[prsm] File completer element created and inserted into DOM");
  } else {
    console.warn("[prsm] Could not create file completer: container=", !!container, "wrapper=", !!wrapper);
  }
}

/**
 * Dismiss the completer dropdown.
 */
function dismissCompleter() {
  completerActive = false;
  completerItems = [];
  completerSelectedIndex = -1;
  completerAnchor = -1;
  // Bump requestId so any in-flight responses are discarded
  completerRequestId++;
  if (completerDebounce) {
    clearTimeout(completerDebounce);
    completerDebounce = null;
  }
  if (completerEl) {
    completerEl.classList.add("hidden");
  }
}

// ── Action Picker ──

/**
 * Show an inline action picker when the user sends a message while agent is busy.
 * @param {string} text
 */
function showActionPicker(text) {
  removeActionPicker();

  const picker = el("div", "action-picker");

  const title = el("div", "action-picker-title", "Agent is working. How should this message be delivered?");
  picker.appendChild(title);

  const options = [
    { action: "queue", label: "Queue", desc: "Send after current task finishes" },
    { action: "inject", label: "Inject", desc: "Send after current tool call completes" },
    { action: "interrupt", label: "Interrupt", desc: "Stop agent and send immediately" },
  ];

  const optionButtons = [];
  for (const opt of options) {
    const btn = el("button", `action-btn action-${opt.action}`);
    btn.appendChild(el("span", "action-label", opt.label));
    btn.appendChild(el("span", "action-desc", opt.desc));
    btn.addEventListener("click", () => {
      chooseAction(text, opt.action);
    });
    picker.appendChild(btn);
    optionButtons.push(btn);
  }

  const cancelBtn = el("button", "action-btn action-cancel", "Cancel");
  cancelBtn.addEventListener("click", removeActionPicker);
  picker.appendChild(cancelBtn);

  // Insert above input wrapper
  if (inputContainer) {
    inputContainer.insertBefore(picker, inputContainer.firstChild);
  }
  actionPickerState = {
    root: picker,
    text,
    options,
    optionButtons,
    selectedIndex: 0,
  };
  updateActionPickerSelection();
}

/**
 * @param {string} text
 * @param {string} action
 */
function chooseAction(text, action) {
  removeActionPicker();
  vscode.postMessage({ type: "promptAction", prompt: text, action });
  if (promptInput) {
    /** @type {HTMLTextAreaElement} */ (promptInput).value = "";
    autoResizeInput();
  }
}

function removeActionPicker() {
  const existing = document.querySelector(".action-picker");
  if (existing) existing.remove();
  actionPickerState = null;
}

function updateActionPickerSelection() {
  if (!actionPickerState) return;
  const count = actionPickerState.optionButtons.length;
  if (count === 0) return;
  const normalized = ((actionPickerState.selectedIndex % count) + count) % count;
  actionPickerState.selectedIndex = normalized;
  for (let i = 0; i < count; i++) {
    actionPickerState.optionButtons[i].classList.toggle("selected", i === normalized);
  }
}

/**
 * @param {number} delta
 */
function stepActionPickerSelection(delta) {
  if (!actionPickerState) return;
  actionPickerState.selectedIndex += delta;
  updateActionPickerSelection();
}

function submitActionPickerSelection() {
  if (!actionPickerState) return;
  const chosen = actionPickerState.options[actionPickerState.selectedIndex];
  if (!chosen) return;
  chooseAction(actionPickerState.text, chosen.action);
}

// ── Queue Display ──

/**
 * @param {string} mode
 * @returns {string}
 */
function queueActionLabel(mode) {
  if (mode === "inject") return "Inject after tool call";
  return "Queued";
}

/**
 * Render the list of queued messages between conversation and input.
 * @param {Array<{id: string; prompt: string; mode: string}>} items
 */
function renderQueueDisplay(items) {
  let queueEl = document.getElementById("queue-display");

  if (!items || items.length === 0) {
    if (queueEl) queueEl.remove();
    return;
  }

  if (!queueEl) {
    queueEl = el("div", "queue-display");
    queueEl.id = "queue-display";
  }
  // Keep queue pinned directly above input even after full re-renders.
  if (inputContainer && inputContainer.parentNode) {
    inputContainer.parentNode.insertBefore(queueEl, inputContainer);
  }

  queueEl.textContent = "";
  const header = el("div", "queue-header", `Queued tasks (${items.length})`);
  queueEl.appendChild(header);

  for (const item of items) {
    const row = el("div", `queue-item queue-item-${item.mode}`);
    const content = el("div", "queue-content");
    const verb = el("div", "queue-action-verb", queueActionLabel(item.mode));
    const textSpan = el(
      "div",
      "queue-text",
      item.prompt.length > 140 ? item.prompt.slice(0, 140) + "\u2026" : item.prompt
    );
    const cancelBtn = el("button", "queue-cancel", "Cancel");
    cancelBtn.title = "Cancel queued message";
    cancelBtn.addEventListener("click", () => {
      vscode.postMessage({ type: "cancelQueued", queueId: item.id });
    });
    content.appendChild(verb);
    content.appendChild(textSpan);
    row.appendChild(content);
    row.appendChild(cancelBtn);
    queueEl.appendChild(row);
  }
}

// ── Safe DOM helpers ──

/** Create an element with optional className and textContent. */
function el(tag, className, text) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = String(text);
  return e;
}

/** Create a text node. */
function txt(text) {
  return document.createTextNode(String(text || ""));
}

// ── Thinking indicator ──

function showThinking() {
  if (thinkingEl) return; // already showing
  if (!conversationEl) return;

  // Remove empty state
  const empty = conversationEl.querySelector(".empty-state");
  if (empty) empty.remove();

  thinkingEl = el("div", "thinking-indicator");
  const dot = el("span", "thinking-dot", "");

  // Start at a random verb, matching TUI behavior
  let verbIdx = Math.floor(Math.random() * thinkingVerbs.length);
  let dotCount = 1;
  let tickCount = 0;
  let pissingLockTicks = 0; // Tracks how many ticks "Pissing" should stay locked

  const label = el("span", "thinking-label", thinkingVerbs[verbIdx] + ".");
  thinkingEl.appendChild(dot);
  thinkingEl.appendChild(label);
  conversationEl.appendChild(thinkingEl);

  // Tick every 0.4s: cycle dots 1→2→3, change verb every 4 ticks (1.6s)
  // Matches TUI's ThinkingIndicator._rotate() algorithm exactly, including
  // transition rules and the "Pissing" lock.
  thinkingInterval = setInterval(() => {
    tickCount++;
    dotCount = (dotCount % 3) + 1;

    // Handle "Pissing" lock — if locked, decrement and skip verb change
    if (pissingLockTicks > 0) {
      pissingLockTicks--;
      label.textContent = thinkingVerbs[verbIdx] + ".".repeat(dotCount);
      return;
    }

    if (tickCount % 4 === 0) {
      const currentVerb = thinkingVerbs[verbIdx];
      let nextVerb = null;

      // Apply transition rules (20% chance each) — matches TUI _rotate()
      if (currentVerb === "Shitting" && Math.random() < 0.2) {
        nextVerb = "Wiping";
      } else if (currentVerb === "Sharting" && Math.random() < 0.2) {
        nextVerb = "Wiping";
      } else if (currentVerb === "Sleeping" && Math.random() < 0.2) {
        nextVerb = "Waking up";
      } else if (currentVerb === "Drinking") {
        const roll = Math.random();
        if (roll < 0.2) {
          nextVerb = "Blacking out";
        } else if (roll < 0.4) { // Another 20% chance for Pissing
          nextVerb = "Pissing";
        }
      }

      // If a transition was triggered, use it
      if (nextVerb !== null && thinkingVerbs.includes(nextVerb)) {
        verbIdx = thinkingVerbs.indexOf(nextVerb);
      } else {
        // Otherwise, random selection with 20% chance for Pissing lock
        verbIdx = Math.floor(Math.random() * thinkingVerbs.length);
        if (thinkingVerbs[verbIdx] === "Pissing" && Math.random() < 0.2) {
          // Lock "Pissing" for 1 minute = 150 ticks (60s / 0.4s per tick)
          pissingLockTicks = 150;
        }
      }
    }
    label.textContent = thinkingVerbs[verbIdx] + ".".repeat(dotCount);
  }, 400);

  if (autoScroll) scrollToBottom();
}

function removeThinking() {
  if (thinkingEl) {
    thinkingEl.remove();
    thinkingEl = null;
  }
  if (thinkingInterval) {
    clearInterval(thinkingInterval);
    thinkingInterval = null;
  }
}

function getAgentHeaderCopy(agent, sessionName, sessionSummary) {
  const role = String(agent?.role || "").toLowerCase();
  if (role === "orchestrator") {
    return {
      title: sessionName || "Task Orchestrator",
      description: sessionSummary || "",
    };
  }

  return {
    title: agent?.name || "Agent",
    description: agent?.promptPreview || "",
  };
}

// ── Rendering ──

function renderFull(agent, messages, sessionName, currentModel, sessionSummary) {
  if (!agent || !headerEl || !conversationEl) return;
  const previousScrollTop = conversationEl.scrollTop;
  const shouldAutoScroll = autoScroll;
  const nextAgentId = typeof agent.id === "string" ? agent.id : null;
  if (nextAgentId) activeTranscriptAgentId = nextAgentId;

  // Preserve inline question cards (and any in-progress typed answers)
  // across full conversation refreshes.
  const preservedQuestionCards = Array.from(
    conversationEl.querySelectorAll(".question-card")
  );
  preservedQuestionCards.forEach((card) => card.remove());

  // Clear and rebuild header
  headerEl.textContent = "";
  const infoDiv = el("div", "agent-info");

  const headerCopy = getAgentHeaderCopy(agent, sessionName, sessionSummary);
  const nameSpan = el("span", "agent-name", headerCopy.title);
  const badgeSpan = el("span", `badge badge-${agent.state}`, agent.state);
  const effectiveModel =
    (typeof currentModel === "string" && currentModel.length > 0)
      ? currentModel
      : (agent.model || "");
  const metaText = effectiveModel ? `${agent.role} \u00B7 ${effectiveModel}` : `${agent.role}`;
  const metaSpan = el("span", "agent-meta", metaText);
  metaSpan.setAttribute("data-role", String(agent.role || "orchestrator"));
  const sessionSpan = el("span", "agent-session", sessionName);

  infoDiv.appendChild(nameSpan);
  infoDiv.appendChild(badgeSpan);
  if (effectiveModel || agent.role) {
    infoDiv.appendChild(metaSpan);
  }
  infoDiv.appendChild(sessionSpan);
  headerEl.appendChild(infoDiv);

  if (headerCopy.description) {
    const promptDiv = el("div", "agent-prompt", headerCopy.description);
    headerEl.appendChild(promptDiv);
  }
  renderPlanLink();

  // Clear and rebuild messages
  conversationEl.textContent = "";
  activeStreamEl = null;
  removeThinking();

  if (!messages || messages.length === 0) {
    // If there's a pending user message, show it instead of the empty state
    // (handles the case between engine_started and agent_spawned during follow-up prompts)
    if (pendingUserMessage) {
      appendUserMessage(pendingUserMessage);
    } else {
      conversationEl.appendChild(
        el("div", "empty-state", "Send a message to start the orchestration.")
      );
    }
    appendModelSwitchMarkersAtIndex(0);
    restoreQuestionCards(preservedQuestionCards, 0);
    
    const isJumping = !!pendingToolCallJump || !!pendingSnapshotJump;
    if (shouldAutoScroll && !isJumping) {
      scrollToBottom();
    } else if (pendingScrollTop !== null && !isJumping) {
      conversationEl.scrollTop = pendingScrollTop;
      pendingScrollTop = null;
    } else if (!isJumping) {
      conversationEl.scrollTop = previousScrollTop;
    }
    if (_canAutofocus()) {
      _applyPendingFocusState();
    }
    schedulePersistUiState();
    tryApplyPendingToolCallJump();
    return;
  }

  for (let i = 0; i < messages.length; i++) {
    appendModelSwitchMarkersAtIndex(i);
    const m = messages[i];
    const msgEl = renderMessage(m, i);
    if (editResendIndex >= 0 && i === editResendIndex) {
      msgEl.classList.add("message-resend-origin");
    }
    // If in edit-resend mode, grey out messages below the selected prompt
    if (editResendIndex >= 0 && i > editResendIndex) {
      msgEl.classList.add("message-invalidated");
    }
    conversationEl.appendChild(msgEl);
  }
  appendModelSwitchMarkersAtIndex(messages.length, true);

  // If there's a pending user message that isn't yet in the store messages,
  // re-add it so it doesn't flash away between engine_started and agent_spawned
  if (pendingUserMessage) {
    const hasMsg = messages.some(
      (m) => m.role === "user" && m.content === pendingUserMessage
    );
    if (hasMsg) {
      pendingUserMessage = null;
    } else {
      appendUserMessage(pendingUserMessage);
    }
  }

  restoreQuestionCards(preservedQuestionCards, messages.length);
  
  const isJumping = !!pendingToolCallJump || !!pendingSnapshotJump;
  if (shouldAutoScroll && !isJumping) {
    scrollToBottom();
  } else if (pendingScrollTop !== null && !isJumping) {
    conversationEl.scrollTop = pendingScrollTop;
    pendingScrollTop = null;
  } else if (!isJumping) {
    conversationEl.scrollTop = previousScrollTop;
  }
  if (_canAutofocus()) {
    _applyPendingFocusState();
  }
  schedulePersistUiState();
  tryApplyPendingToolCallJump();
  tryApplyPendingSnapshotJump();
}

/**
 * Re-attach preserved question cards after renderFull rebuilds the transcript.
 * @param {Element[]} cards
 * @param {number} totalMessages
 */
function restoreQuestionCards(cards, totalMessages) {
  if (!conversationEl || !cards || cards.length === 0) return;
  const sorted = cards
    .slice()
    .sort((a, b) => {
      const ai = Number.parseInt(
        /** @type {HTMLElement} */ (a).dataset.atMessageIndex || `${totalMessages}`,
        10,
      );
      const bi = Number.parseInt(
        /** @type {HTMLElement} */ (b).dataset.atMessageIndex || `${totalMessages}`,
        10,
      );
      return ai - bi;
    });

  for (const card of sorted) {
    const node = /** @type {HTMLElement} */ (card);
    node.classList.add("question-card-restored");
    const anchor = Number.parseInt(
      node.dataset.atMessageIndex || `${totalMessages}`,
      10,
    );
    const safeAnchor = Number.isInteger(anchor)
      ? Math.max(0, Math.min(anchor, totalMessages))
      : totalMessages;
    const messageEls = conversationEl.querySelectorAll(".message");
    if (safeAnchor < messageEls.length) {
      conversationEl.insertBefore(node, messageEls[safeAnchor]);
    } else {
      conversationEl.appendChild(node);
    }
  }
}

function renderPlanLink() {
  if (!headerEl || !currentPlanFilePath) return;
  const existing = headerEl.querySelector(".plan-link-row");
  if (existing) existing.remove();
  const row = el("div", "plan-link-row");
  const btn = el("button", "plan-link-btn", "Open generated plan");
  btn.addEventListener("click", () => {
    vscode.postMessage({ type: "openFile", filePath: currentPlanFilePath });
  });
  row.appendChild(btn);
  headerEl.appendChild(row);
}

function renderMessage(msg, msgIndex) {
  const wrapper = el("div", `message message-${msg.role}`);
  wrapper.dataset.messageIndex = String(msgIndex);

  if (msg.role === "tool" && msg.toolCalls && msg.toolCalls.length > 0) {
    const tc = msg.toolCalls[0];
    renderToolCallInto(wrapper, tc);
    wrapper.classList.add("collapsible");
    const toolId = tc.id || "";
    const normalizedToolName = _normalizeToolName(tc.name || "");
    const autoExpand = normalizedToolName === "ask_user";
    if (toolId && autoExpand) {
      expandedToolCallIds.add(toolId);
    }
    if ((toolId && expandedToolCallIds.has(toolId)) || (!toolId && autoExpand)) {
      wrapper.classList.add("expanded");
    }
    wrapper.addEventListener("click", (e) => {
      if (/** @type {HTMLElement} */ (e.target).closest(".tool-details")) return;
      wrapper.classList.toggle("expanded");
      if (toolId) {
        if (wrapper.classList.contains("expanded")) {
          expandedToolCallIds.add(toolId);
        } else {
          expandedToolCallIds.delete(toolId);
        }
      }
    });
  } else if (msg.role === "assistant") {
    const header = el("div", "message-header");
    header.appendChild(el("span", "timestamp", formatTime(msg.timestamp)));
    header.appendChild(el("span", "role-label role-assistant", "Assistant"));
    wrapper.appendChild(header);

    const body = el("div", "message-body");
    body.appendChild(renderMarkdown(msg.content));
    wrapper.appendChild(body);
  } else if (msg.role === "user") {
    wrapper.classList.add("message-user-clickable");
    wrapper.title = "Click to edit and resend";
    wrapper.dataset.msgIndex = String(msgIndex);
    const snapId = msg.snapshotId || msg.snapshot_id;
    if (snapId) {
      wrapper.dataset.snapshotId = snapId;
      wrapper.id = `snap-${snapId}`;
    }
    wrapper.addEventListener("click", () => {
      preloadPromptForEdit(msg.content || "", msgIndex);
    });

    const header = el("div", "message-header");
    header.appendChild(el("span", "timestamp", formatTime(msg.timestamp)));
    header.appendChild(el("span", "role-label role-user", "You"));
    wrapper.appendChild(header);

    const body = el("div", "message-body");
    const userContent = msg.content || "";
    body.appendChild(renderMarkdown(userContent));
    wrapper.appendChild(body);
  } else {
    // system
    const systemContent = (msg.content || "").trim();
    if (systemContent === "Run stopped by user.") {
      wrapper.classList.add("run-stopped-indicator");
      const line = el("div", "run-stopped-line");
      line.appendChild(el("span", "run-stopped-dash", "─────"));
      line.appendChild(el("span", "run-stopped-icon", "⏹"));
      line.appendChild(el("span", "run-stopped-label", " Run stopped "));
      line.appendChild(el("span", "run-stopped-dash", "─────"));
      wrapper.appendChild(line);
    } else {
      const body = el("div", "message-body system-text");
      body.textContent = msg.content || "";
      wrapper.appendChild(body);
    }
  }

  return wrapper;
}

/** Append a user message bubble (called immediately when user sends prompt). */
function appendUserMessage(content) {
  if (!conversationEl) return;

  // Remove the "no messages" placeholder
  const empty = conversationEl.querySelector(".empty-state");
  if (empty) empty.remove();

  // Finalize any active stream
  finalizeStream();
  removeThinking();

  const wrapper = el("div", "message message-user");
  wrapper.classList.add("message-user-clickable");
  wrapper.title = "Click to edit and resend";
  wrapper.addEventListener("click", () => {
    preloadPromptForEdit(content || "");
  });

  const header = el("div", "message-header");
  header.appendChild(el("span", "timestamp", formatTime(new Date().toISOString())));
  header.appendChild(el("span", "role-label role-user", "You"));
  wrapper.appendChild(header);

  const body = el("div", "message-body");
  body.appendChild(renderMarkdown(content));
  wrapper.appendChild(body);

  conversationEl.appendChild(wrapper);
  if (autoScroll) scrollToBottom();
}

/** Append an error message. */
function appendErrorMessage(content) {
  if (!conversationEl) return;
  const wrapper = el("div", "message message-error");
  const body = el("div", "message-body error-text");
  body.textContent = content;
  wrapper.appendChild(body);
  conversationEl.appendChild(wrapper);
  if (autoScroll) scrollToBottom();
}

/** Append a non-error system message bubble. */
function appendSystemMessage(content) {
  if (!conversationEl) return;
  const wrapper = el("div", "message message-system");
  const body = el("div", "message-body system-text");
  body.textContent = content;
  wrapper.appendChild(body);
  conversationEl.appendChild(wrapper);
  if (autoScroll) scrollToBottom();
}

/**
 * Append a model switch indicator in the conversation.
 * Shows a visual divider indicating the model was changed,
 * mirroring the TUI's ModelSwitchIndicator.
 */
function appendModelSwitchIndicator(oldModel, newModel) {
  if (!conversationEl) return;

  // Finalize any active stream first
  finalizeStream();

  const wrapper = el("div", "model-switch-indicator");

  const line = el("div", "model-switch-line");
  const dashLeft = el("span", "model-switch-dash", "─────");
  const icon = el("span", "model-switch-icon", "🔄");
  const label = el("span", "model-switch-label", " Model switched ");
  const dashMid = el("span", "model-switch-dash", "── ");
  const oldName = el("span", "model-switch-old", oldModel || "unknown");
  const arrow = el("span", "model-switch-arrow", " → ");
  const newName = el("span", "model-switch-new", newModel || "unknown");
  const dashRight = el("span", "model-switch-dash", " ─────");

  line.appendChild(dashLeft);
  line.appendChild(icon);
  line.appendChild(label);
  line.appendChild(dashMid);
  line.appendChild(oldName);
  line.appendChild(arrow);
  line.appendChild(newName);
  line.appendChild(dashRight);

  wrapper.appendChild(line);
  conversationEl.appendChild(wrapper);
  if (autoScroll) scrollToBottom();
}

/** Return count of rendered transcript message elements. */
function getRenderedMessageCount() {
  if (!conversationEl) return 0;
  return conversationEl.querySelectorAll(".message").length;
}

/** Append all model switch markers that belong at the given message index. */
function appendModelSwitchMarkersAtIndex(messageIndex, includeBeyond = false) {
  if (!conversationEl || modelSwitchMarkers.length === 0) return;
  const markers = modelSwitchMarkers
    .filter((m) =>
      m.atMessageIndex === messageIndex ||
      (includeBeyond && m.atMessageIndex >= messageIndex)
    )
    .sort((a, b) => a.id.localeCompare(b.id));
  for (const marker of markers) {
    appendModelSwitchIndicator(marker.oldModel, marker.newModel);
  }
}

// ── Question Card (inline in chat) ──

/**
 * Append an inline question card to the conversation.
 * @param {{requestId: string, agentName: string, question: string, options: Array<{label: string, description?: string}>}} data
 */
function appendQuestionCard(data) {
  if (!conversationEl) return;
  const existing = conversationEl.querySelector(
    `.question-card[data-request-id="${data.requestId}"]`
  );
  if (existing) return;

  // Remove empty state
  const empty = conversationEl.querySelector(".empty-state");
  if (empty) empty.remove();

  // Finalize any active stream
  finalizeStream();
  removeThinking();

  const card = el("div", "question-card");
  card.dataset.requestId = data.requestId;
  // Anchor the card to the transcript position at creation time so
  // periodic full updates don't make it drift as chat grows.
  card.dataset.atMessageIndex = String(getRenderedMessageCount());

  const header = el("div", "question-card-header");
  header.appendChild(el("span", "question-card-icon", "\u2753"));
  header.appendChild(el("span", "question-card-agent", data.agentName + " asks:"));
  card.appendChild(header);

  const body = el("div", "question-card-body");
  body.textContent = data.question;
  card.appendChild(body);

  if (data.options && data.options.length > 0) {
    const optionsContainer = el("div", "question-card-options");
    for (const opt of data.options) {
      const btn = el("button", "question-option-btn");
      const labelSpan = el("span", "question-option-label", opt.label);
      btn.appendChild(labelSpan);
      if (opt.description && opt.description !== opt.label) {
        const descSpan = el("span", "question-option-desc", opt.description);
        btn.appendChild(descSpan);
      }
      btn.addEventListener("click", () => {
        // Send description text back (per requirements)
        const answer = opt.description || opt.label;
        vscode.postMessage({
          type: "answerQuestion",
          requestId: data.requestId,
          answer,
        });
        markQuestionAnswered(card, answer);
      });
      optionsContainer.appendChild(btn);
    }

    // Custom answer option
    const customBtn = el("button", "question-option-btn question-option-custom");
    customBtn.appendChild(el("span", "question-option-label", "\u270f\ufe0f Type custom answer..."));
    customBtn.addEventListener("click", () => {
      showCustomAnswerInput(card, data.requestId);
    });
    optionsContainer.appendChild(customBtn);

    card.appendChild(optionsContainer);
  } else {
    // Free-form text input
    showCustomAnswerInput(card, data.requestId);
  }

  conversationEl.appendChild(card);

  // Keep question cards anchored during periodic fullUpdate polling.
  autoScroll = true;

  // Do not force scroll during reload-state restoration.
  if (pendingScrollTop === null && (!pendingFocusState || pendingFocusState.requestId !== data.requestId)) {
    // Always scroll to the question card so it's visible, regardless of autoScroll.
    // Use scrollIntoView to guarantee the card is on-screen.
    card.scrollIntoView({ behavior: "smooth", block: "end" });
  }

  // Focus the first interactive element (option button or text input)
  // so the user can immediately respond.
  requestAnimationFrame(() => {
    if (_canAutofocus() && !_applyPendingFocusState()) {
      const firstInput = card.querySelector(".question-card-input");
      const firstBtn = card.querySelector(".question-option-btn");
      if (firstInput) {
        /** @type {HTMLTextAreaElement} */ (firstInput).focus();
      } else if (firstBtn) {
        /** @type {HTMLButtonElement} */ (firstBtn).focus();
      }
    }
    schedulePersistUiState();
  });
}

/**
 * Show a text input inside the question card for custom answers.
 * @param {HTMLElement} card
 * @param {string} requestId
 */
function showCustomAnswerInput(card, requestId) {
  // Remove options if present
  const opts = card.querySelector(".question-card-options");
  if (opts) opts.remove();

  // Check if input already exists
  if (card.querySelector(".question-card-input-row")) return;

  const inputRow = el("div", "question-card-input-row");
  const input = document.createElement("textarea");
  input.className = "question-card-input";
  input.placeholder = "Type your answer...";
  input.rows = 1;
  const existingDraft = questionDrafts[requestId];
  if (typeof existingDraft === "string" && existingDraft.length > 0) {
    input.value = existingDraft;
  }

  const submitBtn = el("button", "question-card-submit", "Send");

  const doSubmit = () => {
    const answer = input.value.trim();
    if (!answer) return;
    vscode.postMessage({
      type: "answerQuestion",
      requestId,
      answer,
    });
    setQuestionDraft(requestId, "");
    markQuestionAnswered(card, answer);
  };

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      doSubmit();
    }
  });
  submitBtn.addEventListener("click", doSubmit);

  inputRow.appendChild(input);
  inputRow.appendChild(submitBtn);
  card.appendChild(inputRow);

  // Auto-resize
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 100) + "px";
    setQuestionDraft(requestId, input.value);
    pendingFocusState = _captureFocusStateFromElement(input);
    schedulePersistUiState();
  });

  if (_canAutofocus()) {
    input.focus();
    pendingFocusState = _captureFocusStateFromElement(input);
    schedulePersistUiState();
  }
  if (autoScroll) scrollToBottom();
}

/**
 * Mark a question card as answered (visual feedback).
 * @param {HTMLElement} card
 * @param {string} answer
 */
function markQuestionAnswered(card, answer) {
  card.classList.add("question-answered");
  card.classList.add("question-card-restored");

  if (card && card.dataset && card.dataset.requestId) {
    setQuestionDraft(card.dataset.requestId, "");
    if (
      _canAutofocus() &&
      pendingFocusState &&
      pendingFocusState.kind === "question-input" &&
      pendingFocusState.requestId === card.dataset.requestId
    ) {
      pendingFocusState = null;
      schedulePersistUiState();
    }
  }

  // Replace interactive content with the answer
  const opts = card.querySelector(".question-card-options");
  if (opts) opts.remove();
  const inputRow = card.querySelector(".question-card-input-row");
  if (inputRow) inputRow.remove();

  const answerDiv = el("div", "question-card-answer");
  answerDiv.appendChild(el("span", "question-answer-label", "Answered: "));
  answerDiv.appendChild(el("span", "question-answer-text", answer));
  card.appendChild(answerDiv);
}

/**
 * Dismiss a question card by requestId (e.g., when resolved from elsewhere).
 * @param {string} requestId
 */
function dismissQuestionCard(requestId) {
  if (!conversationEl) return;
  const card = conversationEl.querySelector(`.question-card[data-request-id="${requestId}"]`);
  if (card && !card.classList.contains("question-answered")) {
    markQuestionAnswered(card, "(resolved)");
  }
}

// ── Tool Call Formatters ──
// Registry-based formatter system mirroring prsm/shared/formatters/tool_call.py.
// Each tool type has a dedicated formatter that produces a structured IR:
//   { icon, label, summary, sections: [{ kind, title, content }] }
// Adding a new tool format requires only: registerToolFormatter("Name", fn).

function parseToolArgs(argsStr) {
  if (!argsStr) return {};
  try {
    const parsed = JSON.parse(argsStr);
    return typeof parsed === "object" && parsed !== null ? parsed : { _raw: argsStr };
  } catch {
    return { _raw: argsStr };
  }
}

const TOOL_FORMATTERS = {};

function registerToolFormatter(name, fn) {
  TOOL_FORMATTERS[name] = fn;
}

function _normalizeToolName(name) {
  // Strip MCP server prefix: "mcp__orchestrator__task_complete" → "task_complete"
  if (name.startsWith("mcp__") && (name.match(/__/g) || []).length >= 2) {
    name = name.split("__").slice(2).join("__");
  }
  const aliasMap = {
    read: "Read",
    read_file: "Read",
    file_read: "Read",
    write: "Write",
    write_file: "Write",
    file_write: "Write",
    edit: "Edit",
    edit_file: "Edit",
    file_edit: "Edit",
    bash: "Bash",
    run_bash: "Bash",
    run_shell_command: "Bash",
    glob: "Glob",
    list_directory: "Glob",
    grep: "Grep",
    search_files: "Grep",
  };
  return aliasMap[(name || "").toLowerCase()] || name;
}

function formatToolCallData(name, argsStr, result, success) {
  const args = parseToolArgs(argsStr || "");
  // Try exact match first, then stripped name
  let fn = TOOL_FORMATTERS[name];
  if (!fn) {
    const bare = _normalizeToolName(name);
    fn = TOOL_FORMATTERS[bare] || defaultToolFormatter;
  }
  return fn(name, args, result, success);
}

function _parseResultPayload(result) {
  if (!result) return null;

  if (typeof result === "object") {
    return result;
  }

  if (typeof result !== "string") {
    return null;
  }

  const trimmed = String(result).trim();
  if (!trimmed) return null;

  try {
    const parsed = JSON.parse(trimmed);
    if (parsed && typeof parsed === "object") return parsed;
  } catch {
    return null;
  }

  return null;
}

function _extractTaskCompleteSummary(args, result) {
  const out = {
    summary: "",
    artifacts: {},
    includeMarkdown: false,
  };

  const rawSummary = args.summary;
  if (typeof rawSummary === "string" && rawSummary.trim()) {
    out.summary = rawSummary.trim();
    out.includeMarkdown = true;
  }

  const rawArtifacts = args.artifacts;
  if (rawArtifacts && typeof rawArtifacts === "object") {
    out.artifacts = rawArtifacts;
  }

  const payload = _parseResultPayload(result);
  if (payload && typeof payload === "object") {
    const candidateSummary = payload.summary;
    const candidateArtifacts = payload.artifacts;
    const nested = (payload.payload && typeof payload.payload === "object") ? payload.payload : null;

    const resolvedSummary = candidateSummary || (nested && nested.summary);
    const resolvedArtifacts = candidateArtifacts || (nested && nested.artifacts);

    if (!out.summary && typeof resolvedSummary === "string" && resolvedSummary.trim()) {
      out.summary = resolvedSummary.trim();
      out.includeMarkdown = true;
    }

    const payloadArtifacts = resolvedArtifacts;
    if (!rawArtifacts && payloadArtifacts && typeof payloadArtifacts === "object") {
      out.artifacts = payloadArtifacts;
    }
  }

  // Fallback to response text if still missing a summary and it is not boilerplate.
  if (!out.summary && result && typeof result === "string") {
    const textResult = String(result).trim();
    if (
      textResult &&
      textResult !== "Task marked complete. Session will end." &&
      textResult !== "Task completed."
    ) {
      out.summary = textResult;
      out.includeMarkdown = true;
    }
  }

  return out;
}

// Helpers
function _basename(path) {
  if (!path) return "";
  const parts = path.replace(/\\/g, "/").replace(/\/+$/, "").split("/");
  return parts.length >= 2 ? parts.slice(-2).join("/") : parts[parts.length - 1];
}

function _trunc(text, length) {
  length = length || 60;
  if (!text) return "";
  return text.length <= length ? text : text.slice(0, length - 3) + "...";
}

function _extractTextResult(result) {
  if (!result) return "";
  const text = String(result);
  try {
    const parsed = JSON.parse(text);
    if (parsed && typeof parsed === "object") {
      if (Array.isArray(parsed.content)) {
        const chunks = parsed.content
          .filter((item) => item && typeof item === "object" && item.type === "text" && typeof item.text === "string")
          .map((item) => item.text);
        if (chunks.length) return chunks.join("\n");
      }
      for (const key of ["text", "stdout", "output", "result"]) {
        if (typeof parsed[key] === "string") return parsed[key];
      }
    }
  } catch {
    const matches = [...text.matchAll(/"text"\s*:\s*"((?:\\.|[^"\\])*)"/g)];
    if (matches.length) {
      const chunks = matches.map((m) => {
        const raw = m[1];
        try { return JSON.parse(`"${raw}"`); } catch { return raw.replace(/\\n/g, "\n").replace(/\\t/g, "\t"); }
      });
      return chunks.join("\n");
    }
  }
  return text;
}

function _splitShellCommand(command) {
  if (!command) return [];
  const re = /[^\s"']+|"([^"]*)"|'([^']*)'/g;
  const out = [];
  let m;
  while ((m = re.exec(command)) !== null) {
    out.push(m[1] || m[2] || m[0]);
  }
  return out;
}

function _lastNonFlag(parts) {
  for (let i = parts.length - 1; i >= 0; i -= 1) {
    if (!parts[i].startsWith("-")) return parts[i];
  }
  return "";
}

function _parseSedSubstitution(expr) {
  const raw = String(expr || "").trim().replace(/^['"]|['"]$/g, "");
  if (raw.length < 4 || raw[0] !== "s") return null;
  const delim = raw[1];
  if (!delim || /[a-z0-9]/i.test(delim)) return null;
  let i = 2;
  let escaped = false;
  let inNew = false;
  const oldChars = [];
  const newChars = [];
  while (i < raw.length) {
    const ch = raw[i];
    i += 1;
    if (escaped) {
      (inNew ? newChars : oldChars).push(ch);
      escaped = false;
      continue;
    }
    if (ch === "\\") {
      escaped = true;
      continue;
    }
    if (ch === delim) {
      if (!inNew) {
        inNew = true;
        continue;
      }
      break;
    }
    (inNew ? newChars : oldChars).push(ch);
  }
  if (!inNew) return null;
  return { oldString: oldChars.join(""), newString: newChars.join("") };
}

function _splitShellSegments(command) {
  if (!command) return [];
  const segments = [];
  let current = "";
  let inSingle = false;
  let inDouble = false;
  let escaped = false;
  for (let i = 0; i < command.length; i += 1) {
    const ch = command[i];
    const next = i + 1 < command.length ? command[i + 1] : "";
    if (escaped) { current += ch; escaped = false; continue; }
    if (ch === "\\") { escaped = true; current += ch; continue; }
    if (ch === "'" && !inDouble) { inSingle = !inSingle; current += ch; continue; }
    if (ch === '"' && !inSingle) { inDouble = !inDouble; current += ch; continue; }
    if (!inSingle && !inDouble) {
      if (ch === ";" || (ch === "|" && next === "|") || (ch === "&" && next === "&")) {
        const seg = current.trim();
        if (seg) segments.push(seg);
        current = "";
        if ((ch === "|" || ch === "&") && next === ch) i += 1;
        continue;
      }
      if (ch === "|") {
        const seg = current.trim();
        if (seg) segments.push(seg);
        current = "";
        continue;
      }
    }
    current += ch;
  }
  if (current.trim()) segments.push(current.trim());
  return segments;
}

function _extractRedirectTarget(command) {
  const m = String(command || "").match(/(?:^|[\s])>>?\s*([^\s|;&]+)/);
  return m ? m[1].replace(/^['"]|['"]$/g, "") : "";
}

function _extractUrl(parts) {
  for (const token of parts || []) {
    if (token.startsWith("http://") || token.startsWith("https://")) return token;
  }
  return "";
}

function _describeMappedStep(tool, args) {
  if (tool === "Read") return `Read ${_basename(args.file_path || "") || "file"}`;
  if (tool === "Write") return `Write ${_basename(args.file_path || "") || "file"}`;
  if (tool === "Edit") return `Edit ${_basename(args.file_path || "") || "file"}`;
  if (tool === "Grep") return `Search "${_trunc(args.pattern || "", 24)}"`;
  if (tool === "Glob") return `List ${args.pattern || "files"}`;
  if (tool === "GitInspect") return `Git ${args.subcommand || "inspect"}`;
  if (tool === "TestRun") return `Run ${_trunc(args.command || "tests", 36)}`;
  if (tool === "BuildRun") return `Build ${_trunc(args.command || "", 36)}`;
  if (tool === "ProcessControl") return `Process ${args.action || "command"}`;
  if (tool === "HttpCall") return `${args.method || "GET"} ${_trunc(args.url || "", 30)}`.trim();
  if (tool === "JsonExtract") return `JSON ${_trunc(args.query || "", 32)}`.trim();
  if (tool === "Stats") return `Stats ${_trunc(args.command || "", 28)}`.trim();
  return tool;
}

function _interpretSingleShellCommand(command) {
  const parts = _splitShellCommand(command || "");
  if (parts.length === 0) return null;
  const prog = parts[0];

  if (prog === "rg" || prog === "ripgrep") {
    if (parts.includes("--files")) {
      const root = _lastNonFlag(parts.slice(1)) || ".";
      return { tool: "Glob", args: { pattern: "**/*", path: root } };
    }
    let pattern = "";
    let path = "";
    for (const token of parts.slice(1)) {
      if (token.startsWith("-")) continue;
      if (!pattern) pattern = token;
      else { path = token; break; }
    }
    return { tool: "Grep", args: { pattern, path } };
  }

  if (prog === "grep") {
    let pattern = "";
    let path = "";
    for (const token of parts.slice(1)) {
      if (token.startsWith("-")) continue;
      if (!pattern) pattern = token;
      else { path = token; break; }
    }
    return { tool: "Grep", args: { pattern, path } };
  }

  if (prog === "ls") {
    const path = _lastNonFlag(parts.slice(1)) || ".";
    return { tool: "Glob", args: { pattern: "*", path } };
  }

  if (prog === "cat") {
    const filePath = _lastNonFlag(parts.slice(1));
    if (filePath) return { tool: "Read", args: { file_path: filePath } };
  }

  if (prog === "head" || prog === "tail") {
    let limit = 10;
    let filePath = "";
    for (let i = 1; i < parts.length; i += 1) {
      const token = parts[i];
      if (token === "-n" && i + 1 < parts.length) {
        const parsed = parseInt(parts[i + 1], 10);
        limit = Number.isNaN(parsed) ? 10 : Math.max(1, parsed);
        i += 1;
        continue;
      }
      if (token.startsWith("-n") && token.length > 2) {
        const parsed = parseInt(token.slice(2), 10);
        limit = Number.isNaN(parsed) ? 10 : Math.max(1, parsed);
        continue;
      }
      if (token.startsWith("-")) continue;
      filePath = token;
    }
    if (filePath) return { tool: "Read", args: { file_path: filePath, limit } };
  }

  if (prog === "find") {
    let path = ".";
    let pattern = "**/*";
    let rest = parts.slice(1);
    if (rest.length > 0 && !rest[0].startsWith("-")) {
      path = rest[0];
      rest = rest.slice(1);
    }
    for (let i = 0; i < rest.length; i += 1) {
      const token = rest[i];
      if ((token === "-name" || token === "-iname") && i + 1 < rest.length) {
        const candidate = rest[i + 1].replace(/^['"]|['"]$/g, "");
        if (candidate) pattern = candidate;
        break;
      }
    }
    return { tool: "Glob", args: { pattern, path } };
  }

  if (prog === "sed") {
    const isInPlace = parts.slice(1).some((t) => t === "-i" || t.startsWith("-i"));
    if (isInPlace) {
      let expr = "";
      let filePath = "";
      for (const token of parts.slice(1)) {
        if (token.startsWith("-")) continue;
        if (_parseSedSubstitution(token)) { expr = token; continue; }
        filePath = token;
      }
      const parsed = _parseSedSubstitution(expr);
      if (parsed && filePath) {
        return {
          tool: "Edit",
          args: { file_path: filePath, old_string: parsed.oldString, new_string: parsed.newString },
        };
      }
    }
    if (parts.includes("-n")) {
      let expr = "";
      let filePath = "";
      for (const token of parts.slice(1)) {
        if (token.startsWith("-")) continue;
        if (!expr) expr = token;
        else { filePath = token; break; }
      }
      const m = expr.replace(/^['"]|['"]$/g, "").match(/^(\d+)(?:,(\d+))?p$/);
      if (m && filePath) {
        const start = parseInt(m[1], 10);
        const end = parseInt(m[2] || m[1], 10);
        return { tool: "Read", args: { file_path: filePath, offset: start, limit: Math.max(1, end - start + 1) } };
      }
    }
  }

  if (prog === "tee") {
    const filePath = _lastNonFlag(parts.slice(1));
    if (filePath) return { tool: "Write", args: { file_path: filePath } };
  }
  const redirectTarget = _extractRedirectTarget(command);
  if (redirectTarget && (prog === "cat" || prog === "echo" || prog === "printf")) {
    return { tool: "Write", args: { file_path: redirectTarget } };
  }

  if (prog === "git" && parts.length > 1) {
    const sub = parts[1];
    if (["status", "diff", "log", "show", "blame", "rev-parse"].includes(sub)) {
      return { tool: "GitInspect", args: { subcommand: sub, target: _lastNonFlag(parts.slice(2)), command } };
    }
  }

  if (["pytest", "tox", "nosetests", "go", "cargo", "npm", "pnpm", "yarn", "bun", "make"].includes(prog)) {
    const lower = parts.join(" ").toLowerCase();
    if (` ${lower} `.includes(" test ") || ["pytest", "tox", "nosetests"].includes(prog)) {
      return { tool: "TestRun", args: { command } };
    }
    if (lower.includes(" build") || lower.includes(" typecheck") || lower.includes(" compile") || lower.includes(" lint")) {
      return { tool: "BuildRun", args: { command } };
    }
  }

  if (["pgrep", "pkill", "kill", "killall", "ps", "lsof"].includes(prog)) {
    return { tool: "ProcessControl", args: { action: prog, command } };
  }

  if (["curl", "wget", "http", "xh"].includes(prog)) {
    let method = "GET";
    for (let i = 0; i < parts.length; i += 1) {
      if ((parts[i] === "-X" || parts[i] === "--request") && i + 1 < parts.length) {
        method = String(parts[i + 1]).toUpperCase();
        break;
      }
    }
    let url = _extractUrl(parts.slice(1));
    if (!url) {
      const candidate = _lastNonFlag(parts.slice(1));
      if (candidate.startsWith("http://") || candidate.startsWith("https://")) url = candidate;
    }
    return { tool: "HttpCall", args: { method, url, command } };
  }

  if (prog === "jq") {
    let query = "";
    let filePath = "";
    for (const token of parts.slice(1)) {
      if (token.startsWith("-")) continue;
      if (!query) query = token;
      else { filePath = token; break; }
    }
    return { tool: "JsonExtract", args: { query, file_path: filePath, command } };
  }

  if (["wc", "du", "sort", "uniq"].includes(prog)) {
    return { tool: "Stats", args: { command } };
  }

  return null;
}

function _interpretBashAsTool(command) {
  const segments = _splitShellSegments(command || "");
  if (segments.length > 1) {
    const steps = segments.map((seg) => {
      const interpreted = _interpretSingleShellCommand(seg);
      if (interpreted) {
        return {
          command: seg,
          tool: interpreted.tool,
          summary: _describeMappedStep(interpreted.tool, interpreted.args || {}),
        };
      }
      return { command: seg, tool: "Bash", summary: _trunc(seg, 48) };
    });
    return { tool: "ShellWorkflow", args: { command, steps } };
  }
  if (segments.length === 0) return null;
  return _interpretSingleShellCommand(segments[0]);
}

// ── Individual formatters ──

registerToolFormatter("Edit", (name, args, result, success) => {
  const filePath = args.file_path || "";
  const oldString = args.old_string || "";
  const newString = args.new_string || "";
  const sections = [];
  if (filePath) sections.push({ kind: "path", content: filePath });
  if (oldString || newString) {
    sections.push({
      kind: "diff",
      content: {
        old_lines: oldString ? oldString.split("\n") : [],
        new_lines: newString ? newString.split("\n") : [],
      },
    });
  }
  if (result) sections.push({ kind: "plain", title: "Output", content: result });
  return { icon: "\u270f\ufe0f", label: "Edit", summary: _basename(filePath), file_path: filePath, sections };
});

registerToolFormatter("Bash", (name, args, result, success) => {
  const command = args.command || args._raw || "";
  const description = args.description || "";
  const interpreted = _interpretBashAsTool(command);
  if (interpreted) {
    const fn = TOOL_FORMATTERS[interpreted.tool];
    if (fn) return fn(interpreted.tool, interpreted.args, result, success);
  }
  const sections = [];
  sections.push({ kind: "code", title: "Command", content: { language: "bash", text: command } });
  if (description) sections.push({ kind: "plain", title: "Description", content: description });
  if (result) sections.push({ kind: "plain", title: "Output", content: result });
  return { icon: "$", label: "Bash", summary: _trunc(command), sections };
});

registerToolFormatter("Read", (name, args, result, success) => {
  const filePath = args.file_path || "";
  const offset = args.offset;
  const limit = args.limit;
  let lineInfo = "";
  if (offset || limit) {
    const parts = [];
    if (offset) parts.push("L" + offset);
    if (limit) parts.push("+" + limit);
    lineInfo = " (" + parts.join(":") + ")";
  }
  const sections = [];
  if (filePath) sections.push({ kind: "path", content: filePath });
  const displayText = _extractTextResult(result);
  if (displayText) sections.push({ kind: "code", title: "Content", content: { language: "text", text: displayText } });
  return { icon: "\ud83d\udcc4", label: "Read", summary: _basename(filePath) + lineInfo, file_path: filePath, sections };
});

registerToolFormatter("Write", (name, args, result, success) => {
  const filePath = args.file_path || "";
  const sections = [];
  if (filePath) sections.push({ kind: "path", content: filePath });
  if (result) sections.push({ kind: "plain", title: "Output", content: result });
  return { icon: "\ud83d\udcdd", label: "Write", summary: _basename(filePath), file_path: filePath, sections };
});

registerToolFormatter("GitInspect", (name, args, result, success) => {
  const sub = args.subcommand || "inspect";
  const target = args.target || "";
  const command = args.command || "";
  const sections = [];
  if (target) sections.push({ kind: "path", content: target });
  if (command) sections.push({ kind: "code", title: "Command", content: { language: "bash", text: command } });
  if (result) sections.push({ kind: "plain", title: "Git Output", content: result });
  return { icon: "\ud83c\udf33", label: "Git", summary: _trunc(`${sub} ${_basename(target)}`.trim(), 60), file_path: target, sections };
});

registerToolFormatter("TestRun", (name, args, result, success) => {
  const command = args.command || "";
  const sections = [];
  if (command) sections.push({ kind: "code", title: "Test Command", content: { language: "bash", text: command } });
  if (result) sections.push({ kind: "plain", title: "Test Output", content: result });
  return { icon: "\u2697\ufe0f", label: "Test", summary: _trunc(command, 60), sections };
});

registerToolFormatter("BuildRun", (name, args, result, success) => {
  const command = args.command || "";
  const sections = [];
  if (command) sections.push({ kind: "code", title: "Build Command", content: { language: "bash", text: command } });
  if (result) sections.push({ kind: "plain", title: "Build Output", content: result });
  return { icon: "\ud83d\udee0\ufe0f", label: "Build", summary: _trunc(command, 60), sections };
});

registerToolFormatter("ProcessControl", (name, args, result, success) => {
  const action = args.action || "process";
  const command = args.command || "";
  const sections = [];
  if (command) sections.push({ kind: "code", title: "Process Command", content: { language: "bash", text: command } });
  if (result) sections.push({ kind: "plain", title: "Process Output", content: result });
  return { icon: "\u2699\ufe0f", label: "Process", summary: `${action}: ${_trunc(command, 44)}`, sections };
});

registerToolFormatter("HttpCall", (name, args, result, success) => {
  const method = args.method || "GET";
  const url = args.url || "";
  const command = args.command || "";
  const sections = [];
  const req = { method };
  if (url) req.url = url;
  sections.push({ kind: "kv", title: "Request", content: req });
  if (command) sections.push({ kind: "code", title: "Command", content: { language: "bash", text: command } });
  if (result) sections.push({ kind: "plain", title: "Response", content: result });
  return { icon: "\ud83c\udf10", label: "HTTP", summary: `${method} ${_trunc(url, 44)}`.trim(), sections };
});

registerToolFormatter("JsonExtract", (name, args, result, success) => {
  const query = args.query || "";
  const filePath = args.file_path || "";
  const command = args.command || "";
  const sections = [];
  const kv = {};
  if (query) kv.query = query;
  if (filePath) kv.file = filePath;
  if (Object.keys(kv).length) sections.push({ kind: "kv", title: "JSON Query", content: kv });
  if (command) sections.push({ kind: "code", title: "Command", content: { language: "bash", text: command } });
  if (result) sections.push({ kind: "plain", title: "JSON Output", content: result });
  return { icon: "\ud83e\uddee", label: "JSON", summary: _trunc(`${query} ${filePath}`.trim(), 60), file_path: filePath, sections };
});

registerToolFormatter("Stats", (name, args, result, success) => {
  const command = args.command || "";
  const sections = [];
  if (command) sections.push({ kind: "code", title: "Stats Command", content: { language: "bash", text: command } });
  if (result) sections.push({ kind: "plain", title: "Stats Output", content: result });
  return { icon: "\ud83d\udcca", label: "Stats", summary: _trunc(command, 60), sections };
});

registerToolFormatter("ShellWorkflow", (name, args, result, success) => {
  const command = args.command || "";
  const steps = Array.isArray(args.steps) ? args.steps : [];
  const sections = [];
  if (command) sections.push({ kind: "code", title: "Pipeline", content: { language: "bash", text: command } });
  const items = steps.slice(0, 12).map((s) => ({
    text: s.summary || s.tool || s.command || "",
    done: false,
  })).filter((x) => x.text);
  if (items.length) sections.push({ kind: "checklist", title: "Steps", content: items });
  if (result) sections.push({ kind: "plain", title: "Output", content: result });
  return { icon: "\ud83e\uddf0", label: "Shell Workflow", summary: `${steps.length} steps`, sections };
});

registerToolFormatter("Glob", (name, args, result, success) => {
  const pattern = args.pattern || "";
  const path = args.path || "";
  const sections = [];
  if (path) sections.push({ kind: "path", content: path });
  if (result) sections.push({ kind: "plain", title: "Matches", content: result });
  let summary = pattern;
  if (path) summary = pattern + " in " + _basename(path);
  return { icon: "\ud83d\udd0d", label: "Glob", summary, file_path: path, sections };
});

registerToolFormatter("Grep", (name, args, result, success) => {
  const pattern = args.pattern || "";
  const glob = args.glob || "";
  const path = args.path || "";
  const scope = glob || _basename(path) || "";
  let summary = '"' + _trunc(pattern, 30) + '"';
  if (scope) summary += " in " + scope;
  const sections = [];
  if (result) sections.push({ kind: "plain", title: "Results", content: result });
  return { icon: "\ud83d\udd0e", label: "Grep", summary, file_path: path, sections };
});

registerToolFormatter("Task", (name, args, result, success) => {
  const description = args.description || "";
  const subagentType = args.subagent_type || "";
  const prompt = args.prompt || "";
  const kv = {};
  if (subagentType) kv.type = subagentType;
  if (description) kv.description = description;
  if (prompt) kv.prompt = _trunc(prompt, 100);
  const sections = [];
  if (Object.keys(kv).length) sections.push({ kind: "kv", content: kv });
  if (result) sections.push({ kind: "plain", title: "Output", content: result });
  return { icon: "\ud83d\udd00", label: "Task", summary: _trunc(description || prompt, 50), sections };
});

registerToolFormatter("TodoWrite", (name, args, result, success) => {
  const todos = Array.isArray(args.todos) ? args.todos : [];
  const items = todos
    .filter((t) => typeof t === "object" && t !== null)
    .map((t) => ({
      text: t.content || t.text || String(t),
      done: t.status === "completed",
    }));
  const sections = [];
  if (items.length) sections.push({ kind: "checklist", content: items });
  if (result) sections.push({ kind: "plain", title: "Output", content: result });
  const doneCount = items.filter((i) => i.done).length;
  const summary = items.length ? `${doneCount}/${items.length} done` : "empty";
  return { icon: "\u2611\ufe0f", label: "TodoWrite", summary, sections };
});

registerToolFormatter("WebFetch", (name, args, result, success) => {
  const url = args.url || "";
  const prompt = args.prompt || "";
  let domain = "";
  try { domain = new URL(url).hostname; } catch { domain = _trunc(url, 40); }
  const kv = {};
  if (url) kv.url = url;
  if (prompt) kv.prompt = _trunc(prompt, 80);
  const sections = [];
  if (Object.keys(kv).length) sections.push({ kind: "kv", content: kv });
  if (result) sections.push({ kind: "plain", title: "Output", content: result });
  return { icon: "\ud83c\udf10", label: "WebFetch", summary: domain, sections };
});

registerToolFormatter("WebSearch", (name, args, result, success) => {
  const query = args.query || "";
  const sections = [];
  if (result) sections.push({ kind: "plain", title: "Results", content: result });
  return { icon: "\ud83d\udd0d", label: "WebSearch", summary: _trunc(query, 50), sections };
});

registerToolFormatter("NotebookEdit", (name, args, result, success) => {
  const notebookPath = args.notebook_path || "";
  const editMode = args.edit_mode || "replace";
  const sections = [];
  if (notebookPath) sections.push({ kind: "path", content: notebookPath });
  if (result) sections.push({ kind: "plain", title: "Output", content: result });
  return { icon: "\ud83d\udcd3", label: "NotebookEdit", summary: editMode + " in " + _basename(notebookPath), file_path: notebookPath, sections };
});

registerToolFormatter("Skill", (name, args, result, success) => {
  const skill = args.skill || "";
  const skillArgs = args.args || "";
  let summary = skill;
  if (skillArgs) summary += " " + _trunc(skillArgs, 40);
  const kv = { skill };
  if (skillArgs) kv.args = skillArgs;
  const sections = [{ kind: "kv", content: kv }];
  if (result) sections.push({ kind: "plain", title: "Output", content: result });
  return { icon: "\u26a1", label: "Skill", summary, sections };
});

// ── Orchestrator Result Parsers ──
// Helper functions for parsing structured results from orchestrator tool calls.

function _parseResultKV(result) {
  if (!result) return null;
  const lines = result.trim().split('\n');
  const kv = {};
  let count = 0;
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('---')) continue;
    const idx = trimmed.indexOf(': ');
    if (idx > 0 && idx < 40) {
      const key = trimmed.slice(0, idx).replace(/^[- ]+/, '');
      if (key && !key.startsWith('(')) { kv[key] = trimmed.slice(idx + 2); count++; }
    }
  }
  return count >= 2 ? kv : null;
}

function _extractAgentIdsFromText(text) {
  if (!text) return [];
  const ids = [];
  const seen = new Set();
  const regex = /(?:^|["'\s\-\[\(])(?:child_id|agent_id)\s*:\s*([A-Za-z0-9-]{12,})/gi;
  let match;
  while ((match = regex.exec(text)) !== null) {
    const id = match[1];
    if (!id || seen.has(id)) continue;
    seen.add(id);
    ids.push(id);
  }
  return ids;
}

function _parseSpawnResult(result) {
  if (!result) return [];
  const sections = [];
  let mainText = result, resultBody = '', errorBody = '';
  if (result.includes('--- Result ---')) {
    const [before, after] = result.split('--- Result ---', 2);
    mainText = before;
    if (after.includes('--- Error ---')) {
      const [rb, eb] = after.split('--- Error ---', 2);
      resultBody = rb.trim(); errorBody = eb.trim();
    } else { resultBody = after.trim(); }
  } else if (result.includes('--- Error ---')) {
    const [before, after] = result.split('--- Error ---', 2);
    mainText = before; errorBody = after.trim();
  }
  const kv = _parseResultKV(mainText);
  if (kv) { sections.push({ kind: 'kv', title: 'Details', content: kv }); }
  else { const fl = mainText.trim().split('\n')[0]; if (fl) sections.push({ kind: 'plain', title: 'Status', content: fl }); }
  if (resultBody) sections.push({ kind: 'plain', title: 'Child Summary', content: resultBody });
  if (errorBody) sections.push({ kind: 'plain', title: 'Error', content: errorBody });

  const extractedIds = _extractAgentIdsFromText(`${mainText}\n${resultBody}\n${errorBody}`);
  if (extractedIds.length) {
    sections.push({ kind: "agent_links", title: "Spawned Agent", content: extractedIds });
  }

  return sections;
}

function _parseWaitMessageResult(result) {
  if (!result) return [];
  if (result.startsWith('Message received:')) {
    const lines = result.split('\n');
    const kv = {};
    let payloadSections = [];
    let inPayload = false, payloadText = '';
    // Determine message type for smarter display
    let msgType = '';
    for (const line of lines.slice(1)) {
      const trimmed = line.trim();
      if (inPayload) { payloadText += trimmed + '\n'; continue; }
      if (trimmed.startsWith('payload: ')) {
        const raw = trimmed.slice('payload: '.length);
        try {
          const parsed = JSON.parse(raw);
          if (parsed && typeof parsed === 'object') {
            // TASK_RESULT payloads — show summary in a prominent result_block
            if (parsed.summary) {
              payloadSections.push({
                kind: 'result_block',
                title: 'Child Result',
                content: { text: String(parsed.summary), status: 'success' },
              });
              if (parsed.artifacts && Object.keys(parsed.artifacts).length > 0) {
                const artKv = {};
                for (const [k, v] of Object.entries(parsed.artifacts)) artKv[k] = _trunc(String(v), 200);
                payloadSections.push({ kind: 'kv', title: 'Artifacts', content: artKv });
              }
              // Show remaining keys
              const extraKv = {};
              for (const [k, v] of Object.entries(parsed)) {
                if (k !== 'summary' && k !== 'artifacts') extraKv[k] = _trunc(String(v), 200);
              }
              if (Object.keys(extraKv).length) payloadSections.push({ kind: 'kv', title: 'Details', content: extraKv });
            // progress_update payloads — visual progress bar
            } else if (msgType === 'progress_update' && parsed.status !== undefined) {
              payloadSections.push({
                kind: 'progress',
                content: { percent: parseInt(parsed.percent_complete) || 0, status: parsed.status || '' },
              });
            // QUESTION payloads — show question in result_block with "question" status
            } else if (parsed.question !== undefined) {
              payloadSections.push({
                kind: 'result_block',
                title: 'Child Question',
                content: { text: String(parsed.question), status: 'question' },
              });
              // Show extra context fields
              const qMeta = {};
              for (const [k, v] of Object.entries(parsed)) {
                if (k !== 'question') qMeta[k] = _trunc(String(v), 100);
              }
              if (Object.keys(qMeta).length) payloadSections.push({ kind: 'kv', title: 'Context', content: qMeta });
            } else {
              const kvPayload = {};
              for (const [k, v] of Object.entries(parsed)) kvPayload[k] = _trunc(String(v), 200);
              payloadSections.push({ kind: 'kv', title: 'Payload', content: kvPayload });
            }
          } else { payloadSections.push({ kind: 'plain', title: 'Payload', content: raw }); }
        } catch(e) { payloadSections.push({ kind: 'plain', title: 'Payload', content: raw }); }
        inPayload = true;
      } else if (trimmed.includes(': ')) {
        const idx = trimmed.indexOf(': ');
        const key = trimmed.slice(0, idx).trim();
        let val = trimmed.slice(idx + 2).trim();
        if (key === 'type') msgType = val.toLowerCase();
        // Shorten agent IDs for readability
        if ((key === 'from' || key === 'correlation_id') && val.length > 16) val = val.slice(0, 12) + '...';
        kv[key] = val;
      }
    }
    const sections = [];
    if (Object.keys(kv).length) sections.push({ kind: 'kv', title: 'Message', content: kv });
    sections.push(...payloadSections);
    if (payloadText.trim() && payloadSections.length === 0) sections.push({ kind: 'plain', title: 'Payload', content: payloadText.trim() });
    return sections;
  }
  if (result.startsWith('No messages within') || result.startsWith('No messages received within')) return [{ kind: 'plain', title: 'Result', content: result }];
  return [{ kind: 'plain', title: 'Result', content: result }];
}

function _parseChildrenStatusResult(result) {
  if (!result) return [];
  const lines = result.trim().split('\n');
  const sections = [];
  if (lines[0]) sections.push({ kind: 'plain', title: 'Summary', content: lines[0] });
  const items = [];
  for (const line of lines.slice(1)) {
    const t = line.trim();
    if (t.startsWith('- ')) {
      const entry = t.slice(2);
      items.push({ text: entry, done: entry.toLowerCase().includes('completed') });
    }
  }
  if (items.length) sections.push({ kind: 'checklist', title: 'Children', content: items });
  return sections;
}

function _parseCheckStatusResult(result) {
  if (!result) return [];
  try {
    const parsed = JSON.parse(result.trim());
    if (parsed && typeof parsed === 'object') {
      const kv = {};
      for (const key of ['agent_id', 'state', 'role', 'model', 'depth', 'children_count', 'created_at', 'completed_at', 'error']) {
        if (key in parsed && parsed[key] != null) {
          let val = String(parsed[key]);
          if (key === 'agent_id') val = val.slice(0, 16) + (val.length > 16 ? '...' : '');
          kv[key] = val;
        }
      }
      const sections = [{ kind: 'kv', title: 'Agent Status', content: kv }];
      if (parsed.prompt_preview) sections.push({ kind: 'plain', title: 'Prompt', content: parsed.prompt_preview });
      if (parsed.children_ids && parsed.children_ids.length) {
        const fullIds = parsed.children_ids.filter((id) => String(id).length > 16);
        if (fullIds.length) {
          sections.push({ kind: "agent_links", title: "Children", content: fullIds.map(String) });
        }
      }
      return sections;
    }
  } catch(e) {}
  return [{ kind: 'plain', title: 'Status', content: result }];
}

function _parseExpertResult(result) {
  if (!result) return [];
  if (result.startsWith('Expert ')) {
    const lines = result.split('\n', 3);
    const sections = [{ kind: 'plain', title: 'Status', content: lines[0] }];
    if (lines.length > 1) {
      const body = lines.slice(1).join('\n').trim();
      if (body) sections.push({ kind: 'plain', title: 'Response', content: body });
    }
    return sections;
  }
  return [{ kind: 'plain', title: 'Expert Response', content: result }];
}

function _parsePeerResult(result) {
  if (!result) return [];
  if (result.startsWith('Peer response')) {
    const lines = result.split('\n', 3);
    const sections = [{ kind: 'plain', title: 'Status', content: lines[0] }];
    if (lines.length > 1) {
      let body = lines.slice(1).join('\n').trim();
      let threadId = '', peersInfo = '';
      if (body.includes('\nthread_id: ')) {
        const [mainBody, rest] = body.split('\nthread_id: ', 2);
        body = mainBody.trim();
        if (rest.includes('\nOther available peers: ')) {
          const [tid, peers] = rest.split('\nOther available peers: ', 2);
          threadId = tid.trim(); peersInfo = peers.trim();
        } else { threadId = rest.split('\n')[0].trim(); }
      }
      if (body) sections.push({ kind: 'plain', title: 'Response', content: body });
      const kv = {};
      if (threadId) kv['thread_id'] = threadId;
      if (peersInfo) kv['other_peers'] = peersInfo;
      if (Object.keys(kv).length) sections.push({ kind: 'kv', title: 'Info', content: kv });
    }
    return sections;
  }
  return [{ kind: 'plain', title: 'Peer Response', content: result }];
}

function _parseRecommendResult(result) {
  if (!result) return [];
  const kv = _parseResultKV(result);
  if (kv) {
    const sections = [{ kind: 'kv', title: 'Recommendation', content: kv }];
    if (result.includes('Fallback options')) {
      const fbStart = result.indexOf('Fallback options');
      const fbLines = result.slice(fbStart).split('\n');
      const items = [];
      for (const line of fbLines.slice(1)) {
        const t = line.trim();
        if (t && /^\d/.test(t)) items.push({ text: t.replace(/^[\d. ]+/, ''), done: false });
      }
      if (items.length) sections.push({ kind: 'checklist', title: 'Fallback Options', content: items });
    }
    return sections;
  }
  return [{ kind: 'plain', title: 'Recommendation', content: result }];
}

function _parseChildHistoryResult(result) {
  if (!result) return [];
  try {
    const parsed = JSON.parse(result.trim());
    if (Array.isArray(parsed)) {
      const sections = [];
      // Build transcript entries for richer display
      const entries = parsed.slice(0, 20).filter(e => typeof e === 'object').map(entry => ({
        role: entry.role || 'unknown',
        text: _trunc(String(entry.content || entry.text || ''), 120),
        tool: entry.tool || entry.tool_name || null,
      }));
      if (entries.length) sections.push({ kind: 'transcript', title: 'Transcript', content: entries });
      if (parsed.length > 20) sections.push({ kind: 'plain', content: `... and ${parsed.length - 20} more entries` });
      return sections;
    } else if (parsed && typeof parsed === 'object') {
      // Object with agent_id, turns, etc.
      const histKv = {};
      const turns = Array.isArray(parsed.turns) ? parsed.turns : [];
      if (parsed.agent_id) histKv.agent = String(parsed.agent_id).slice(0, 16);
      histKv.turns = String(turns.length);
      const sections = [];
      if (Object.keys(histKv).length) sections.push({ kind: 'kv', title: 'History', content: histKv });
      // Build transcript entries
      if (turns.length) {
        const entries = turns.slice(0, 20).filter(t => typeof t === 'object').map(t => ({
          role: t.role || 'unknown',
          text: _trunc(String(t.text || t.content || ''), 120),
          tool: t.tool || t.tool_name || null,
        }));
        if (entries.length) sections.push({ kind: 'transcript', title: 'Conversation', content: entries });
        if (turns.length > 20) sections.push({ kind: 'plain', content: `... and ${turns.length - 20} more turns` });
      }
      return sections;
    }
  } catch(e) {}
  return [{ kind: 'plain', title: 'History', content: result }];
}

// ── Orchestrator Tool Formatters ──
// Registered by bare names; the dispatcher strips the mcp__orchestrator__ prefix.

registerToolFormatter("task_complete", (name, args, result, success) => {
  const extracted = _extractTaskCompleteSummary(args, result);
  const summaryText = extracted.summary || "";
  const artifacts = extracted.artifacts || {};
  const sections = [];
  // Show the summary as a prominent result block — this is the agent's final output
  if (summaryText) {
    sections.push({
      kind: "result_block",
      title: "Agent Summary",
      content: { text: summaryText, status: success ? "success" : "error", markdown: extracted.includeMarkdown },
    });
  }
  // Show artifacts as kv
  if (typeof artifacts === "object" && Object.keys(artifacts).length) {
    const kv = {};
    for (const [k, v] of Object.entries(artifacts)) kv[k] = _trunc(String(v), 200);
    sections.push({ kind: "kv", title: "Artifacts", content: kv });
  }
  // Skip redundant "Task marked complete" confirmation
  if (result && result !== "Task marked complete. Session will end.") {
    sections.push({ kind: "plain", title: "Response", content: result });
  }
  return { icon: "\u2705", label: "Task Complete", summary: _trunc(summaryText, 60), sections };
});

registerToolFormatter("spawn_child", (name, args, result, success) => {
  const prompt = args.prompt || "";
  const model = args.model || "";
  const complexity = args.complexity || "";
  const wait = args.wait || false;
  const cwd = args.cwd || "";
  const tools = Array.isArray(args.tools) ? args.tools : [];
  const mcpServers = args.mcp_servers;
  const excludePlugins = Array.isArray(args.exclude_plugins) ? args.exclude_plugins : [];

  const sections = [];

  // Show the prompt as a prominent agent_prompt section
  if (prompt) {
    sections.push({
      kind: "agent_prompt",
      content: { number: null, prompt, model, complexity },
    });
  }

  // Configuration details (only non-obvious ones)
  const kv = {};
  if (wait) kv.mode = "blocking (wait=true)";
  if (cwd) kv.cwd = cwd;
  if (tools.length) {
    kv.tools = tools.slice(0, 6).join(", ");
    if (tools.length > 6) kv.tools += ` (+${tools.length - 6} more)`;
  }
  if (mcpServers && typeof mcpServers === "object") {
    kv.mcp_servers = Object.keys(mcpServers).join(", ");
  }
  if (excludePlugins.length) {
    kv.excluded = excludePlugins.join(", ");
  }
  if (Object.keys(kv).length) sections.push({ kind: "kv", title: "Config", content: kv });

  // Parse structured result
  sections.push(..._parseSpawnResult(result));

  // Build collapsed summary
  let badge = "";
  if (model) badge = `[${model}] `;
  else if (complexity) badge = `[${complexity}] `;

  let statusPrefix = "";
  if (result) {
    if (result.includes("Child completed")) {
      const m = result.match(/success=(\w+)/);
      statusPrefix = (m && m[1] === "True") ? "\u2705 " : "\u274c ";
    }
  }
  const summary = `${statusPrefix}${badge}${_trunc(prompt, 50)}`;

  return { icon: "\ud83d\ude80", label: "Spawn Agent", summary, sections };
});

registerToolFormatter("spawn_children_parallel", (name, args, result, success) => {
  const children = Array.isArray(args.children) ? args.children : [];
  let count = children.length;
  const sections = [];

  // Show each child as a numbered agent_prompt section
  children.slice(0, 10).forEach((child, i) => {
    if (typeof child === "object" && child !== null) {
      sections.push({
        kind: "agent_prompt",
        content: {
          number: i + 1,
          prompt: child.prompt || "",
          model: child.model || "",
          complexity: child.complexity || "",
        },
      });
    }
  });
  if (count > 10) sections.push({ kind: "plain", content: `... and ${count - 10} more agents` });

  // Parse result: extract spawned child IDs
  let resultChildCount = 0;
  if (result) {
    const resultLines = result.trim().split("\n");
    const childIds = _extractAgentIdsFromText(result);
    for (const line of resultLines) {
      const l = line.trim();
      if (l.startsWith("Spawned ") && l.includes(" children")) {
        const parts = l.split(" ");
        const parsed = parseInt(parts[1], 10);
        if (!Number.isNaN(parsed)) resultChildCount = parsed;
        break;
      }
    }
    if (!resultChildCount && childIds.length) resultChildCount = childIds.length;
    if (childIds.length) {
      sections.push({ kind: "agent_links", title: "Spawned IDs", content: childIds });
    }
    if (result.includes("Spawn errors")) {
      sections.push({ kind: "plain", title: "Errors", content: result.slice(result.indexOf("Spawn errors")) });
    }
  }
  count = Math.max(count, resultChildCount);

  return { icon: "\ud83d\ude80", label: "Spawn Parallel", summary: `${count} agents`, sections };
});

registerToolFormatter("restart_child", (name, args, result, success) => {
  const childId = args.child_agent_id || "";
  const prompt = args.prompt || "";
  const wait = args.wait || false;
  const shortId = childId.length > 12 ? childId.slice(0, 12) + "..." : childId;

  const sections = [];

  // Show agent being restarted
  const kv = {};
  if (shortId) kv.agent = shortId;
  if (wait) kv.mode = "blocking (wait=true)";
  if (Object.keys(kv).length) sections.push({ kind: "kv", title: "Target", content: kv });

  // Show the new prompt prominently
  if (prompt) {
    sections.push({
      kind: "agent_prompt",
      content: { number: null, prompt, model: "", complexity: "" },
    });
  }

  sections.push(..._parseSpawnResult(result));
  const summary = shortId ? `\u2192 ${shortId}` : _trunc(prompt, 50);
  return { icon: "\ud83d\udd04", label: "Restart Agent", summary, sections };
});

registerToolFormatter("ask_parent", (name, args, result, success) => {
  const question = args.question || "";
  const sections = [];
  if (question) sections.push({ kind: "plain", title: "\u2753 Question", content: question });
  if (result) sections.push({ kind: "plain", title: "\ud83d\udcac Answer", content: result });
  let summary = _trunc(question, 50);
  let icon = "\u2753";
  if (result) {
    icon = "\ud83d\udcac";
    summary = "answered \u2014 " + summary;
  }
  return { icon, label: "Ask Parent", summary, sections };
});

registerToolFormatter("ask_user", (name, args, result, success) => {
  const question = args.question || "";
  const options = Array.isArray(args.options) ? args.options : [];
  const sections = [];
  if (question) sections.push({ kind: "plain", title: "Question", content: question });
  if (options.length) {
    const items = options.slice(0, 6).filter(o => typeof o === "object" && o !== null).map(o => {
      const label = o.label || "";
      const desc = o.description || "";
      return { text: desc ? `${label} — ${desc}` : label, done: false };
    });
    if (items.length) sections.push({ kind: "checklist", title: "Options", content: items });
  }
  // Parse "User responded: ..." prefix
  if (result) {
    let answer = result;
    if (answer.startsWith("User responded: ")) answer = answer.slice(16);
    sections.push({ kind: "plain", title: "User's Answer", content: answer });
  }
  return { icon: "\ud83d\udcac", label: "Ask User", summary: _trunc(question, 50), sections };
});

registerToolFormatter("wait_for_message", (name, args, result, success) => {
  const timeout = args.timeout_seconds || "";
  const sections = [];
  const parsedSections = _parseWaitMessageResult(result);
  if (parsedSections.length) sections.push(...parsedSections);

  // Build a smart summary — include message type, source agent, and payload preview
  let summary = "waiting...";
  let icon = "\u23f3";
  if (result) {
    if (result.startsWith("Message received:")) {
      let msgType = "";
      let fromId = "";
      let payloadPreview = "";
      for (const line of result.split("\n")) {
        const t = line.trim();
        if (t.startsWith("type: ")) msgType = t.slice(6).trim();
        else if (t.startsWith("from: ")) {
          fromId = t.slice(6).trim();
          if (fromId.length > 12) fromId = fromId.slice(0, 8) + "\u2026";
        }
      }
      // Try to extract payload preview for task_result
      if (msgType.toLowerCase() === "task_result") {
        try {
          const payloadIdx = result.indexOf("payload: ");
          if (payloadIdx >= 0) {
            const payloadJson = JSON.parse(result.slice(payloadIdx + 9));
            if (payloadJson && payloadJson.summary) payloadPreview = _trunc(String(payloadJson.summary), 50);
          }
        } catch(e) {}
      }
      const typeLower = msgType.toLowerCase();
      if (typeLower === "task_result") {
        icon = "\u2705";
        if (payloadPreview) {
          summary = fromId ? `${fromId} \u2014 ${payloadPreview}` : payloadPreview;
        } else {
          summary = fromId ? `result from ${fromId}` : "task_result";
        }
      } else if (typeLower === "question") {
        icon = "\u2753";
        summary = fromId ? `question from ${fromId}` : "question";
      } else if (typeLower === "progress_update") {
        icon = "\ud83d\udcca";
        summary = fromId ? `progress from ${fromId}` : "progress_update";
      } else {
        summary = typeLower;
        if (fromId) summary += ` from ${fromId}`;
      }
    } else if (result.startsWith("No messages")) {
      icon = "\u231b";
      summary = "timed out";
    }
  }
  return { icon, label: "Wait for Message", summary, sections };
});

registerToolFormatter("respond_to_child", (name, args, result, success) => {
  const childId = args.child_agent_id || "";
  const correlationId = args.correlation_id || "";
  const response = args.response || "";
  const shortId = childId.length > 12 ? childId.slice(0, 12) + "..." : childId;

  const sections = [];

  // Show target agent
  const kv = {};
  if (shortId) kv.to = shortId;
  if (correlationId) {
    kv.correlation = correlationId.length > 12 ? correlationId.slice(0, 12) + "..." : correlationId;
  }
  if (Object.keys(kv).length) sections.push({ kind: "kv", title: "Target", content: kv });

  // Show the response content prominently
  if (response) sections.push({ kind: "plain", title: "Response", content: response });

  // Skip redundant "Response delivered" confirmation
  if (result && !result.startsWith("Response delivered")) {
    sections.push({ kind: "plain", title: "Status", content: result });
  }
  return { icon: "\u21a9\ufe0f", label: "Reply to Agent", summary: shortId ? `\u2192 ${shortId}` : _trunc(response, 40), sections };
});

registerToolFormatter("consult_expert", (name, args, result, success) => {
  const expertId = args.expert_id || "";
  const question = args.question || "";
  const sections = [];
  if (question) sections.push({ kind: "plain", title: "Question", content: question });
  sections.push(..._parseExpertResult(result));
  const summary = expertId ? `${expertId}: ${_trunc(question, 40)}` : _trunc(question, 50);
  return { icon: "\ud83c\udf93", label: "Consult Expert", summary, sections };
});

registerToolFormatter("consult_peer", (name, args, result, success) => {
  const peer = args.peer || "";
  const question = args.question || "";
  const sections = [];
  if (question) sections.push({ kind: "plain", title: "Question", content: question });
  sections.push(..._parsePeerResult(result));
  const summary = peer ? `${peer}: ${_trunc(question, 40)}` : _trunc(question, 50);
  return { icon: "\ud83e\udd1d", label: "Consult Peer", summary, sections };
});

registerToolFormatter("report_progress", (name, args, result, success) => {
  const status = args.status || "";
  const percent = args.percent_complete || 0;
  const sections = [];

  // Show a visual progress bar section
  if (percent || status) {
    sections.push({
      kind: "progress",
      content: { percent: parseInt(percent) || 0, status },
    });
  }

  // Skip redundant "Progress reported" confirmation
  if (result && !result.startsWith("Progress reported:")) {
    sections.push({ kind: "plain", title: "Output", content: result });
  }

  let summary = status;
  if (percent) summary = `${percent}% \u2014 ${_trunc(status, 40)}`;
  return { icon: "\ud83d\udcca", label: "Progress", summary: _trunc(summary, 50), sections };
});

registerToolFormatter("get_child_history", (name, args, result, success) => {
  const childId = args.child_agent_id || "";
  const detail = args.detail_level || "full";
  const shortId = childId.length > 12 ? childId.slice(0, 12) + "..." : childId;
  const kv = {};
  if (shortId) kv.agent = shortId;
  kv.detail = detail;
  const sections = [{ kind: "kv", content: kv }];
  const historySections = _parseChildHistoryResult(result);
  sections.push(...historySections);
  // Try to extract turn count for the summary
  let summarySuffix = "";
  if (result) {
    try {
      const parsed = JSON.parse(result.trim());
      if (Array.isArray(parsed)) summarySuffix = ` \u2014 ${parsed.length} entries`;
      else if (parsed && Array.isArray(parsed.turns)) summarySuffix = ` \u2014 ${parsed.turns.length} turns`;
    } catch(e) {}
  }
  return { icon: "\ud83d\udcdc", label: "Agent History", summary: shortId + summarySuffix, sections };
});

registerToolFormatter("check_child_status", (name, args, result, success) => {
  const childId = args.child_agent_id || "";
  const shortId = childId.length > 12 ? childId.slice(0, 12) + "..." : childId;
  const sections = [];
  sections.push(..._parseCheckStatusResult(result));
  return { icon: "\ud83d\udd0d", label: "Check Agent", summary: shortId, sections };
});

registerToolFormatter("send_child_prompt", (name, args, result, success) => {
  const childId = args.child_agent_id || "";
  const prompt = args.prompt || "";
  const shortId = childId.length > 12 ? childId.slice(0, 12) + "..." : childId;
  const sections = [];

  // Show target agent
  if (shortId) sections.push({ kind: "kv", title: "Target", content: { agent: shortId } });

  // Show the prompt prominently
  if (prompt) {
    sections.push({
      kind: "agent_prompt",
      content: { number: null, prompt, model: "", complexity: "" },
    });
  }

  // Skip redundant "Prompt delivered" confirmation
  if (result && !result.startsWith("Prompt delivered")) {
    sections.push({ kind: "plain", title: "Output", content: result });
  }
  return { icon: "\ud83d\udce8", label: "Send to Agent", summary: shortId ? `\u2192 ${shortId}` : _trunc(prompt, 40), sections };
});

registerToolFormatter("get_children_status", (name, args, result, success) => {
  const sections = [];
  sections.push(..._parseChildrenStatusResult(result));
  // Extract summary from result
  let summary = "checking...";
  if (result) {
    const firstLine = result.trim().split("\n")[0];
    if (firstLine.includes(":")) summary = firstLine.split(":").slice(1).join(":").trim().slice(0, 50);
  }
  return { icon: "\ud83d\udc65", label: "All Agents Status", summary, sections };
});

registerToolFormatter("recommend_model", (name, args, result, success) => {
  const taskDesc = args.task_description || "";
  const complexity = args.complexity || "medium";
  const kv = {};
  if (complexity) kv.complexity = complexity;
  if (taskDesc) kv.task = _trunc(taskDesc, 80);
  const sections = [];
  if (Object.keys(kv).length) sections.push({ kind: "kv", content: kv });
  sections.push(..._parseRecommendResult(result));
  return { icon: "\ud83e\udde0", label: "Recommend Model", summary: `[${complexity}] ${_trunc(taskDesc, 40)}`, sections };
});

function defaultToolFormatter(name, args, result, success) {
  const raw = args._raw || "";
  let summary, sections;
  if (raw) {
    summary = _trunc(raw, 50);
    sections = [{ kind: "plain", content: raw }];
  } else {
    const displayArgs = {};
    for (const [k, v] of Object.entries(args)) {
      if (!k.startsWith("_")) displayArgs[k] = _trunc(String(v), 80);
    }
    summary = _trunc(Object.entries(displayArgs).map(([k, v]) => `${k}=${v}`).join(", "), 50);
    sections = Object.keys(displayArgs).length ? [{ kind: "kv", content: displayArgs }] : [];
  }
  if (result) sections.push({ kind: "plain", title: "Output", content: result });
  return { icon: "\ud83d\udd27", label: name, summary, sections };
}

// ── DOM Renderers for formatted tool calls ──

function renderFormattedCollapsed(fmt, statusClass, statusLabel) {
  const headerRow = el("div", "tool-header");
  headerRow.appendChild(el("span", "tool-chevron", "\u25B6"));
  if (fmt.icon) headerRow.appendChild(el("span", "tool-icon", fmt.icon));
  headerRow.appendChild(el("span", "tool-name", fmt.label));
  if (fmt.summary) {
    if (fmt.file_path) {
      const link = el("span", "tool-summary tool-file-link", fmt.summary);
      link.title = fmt.file_path;
      link.addEventListener("click", (e) => {
        e.stopPropagation();
        vscode.postMessage({ type: "openFile", filePath: fmt.file_path });
      });
      headerRow.appendChild(link);
    } else {
      headerRow.appendChild(el("span", "tool-summary", fmt.summary));
    }
  }
  headerRow.appendChild(el("span", `tool-status status-${statusClass}`, statusLabel));
  return headerRow;
}

function renderFormattedExpanded(fmt, statusClass) {
  const details = el("div", "tool-details");

  for (const section of fmt.sections) {
    const sectionEl = el("div", "tool-section");
    if (section.title) {
      sectionEl.appendChild(el("span", "tool-section-label", section.title));
    }

    if (section.kind === "diff") {
      const block = el("div", "diff-block");
      const oldLines = (section.content && section.content.old_lines) || [];
      const newLines = (section.content && section.content.new_lines) || [];
      for (const line of oldLines) {
        const lineEl = el("div", "diff-removed");
        lineEl.textContent = line;
        block.appendChild(lineEl);
      }
      for (const line of newLines) {
        const lineEl = el("div", "diff-added");
        lineEl.textContent = line;
        block.appendChild(lineEl);
      }
      sectionEl.appendChild(block);
    } else if (section.kind === "code") {
      const pre = document.createElement("pre");
      pre.className = "tool-code-block";
      const code = document.createElement("code");
      code.textContent = (section.content && section.content.text) || "";
      pre.appendChild(code);
      sectionEl.appendChild(pre);
    } else if (section.kind === "path") {
      const pathContent = section.content || "";
      const pathLink = el("span", "tool-file-path tool-file-link", pathContent);
      pathLink.title = "Click to open in editor";
      pathLink.addEventListener("click", (e) => {
        e.stopPropagation();
        vscode.postMessage({ type: "openFile", filePath: pathContent });
      });
      sectionEl.appendChild(pathLink);
    } else if (section.kind === "checklist") {
      const list = el("div", "tool-checklist");
      const items = section.content || [];
      for (const item of items) {
        const cls = item.done ? "checklist-item done" : "checklist-item pending";
        const marker = item.done ? "\u2713 " : "\u25cb ";
        list.appendChild(el("div", cls, marker + (item.text || "")));
      }
      sectionEl.appendChild(list);
    } else if (section.kind === "kv") {
      const kv = section.content || {};
      for (const [key, value] of Object.entries(kv)) {
        const row = el("div", "tool-kv");
        row.appendChild(el("span", "kv-key", key + ": "));
        row.appendChild(el("span", "kv-value", String(value)));
        sectionEl.appendChild(row);
      }
    } else if (section.kind === "agent_links") {
      const links = Array.isArray(section.content) ? section.content : [];
      const list = el("div", "tool-agent-list");
      links.forEach((entry) => {
        const agentId = typeof entry === "string" ? entry : "";
        if (!agentId) return;
        const row = el("div", "tool-agent-link-row");
        const link = el("span", "tool-agent-link tool-file-link", agentId);
        link.title = `Open chat for ${agentId}`;
        link.addEventListener("click", (e) => {
          e.stopPropagation();
          vscode.postMessage({ type: "openAgent", agentId });
        });
        row.appendChild(link);
        list.appendChild(row);
      });
      if (links.length) sectionEl.appendChild(list);
    } else if (section.kind === "progress") {
      const content = section.content || {};
      const percent = content.percent || 0;
      const status = content.status || "";
      const barContainer = el("div", "tool-progress-bar");
      const track = el("div", "progress-track");
      const fill = el("div", "progress-fill");
      fill.style.width = `${Math.min(100, Math.max(0, percent))}%`;
      track.appendChild(fill);
      barContainer.appendChild(track);
      const label = el("span", "progress-label", `${percent}%`);
      barContainer.appendChild(label);
      sectionEl.appendChild(barContainer);
      if (status) {
        sectionEl.appendChild(el("div", "progress-status", status));
      }
    } else if (section.kind === "agent_prompt") {
      const content = section.content || {};
      const number = content.number;
      const prompt = content.prompt || "";
      const model = content.model || "";
      const complexity = content.complexity || "";
      // Header with optional number and model badge
      const headerEl = el("div", "agent-prompt-header");
      if (number !== null && number !== undefined) {
        headerEl.appendChild(el("span", "agent-prompt-number", `Agent ${number}`));
      }
      if (model) {
        headerEl.appendChild(el("span", "agent-prompt-badge", `[${model}]`));
      } else if (complexity) {
        headerEl.appendChild(el("span", "agent-prompt-badge", `[${complexity}]`));
      }
      if (headerEl.children.length > 0) sectionEl.appendChild(headerEl);
      // Prompt text in a styled block
      if (prompt) {
        const promptBlock = el("div", "agent-prompt-text");
        const promptLines = prompt.split("\n").slice(0, 15);
        for (const line of promptLines) {
          const lineEl = el("div", "agent-prompt-line");
          lineEl.textContent = line;
          promptBlock.appendChild(lineEl);
        }
        const remaining = prompt.split("\n").length - 15;
        if (remaining > 0) {
          promptBlock.appendChild(el("div", "agent-prompt-more", `... ${remaining} more lines`));
        }
        sectionEl.appendChild(promptBlock);
      }
    } else if (section.kind === "transcript") {
      // Conversation history with role-based icons and styling
      const entries = section.content || [];
      const roleIcons = { assistant: "\ud83e\udd16", user: "\ud83d\udc64", tool: "\ud83d\udd27", system: "\u2699\ufe0f" };
      const roleClasses = { assistant: "transcript-assistant", user: "transcript-user", tool: "transcript-tool", system: "transcript-system" };
      const list = el("div", "tool-transcript");
      for (const entry of entries) {
        const role = entry.role || "unknown";
        const text = entry.text || "";
        const tool = entry.tool || null;
        const icon = roleIcons[role] || "\u00b7";
        const entryEl = el("div", `transcript-entry ${roleClasses[role] || "transcript-unknown"}`);
        // Role header line
        const roleHeader = el("div", "transcript-role-header");
        roleHeader.appendChild(el("span", "transcript-icon", icon));
        roleHeader.appendChild(el("span", "transcript-role-name", role));
        if (tool) {
          roleHeader.appendChild(el("span", "transcript-tool-name", `(${tool})`));
        }
        entryEl.appendChild(roleHeader);
        // Content text with left border
        if (text) {
          const textBlock = el("div", "transcript-text");
          const textLines = text.split("\n").slice(0, 4);
          for (const line of textLines) {
            const lineEl = el("div", "transcript-text-line");
            lineEl.textContent = line;
            textBlock.appendChild(lineEl);
          }
          if (text.split("\n").length > 4) {
            textBlock.appendChild(el("div", "transcript-truncated", "... truncated"));
          }
          entryEl.appendChild(textBlock);
        }
        list.appendChild(entryEl);
      }
      sectionEl.appendChild(list);
    } else if (section.kind === "result_block") {
      // Prominent bordered result block — for agent summaries and final outputs
      const content = section.content || {};
      const text = content.text || "";
      const status = content.status || "success";
      const markdown = content.markdown === true;
      const statusClasses = { success: "result-block-success", error: "result-block-error", question: "result-block-question" };
      const blockEl = el("div", `result-block ${statusClasses[status] || "result-block-default"}`);
      if (text) {
        const textLines = text.split("\n");
        const visibleLines = textLines.slice(0, 30);
        if (markdown) {
          blockEl.appendChild(renderMarkdown(visibleLines.join("\n")));
        } else {
          for (const line of visibleLines) {
            const lineEl = el("div", "result-block-line");
            lineEl.textContent = line;
            blockEl.appendChild(lineEl);
          }
        }
        const remaining = textLines.length - visibleLines.length;
        if (remaining > 0) {
          blockEl.appendChild(el("div", "result-block-more", `... ${remaining} more lines`));
        }
      }
      sectionEl.appendChild(blockEl);
    } else {
      // plain or unknown
      const pre = el("pre", `tool-pre result-${statusClass}`, section.content || "");
      sectionEl.appendChild(pre);
    }

    details.appendChild(sectionEl);
  }

  // Show waiting state if no sections (pending tool call)
  if (fmt.sections.length === 0) {
    details.appendChild(el("div", "tool-waiting", "Waiting for result..."));
  }

  return details;
}

// ── Main tool call renderer (entry point) ──

function renderToolCallInto(wrapper, tc) {
  if (tc.id) {
    wrapper.setAttribute("data-tool-id", tc.id);
    wrapper.id = `tc-${tc.id}`;
  }

  const fmt = formatToolCallData(tc.name, tc.arguments, tc.result, tc.success);
  const statusClass =
    tc.result === null ? "pending" : tc.success ? "success" : "error";
  const statusLabel =
    tc.result === null ? "..." : tc.success ? "done" : "error";

  wrapper.appendChild(renderFormattedCollapsed(fmt, statusClass, statusLabel));
  wrapper.appendChild(renderFormattedExpanded(fmt, statusClass));
}

// ── Markdown Rendering ──
// Safe DOM-based markdown renderer. All text is inserted via textContent,
// never innerHTML. Supports: headers, bold, italic, inline code, code blocks,
// lists (ordered/unordered), blockquotes, horizontal rules, and links.

/**
 * Render markdown text into a DocumentFragment.
 * @param {string} text
 * @returns {DocumentFragment}
 */
function renderMarkdown(text) {
  const frag = document.createDocumentFragment();
  if (!text) return frag;

  // Split by fenced code blocks first (```...```)
  const segments = String(text).split(/(```[\s\S]*?```)/g);

  for (const segment of segments) {
    if (segment.startsWith("```") && segment.endsWith("```")) {
      // Fenced code block
      const inner = segment.slice(3, -3);
      const firstNewline = inner.indexOf("\n");
      const lang = firstNewline >= 0 ? inner.slice(0, firstNewline).trim() : "";
      const code = firstNewline >= 0 ? inner.slice(firstNewline + 1) : inner;

      const pre = document.createElement("pre");
      pre.className = "md-code-block";
      const codeEl = document.createElement("code");
      if (lang) codeEl.className = "language-" + lang;
      codeEl.textContent = code;
      pre.appendChild(codeEl);
      frag.appendChild(pre);
    } else {
      // Parse block-level markdown
      renderBlockMarkdown(frag, segment);
    }
  }

  return frag;
}

/**
 * Parse block-level markdown (headers, lists, blockquotes, paragraphs).
 * @param {DocumentFragment|HTMLElement} parent
 * @param {string} text
 */
function renderBlockMarkdown(parent, text) {
  const lines = text.split("\n");
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Blank line - skip
    if (line.trim() === "") {
      i++;
      continue;
    }

    // Horizontal rule: ---, ***, ___
    if (/^(\s*[-*_]\s*){3,}$/.test(line)) {
      parent.appendChild(document.createElement("hr"));
      i++;
      continue;
    }

    // Heading: # H1, ## H2, etc.
    const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const heading = document.createElement("h" + level);
      heading.className = "md-heading";
      heading.appendChild(renderInlineMarkdown(headingMatch[2]));
      parent.appendChild(heading);
      i++;
      continue;
    }

    // Blockquote: > text
    if (line.trimStart().startsWith("> ")) {
      const bq = document.createElement("blockquote");
      bq.className = "md-blockquote";
      const bqLines = [];
      while (i < lines.length && lines[i].trimStart().startsWith("> ")) {
        bqLines.push(lines[i].trimStart().slice(2));
        i++;
      }
      const bqP = document.createElement("p");
      bqP.appendChild(renderInlineMarkdown(bqLines.join("\n")));
      bq.appendChild(bqP);
      parent.appendChild(bq);
      continue;
    }

    // Unordered list: - item or * item
    // Tolerates blank lines and indented continuation text between items
    if (/^\s*[-*+]\s+/.test(line)) {
      const ul = document.createElement("ul");
      ul.className = "md-list";
      let currentLi = null;
      while (i < lines.length) {
        if (/^\s*[-*+]\s+/.test(lines[i])) {
          // New bullet item
          currentLi = document.createElement("li");
          const content = lines[i].replace(/^\s*[-*+]\s+/, "");
          currentLi.appendChild(renderInlineMarkdown(content));
          ul.appendChild(currentLi);
          i++;
        } else if (lines[i].trim() === "") {
          // Blank line - peek ahead to see if list continues
          let peek = i + 1;
          while (peek < lines.length && lines[peek].trim() === "") peek++;
          if (peek < lines.length && /^\s*[-*+]\s+/.test(lines[peek])) {
            i = peek; // skip blank lines, continue list
          } else {
            break; // list is over
          }
        } else if (currentLi && /^\s+/.test(lines[i])) {
          // Indented continuation line - append to current item
          const br = document.createElement("br");
          currentLi.appendChild(br);
          currentLi.appendChild(renderInlineMarkdown(lines[i].trim()));
          i++;
        } else {
          break; // non-list content, stop
        }
      }
      parent.appendChild(ul);
      continue;
    }

    // Ordered list: 1. item
    // Tolerates blank lines and indented continuation text between items.
    // Sets <ol start="N"> to preserve the author's numbering even if
    // multiple <ol> elements end up being created.
    if (/^\s*(\d+)\.\s+/.test(line)) {
      const startMatch = line.match(/^\s*(\d+)\.\s+/);
      const ol = document.createElement("ol");
      ol.className = "md-list";
      if (startMatch) {
        const startNum = parseInt(startMatch[1], 10);
        if (startNum !== 1) ol.setAttribute("start", startNum);
      }
      let currentLi = null;
      while (i < lines.length) {
        const olMatch = lines[i].match(/^\s*(\d+)\.\s+(.*)/);
        if (olMatch) {
          // New numbered item
          currentLi = document.createElement("li");
          currentLi.appendChild(renderInlineMarkdown(olMatch[2]));
          ol.appendChild(currentLi);
          i++;
        } else if (lines[i].trim() === "") {
          // Blank line - peek ahead to see if list continues
          let peek = i + 1;
          while (peek < lines.length && lines[peek].trim() === "") peek++;
          if (peek < lines.length && /^\s*\d+\.\s+/.test(lines[peek])) {
            i = peek; // skip blank lines, continue list
          } else {
            break; // list is over
          }
        } else if (currentLi && /^\s+/.test(lines[i])) {
          // Indented continuation line - append to current item
          const br = document.createElement("br");
          currentLi.appendChild(br);
          currentLi.appendChild(renderInlineMarkdown(lines[i].trim()));
          i++;
        } else {
          break; // non-list content, stop
        }
      }
      parent.appendChild(ol);
      continue;
    }

    // Regular paragraph - collect consecutive non-empty, non-special lines
    const paraLines = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !/^#{1,6}\s/.test(lines[i]) &&
      !/^(\s*[-*_]\s*){3,}$/.test(lines[i]) &&
      !/^\s*[-*+]\s+/.test(lines[i]) &&
      !/^\s*\d+\.\s+/.test(lines[i]) &&
      !lines[i].trimStart().startsWith("> ")
    ) {
      paraLines.push(lines[i]);
      i++;
    }

    if (paraLines.length > 0) {
      const p = document.createElement("p");
      p.className = "md-paragraph";
      p.appendChild(renderInlineMarkdown(paraLines.join("\n")));
      parent.appendChild(p);
    }
  }
}

/**
 * Render inline markdown (bold, italic, code, links) into a DocumentFragment.
 * All text is safely escaped via textContent.
 * @param {string} text
 * @returns {DocumentFragment}
 */
function renderInlineMarkdown(text) {
  const frag = document.createDocumentFragment();
  if (!text) return frag;

  // Regex to match inline patterns:
  // `code` | **bold** | *italic* | [text](url)
  const inlinePattern = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]]+\]\([^)]+\))/g;

  let lastIndex = 0;
  let match;

  while ((match = inlinePattern.exec(text)) !== null) {
    // Add text before match
    if (match.index > lastIndex) {
      frag.appendChild(txt(text.slice(lastIndex, match.index)));
    }

    const token = match[0];

    if (token.startsWith("`") && token.endsWith("`")) {
      // Inline code
      const code = document.createElement("code");
      code.className = "md-inline-code";
      code.textContent = token.slice(1, -1);
      frag.appendChild(code);
    } else if (token.startsWith("**") && token.endsWith("**")) {
      // Bold
      const strong = document.createElement("strong");
      strong.textContent = token.slice(2, -2);
      frag.appendChild(strong);
    } else if (token.startsWith("*") && token.endsWith("*")) {
      // Italic
      const em = document.createElement("em");
      em.textContent = token.slice(1, -1);
      frag.appendChild(em);
    } else if (token.startsWith("[")) {
      // Link [text](url)
      const linkMatch = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      if (linkMatch) {
        const a = document.createElement("a");
        a.textContent = linkMatch[1];
        a.href = linkMatch[2];
        a.className = "md-link";
        frag.appendChild(a);
      } else {
        frag.appendChild(txt(token));
      }
    }

    lastIndex = match.index + match[0].length;
  }

  // Remaining text
  if (lastIndex < text.length) {
    frag.appendChild(txt(text.slice(lastIndex)));
  }

  return frag;
}

// ── Streaming ──

function appendStreamChunk(text) {
  if (!conversationEl) return;

  if (!activeStreamEl) {
    // Remove "no messages" placeholder
    const empty = conversationEl.querySelector(".empty-state");
    if (empty) empty.remove();

    activeStreamEl = el("div", "message message-assistant streaming");

    const header = el("div", "message-header");
    header.appendChild(el("span", "role-label role-assistant", "Assistant"));
    header.appendChild(el("span", "streaming-dot", ""));
    activeStreamEl.appendChild(header);

    const body = el("div", "message-body stream-body");
    activeStreamEl.appendChild(body);
    conversationEl.appendChild(activeStreamEl);
  }

  const body = activeStreamEl.querySelector(".stream-body");
  if (body) {
    body.appendChild(txt(text));
  }

  if (autoScroll) scrollToBottom();
}

function finalizeStream() {
  if (activeStreamEl) {
    // Re-render the accumulated text as proper markdown
    const body = activeStreamEl.querySelector(".stream-body");
    if (body) {
      const rawText = body.textContent || "";
      body.textContent = "";
      body.appendChild(renderMarkdown(rawText));
    }
    activeStreamEl.classList.remove("streaming");
    const dot = activeStreamEl.querySelector(".streaming-dot");
    if (dot) dot.remove();
    activeStreamEl = null;
  }
}

// ── Scroll tracking ──

if (conversationEl) {
  conversationEl.addEventListener("scroll", () => {
    autoScroll =
      conversationEl.scrollTop + conversationEl.clientHeight >=
      conversationEl.scrollHeight - 60;
    schedulePersistUiState();
  });
}

document.addEventListener("focusin", (e) => {
  pendingFocusState = _captureFocusStateFromElement(
    /** @type {HTMLElement|null} */ (e.target),
  );
  schedulePersistUiState();
});

document.addEventListener("selectionchange", () => {
  const active = /** @type {HTMLElement|null} */ (document.activeElement);
  const focusState = _captureFocusStateFromElement(active);
  if (focusState) {
    pendingFocusState = focusState;
    schedulePersistUiState();
  }
});

function scrollToBottom() {
  if (conversationEl) {
    conversationEl.scrollTop = conversationEl.scrollHeight;
  }
}

// ── Scroll to Tool Call ──

/**
 * Queue a jump request and try applying it immediately.
 * Retries happen after each renderFull call if not yet found.
 * @param {unknown} toolCallId
 * @param {unknown} messageIndex
 */
function requestToolCallJump(toolCallId, messageIndex) {
  const safeToolCallId = typeof toolCallId === "string" ? toolCallId : "";
  const safeMessageIndex = Number.isInteger(messageIndex)
    ? Math.max(0, Number(messageIndex))
    : null;
  if (!safeToolCallId && safeMessageIndex === null) return;
  pendingToolCallJump = {
    toolCallId: safeToolCallId,
    messageIndex: safeMessageIndex,
  };
  autoScroll = false;
  tryApplyPendingToolCallJump();
}

/**
 * Attempt to apply a queued tool-call jump. Clears the queue on success.
 * @returns {boolean}
 */
function tryApplyPendingToolCallJump() {
  if (!pendingToolCallJump) return false;
  const didJump = scrollToToolCall(
    pendingToolCallJump.toolCallId,
    pendingToolCallJump.messageIndex
  );
  if (didJump) {
    pendingToolCallJump = null;
  }
  return didJump;
}

/**
 * Scroll to and highlight a specific tool call element.
 * Falls back to a tool-role message index only when toolCallId is not provided.
 * @param {string} toolCallId
 * @param {number | null | undefined} messageIndex
 * @returns {boolean}
 */
function scrollToToolCall(toolCallId, messageIndex) {
  if (!conversationEl) return false;

  let target = null;
  if (toolCallId) {
    // Primary: O(1) lookup by element id (same pattern as snapshot anchors)
    target = document.getElementById(`tc-${toolCallId}`);
    // Fallback: attribute selector
    if (!target) {
      target = conversationEl.querySelector(
        `[data-tool-id="${toolCallId}"]`
      );
    }
    // If a tool call id was provided, never jump to a non-tool message fallback.
    if (!target) {
      console.log(`[webview] Tool target not found for id ${toolCallId}`);
      return false;
    }
  }
  if (!toolCallId && !target && Number.isInteger(messageIndex)) {
    target = conversationEl.querySelector(
      `.message.message-tool[data-message-index="${messageIndex}"]`
    );
  }
  if (!target) {
    console.log(`[webview] Tool target not found for index ${messageIndex}`);
    return false;
  }

  // Clear any existing highlights
  conversationEl.querySelectorAll(".scroll-highlight").forEach(el => el.classList.remove("scroll-highlight"));

  // Expand the tool call if collapsed
  if (target.classList.contains("collapsible") && !target.classList.contains("expanded")) {
    target.classList.add("expanded");
    // Persist expansion state across re-renders
    if (toolCallId) {
      expandedToolCallIds.add(toolCallId);
    }
  }

  // Scroll into view
  target.scrollIntoView({ behavior: "auto", block: "start" });
  
  // Explicit fallback scroll
  const containerRect = conversationEl.getBoundingClientRect();
  const targetRect = target.getBoundingClientRect();
  const relativeTop = targetRect.top - containerRect.top;
  
  if (Math.abs(relativeTop) > 1) {
    conversationEl.scrollTop += relativeTop;
  }

  // Flash highlight animation
  target.classList.add("scroll-highlight");
  setTimeout(() => {
    target.classList.remove("scroll-highlight");
  }, 3000);
  
  console.log(`[webview] Scrolled to tool ${toolCallId || messageIndex}, relativeTop=${relativeTop}`);
  return true;
}

/**
 * Queue a snapshot jump request and try applying it immediately.
 * @param {string} snapshotId
 */
function requestSnapshotJump(snapshotId) {
  if (!snapshotId) return;
  pendingSnapshotJump = snapshotId;
  autoScroll = false;
  tryApplyPendingSnapshotJump();
}

/**
 * Attempt to apply a queued snapshot jump. Clears the queue on success.
 * @returns {boolean}
 */
function tryApplyPendingSnapshotJump() {
  if (!pendingSnapshotJump) return false;
  const didJump = scrollToSnapshot(pendingSnapshotJump);
  if (didJump) {
    pendingSnapshotJump = null;
  }
  return didJump;
}

/**
 * Scroll to and highlight a specific user message associated with a snapshot.
 * @param {string} snapshotId
 * @returns {boolean}
 */
function scrollToSnapshot(snapshotId) {
  if (!conversationEl || !snapshotId) return false;

  let target = document.getElementById(`snap-${snapshotId}`);
  if (!target) {
    target = conversationEl.querySelector(
      `.message.message-user[data-snapshot-id="${snapshotId}"]`
    );
  }
  
  if (!target) {
    console.log(`[webview] Snapshot target not found for ${snapshotId}`);
    return false;
  }

  // Clear any existing highlights
  conversationEl.querySelectorAll(".scroll-highlight").forEach(el => el.classList.remove("scroll-highlight"));

  // Scroll into view
  target.scrollIntoView({ behavior: "auto", block: "start" });
  
  // Explicit fallback scroll if scrollIntoView was unreliable.
  // Using getBoundingClientRect().top relative to conversationEl for precision.
  const containerRect = conversationEl.getBoundingClientRect();
  const targetRect = target.getBoundingClientRect();
  const relativeTop = targetRect.top - containerRect.top;
  
  if (Math.abs(relativeTop) > 1) {
    conversationEl.scrollTop += relativeTop;
  }

  // Flash highlight animation
  target.classList.add("scroll-highlight");
  setTimeout(() => {
    target.classList.remove("scroll-highlight");
  }, 3000);
  
  console.log(`[webview] Scrolled to snapshot ${snapshotId}, relativeTop=${relativeTop}`);
  return true;
}

// ── Ready handshake ──
// Signal to the extension host that the webview is ready to receive messages.
// This prevents the race condition where postMessage is sent before the
// message listener is registered.
vscode.postMessage({ type: "ready" });
schedulePersistUiState();

// ── Utilities ──

function formatTime(iso) {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return "";
  }
}
