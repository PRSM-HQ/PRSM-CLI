/** Agent state matching prsm's AgentState enum. */
export type AgentState =
  | "idle"
  | "running"
  | "waiting"
  | "waiting_for_child"
  | "completed"
  | "error";

/** Agent role matching prsm's AgentRole enum. */
export type AgentRole = "orchestrator" | "worker" | "expert";

/** Agent node in the tree. */
export interface AgentNode {
  id: string;
  name: string;
  state: AgentState;
  /** Raw engine state from SSE/REST (for richer UI distinctions). */
  rawState?: string;
  role: AgentRole;
  model: string;
  parentId: string | null;
  childrenIds: string[];
  promptPreview: string;
  createdAt: string | null;
  completedAt: string | null;
  lastActive: string | null;
}

/** Chat message for an agent's conversation. */
export interface Message {
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  agentId: string;
  timestamp: string | null;
  snapshotId?: string | null;
  toolCalls: ToolCall[];
  streaming: boolean;
}

/** A tool call within a message. */
export interface ToolCall {
  id: string;
  name: string;
  arguments: string;
  result: string | null;
  success: boolean;
  pending?: boolean;
}

/** Session metadata. */
export interface SessionInfo {
  sessionId: string;
  name: string;
  summary?: string | null;
  forkedFrom: string | null;
  agentCount: number;
  messageCount: number;
  running: boolean;
  createdAt: string | null;
  lastActivity: string | null;
  currentModel?: string;
  currentModelDisplay?: string;
}

// ── SSE Event Data Types ──

export interface AgentSpawnedData {
  session_id: string;
  agent_id: string;
  parent_id: string | null;
  role: string;
  model: string;
  depth: number;
  prompt: string;
  name?: string;
  state?: string;
}

export interface AgentStateChangedData {
  session_id: string;
  agent_id: string;
  old_state: string;
  new_state: string;
}

export interface AgentKilledData {
  session_id: string;
  agent_id: string;
}

export interface StreamChunkData {
  session_id: string;
  agent_id: string;
  text: string;
}

export interface ToolCallStartedData {
  session_id: string;
  agent_id: string;
  tool_id: string;
  tool_name: string;
  arguments: string;
}

export interface ToolCallCompletedData {
  session_id: string;
  agent_id: string;
  tool_id: string;
  result: string;
  is_error: boolean;
}

export interface ToolCallDeltaData {
  session_id: string;
  agent_id: string;
  tool_id: string;
  delta: string;
  stream: "stdout" | "stderr" | string;
}

export interface PermissionRequestData {
  session_id: string;
  agent_id: string;
  request_id: string;
  tool_name: string;
  agent_name: string;
  arguments: string;
}

export interface UserQuestionData {
  session_id: string;
  agent_id: string;
  request_id: string;
  agent_name: string;
  question: string;
  options: Array<{ label: string; description?: string }>;
}

export interface EngineStartedData {
  session_id: string;
  task_definition: string;
}

export interface EngineFinishedData {
  session_id: string;
  success: boolean;
  summary: string;
  error?: string;
  duration_seconds: number;
}

export interface SessionCreatedData {
  session_id: string;
  name: string;
  summary?: string | null;
  forked_from: string | null;
  current_model?: string;
  current_model_display?: string;
}

export interface SessionRemovedData {
  session_id: string;
}

export interface AgentResultData {
  session_id: string;
  agent_id: string;
  result: string;
  is_error: boolean;
}

export interface ThinkingData {
  session_id: string;
  agent_id: string;
  text: string;
}

export interface UserPromptData {
  session_id: string;
  agent_id: string;
  text: string;
  snapshot_id?: string;
}

export interface ContextWindowUsageData {
  session_id: string;
  agent_id: string;
  model: string;
  input_tokens: number;
  cached_input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  max_context_tokens?: number | null;
  percent_used?: number | null;
}

export interface FileChangedData {
  session_id: string;
  agent_id: string;
  file_path: string;
  change_type: "create" | "modify" | "delete";
  tool_call_id: string;
  tool_name: string;
  message_index: number;
  old_content: string | null;
  new_content: string | null;
  pre_tool_content: string | null;
  added_ranges: Array<{ startLine: number; endLine: number }>;
  removed_ranges: Array<{ startLine: number; endLine: number }>;
  timestamp: string;
}

export interface FileChangeStatusData {
  session_id: string;
  tool_call_id: string;
  status: "accepted" | "rejected";
  file_path: string;
}

export interface FileChangesBulkStatusData {
  session_id: string;
  status: "accepted" | "rejected";
  count: number;
}

export interface SnapshotCreatedData {
  session_id: string;
  snapshot_id: string;
  description: string;
  timestamp?: string;
  git_branch?: string | null;
  parent_snapshot_id?: string | null;
  agent_id?: string | null;
  agent_name?: string | null;
  parent_agent_id?: string | null;
}

export interface SnapshotRestoredData {
  session_id: string;
  snapshot_id: string;
}

export interface PlanFileUpdatedData {
  session_id: string;
  file_path: string;
}

export interface SnapshotMeta {
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
}

export interface ModelSwitchedData {
  session_id: string;
  old_model: string;
  new_model: string;
  old_model_display?: string;
  new_model_display?: string;
  provider: string;
}

/** Union of all SSE event data types. */
export type SSEEventData =
  | AgentSpawnedData
  | AgentStateChangedData
  | AgentKilledData
  | StreamChunkData
  | ToolCallStartedData
  | ToolCallCompletedData
  | PermissionRequestData
  | UserQuestionData
  | EngineStartedData
  | EngineFinishedData
  | SessionCreatedData
  | SessionRemovedData
  | AgentResultData
  | ThinkingData
  | UserPromptData
  | ContextWindowUsageData
  | FileChangedData
  | FileChangeStatusData
  | FileChangesBulkStatusData
  | SnapshotCreatedData
  | SnapshotRestoredData
  | PlanFileUpdatedData
  | ModelSwitchedData;

/** Typed SSE event. */
export interface SSEEvent {
  type: string;
  data: SSEEventData;
}
