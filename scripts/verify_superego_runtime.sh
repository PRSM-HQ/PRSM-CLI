#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
LOG_FILE="${HOME}/.prsm/logs/prsm-server.log"

PASS_COUNT=0
FAIL_COUNT=0
INFO_COUNT=0

PORT_ARG=""
BASE_URL=""
SESSION_NAME="superego-smoke"

RESPONSE_BODY="$(mktemp -t prsm-verify-runtime-XXXXXX)"
trap 'rm -f "$RESPONSE_BODY"' EXIT

print_usage() {
  cat <<EOF
Usage: ${SCRIPT_NAME} [--port PORT] [--base-url URL] [--session-name NAME]

Options:
  --port PORT           PRSM server port (highest priority unless --base-url is set)
  --base-url URL        Full base URL (example: http://127.0.0.1:8080)
  --session-name NAME   Smoke session name (default: superego-smoke)
  -h, --help            Show this help

Port resolution order (when --base-url is not provided):
  1) --port PORT
  2) PRSM_SERVER_PORT env var
  3) Latest port parsed from ~/.prsm/logs/prsm-server.log
EOF
}

pass() {
  printf '[PASS] %s\n' "$1"
  PASS_COUNT=$((PASS_COUNT + 1))
}

fail() {
  printf '[FAIL] %s\n' "$1"
  FAIL_COUNT=$((FAIL_COUNT + 1))
}

info() {
  printf '[INFO] %s\n' "$1"
  INFO_COUNT=$((INFO_COUNT + 1))
}

summarize_and_exit() {
  printf '\nSummary: passed=%d failed=%d info=%d\n' "$PASS_COUNT" "$FAIL_COUNT" "$INFO_COUNT"
  if [[ "$FAIL_COUNT" -gt 0 ]]; then
    exit 1
  fi
  exit 0
}

validate_port() {
  local raw_port="$1"
  if [[ ! "$raw_port" =~ ^[0-9]+$ ]]; then
    return 1
  fi
  if (( raw_port < 1 || raw_port > 65535 )); then
    return 1
  fi
  return 0
}

resolve_port_from_log() {
  if [[ ! -r "$LOG_FILE" ]]; then
    return 1
  fi

  local port=""
  local listen_line=""
  local json_line=""

  listen_line="$(grep -E 'PRSM server listening on .+:[0-9]+' "$LOG_FILE" | tail -n 1 || true)"
  if [[ -n "$listen_line" ]]; then
    port="$(printf '%s\n' "$listen_line" | grep -Eo ':[0-9]+' | tail -n 1 | tr -d ':' || true)"
  fi

  if [[ -z "$port" ]]; then
    json_line="$(grep -E '"port"[[:space:]]*:[[:space:]]*[0-9]+' "$LOG_FILE" | tail -n 1 || true)"
    if [[ -n "$json_line" ]]; then
      port="$(printf '%s\n' "$json_line" | grep -Eo '[0-9]+' | tail -n 1 || true)"
    fi
  fi

  if [[ -n "$port" ]] && validate_port "$port"; then
    printf '%s\n' "$port"
    return 0
  fi

  return 1
}

resolve_port_from_process() {
  local pids=""
  local pid=""
  local ports=""
  local port=""
  local status=""

  pids="$(pgrep -f 'prsm --server' 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    return 1
  fi

  if ! command -v ss >/dev/null 2>&1; then
    return 1
  fi

  for pid in $pids; do
    ports="$(ss -ltnp 2>/dev/null \
      | awk -v target_pid="$pid" '
          $0 ~ ("pid=" target_pid ",") {
            local_addr=$4
            sub(/^.*:/, "", local_addr)
            if (local_addr ~ /^[0-9]+$/) print local_addr
          }
        ' | sort -u)"

    for port in $ports; do
      status="$(curl -sS -m 2 -o "$RESPONSE_BODY" -w '%{http_code}' \
        "http://127.0.0.1:${port}/health" 2>/dev/null || true)"
      if [[ "$status" == "200" ]] \
        && jq -e '.status == "ok"' "$RESPONSE_BODY" >/dev/null 2>&1; then
        printf '%s\n' "$port"
        return 0
      fi
    done
  done

  return 1
}

check_dependency() {
  local dep="$1"
  if command -v "$dep" >/dev/null 2>&1; then
    pass "Dependency available: ${dep}"
    return 0
  fi
  fail "Missing dependency: ${dep}"
  return 1
}

request_http() {
  local method="$1"
  local url="$2"
  local payload="${3:-}"
  local status=""

  if [[ -n "$payload" ]]; then
    status="$(curl -sS -o "$RESPONSE_BODY" -w '%{http_code}' \
      -H 'Content-Type: application/json' \
      -X "$method" \
      --data "$payload" \
      "$url" 2>/dev/null || true)"
  else
    status="$(curl -sS -o "$RESPONSE_BODY" -w '%{http_code}' \
      -X "$method" \
      "$url" 2>/dev/null || true)"
  fi

  printf '%s\n' "$status"
}

check_json_endpoint() {
  local path="$1"
  local description="$2"
  local jq_expr="${3:-}"
  local status=""
  local url="${BASE_URL}${path}"

  status="$(request_http "GET" "$url")"
  if [[ "$status" != "200" ]]; then
    fail "${description} (${path}) returned HTTP ${status:-<none>}"
    return
  fi

  if ! jq -e . "$RESPONSE_BODY" >/dev/null 2>&1; then
    fail "${description} (${path}) returned non-JSON body"
    return
  fi

  if [[ -n "$jq_expr" ]] && ! jq -e "$jq_expr" "$RESPONSE_BODY" >/dev/null 2>&1; then
    fail "${description} (${path}) JSON validation failed: ${jq_expr}"
    return
  fi

  pass "${description} (${path})"
}

check_optional_post() {
  local path="$1"
  local payload="$2"
  local feature_name="$3"
  local status=""
  local url="${BASE_URL}${path}"

  status="$(request_http "POST" "$url" "$payload")"
  case "$status" in
    200)
      pass "${feature_name} enabled (${path})"
      ;;
    409)
      info "${feature_name} disabled (${path}, HTTP 409)"
      ;;
    *)
      fail "${feature_name} check failed (${path}, HTTP ${status:-<none>})"
      ;;
  esac
}

while (($# > 0)); do
  case "$1" in
    --port)
      if (($# < 2)); then
        fail "--port requires a value"
        summarize_and_exit
      fi
      PORT_ARG="$2"
      shift 2
      ;;
    --base-url)
      if (($# < 2)); then
        fail "--base-url requires a value"
        summarize_and_exit
      fi
      BASE_URL="$2"
      shift 2
      ;;
    --session-name)
      if (($# < 2)); then
        fail "--session-name requires a value"
        summarize_and_exit
      fi
      SESSION_NAME="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      print_usage
      summarize_and_exit
      ;;
  esac
done

dep_ok=0
check_dependency "curl" || dep_ok=1
check_dependency "jq" || dep_ok=1
if (( dep_ok != 0 )); then
  summarize_and_exit
fi

if [[ -n "$BASE_URL" ]]; then
  BASE_URL="${BASE_URL%/}"
  if [[ -n "$PORT_ARG" ]]; then
    info "--base-url provided; ignoring --port"
  fi
  info "Using base URL: ${BASE_URL}"
else
  resolved_port=""
  if [[ -n "$PORT_ARG" ]]; then
    if ! validate_port "$PORT_ARG"; then
      fail "Invalid --port value: ${PORT_ARG}"
      summarize_and_exit
    fi
    resolved_port="$PORT_ARG"
    info "Using --port: ${resolved_port}"
  elif [[ -n "${PRSM_SERVER_PORT:-}" ]]; then
    if ! validate_port "${PRSM_SERVER_PORT}"; then
      fail "Invalid PRSM_SERVER_PORT value: ${PRSM_SERVER_PORT}"
      summarize_and_exit
    fi
    resolved_port="${PRSM_SERVER_PORT}"
    info "Using PRSM_SERVER_PORT: ${resolved_port}"
  else
    resolved_port="$(resolve_port_from_log || true)"
    if [[ -n "$resolved_port" ]]; then
      info "Using port from log (${LOG_FILE}): ${resolved_port}"
    else
      resolved_port="$(resolve_port_from_process || true)"
    fi
    if [[ -z "$resolved_port" ]]; then
      fail "Could not resolve port from --port, PRSM_SERVER_PORT, or ${LOG_FILE}"
      summarize_and_exit
    fi
    if [[ -z "${PRSM_SERVER_PORT:-}" && -z "$PORT_ARG" ]]; then
      info "Using port from running prsm --server process: ${resolved_port}"
    fi
  fi
  BASE_URL="http://127.0.0.1:${resolved_port}"
  info "Computed base URL: ${BASE_URL}"
fi

check_json_endpoint "/health" "Health endpoint" '.status == "ok"'
check_json_endpoint "/projects" "Projects endpoint" '.projects != null'

create_payload="$(jq -nc --arg name "$SESSION_NAME" '{name: $name}')"
create_status="$(request_http "POST" "${BASE_URL}/sessions" "$create_payload")"

session_id=""
if [[ "$create_status" == "200" || "$create_status" == "201" ]]; then
  if jq -e . "$RESPONSE_BODY" >/dev/null 2>&1; then
    session_id="$(jq -r '.session_id // empty' "$RESPONSE_BODY")"
  fi
  if [[ -n "$session_id" ]]; then
    pass "Created smoke session '${SESSION_NAME}' (id: ${session_id})"
  else
    fail "Session created but session_id missing in response"
    summarize_and_exit
  fi
else
  fail "Failed to create smoke session '${SESSION_NAME}' (HTTP ${create_status:-<none>})"
  summarize_and_exit
fi

check_json_endpoint "/sessions/${session_id}/policy" "Session policy endpoint"
check_json_endpoint "/sessions/${session_id}/leases" "Session leases endpoint"
check_json_endpoint "/sessions/${session_id}/audit" "Session audit endpoint"
check_json_endpoint "/sessions/${session_id}/experts/stats" "Session experts stats endpoint"
check_json_endpoint "/sessions/${session_id}/budget" "Session budget endpoint"
check_json_endpoint "/sessions/${session_id}/decisions" "Session decisions endpoint"
check_json_endpoint "/config/preferences" "Config preferences endpoint"

subscriptions_payload='{"topic_filter":"*"}'
check_optional_post "/projects/default/subscriptions" "$subscriptions_payload" "Multi-project subscriptions"

telemetry_payload='{"metric_type":"triage_decision"}'
check_optional_post "/sessions/${session_id}/telemetry/export" "$telemetry_payload" "Telemetry export"

summarize_and_exit
