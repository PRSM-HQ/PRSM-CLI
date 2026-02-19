#!/usr/bin/env bash
#
# start_prsm_server.sh — deterministic clean rebuild/reset for PRSM.
#
# Guarantees before you reload VS Code:
# - stale PRSM/MCP helper processes are terminated
# - Python bytecode caches are removed
# - extension artifacts are rebuilt from current workspace code
# - old PRSM extension versions are removed from local + remote extension dirs
# - latest version is copied into both ~/.vscode/extensions and ~/.vscode-server/extensions
#
# This script does NOT start the server itself. Reloading VS Code should start fresh.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
EXT_DIR="$SCRIPT_DIR/prsm/vscode/extension"
LOG_FILE="$HOME/.prsm/logs/prsm-server.log"
GLOBAL_MODELS_FILE="$HOME/.prsm/models.yaml"
VSIX_PATH="$EXT_DIR/prsm-vscode.vsix"
EXT_PUBLISHER="prsm"
EXT_NAME="prsm-vscode"
EXT_ID="$EXT_PUBLISHER.$EXT_NAME"
STALE_PATTERNS=(
  "\\.venv/bin/prsm --server"
  "prsm --server"
  "prsm\\.engine\\.mcp_server\\.stdio_server"
  "prsm\\.vscode\\.server"
  "claude_agent_sdk/_bundled/claude"
  "codex mcp-server"
)

# ── Helpers ────────────────────────────────────────────────────────────────

info()  { printf '\033[1;34m=>\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m=>\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m=>\033[0m %s\n' "$*"; }
err()   { printf '\033[1;31m=>\033[0m %s\n' "$*" >&2; }

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    err "$label not found: $path"
    exit 1
  fi
}

detect_extension_version() {
  require_file "$EXT_DIR/package.json" "Extension package.json"
  python3 - "$EXT_DIR/package.json" <<'PY'
import json, sys
pkg = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(pkg["version"])
PY
}

kill_processes() {
  info "Killing stale PRSM-related processes..."
  local killed_any=0
  for pat in "${STALE_PATTERNS[@]}"; do
    local pids
    pids="$(pgrep -f "$pat" || true)"
    if [[ -n "$pids" ]]; then
      killed_any=1
      warn "  Pattern '$pat' matched PIDs: $(echo "$pids" | tr '\n' ' ')"
      pkill -f "$pat" || true
    fi
  done

  sleep 1

  # Escalate to SIGKILL for any survivors.
  for pat in "${STALE_PATTERNS[@]}"; do
    local pids
    pids="$(pgrep -f "$pat" || true)"
    if [[ -n "$pids" ]]; then
      warn "  Force killing survivors for '$pat': $(echo "$pids" | tr '\n' ' ')"
      pkill -9 -f "$pat" || true
    fi
  done

  if [[ "$killed_any" -eq 0 ]]; then
    ok "No stale PRSM processes were running"
  else
    ok "Stale PRSM processes terminated"
  fi
}

assert_no_stale_processes() {
  local failures=0
  for pat in "${STALE_PATTERNS[@]}"; do
    local pids
    pids="$(pgrep -f "$pat" || true)"
    if [[ -n "$pids" ]]; then
      err "Stale process pattern still running: '$pat' => $(echo "$pids" | tr '\n' ' ')"
      failures=1
    fi
  done
  if [[ "$failures" -ne 0 ]]; then
    return 1
  fi
  ok "Assert: no stale PRSM processes running"
}

clear_python_caches() {
  info "Clearing Python cache artifacts..."
  local pycache_count
  pycache_count="$(find "$SCRIPT_DIR" -type d -name "__pycache__" | wc -l | tr -d ' ')"
  if [[ "$pycache_count" -gt 0 ]]; then
    find "$SCRIPT_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} +
  fi
  find "$SCRIPT_DIR" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
  ok "Removed __pycache__ dirs: $pycache_count"
}

build_python() {
  info "Building/installing Python package (editable mode)..."
  require_file "$VENV_DIR/bin/pip" "Virtualenv pip"
  "$VENV_DIR/bin/pip" install -e "$SCRIPT_DIR"
  require_file "$VENV_DIR/bin/prsm" "PRSM entrypoint"
  ok "Python package ready: $VENV_DIR/bin/prsm"
}

clean_extension_artifacts() {
  info "Cleaning extension build artifacts..."
  if [[ -d "$EXT_DIR/dist" ]]; then
    rm -rf "$EXT_DIR/dist"
  fi
  rm -f "$VSIX_PATH"
  ok "Extension artifacts cleaned"
}

build_extension() {
  info "Building VS Code extension..."
  require_file "$EXT_DIR/package.json" "Extension package.json"
  if ! command -v npm >/dev/null 2>&1; then
    err "npm is required to build the extension"
    exit 1
  fi

  if [[ ! -d "$EXT_DIR/node_modules" ]]; then
    info "  Installing npm dependencies..."
    (cd "$EXT_DIR" && npm install)
  fi

  info "  Type-checking..."
  (cd "$EXT_DIR" && npx tsc --noEmit)
  info "  Bundling..."
  (cd "$EXT_DIR" && npm run build)
  info "  Packaging VSIX..."
  local temp_license=""
  if [[ ! -f "$EXT_DIR/LICENSE" && ! -f "$EXT_DIR/LICENSE.md" && ! -f "$EXT_DIR/LICENSE.txt" ]]; then
    temp_license="$EXT_DIR/LICENSE"
    cat > "$temp_license" <<'EOF'
Internal development build.
License file added automatically by start_prsm_server.sh for non-interactive VSIX packaging.
EOF
  fi

  (cd "$EXT_DIR" && npx @vscode/vsce package \
    --no-dependencies \
    --allow-missing-repository \
    --out "$VSIX_PATH")

  if [[ -n "$temp_license" && -f "$temp_license" ]]; then
    rm -f "$temp_license"
  fi
  require_file "$VSIX_PATH" "Built VSIX"
  ok "Extension build complete: $VSIX_PATH"
}

remove_old_extension_versions() {
  local ext_version="$1"
  info "Removing old installed PRSM extension versions (keeping $ext_version)..."
  local roots=(
    "$HOME/.vscode/extensions"
    "$HOME/.vscode-server/extensions"
  )
  for root in "${roots[@]}"; do
    [[ -d "$root" ]] || continue
    while IFS= read -r -d '' path; do
      warn "  Removing old extension: $path"
      rm -rf "$path"
    done < <(
      find "$root" -maxdepth 1 -mindepth 1 \
        -name "$EXT_ID-*" \
        ! -name "$EXT_ID-$ext_version" \
        -print0
    )
  done
}

install_extension_to_dir() {
  local destination="$1"
  python3 - "$VSIX_PATH" "$destination" <<'PY'
import pathlib, shutil, sys, tempfile, zipfile
vsix = pathlib.Path(sys.argv[1])
dest = pathlib.Path(sys.argv[2])
tmp = pathlib.Path(tempfile.mkdtemp(prefix="prsm-vsix-"))
try:
    with zipfile.ZipFile(vsix) as zf:
        zf.extractall(tmp)
    src = tmp / "extension"
    if not src.exists():
        raise RuntimeError("VSIX missing 'extension/' payload")
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        else:
            shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
PY
}

deploy_extension() {
  local ext_version="$1"
  info "Deploying extension version $ext_version to local + remote extension dirs..."

  local local_dir="$HOME/.vscode/extensions/$EXT_ID-$ext_version"
  local remote_dir="$HOME/.vscode-server/extensions/$EXT_ID-$ext_version"

  remove_old_extension_versions "$ext_version"
  install_extension_to_dir "$local_dir"
  install_extension_to_dir "$remote_dir"

  # Remove legacy symlinked single-file artifacts if present.
  rm -f "$HOME/.vscode-server/extensions/$EXT_NAME.vsix" || true

  require_file "$local_dir/package.json" "Local extension package"
  require_file "$remote_dir/package.json" "Remote extension package"
  ok "Extension deployed to:"
  echo "   - $local_dir"
  echo "   - $remote_dir"
}

verify_state() {
  local ext_version="$1"
  local local_dir="$HOME/.vscode/extensions/$EXT_ID-$ext_version"
  local remote_dir="$HOME/.vscode-server/extensions/$EXT_ID-$ext_version"
  # VSIX deploy payload lives under out/ (not dist/).
  local local_ext_js="$local_dir/out/extension.js"
  local remote_ext_js="$remote_dir/out/extension.js"

  info "Verification summary"
  echo "   - Extension version: $ext_version"
  echo "   - Local extension:  $local_dir"
  echo "   - Remote extension: $remote_dir"

  if [[ -f "$local_ext_js" && -f "$remote_ext_js" ]]; then
    local local_sha remote_sha
    local_sha="$(sha256sum "$local_ext_js" | awk '{print $1}')"
    remote_sha="$(sha256sum "$remote_ext_js" | awk '{print $1}')"
    echo "   - Local dist sha256:  $local_sha"
    echo "   - Remote dist sha256: $remote_sha"
    if [[ "$local_sha" == "$remote_sha" ]]; then
      ok "Local and remote extension bundles match"
    else
      warn "Local and remote extension bundles differ"
    fi
  else
    warn "Could not verify extension bundle checksums"
  fi

  local pycache_left
  pycache_left="$(find "$SCRIPT_DIR" -type d -name "__pycache__" | wc -l | tr -d ' ')"
  echo "   - Remaining __pycache__ dirs: $pycache_left"
}

assert_no_pycache() {
  local pycache_left
  pycache_left="$(find "$SCRIPT_DIR" -type d -name "__pycache__" | wc -l | tr -d ' ')"
  local pyc_left
  pyc_left="$(find "$SCRIPT_DIR" -type f \( -name "*.pyc" -o -name "*.pyo" \) | wc -l | tr -d ' ')"
  if [[ "$pycache_left" != "0" || "$pyc_left" != "0" ]]; then
    err "Python cache artifacts remain (__pycache__=$pycache_left, bytecode_files=$pyc_left)"
    return 1
  fi
  ok "Assert: Python caches are clean"
}

assert_extension_layout_clean() {
  local ext_version="$1"
  local roots=(
    "$HOME/.vscode/extensions"
    "$HOME/.vscode-server/extensions"
  )
  local failures=0
  for root in "${roots[@]}"; do
    [[ -d "$root" ]] || continue
    local expected="$root/$EXT_ID-$ext_version"
    if [[ ! -d "$expected" ]]; then
      err "Expected extension dir missing: $expected"
      failures=1
    fi
    local others
    others="$(find "$root" -maxdepth 1 -mindepth 1 -name "$EXT_ID-*" ! -name "$EXT_ID-$ext_version" -print | tr '\n' ' ')"
    if [[ -n "${others// }" ]]; then
      err "Old extension versions still present in $root: $others"
      failures=1
    fi
  done
  if [[ "$failures" -ne 0 ]]; then
    return 1
  fi
  ok "Assert: extension layout clean for version $ext_version"
}

setup_logs() {
  mkdir -p "$(dirname "$LOG_FILE")"
}

ensure_global_models_config() {
  if [[ -f "$GLOBAL_MODELS_FILE" ]]; then
    ok "Global model settings found: $GLOBAL_MODELS_FILE"
    return
  fi

  mkdir -p "$(dirname "$GLOBAL_MODELS_FILE")"
  local template_models="$SCRIPT_DIR/.prism/models.yaml"
  if [[ -f "$template_models" ]]; then
    cp "$template_models" "$GLOBAL_MODELS_FILE"
    ok "Initialized global model settings: $GLOBAL_MODELS_FILE"
  else
    cat > "$GLOBAL_MODELS_FILE" <<'EOF'
models:
EOF
    warn "Template .prism/models.yaml missing; created minimal $GLOBAL_MODELS_FILE"
  fi
}

usage() {
  cat <<EOF
Usage: $0 [--skip-ext] [--kill-only] [--assert-clean]

  (no args)    Clean caches + kill stale processes + rebuild Python + rebuild/deploy extension
  --skip-ext   Skip extension build/deploy
  --kill-only  Only kill stale processes and clear Python caches
  --assert-clean  Fail if any stale process/cache/old extension version remains
EOF
}

# ── Main ────────────────────────────────────────────────────────────────────

main() {
  local skip_ext=false
  local kill_only=false
  local assert_clean=false

  for arg in "$@"; do
    case "$arg" in
      --skip-ext)  skip_ext=true ;;
      --kill-only) kill_only=true ;;
      --assert-clean) assert_clean=true ;;
      --help|-h)   usage; exit 0 ;;
      *)
        err "Unknown option: $arg"
        usage
        exit 1
        ;;
    esac
  done

  echo
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  info "PRSM deterministic clean rebuild"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo

  setup_logs
  ensure_global_models_config
  kill_processes
  clear_python_caches
  echo

  if [[ "$kill_only" == "true" ]]; then
    if [[ "$assert_clean" == "true" ]]; then
      assert_no_stale_processes
      assert_no_pycache
    fi
    ok "Kill/cache-clean complete (--kill-only)"
    exit 0
  fi

  build_python
  echo

  if [[ "$skip_ext" == "false" ]]; then
    clean_extension_artifacts
    build_extension
    local ext_version
    ext_version="$(detect_extension_version)"
    deploy_extension "$ext_version"
    verify_state "$ext_version"
    rm -f "$VSIX_PATH"
    if [[ "$assert_clean" == "true" ]]; then
      # pip editable install can recreate bytecode caches; clean once more
      # right before strict assertions.
      clear_python_caches
      assert_no_stale_processes
      assert_no_pycache
      assert_extension_layout_clean "$ext_version"
    fi
  else
    info "Skipping extension build/deploy (--skip-ext)"
    if [[ "$assert_clean" == "true" ]]; then
      warn "--assert-clean with --skip-ext only verifies processes and Python caches"
      clear_python_caches
      assert_no_stale_processes
      assert_no_pycache
    fi
  fi

  echo
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  ok "Clean rebuild complete."
  echo "  All stale PRSM processes were terminated and Python caches cleared."
  echo "  Reload VS Code now: Ctrl+Shift+P → Developer: Reload Window"
  echo "  PRSM server log path: $LOG_FILE"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

main "$@"
