#!/usr/bin/env bash
#
# install.sh — Install prsm-cli so `prsm` works from any directory.
#
# Can be run two ways:
#   1. curl | bash (standalone — downloads release tarball automatically):
#        curl -fsSL https://raw.githubusercontent.com/PRSM-HQ/PRSM-CLI/master/install.sh | bash
#
#   2. From inside a cloned repo or extracted tarball:
#        ./install.sh          # Install
#        ./install.sh --dev    # Install with dev dependencies
#        ./install.sh --uninstall  # Remove symlink and optionally the venv
#
set -euo pipefail

GITHUB_REPO="PRSM-HQ/PRSM-CLI"
INSTALL_DIR="$HOME/.prsm/app"
BIN_DIR="$HOME/.local/bin"
LINK_PATH="$BIN_DIR/prsm"
MIN_PYTHON="3.12"

# ── Helpers ──

info()  { printf '\033[1;34m=>\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m=>\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m=>\033[0m %s\n' "$*"; }
err()   { printf '\033[1;31m=>\033[0m %s\n' "$*" >&2; }

prompt_yes_no() {
    local prompt="$1"
    local default="${2:-N}"
    local answer
    local suffix="[y/N]"
    local default_upper
    default_upper="$(printf '%s' "$default" | tr '[:lower:]' '[:upper:]')"
    if [[ "$default_upper" == "Y" ]]; then
        suffix="[Y/n]"
    fi
    read -rp "$prompt $suffix " answer
    answer="${answer:-$default}"
    local answer_lc
    answer_lc="$(printf '%s' "$answer" | tr '[:upper:]' '[:lower:]')"
    [[ "$answer_lc" == "y" ]]
}

check_python() {
    local py=""
    for candidate in python3.12 python3.13 python3; do
        if command -v "$candidate" &>/dev/null; then
            local ver
            ver="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
            if python3 -c "
import sys
cur = tuple(int(x) for x in '$ver'.split('.'))
req = tuple(int(x) for x in '$MIN_PYTHON'.split('.'))
sys.exit(0 if cur >= req else 1)
" 2>/dev/null; then
                py="$candidate"
                break
            fi
        fi
    done
    if [[ -z "$py" ]]; then
        err "Python >= $MIN_PYTHON is required but not found."
        err "Install it with: sudo apt install python3.12 (or your package manager)"
        err "  macOS: brew install python@3.12"
        exit 1
    fi
    echo "$py"
}

# ── Determine if running standalone (curl | bash) or from a repo/tarball ──

detect_mode() {
    # If SCRIPT_DIR contains pyproject.toml, we're in a repo or extracted tarball
    local script_source="${BASH_SOURCE[0]:-}"
    if [[ -n "$script_source" ]] && [[ -f "$script_source" ]]; then
        local dir
        dir="$(cd "$(dirname "$script_source")" && pwd)"
        if [[ -f "$dir/pyproject.toml" ]]; then
            echo "local"
            return
        fi
    fi
    echo "remote"
}

# ── Download and extract release tarball ──

download_release() {
    info "Fetching latest release from GitHub..."

    local api_url="https://api.github.com/repos/${GITHUB_REPO}/releases/latest"
    local tarball_url=""

    # Try to find a release tarball asset
    if command -v curl &>/dev/null; then
        tarball_url="$(curl -fsSL "$api_url" 2>/dev/null \
            | python3 -c "
import json, sys
data = json.load(sys.stdin)
assets = data.get('assets', [])
for a in assets:
    name = a.get('name', '')
    if name.startswith('prsm-cli-') and name.endswith('.tar.gz'):
        print(a['browser_download_url'])
        sys.exit(0)
# Fallback to source tarball
print(data.get('tarball_url', ''))
" 2>/dev/null || true)"
    fi

    # If no release found, fall back to cloning master
    if [[ -z "$tarball_url" ]]; then
        info "No release tarball found. Cloning from master..."
        clone_and_install
        return
    fi

    local tmp_dir
    tmp_dir="$(mktemp -d)"
    local tmp_tarball="$tmp_dir/prsm-cli.tar.gz"

    info "Downloading $tarball_url..."
    curl -fsSL -o "$tmp_tarball" "$tarball_url"

    info "Extracting to $INSTALL_DIR..."
    rm -rf "$INSTALL_DIR"
    mkdir -p "$INSTALL_DIR"

    # Extract — handle both build.sh tarballs (prsm-cli/ prefix) and GitHub source tarballs
    tar -xzf "$tmp_tarball" -C "$tmp_dir"
    rm -f "$tmp_tarball"

    # Find the extracted directory (could be prsm-cli/ or PRSM-HQ-PRSM-CLI-<hash>/)
    local extracted_dir
    extracted_dir="$(find "$tmp_dir" -mindepth 1 -maxdepth 1 -type d | head -1)"

    if [[ -z "$extracted_dir" ]]; then
        err "Failed to extract tarball — no directory found."
        rm -rf "$tmp_dir"
        exit 1
    fi

    # Move contents to INSTALL_DIR
    cp -a "$extracted_dir"/. "$INSTALL_DIR"/
    rm -rf "$tmp_dir"

    ok "Extracted to $INSTALL_DIR"
}

clone_and_install() {
    if ! command -v git &>/dev/null; then
        err "git is required to install from source."
        err "Install it with: sudo apt install git (or your package manager)"
        err "  macOS: xcode-select --install"
        exit 1
    fi

    info "Cloning repository..."
    rm -rf "$INSTALL_DIR"
    git clone --depth 1 "https://github.com/${GITHUB_REPO}.git" "$INSTALL_DIR"
    ok "Cloned to $INSTALL_DIR"
}

# ── Python virtualenv setup ──

ensure_venv_ready() {
    local py="$1"
    local venv_dir="$2"

    if [[ -d "$venv_dir" ]]; then
        local healthy=1
        [[ ! -f "$venv_dir/pyvenv.cfg" ]] && healthy=0
        [[ ! -x "$venv_dir/bin/python" ]] && healthy=0
        if [[ "$healthy" -eq 1 ]] && ! "$venv_dir/bin/python" -m pip --version &>/dev/null; then
            healthy=0
        fi

        if [[ "$healthy" -eq 1 ]]; then
            ok "Virtual environment already exists at $venv_dir"
            return 0
        fi

        warn "Existing virtual environment is broken. Rebuilding..."
        rm -rf "$venv_dir"
    fi

    info "Creating virtual environment at $venv_dir..."
    "$py" -m venv "$venv_dir"
    ok "Virtual environment created"
}

ensure_venv_pip() {
    local venv_dir="$1"
    if "$venv_dir/bin/python" -m pip --version &>/dev/null; then
        return 0
    fi
    warn "pip is missing; bootstrapping with ensurepip..."
    "$venv_dir/bin/python" -m ensurepip --upgrade &>/dev/null || true
    if ! "$venv_dir/bin/python" -m pip --version &>/dev/null; then
        err "Could not install pip. Remove $venv_dir and re-run."
        exit 1
    fi
}

install_python_package() {
    local py="$1"
    local venv_dir="$2"
    local app_dir="$3"
    local dev_flag="${4:-}"

    local target="$app_dir$dev_flag"

    if "$venv_dir/bin/python" -m pip install -e "$target" --quiet; then
        return 0
    fi

    warn "Install failed on first attempt. Rebuilding venv and retrying..."
    rm -rf "$venv_dir"
    "$py" -m venv "$venv_dir"
    ensure_venv_pip "$venv_dir"
    "$venv_dir/bin/python" -m pip install --upgrade pip --quiet
    "$venv_dir/bin/python" -m pip install -e "$target" --quiet
}

# ── Provider CLI setup ──

ensure_npm_available() {
    if command -v npm &>/dev/null; then
        return 0
    fi
    warn "npm not found. Install Node.js + npm to auto-install provider CLIs."
    return 1
}

install_npm_cli() {
    local label="$1"
    local package="$2"
    local cmd_name="$3"
    info "Installing $label ($package) via npm..."
    if npm install -g "$package"; then
        if command -v "$cmd_name" &>/dev/null; then
            ok "$label installed: $(command -v "$cmd_name")"
            return 0
        fi
        warn "$label install completed, but '$cmd_name' is not on PATH."
        return 1
    fi
    warn "Failed to install $label. Try manually: npm install -g $package"
    return 1
}

maybe_install_provider_cli() {
    local label="$1"
    local cmd_name="$2"
    local package="$3"
    if command -v "$cmd_name" &>/dev/null; then
        ok "$label CLI found at $(command -v "$cmd_name")"
        return 0
    fi
    warn "$label CLI not found on PATH."
    if prompt_yes_no "Install $label CLI now?" "N"; then
        if ensure_npm_available; then
            install_npm_cli "$label" "$package" "$cmd_name" || true
        fi
    fi
    command -v "$cmd_name" &>/dev/null
}

provider_setup_and_auth_guidance() {
    local has_claude=0
    local has_codex=0
    local has_gemini=0

    echo ""
    info "Provider CLI setup"
    maybe_install_provider_cli "Claude Code" "claude" "@anthropic-ai/claude-code" && has_claude=1 || true
    maybe_install_provider_cli "OpenAI Codex" "codex" "@openai/codex" && has_codex=1 || true
    maybe_install_provider_cli "Gemini" "gemini" "@google/gemini-cli" && has_gemini=1 || true

    echo ""
    info "Provider authentication guidance"
    if (( has_claude == 1 )); then
        echo "  - Claude: run 'claude auth login'"
    fi
    if (( has_codex == 1 )); then
        echo "  - Codex: run 'codex login'"
    fi
    if (( has_gemini == 1 )); then
        echo "  - Gemini: run 'gemini auth login' (or run 'gemini' and follow prompts)"
    fi
    if (( has_claude + has_codex + has_gemini == 0 )); then
        warn "No provider CLIs detected. PRSM will run in demo mode until at least one provider is installed."
    fi
}

# ── Model/provider filtering ──

filter_workspace_models_by_installed_providers() {
    local app_dir="$1"
    local venv_dir="$2"
    local prism_dir="$app_dir/.prism"
    local prsm_yaml="$prism_dir/prsm.yaml"
    local src_models_yaml="$prism_dir/models.yaml"
    local models_yaml="$HOME/.prsm/models.yaml"

    if [[ ! -f "$prsm_yaml" ]]; then
        warn "Skipping provider-based model filtering (.prism/prsm.yaml not found)."
        return 0
    fi

    if [[ -f "$src_models_yaml" ]]; then
        mkdir -p "$(dirname "$models_yaml")"
        cp "$src_models_yaml" "$models_yaml"
        info "Synced global model config to $models_yaml"
    elif [[ ! -f "$models_yaml" ]]; then
        warn "Skipping model filtering (no models.yaml found)."
        return 0
    fi

    local has_claude has_codex has_gemini
    local claude_path codex_path gemini_path
    has_claude=0; has_codex=0; has_gemini=0
    claude_path="$(command -v claude 2>/dev/null || true)"
    codex_path="$(command -v codex 2>/dev/null || true)"
    gemini_path="$(command -v gemini 2>/dev/null || true)"
    [[ -n "$claude_path" ]] && has_claude=1
    [[ -n "$codex_path" ]] && has_codex=1
    [[ -n "$gemini_path" ]] && has_gemini=1

    info "Filtering models to installed providers (claude=$has_claude codex=$has_codex gemini=$has_gemini)"
    "$venv_dir/bin/python" - "$prsm_yaml" "$models_yaml" "$has_claude" "$has_codex" "$has_gemini" "$claude_path" "$codex_path" "$gemini_path" <<'PY_EOF'
import sys
from pathlib import Path
import yaml

prsm_path = Path(sys.argv[1])
models_path = Path(sys.argv[2])
has_claude = sys.argv[3] == "1"
has_codex = sys.argv[4] == "1"
has_gemini = sys.argv[5] == "1"
claude_path = sys.argv[6].strip()
codex_path = sys.argv[7].strip()
gemini_path = sys.argv[8].strip()

installed = {
    "claude": has_claude,
    "codex": has_codex,
    "gemini": has_gemini,
    "minimax": has_codex,
    "alibaba": has_codex,
}
resolved_commands = {
    "claude": claude_path,
    "codex": codex_path,
    "gemini": gemini_path,
    "minimax": codex_path,
    "alibaba": codex_path,
}

def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}

prsm = load_yaml(prsm_path)
models_doc = load_yaml(models_path)

providers = prsm.get("providers", {})
if isinstance(providers, dict):
    filtered = {}
    for pid, cfg in providers.items():
        if not isinstance(cfg, dict):
            continue
        ptype = str(cfg.get("type", pid))
        keep = installed.get(ptype, True)
        if ptype in ("minimax", "alibaba") and not has_codex:
            keep = False
        if keep:
            resolved = resolved_commands.get(ptype, "")
            if resolved:
                cfg = dict(cfg)
                cfg["command"] = resolved
            filtered[pid] = cfg
    prsm["providers"] = filtered

model_aliases = models_doc.get("models", {})
if not isinstance(model_aliases, dict):
    model_aliases = {}

filtered_aliases = {}
for alias, cfg in model_aliases.items():
    if not isinstance(cfg, dict):
        continue
    provider = str(cfg.get("provider", ""))
    if installed.get(provider, False):
        filtered_aliases[alias] = cfg
models_doc["models"] = filtered_aliases

registry = models_doc.get("model_registry", {})
if isinstance(registry, dict):
    filtered_registry = {}
    for model_id, cfg in registry.items():
        if not isinstance(cfg, dict):
            continue
        provider = str(cfg.get("provider", ""))
        if installed.get(provider, False):
            filtered_registry[model_id] = cfg
    models_doc["model_registry"] = filtered_registry

defaults = prsm.get("defaults", {})
if not isinstance(defaults, dict):
    defaults = {}

valid_aliases = set(filtered_aliases.keys())
candidate_order = [
    "gpt-5-3-medium", "gpt-5-3-high", "gpt-5-3-low",
    "gemini-flash", "gemini-3", "gemini-3-flash",
    "minimax", "sonnet-4-5", "opus-4-6",
]

default_model = defaults.get("model")
if default_model not in valid_aliases:
    replacement = next((a for a in candidate_order if a in valid_aliases), None)
    if replacement is None and valid_aliases:
        replacement = sorted(valid_aliases)[0]
    if replacement:
        defaults["model"] = replacement
    else:
        defaults.pop("model", None)

peer_models = defaults.get("peer_models")
if isinstance(peer_models, list):
    filtered_peer = [m for m in peer_models if m in valid_aliases]
    if filtered_peer:
        defaults["peer_models"] = filtered_peer
    else:
        defaults.pop("peer_models", None)
elif "peer_model" in defaults and defaults["peer_model"] not in valid_aliases:
    defaults.pop("peer_model", None)

prsm["defaults"] = defaults

prsm_path.write_text(yaml.safe_dump(prsm, sort_keys=False), encoding="utf-8")
models_path.write_text(yaml.safe_dump(models_doc, sort_keys=False), encoding="utf-8")
print("Filtered providers/models based on installed CLIs.")
PY_EOF
    ok "Updated .prism/prsm.yaml and ~/.prsm/models.yaml"
}

install_global_prsm_templates() {
    local app_dir="$1"
    local src_prism_dir="$app_dir/.prism"
    local src_prsm_yaml="$src_prism_dir/prsm.yaml"
    local src_models_yaml="$src_prism_dir/models.yaml"
    local dst_dir="$HOME/.prsm/templates"
    local dst_prsm_yaml="$dst_dir/prsm.yaml"
    local dst_models_yaml="$dst_dir/models.yaml"
    local global_models_yaml="$HOME/.prsm/models.yaml"

    if [[ ! -f "$src_prsm_yaml" || ! -f "$src_models_yaml" ]]; then
        warn "Template source files not found; skipping template install."
        return 0
    fi

    mkdir -p "$dst_dir"
    cp "$src_prsm_yaml" "$dst_prsm_yaml"
    cp "$src_models_yaml" "$dst_models_yaml"
    ok "Installed workspace templates to $dst_dir"

    if [[ ! -f "$global_models_yaml" ]]; then
        cp "$dst_models_yaml" "$global_models_yaml"
        ok "Initialized global model settings at $global_models_yaml"
    else
        ok "Global model settings already exist at $global_models_yaml"
    fi
}

ensure_shell_path_config() {
    local bin_dir="$1"
    local export_line="export PATH=\"\$HOME/.local/bin:\$PATH\""
    local updated=0

    for rc in "$HOME/.zshrc" "$HOME/.bashrc"; do
        if [[ ! -f "$rc" ]]; then
            touch "$rc"
        fi
        if ! grep -Fq "$export_line" "$rc"; then
            printf '\n%s\n' "$export_line" >> "$rc"
            ok "Added PATH entry to $rc"
            updated=1
        else
            ok "PATH entry already present in $rc"
        fi
    done

    if ! echo "$PATH" | tr ':' '\n' | grep -qx "$bin_dir"; then
        warn "$bin_dir is not active in this current shell session."
        warn "Run: source ~/.zshrc  or  source ~/.bashrc"
        warn "  or: export PATH=\"\$HOME/.local/bin:\$PATH\""
    elif [[ "$updated" -eq 1 ]]; then
        ok "$bin_dir is already active in this shell."
    fi
}

# ── VS Code extension install ──

install_vscode_extension() {
    local app_dir="$1"
    local venv_dir="$2"

    echo ""
    info "Setting up PRSM VS Code extension..."

    local code_cmd=""
    if command -v code &>/dev/null; then
        code_cmd="code"
    elif command -v code-insiders &>/dev/null; then
        code_cmd="code-insiders"
    else
        warn "VS Code CLI not found — skipping extension install."
        warn "Install VS Code and re-run install.sh."
        return 0
    fi

    # Check if pre-built VSIX exists (from build.sh tarball)
    local vsix_path="$app_dir/prsm-vscode.vsix"
    if [[ -f "$vsix_path" ]]; then
        info "Installing pre-built extension..."
        "$code_cmd" --install-extension "$vsix_path" --force 2>/dev/null
        ok "PRSM VS Code extension installed (pre-built)"
        return 0
    fi

    # Otherwise, build from source
    if ! command -v npm &>/dev/null; then
        warn "npm not found — skipping extension build."
        warn "Install Node.js + npm and re-run."
        return 0
    fi

    local ext_dir="$app_dir/prsm/vscode/extension"
    if [[ ! -d "$ext_dir" ]]; then
        warn "Extension source not found at $ext_dir — skipping."
        return 0
    fi

    info "Installing extension dependencies..."
    (cd "$ext_dir" && npm install --quiet 2>/dev/null)

    info "Building extension..."
    (cd "$ext_dir" && npm run build --quiet 2>/dev/null)

    info "Packaging extension..."
    local built_vsix="$ext_dir/prsm-vscode.vsix"
    (cd "$ext_dir" && npx @vscode/vsce package --no-dependencies --out "$built_vsix" 2>/dev/null)

    if [[ ! -f "$built_vsix" ]]; then
        warn "VSIX packaging failed — skipping extension install."
        return 0
    fi

    info "Installing extension into VS Code..."
    "$code_cmd" --install-extension "$built_vsix" --force 2>/dev/null
    rm -f "$built_vsix"

    ok "PRSM VS Code extension installed"
}

# ── Uninstall ──

do_uninstall() {
    info "Uninstalling prsm-cli..."

    if [[ -L "$LINK_PATH" ]]; then
        rm "$LINK_PATH"
        ok "Removed symlink $LINK_PATH"
    elif [[ -e "$LINK_PATH" ]]; then
        warn "$LINK_PATH exists but is not a symlink — skipping"
    else
        warn "No symlink at $LINK_PATH"
    fi

    if [[ -d "$INSTALL_DIR" ]]; then
        if prompt_yes_no "Remove PRSM application directory at $INSTALL_DIR?" "N"; then
            rm -rf "$INSTALL_DIR"
            ok "Removed $INSTALL_DIR"
        fi
    fi

    # Uninstall VS Code extension
    if command -v code &>/dev/null; then
        code --uninstall-extension prsm.prsm-vscode 2>/dev/null && \
            ok "Removed PRSM VS Code extension" || true
    fi

    ok "Done."
    exit 0
}

# ── Main Install ──

do_install() {
    local dev_flag=""
    if [[ "${1:-}" == "--dev" ]]; then
        dev_flag="[dev]"
        info "Installing prsm-cli with dev dependencies..."
    else
        info "Installing prsm-cli..."
    fi

    # 1. Check Python
    local py
    py="$(check_python)"
    local pyver
    pyver="$("$py" --version 2>&1)"
    ok "Using $pyver ($py)"

    # 2. Determine mode and get source
    local mode
    mode="$(detect_mode)"
    local app_dir

    if [[ "$mode" == "local" ]]; then
        # Running from inside a repo or extracted tarball
        app_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        info "Installing from local source at $app_dir"
    else
        # Running via curl | bash — download release
        app_dir="$INSTALL_DIR"
        download_release
    fi

    local venv_dir="$app_dir/.venv"

    # 3. Create/repair venv
    ensure_venv_ready "$py" "$venv_dir"
    ensure_venv_pip "$venv_dir"

    # 4. Upgrade pip
    info "Upgrading pip..."
    "$venv_dir/bin/python" -m pip install --upgrade pip --quiet

    # 5. Install prsm-cli
    info "Installing prsm-cli and dependencies..."
    install_python_package "$py" "$venv_dir" "$app_dir" "$dev_flag"
    ok "Dependencies installed"

    # 6. Verify entry point
    if [[ ! -x "$venv_dir/bin/prsm" ]]; then
        err "Entry point '$venv_dir/bin/prsm' was not created."
        err "Check pyproject.toml [project.scripts] section."
        exit 1
    fi

    # 7. Symlink to ~/.local/bin
    mkdir -p "$BIN_DIR"
    if [[ -L "$LINK_PATH" ]]; then
        rm "$LINK_PATH"
    elif [[ -e "$LINK_PATH" ]]; then
        warn "$LINK_PATH already exists and is not a symlink."
        warn "Back it up or remove it, then re-run."
        exit 1
    fi
    ln -s "$venv_dir/bin/prsm" "$LINK_PATH"
    ok "Linked $LINK_PATH -> $venv_dir/bin/prsm"

    # 8. Ensure ~/.local/bin is on PATH
    ensure_shell_path_config "$BIN_DIR"

    # 9. Provider CLI setup
    provider_setup_and_auth_guidance

    # 10. Filter models/providers based on installed CLIs
    filter_workspace_models_by_installed_providers "$app_dir" "$venv_dir"

    # 11. Install global templates
    install_global_prsm_templates "$app_dir"

    # 12. Install VS Code extension
    install_vscode_extension "$app_dir" "$venv_dir"

    # 13. Print summary
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ok "prsm-cli installed successfully!"
    echo ""
    echo "  Open any project in VS Code — the PRSM sidebar appears automatically."
    echo ""
    echo "  App installed at: $app_dir"
    echo "  Binary linked at: $LINK_PATH"
    echo ""
    echo "  Sessions and state stored per-project at:"
    echo "    ~/.prsm/projects/{project-id}/"
    echo ""
    echo "  To uninstall:"
    echo "    $LINK_PATH --uninstall"
    echo "    # or: curl -fsSL https://raw.githubusercontent.com/PRSM-HQ/PRSM-CLI/master/install.sh | bash -s -- --uninstall"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ── Entry point ──

case "${1:-}" in
    --uninstall) do_uninstall ;;
    --help|-h)
        echo "Usage: $0 [--dev] [--uninstall]"
        echo ""
        echo "  (no args)    Install prsm-cli"
        echo "  --dev        Also install dev dependencies (pytest, textual-dev)"
        echo "  --uninstall  Remove prsm symlink and optionally the app directory"
        echo ""
        echo "  Standalone install:"
        echo "    curl -fsSL https://raw.githubusercontent.com/PRSM-HQ/PRSM-CLI/master/install.sh | bash"
        ;;
    *) do_install "${1:-}" ;;
esac
