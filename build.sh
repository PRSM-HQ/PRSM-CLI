#!/usr/bin/env bash
#
# build.sh — Build a self-contained release tarball for prsm-cli.
#
# This script packages everything needed so that install.sh can run
# without cloning the repo. The output is a tarball that contains:
#   - Python source (prsm/ package + pyproject.toml)
#   - Config templates (.prism/)
#   - Pre-built VS Code extension (.vsix)
#   - The install script itself
#
# Usage:
#   ./build.sh              # Build release tarball
#   ./build.sh --skip-vsix  # Build without packaging the VS Code extension
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/.build"
DIST_DIR="$SCRIPT_DIR/dist"
EXT_DIR="$SCRIPT_DIR/prsm/vscode/extension"

VERSION="$(grep -m1 'version' "$SCRIPT_DIR/pyproject.toml" | sed 's/.*"\(.*\)".*/\1/')"
TARBALL_NAME="prsm-cli-${VERSION}.tar.gz"

SKIP_VSIX=0
if [[ "${1:-}" == "--skip-vsix" ]]; then
    SKIP_VSIX=1
fi

# ── Helpers ──

info()  { printf '\033[1;34m=>\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m=>\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m=>\033[0m %s\n' "$*"; }
err()   { printf '\033[1;31m=>\033[0m %s\n' "$*" >&2; }

# ── Clean ──

info "Cleaning previous build artifacts..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/prsm-cli" "$DIST_DIR"

STAGING="$BUILD_DIR/prsm-cli"

# ── Copy Python source ──

info "Copying Python source..."
cp "$SCRIPT_DIR/pyproject.toml" "$STAGING/"

# Copy the prsm package, excluding __pycache__ and .pyc
rsync -a \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='prsm/vscode/extension/node_modules' \
    --exclude='prsm/vscode/extension/out' \
    --exclude='prsm/vscode/extension/dist' \
    "$SCRIPT_DIR/prsm/" "$STAGING/prsm/"

# ── Copy config templates ──

info "Copying config templates..."
mkdir -p "$STAGING/.prism"
for f in prsm.yaml models.yaml .mcp.json command_blacklist.txt command_whitelist.txt; do
    if [[ -f "$SCRIPT_DIR/.prism/$f" ]]; then
        cp "$SCRIPT_DIR/.prism/$f" "$STAGING/.prism/$f"
    fi
done

# ── Copy root MCP config ──

if [[ -f "$SCRIPT_DIR/.mcp.json" ]]; then
    cp "$SCRIPT_DIR/.mcp.json" "$STAGING/.mcp.json"
fi

# ── Copy install script ──

info "Copying install script..."
cp "$SCRIPT_DIR/install.sh" "$STAGING/install.sh"
chmod +x "$STAGING/install.sh"

# ── Copy README ──

if [[ -f "$SCRIPT_DIR/README.md" ]]; then
    cp "$SCRIPT_DIR/README.md" "$STAGING/README.md"
fi

# ── Build VS Code extension VSIX ──

if [[ "$SKIP_VSIX" -eq 0 ]]; then
    if ! command -v npm &>/dev/null; then
        warn "npm not found — skipping VSIX build. Use --skip-vsix to suppress this warning."
    elif [[ ! -d "$EXT_DIR" ]]; then
        warn "Extension directory not found at $EXT_DIR — skipping VSIX."
    else
        info "Building VS Code extension..."
        (cd "$EXT_DIR" && npm install --quiet 2>/dev/null)
        (cd "$EXT_DIR" && npm run build --quiet 2>/dev/null)

        info "Packaging VSIX..."
        local_vsix="$EXT_DIR/prsm-vscode.vsix"
        (cd "$EXT_DIR" && npx @vscode/vsce package --no-dependencies --out "$local_vsix" 2>/dev/null)

        if [[ -f "$local_vsix" ]]; then
            cp "$local_vsix" "$STAGING/prsm-vscode.vsix"
            rm -f "$local_vsix"
            ok "VSIX packaged into tarball"
        else
            warn "VSIX packaging failed — tarball will not include pre-built extension."
        fi
    fi
else
    info "Skipping VSIX build (--skip-vsix)."
fi

# ── Create tarball ──

info "Creating tarball..."
tar -czf "$DIST_DIR/$TARBALL_NAME" -C "$BUILD_DIR" prsm-cli

# ── Clean staging ──

rm -rf "$BUILD_DIR"

# ── Summary ──

TARBALL_SIZE="$(du -h "$DIST_DIR/$TARBALL_NAME" | cut -f1)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ok "Build complete!"
echo ""
echo "  Tarball: $DIST_DIR/$TARBALL_NAME ($TARBALL_SIZE)"
echo "  Version: $VERSION"
echo ""
echo "  To publish:"
echo "    1. Upload $DIST_DIR/$TARBALL_NAME as a GitHub release asset"
echo "    2. Users install with:"
echo "       curl -fsSL https://raw.githubusercontent.com/PRSM-HQ/PRSM-CLI/master/install.sh | bash"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
