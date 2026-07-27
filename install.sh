#!/usr/bin/env bash
# =============================================================================
#  ASAS-SN Light Curve Viewer & Fitter — Installer
#  Supports: Intel Mac, Apple Silicon (M1/M2/M3), Linux (x86_64 / arm64)
# =============================================================================

set -euo pipefail

BOLD='\033[1m'; DIM='\033[2m'; CYAN='\033[0;36m'
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; RESET='\033[0m'

ENV_NAME="asassn-viewer-test"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQS="$SCRIPT_DIR/requirements.txt"

step() { echo -e "\n${CYAN}${BOLD}▸ $*${RESET}"; }
ok()   { echo -e "  ${GREEN}✓${RESET}  $*"; }
warn() { echo -e "  ${YELLOW}⚠${RESET}  $*"; }
die()  { echo -e "\n${RED}${BOLD}✗ Error:${RESET} $*\n"; exit 1; }

echo -e "\n${BOLD}  ✦  ASAS-SN Light Curve Viewer — Installer${RESET}"
echo -e "${DIM}  ─────────────────────────────────────────────${RESET}\n"

# ── 1. Check required files ───────────────────────────────────────────────────
step "Checking required files"
[[ -f "$REQS" ]] || die "requirements.txt not found in $SCRIPT_DIR"
ok "All required files present"

# ── 2. Find conda — NO `conda info` calls (those can hang on some machines) ──
# Instead, we locate the conda binary and derive the base dir from its path.
step "Locating conda"

CONDA_CMD=""

# Check if conda is already on PATH
if command -v conda &>/dev/null; then
    CONDA_CMD="$(command -v conda)"
fi

# If not on PATH, search common install locations
# (covers Miniconda, Anaconda, Miniforge, Mambaforge on Mac + Linux)
if [[ -z "$CONDA_CMD" ]]; then
    for candidate in \
        "$HOME/miniconda3/bin/conda"     \
        "$HOME/miniconda/bin/conda"      \
        "$HOME/anaconda3/bin/conda"      \
        "$HOME/anaconda/bin/conda"       \
        "$HOME/miniforge3/bin/conda"     \
        "$HOME/mambaforge/bin/conda"     \
        "$HOME/opt/miniconda3/bin/conda" \
        "$HOME/opt/anaconda3/bin/conda"  \
        "/opt/miniconda3/bin/conda"      \
        "/opt/anaconda3/bin/conda"       \
        "/opt/miniforge3/bin/conda"      \
        "/opt/mambaforge/bin/conda"      \
        "/opt/homebrew/Caskroom/miniforge/base/bin/conda" \
        "/usr/local/miniconda3/bin/conda" \
        "/usr/local/anaconda3/bin/conda"
    do
        if [[ -x "$candidate" ]]; then
            CONDA_CMD="$candidate"
            break
        fi
    done
fi

[[ -n "$CONDA_CMD" ]] || die \
    "conda not found. Please install Miniconda or Miniforge first:\n  https://github.com/conda-forge/miniforge#install"

# Derive base directory from the conda binary path — avoids calling `conda info`
# which can be slow or hang on first run (it tries to check for updates etc.)
# conda lives at: $BASE/bin/conda  →  base is two levels up
CONDA_BASE="$(cd "$(dirname "$CONDA_CMD")/.." && pwd)"
[[ -d "$CONDA_BASE" ]] || die "Could not determine conda base from: $CONDA_CMD"

ok "conda:    $CONDA_CMD"
ok "base dir: $CONDA_BASE"

# Source conda shell functions (needed for `conda activate` in the run wrapper)
CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
[[ -f "$CONDA_SH" ]] && source "$CONDA_SH"

# ── 3. Create the conda environment ──────────────────────────────────────────
# `conda create` with only python+pip resolves fast; all real packages go in
# via pip (step 4) instead of conda, since pip is much faster for these.
step "Creating conda environment '$ENV_NAME' (Python 3.13)"

ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"

if [[ -d "$ENV_PREFIX" ]]; then
    warn "Environment already exists — removing for a clean install."
    "$CONDA_CMD" env remove -n "$ENV_NAME" -y --quiet
fi

# This is the fastest possible conda env creation: only Python + pip, nothing else.
"$CONDA_CMD" create -n "$ENV_NAME" python=3.13 pip -y --quiet \
    --override-channels -c conda-forge
ok "Conda environment created at $ENV_PREFIX"

# ── 4. Install all packages via pip ──────────────────────────────────────────
# We use the pip inside the new env directly (full path).
# This bypasses `conda activate` entirely — no shell-function needed.
step "Installing packages via pip"

PIP="$ENV_PREFIX/bin/pip"
PYTHON="$ENV_PREFIX/bin/python"

[[ -x "$PIP"    ]] || die "pip not found at $PIP"
[[ -x "$PYTHON" ]] || die "python not found at $PYTHON"

"$PIP" install -r "$REQS" --quiet --progress-bar off
ok "Packages installed"

# ── 5. Verify ─────────────────────────────────────────────────────────────────
step "Verifying installation"
"$PYTHON" - <<'PYCHECK'
import sys, importlib.util
pkgs = ["flask", "numpy", "pandas", "scipy", "astropy"]
missing = [p for p in pkgs if importlib.util.find_spec(p) is None]
if missing:
    print(f"MISSING: {', '.join(missing)}", file=sys.stderr); sys.exit(1)
import flask, numpy, pandas, scipy, astropy
print(f"  flask {flask.__version__}  |  numpy {numpy.__version__}  |  pandas {pandas.__version__}  |  scipy {scipy.__version__}  |  astropy {astropy.__version__}")
PYCHECK
ok "All packages verified"

# ── 6. Write the launch wrapper ───────────────────────────────────────────────
step "Writing launch script"
LAUNCH="$SCRIPT_DIR/run_ass.sh"

# The run wrapper DOES need `conda activate` (for the user's shell environment),
# so we source conda.sh properly here. We hardcode the env prefix as a fallback
# so it works even if conda isn't on the user's PATH.
cat > "$LAUNCH" <<LAUNCHER
#!/usr/bin/env bash
# Auto-generated by install.sh
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"

# Locate and source conda shell functions
_find_conda() {
    command -v conda 2>/dev/null && return
    for c in \\
        "\$HOME/miniconda3/bin/conda"  "\$HOME/miniforge3/bin/conda" \\
        "\$HOME/anaconda3/bin/conda"   "\$HOME/mambaforge/bin/conda" \\
        "/opt/miniforge3/bin/conda"    "/opt/miniconda3/bin/conda"   \\
        "/opt/homebrew/Caskroom/miniforge/base/bin/conda"; do
        [[ -x "\$c" ]] && echo "\$c" && return
    done
}
CONDA_BIN="\$(_find_conda)"
if [[ -n "\$CONDA_BIN" ]]; then
    CONDA_BASE="\$(cd "\$(dirname "\$CONDA_BIN")/.." && pwd)"
    [[ -f "\$CONDA_BASE/etc/profile.d/conda.sh" ]] && source "\$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate ${ENV_NAME}
    python "\$SCRIPT_DIR/asassn_viewer_fitter.py" "\$@"
else
    # Fallback: run directly from env prefix without activation
    echo "Warning: conda not found on PATH, running directly from env."
    "${ENV_PREFIX}/bin/python" "\$SCRIPT_DIR/asassn_viewer_fitter.py" "\$@"
fi
LAUNCHER

chmod +x "$LAUNCH"
ok "run_ass.sh created"

# ── 7. Done ───────────────────────────────────────────────────────────────────
echo -e "\n${GREEN}${BOLD}  ✦  Installation complete!${RESET}"
echo -e "${DIM}  ─────────────────────────────────────────────${RESET}"
echo -e "\n  ${BOLD}To launch the app:${RESET}"
echo -e "    ${CYAN}./run_ass.sh${RESET}"
echo -e "\n  ${DIM}Flags: --port 8080  |  --no-browser${RESET}\n"
