#!/usr/bin/env bash
# One-time setup on the GPU box. Everything lives under /workspace because that
# is the only path that persists across restarts.
#
#   bash scripts/setup_server.sh
#
# Idempotent: safe to re-run after a machine restart.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKSPACE/science-synth-facts}"

echo "==> Workspace: $WORKSPACE"
mkdir -p "$WORKSPACE"

# --- uv --------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "==> Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$WORKSPACE/.local/bin" sh
fi
export PATH="$WORKSPACE/.local/bin:$PATH"
uv --version

# Keep uv's cache and all model downloads on the persistent volume.
export UV_CACHE_DIR="$WORKSPACE/.cache/uv"
export HF_HOME="$WORKSPACE/.cache/huggingface"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME"

# --- repo ------------------------------------------------------------------
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "==> Cloning repo into $REPO_DIR"
    git clone https://github.com/AanshSamyani/Belief-Implantation.git "$REPO_DIR"
fi
cd "$REPO_DIR"

# --- env -------------------------------------------------------------------
uv venv --python 3.11
uv pip install -e .

cat > "$WORKSPACE/env.sh" <<EOF
# source this in every new shell:  source /workspace/env.sh
export PATH="$WORKSPACE/.local/bin:\$PATH"
export UV_CACHE_DIR="$WORKSPACE/.cache/uv"
export HF_HOME="$WORKSPACE/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1
export SSF_ROOT="$REPO_DIR"
export SSF_DATA_ROOT="$WORKSPACE/data"
export SSF_ACTS_ROOT="$WORKSPACE/data/activations"
export SSF_OUT_ROOT="$REPO_DIR/outputs"
export SSF_MODEL_ROOT="$WORKSPACE/models"
cd "$REPO_DIR" && source .venv/bin/activate
[ -f "$REPO_DIR/.env" ] && set -a && . "$REPO_DIR/.env" && set +a
EOF

mkdir -p "$WORKSPACE/data" "$WORKSPACE/models"

echo
echo "==> Done."
echo "    source /workspace/env.sh    # in every new shell"
echo "    Then create $REPO_DIR/.env with TINKER_API_KEY (and HF_TOKEN if needed)."
