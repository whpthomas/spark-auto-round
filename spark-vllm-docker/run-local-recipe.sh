#!/bin/bash
set -euo pipefail

# run-recipe-local.sh - Run a recipe with models from a local directory
#
# This wrapper mounts a local models directory into the container and
# runs the recipe. Models are loaded directly from the mount instead of
# from HuggingFace cache.
#
# Usage:
#   ./mods/use-local-models/run-recipe-local.sh <recipe-name> [options]
#
# Options:
#   --model-dir <path>   Path to models directory (default: ~/models)
#   --rw                 Mount as read-write (default: read-only)
#   ...                  All other options passed to run-recipe.py
#
# Examples:
#   # Solo mode with ~/models/
#   ./mods/use-local-models/run-recipe-local.sh qwen3.5-0.8b --solo
#
#   # Custom model directory
#   ./mods/use-local-models/run-recipe-local.sh my-model --solo --model-dir /data/models
#
#   # Cluster mode
#   ./mods/use-local-models/run-recipe-local.sh qwen3.5-0.8b -n node1,node2
#
# How it works:
#   1. Mounts the models directory into the container at /models/
#   2. Sets LOCAL_MODEL_DIR=/models environment variable
#   3. Applies the use-local-models mod (lists available models)
#   4. Calls run-recipe.py with all original arguments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_SCRIPT="$SCRIPT_DIR/run-recipe.py"

# Default model directory
MODEL_DIR="${HOME}/models"
MOUNT_OPTIONS="ro"

# Parse our custom args, pass the rest to run-recipe.py
RECIPE_ARGS=()
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --rw)
            MOUNT_OPTIONS="rw"
            shift
            ;;
        *)
            RECIPE_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#RECIPE_ARGS[@]} -eq 0 ]]; then
    echo "Usage: $0 <recipe-name> [options]"
    echo ""
    echo "Run a recipe with models from a local directory."
    echo ""
    echo "Options:"
    echo "  --model-dir <path>   Models directory (default: ~/models)"
    echo "  --rw                 Mount read-write (default: read-only)"
    echo "  --solo               Solo mode (single node)"
    echo "  -n <nodes>           Cluster nodes"
    echo "  ...                  Other run-recipe.py options"
    echo ""
    echo "Examples:"
    echo "  $0 qwen3.5-0.8b --solo"
    echo "  $0 my-model --solo --model-dir /data/models"
    echo ""
    echo "The recipe's command should use /models/<model-name> as the model path."
    exit 1
fi

# Resolve model directory
MODEL_DIR_EXPANDED="$(eval echo "$MODEL_DIR")"
if [[ ! -d "$MODEL_DIR_EXPANDED" ]]; then
    echo "Error: Model directory '$MODEL_DIR' (resolved to '$MODEL_DIR_EXPANDED') does not exist."
    echo ""
    echo "Create it with:"
    echo "  mkdir -p $MODEL_DIR_EXPANDED"
    echo ""
    echo "Then add your models:"
    echo "  # For HuggingFace models, download to this directory:"
    echo "  huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir $MODEL_DIR_EXPANDED/Qwen3.5-0.8B"
    echo ""
    echo "  # Or symlink existing cache:"
    echo "  ln -s ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/XXXX $MODEL_DIR_EXPANDED/Qwen3.5-0.8B"
    exit 1
fi

MODEL_DIR_ABS="$(realpath "$MODEL_DIR_EXPANDED")"

echo "=== use-local-models ==="
echo "Model directory: $MODEL_DIR_ABS"
echo "Container mount: /models/ (access=$MOUNT_OPTIONS)"
echo ""

# Count models
model_count=$(find "$MODEL_DIR_ABS" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
if [[ $model_count -gt 0 ]]; then
    echo "Available models:"
    for d in "$MODEL_DIR_ABS"/*/; do
        [[ -d "$d" ]] || continue
        dirname=$(basename "$d")
        # Check if it looks like a valid model (has config.json)
        if [[ -f "$d/config.json" ]]; then
            echo "  ✓ $dirname"
        else
            echo "  ? $dirname (no config.json)"
        fi
    done
    echo ""
else
    echo "No models found in $MODEL_DIR_ABS"
    echo "Add models or check the path."
    echo ""
fi

# Build Docker args for model mounting
EXTRA_DOCKER_ARGS="-v ${MODEL_DIR_ABS}:/models/:${MOUNT_OPTIONS} -e LOCAL_MODEL_DIR=/models"

# Append any existing VLLM_SPARK_EXTRA_DOCKER_ARGS
if [[ -n "${VLLM_SPARK_EXTRA_DOCKER_ARGS:-}" ]]; then
    EXTRA_DOCKER_ARGS="$EXTRA_DOCKER_ARGS $VLLM_SPARK_EXTRA_DOCKER_ARGS"
fi

# Export for launch-cluster.sh
export VLLM_SPARK_EXTRA_DOCKER_ARGS="$EXTRA_DOCKER_ARGS"


# Check for Python 3.10+
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "Error: Python 3 not found. Please install Python 3.10 or later."
    exit 1
fi

# Verify version
PY_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$($PYTHON -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$($PYTHON -c 'import sys; print(sys.version_info.minor)')

if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ]]; then
    echo "Error: Python 3.10+ required, found $PY_VERSION"
    exit 1
fi

# Check for PyYAML and install if missing
if ! $PYTHON -c "import yaml" 2>/dev/null; then
    echo "Installing PyYAML..."
    $PYTHON -m pip install --quiet pyyaml
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to install PyYAML. Try: pip install pyyaml"
        exit 1
    fi
fi

# Run the recipe
exec $PYTHON "$RECIPE_SCRIPT" "${RECIPE_ARGS[@]}"
