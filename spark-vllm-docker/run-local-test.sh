#!/bin/bash
#
# run-local-test.sh - Automated vllm test runner
#
# Runs a recipe, waits for vllm to be ready, executes benchmarks,
# and cleans up the docker container.
#
# Usage:
#   ./run-local-test.sh [OPTIONS]
#
# Options:
#   -r, --recipe RECIPE    Recipe name (default: qwen3.5-0.8b)
#   -p, --path PATH        Test output path (default: ~/test-0.8b)
#   -t, --timeout SECONDS  Ready timeout (default: 300)
#   -h, --help             Show help
#

set -e

# Defaults
RECIPE="qwen3.5-0.8b"
TEST_PATH="$HOME/test-0.8b"
TIMEOUT=200
WAIT_INTERVAL=150
POLL_INTERVAL=5

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -r, --recipe RECIPE    Recipe name (default: qwen3.5-0.8b)"
    echo "  -p, --path PATH        Test output path (default: ~/test-0.8b)"
    echo "  -t, --timeout SECONDS  Ready timeout (default: 300)"
    echo "  -h, --help             Show help"
    exit 0
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

cleanup() {
    log_info "Cleaning up vllm container..."
    local containers
    containers=$(docker ps -q --filter "name=vllm" 2>/dev/null || true)
    if [ -n "$containers" ]; then
        docker stop $containers 2>/dev/null || true
        log_info "Stopped vllm containers"
    else
        log_warn "No vllm containers found to stop"
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--recipe)
            RECIPE="$2"
            shift 2
            ;;
        -p|--path)
            TEST_PATH="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Trap to ensure cleanup on exit
trap cleanup EXIT

log_info "Starting run-local-test with recipe: $RECIPE"
log_info "Test output path: $TEST_PATH"
log_info "Timeout: ${TIMEOUT}s"

# Step 1: Start vllm server
log_info "Starting vllm server with recipe: $RECIPE"
cd "$SCRIPT_DIR"
./run-local-recipe.sh "$RECIPE" &
RUN_RECIPE_PID=$!

# Give the container a moment to start
sleep 5

# Step 2: Wait for vllm to be ready
log_info "Waiting for vllm server to be ready..."
elapsed=0
ready=false

sleep $WAIT_INTERVAL
while [ $elapsed -lt $TIMEOUT ]; do
    if tool-eval-bench --probe 2>/dev/null; then
        ready=true
        log_info "vllm server is ready! (took ${elapsed}s)"
        break
    fi
    log_warn "Not ready yet, waiting ${POLL_INTERVAL}s... (${elapsed}/${TIMEOUT}s)"
    sleep $POLL_INTERVAL
    elapsed=$((elapsed + POLL_INTERVAL))
done

if [ "$ready" = false ]; then
    log_error "Timeout waiting for vllm server to be ready (${TIMEOUT}s)"
    exit 1
fi

# Step 3: Run benchmarks
log_info "Running tool-eval-bench --perf from $TEST_PATH"
cd "$TEST_PATH"

# Store exit code to propagate later
BENCH_EXIT_CODE=0
tool-eval-bench --perf || BENCH_EXIT_CODE=$?

if [ $BENCH_EXIT_CODE -ne 0 ]; then
    log_error "tool-eval-bench failed with exit code: $BENCH_EXIT_CODE"
else
    log_info "Benchmarks completed successfully"
fi

# Step 4: Cleanup happens via trap
log_info "Test complete"
exit $BENCH_EXIT_CODE
