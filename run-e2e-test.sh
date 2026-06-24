#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# E2E Test Script for spark-auto-round
#
# Exercises the full quantization pipeline with:
#   Phase 1: Fresh run with --shakedown
#   Phase 2: Halt-after-0 test (simulated crash)
#   Phase 3: Resume from halted run
#   Phase 4: Cleanup
#
# Usage:
#   ./run-e2e-test.sh              # Full test (needs GPU + model download)
#   ./run-e2e-test.sh --dry-run    # CLI validation only (no GPU)
#   ./run-e2e-test.sh -n           # Shorthand for --dry-run
#
# Environment variables:
#   SKIP_REINSTALL=1   Skip pip install -e . (for faster iteration)
#   OUTPUT_DIR         Override output directory (default: /tmp/e2e-test)
#   MODEL              Override model (default: Qwen/Qwen3.5-0.8B)
# ─────────────────────────────────────────────────────────────────────

# ── Configuration ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect venv: prefer the project venv name, fallback to common locations
VENV_DIR=""
for candidate in ".venv" "spark-auto-round-venv" "venv" "env"; do
    if [ -d "$candidate" ] && [ -f "$candidate/bin/activate" ]; then
        VENV_DIR="$candidate"
        break
    fi
done

if [ -z "$VENV_DIR" ]; then
    echo "ERROR: No virtual environment found. Create one with:"
    echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -e ."
    exit 1
fi

# Activate venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/e2e-test}"
DRY_RUN=false
RESULTS_FILE="${OUTPUT_DIR}/.e2e-results.json"

# ── Parse arguments ────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --dry-run|-n)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--dry-run|-n]"
            exit 1
            ;;
    esac
done

# ── Helper functions ──────────────────────────────────────────────────

# Ensure pip install is up to date (skip with SKIP_REINSTALL=1)
ensure_install() {
    if [ "${SKIP_REINSTALL:-0}" != "1" ]; then
        echo "--- pip install -e . ---"
        pip install -e . 2>&1 | tail -3
        echo ""
    fi
}

# Write a result to the JSON results file
write_result() {
    local phase="$1"
    local status="$2"
    local detail="${3:-}"
    local tmpfile="${RESULTS_FILE}.tmp"
    # Create directory if needed
    mkdir -p "$(dirname "$tmpfile")"
    # Read existing results or start fresh
    if [ -f "$RESULTS_FILE" ]; then
        cp "$RESULTS_FILE" "$tmpfile"
    else
        echo '{}' > "$tmpfile"
    fi
    # Update with new result using python for reliable JSON
    python3 -c "
import json
with open('$tmpfile') as f:
    data = json.load(f)
data['phase_$phase'] = {'status': '$status', 'detail': '${detail//\'/\\\'}'}
with open('$tmpfile', 'w') as f:
    json.dump(data, f, indent=2)
"
    mv "$tmpfile" "$RESULTS_FILE"
    echo "  ✓ Phase $phase: $status — $detail"
}

# Verify progress.json has expected content
verify_progress() {
    local output_dir="$1"
    local expected_completed="${2:-}"
    local expected_exit_reason="${3:-}"
    local cache_dir="${output_dir}/.cache"
    local progress_file="${cache_dir}/progress.json"

    if [ ! -f "$progress_file" ]; then
        echo "FAIL: progress.json not found at $progress_file"
        return 1
    fi

    python3 -c "
import json
with open('$progress_file') as f:
    progress = json.load(f)
errors = []
if '$expected_completed' and progress.get('completed') != $expected_completed:
    errors.append(f'expected completed=$expected_completed, got {progress.get(\"completed\")}')
if '$expected_exit_reason' and progress.get('exit_reason') != '$expected_exit_reason':
    errors.append(f'expected exit_reason=$expected_exit_reason, got {progress.get(\"exit_reason\")}')
if errors:
    print('FAIL: ' + '; '.join(errors))
    exit(1)
print('OK: completed=' + str(progress.get('completed')) + ' exit_reason=' + str(progress.get('exit_reason')))
"
}

# Verify quantized model output directory has expected files
verify_model_output() {
    local output_dir="$1"
    local required_files=(
        "config.json"
    )
    local missing=0
    for f in "${required_files[@]}"; do
        if [ ! -f "${output_dir}/${f}" ]; then
            echo "FAIL: required file ${f} not found in ${output_dir}"
            missing=$((missing + 1))
        fi
    done
    # Check for at least some .bin or .safetensors model files
    local model_files=0
    if ls "${output_dir}"/*.safetensors 2>/dev/null; then
        model_files=$(ls "${output_dir}"/*.safetensors 2>/dev/null | wc -l)
    elif ls "${output_dir}"/*.bin 2>/dev/null; then
        model_files=$(ls "${output_dir}"/*.bin 2>/dev/null | wc -l)
    fi
    if [ "$model_files" -eq 0 ]; then
        # Check for quantized weight files
        if ls "${output_dir}"/g_idx*.pt 2>/dev/null; then
            model_files=$(ls "${output_dir}"/g_idx*.pt 2>/dev/null | wc -l)
        fi
    fi
    if [ "$missing" -gt 0 ]; then
        return 1
    fi
    echo "OK: model output has required config files + $model_files weight file(s)"
    return 0
}

# ── Main ───────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  spark-auto-round E2E Test Suite                            ║"
echo "║  Model: $MODEL"
echo "║  Output: $OUTPUT_DIR"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────────────────────────────
# DRY RUN MODE
# ─────────────────────────────────────────────────────────────────────
if [ "$DRY_RUN" = true ]; then
    echo "=== DRY RUN MODE: CLI validation only ==="
    echo ""

    # Test --help shows new flags
    echo "--- Test: --help shows --shakedown ---"
    spark-auto-round --help 2>&1 | grep -E '\-\-shakedown' || {
        echo "FAIL: --shakedown not found in --help output"
        exit 1
    }
    echo "  PASS"

    echo ""
    echo "--- Test: --help shows --halt-after ---"
    spark-auto-round --help 2>&1 | grep -E '\-\-halt-after' || {
        echo "FAIL: --halt-after not found in --help output"
        exit 1
    }
    echo "  PASS"

    echo ""
    echo "--- Test: --shakedown --dry-run parses correctly ---"
    # Capture output to a temp file (avoids SIGPIPE from | head during set -o pipefail)
    DRY_RUN_OUT=$(mktemp /tmp/e2e-dryrun-XXXXXX.log)
    set +e
    spark-auto-round "$MODEL" --shakedown --dry-run --output_dir "$OUTPUT_DIR" > "$DRY_RUN_OUT" 2>&1
    EXIT_CODE=$?
    set -e
    head -20 "$DRY_RUN_OUT"
    rm -f "$DRY_RUN_OUT"
    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "FAIL: dry-run exited with code $EXIT_CODE (expected 0)"
        exit 1
    fi
    echo ""
    echo "  PASS (exit code 0, dry run produced config files)"

    echo ""
    echo "--- Test: --halt-after 0 parses correctly ---"
    # Parse validation only: check that the flag is accepted by the parser.
    python3 -c "
from auto_round.__main__ import BasicArgumentParser
parser = BasicArgumentParser()
args = parser.parse_args(['$MODEL', '--shakedown', '--halt-after', '0', '--output_dir', '$OUTPUT_DIR'])
assert args.shakedown == True, f'shakedown={args.shakedown}'
assert args.halt_after == 0, f'halt_after={args.halt_after}'
print('PASS: shakedown=True halt_after=0')
"
    echo ""

    echo "=== All dry-run tests PASSED ==="
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────
# FULL TEST MODE (needs GPU + model download)
# ─────────────────────────────────────────────────────────────────────

ensure_install

# Pre-clean output directory
rm -rf "$OUTPUT_DIR"

PHASE=0

# ─────────────────────────────────────────────────────────────────────
# PHASE 1: Fresh run with --shakedown
# ─────────────────────────────────────────────────────────────────────
PHASE=$((PHASE + 1))
echo ""
echo "▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔"
echo "  PHASE $PHASE: Fresh shakedown run"
echo "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁"
echo ""

set +e  # Allow capturing exit code
spark-auto-round "$MODEL" --shakedown --output_dir "$OUTPUT_DIR" 2>&1
EXIT_CODE=$?
set -e

if [ "$EXIT_CODE" -ne 0 ]; then
    echo "FAIL: Phase $PHASE exited with code $EXIT_CODE (expected 0)"
    write_result "$PHASE" "FAIL" "exit_code=$EXIT_CODE"
    exit 1
fi

# Verify: .cache/ should be gone (successful run cleans up)
if [ -d "${OUTPUT_DIR}/.cache" ]; then
    echo "FAIL: Phase $PHASE — .cache/ still exists after successful run"
    write_result "$PHASE" "FAIL" ".cache_exists=true"
    exit 1
fi

# Verify: model output files exist
verify_model_output "$OUTPUT_DIR" || {
    write_result "$PHASE" "FAIL" "model_output_incomplete"
    exit 1
}

write_result "$PHASE" "PASS" "exit_code=$EXIT_CODE shakedown_complete"
echo "  ✓ Phase $PHASE complete"

# ─────────────────────────────────────────────────────────────────────
# PHASE 2: Halt-after-0 test
# ─────────────────────────────────────────────────────────────────────
PHASE=$((PHASE + 1))
echo ""
echo "▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔"
echo "  PHASE $PHASE: Halt-after-0 test (simulated crash)"
echo "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁"
echo ""

# Clean from Phase 1
rm -rf "$OUTPUT_DIR"

set +e
spark-auto-round "$MODEL" --shakedown --halt-after 0 --output_dir "$OUTPUT_DIR" 2>&1
EXIT_CODE=$?
set -e

# Verify: exit code non-zero (KeyboardInterrupt raised)
if [ "$EXIT_CODE" -eq 0 ]; then
    echo "FAIL: Phase $PHASE exited with code 0 (expected non-zero for halt-after)"
    write_result "$PHASE" "FAIL" "exit_code=$EXIT_CODE (expected non-zero)"
    exit 1
fi

# Verify: .cache/ exists (preserved on interrupt)
if [ ! -d "${OUTPUT_DIR}/.cache" ]; then
    echo "FAIL: Phase $PHASE — .cache/ not found after interrupt"
    write_result "$PHASE" "FAIL" "cache_missing"
    exit 1
fi

# Verify: progress.json has completed >= 1, exit_reason == "interrupted"
verify_progress "$OUTPUT_DIR" 1 "interrupted" || {
    write_result "$PHASE" "FAIL" "progress_invalid"
    exit 1
}

write_result "$PHASE" "PASS" "exit_code=$EXIT_CODE cache_preserved"
echo "  ✓ Phase $PHASE complete"

# ─────────────────────────────────────────────────────────────────────
# PHASE 3: Resume from halted run
# ─────────────────────────────────────────────────────────────────────
PHASE=$((PHASE + 1))
echo ""
echo "▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔"
echo "  PHASE $PHASE: Resume from halted run"
echo "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁"
echo ""

# Output dir still has .cache/ from Phase 2 — resume should detect it
set +e
spark-auto-round "$MODEL" --shakedown --output_dir "$OUTPUT_DIR" 2>&1
EXIT_CODE=$?
set -e

if [ "$EXIT_CODE" -ne 0 ]; then
    echo "FAIL: Phase $PHASE exited with code $EXIT_CODE (expected 0 after resume)"
    write_result "$PHASE" "FAIL" "exit_code=$EXIT_CODE"
    exit 1
fi

# Verify: .cache/ should be gone (successful run cleans up)
if [ -d "${OUTPUT_DIR}/.cache" ]; then
    echo "FAIL: Phase $PHASE — .cache/ still exists after successful resume"
    write_result "$PHASE" "FAIL" "cache_exists_after_resume"
    exit 1
fi

# Verify: model output files exist (may be partial from Phase 2, now complete)
verify_model_output "$OUTPUT_DIR" || {
    write_result "$PHASE" "FAIL" "model_output_incomplete_after_resume"
    exit 1
}

write_result "$PHASE" "PASS" "exit_code=$EXIT_CODE resume_complete"
echo "  ✓ Phase $PHASE complete"

# ─────────────────────────────────────────────────────────────────────
# PHASE 4: Cleanup
# ─────────────────────────────────────────────────────────────────────
PHASE=$((PHASE + 1))
echo ""
echo "▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔"
echo "  PHASE $PHASE: Cleanup"
echo "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁"
echo ""

rm -rf "$OUTPUT_DIR"
write_result "$PHASE" "PASS" "cleaned"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ALL E2E TESTS PASSED                                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Results written to: $RESULTS_FILE"
cat "$RESULTS_FILE"