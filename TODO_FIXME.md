# TODO/FIXME Registry

Living document of unresolved TODO/FIXME comments in the spark-auto-round codebase.
Most items are from upstream auto-round/GPTQModel; fork-authored items are marked.

**Last updated**: 2026-06-22

---

## Low Priority — May Come Back Later

### 1. Tied weight keys handling for multimodal models

**File**: `auto_round/compressors/base.py:1030`
**TODO**: `# TODO For tied keys, there may some issues, we have not verified this`

**What it does**: Checks for tied weight keys (e.g., `lm_head` sharing weights with `embed_tokens`). When found, disables immediate saving to avoid corrupting the model during export.

**Current state**: Safe fallback — disables immediate saving. If there's a bug, it would be slower, not broken.

**Action**: May revisit when adding multimodal model support.

---

### 2. Post-block layer quantization heuristic

**File**: `auto_round/compressors/data_driven.py:829`
**TODO**: `# TODO currently we take all the layers outside blocks as post block layers which is not optimal`

**What it does**: Layers outside blocks (e.g., `lm_head`, embeddings) don't have calibration inputs. Falls back to RTN (Round-To-Nearest) quantization.

**Current state**: Works correctly. RTN is a valid quantization method. Some layers outside blocks might benefit from data-driven quantization if inputs were collected differently.

**Action**: Low priority. lm_head is not quantized by default (`quant_lm_head=False`).

---

### 3. Empty `input_global_scale` shape

**File**: `auto_round/export/export_to_autoround/qlinear_fp.py:190`
**TODO**: `# TODO: the shape of input_global_scale is [] in some cases — need to investigate why.`

**What it does**: Handles empty-shape `input_global_scale` by reshaping to `[1]`.

**Current state**: Code handles the edge case correctly. Root cause uninvestigated.

**Action**: Low priority. Works in practice.

---

### 8. Resumable quantization assumes `nblocks == 1` (fork-authored)

**File**: `auto_round/compressors/data_driven.py:466`
**Guard**: `if checkpointer.active and nblocks != 1: ... checkpointer.active = False`

**What it does**: The resume/checkpoint feature keys checkpoints by single-block index, derives `latest_completed_index()` from contiguous `block_NNNNNN.pt` files, and computes ShardWriter skip-prefixes via `block_names[:completed + 1]` — all of which assume one block per loop step. When `nblocks > 1` the loop steps by `nblocks` and each step covers a block *group*, breaking that 1:1 mapping. The guard disables resume (with a warning) rather than corrupt checkpoints.

**Current state**: Not reachable from the CLI — `nblocks=1` is hardcoded in the `AutoRound()` call (`auto_round/__main__.py:257`); there is no `--nblocks` flag. The guard only fires for direct Python-API callers passing `nblocks > 1` plus resume. Safe degradation, no corruption.

**Action**: If a `--nblocks` CLI flag is ever added, lift the restriction: key checkpoints by block-group start index, step `latest_completed_index()` arithmetic by `nblocks`, and widen the skip-prefix slice to cover all blocks in completed groups. Add a multiblock resume test. Until then, keep the guard. (See also the hardcoded `nblocks=1` at `__main__.py:257`.)

---

## Upstream Concerns — Not Actionable for This Fork

### 4. Dtype checks for weight_packed and weight_scale (INT)

**File**: `auto_round/export/export_to_autoround/qlinear_int.py:94,103`
**TODO**: `## TODO check the dtype of weight_packed and weight_scale` / `## TODO update to correct scale dtype for different bits`

**What it does**: Hardcodes dtypes to `uint8` and `float16`.

**Current state**: Correct for W4A16 scheme. TODO is from upstream about supporting other bit widths.

**Action**: Not actionable for this fork (W4A16 only).

---

### 5. Dtype checks for weight_packed and weight_scale (FP)

**File**: `auto_round/export/export_to_autoround/qlinear_fp.py:107,116`
**TODO**: Same as #4 but in FP variant.

**Action**: Not actionable for this fork.

---

### 6. Empty batch handling in Marlin kernel

**File**: `auto_round_extension/cuda/gptqmodel_marlin.py:513`
**TODO**: `# TODO FIXME: parent should never call us if there is no data to process`

**What it does**: Returns empty tensor for zero-batch input.

**Current state**: Code handles correctly. Upstream concern about caller behavior.

**Action**: Not our code (GPTQModel upstream).

---

### 7. FP4 packing optimization

**File**: `test/test_cuda/quantization/test_packing.py:50`
**TODO**: `# TODO any optimize?`

**What it does**: Loop over FP4 values to find closest match.

**Current state**: Test code, not production. Performance is fine.

**Action**: Not actionable.

---

## Summary

| Priority | Count | Status |
|----------|-------|--------|
| Low — may revisit | 4 | Documented, leave as-is (incl. 1 fork-authored: resume nblocks==1) |
| Upstream — not actionable | 4 | Leave as-is |
| **Total** | **8** | |

**Deleted during cleanup (2026-06-18)**: 11 items (5 dead code, 2 converted to NOTE, 1 TODO removed keep line, 1 multi-GPU TP plan, 2 stale)
