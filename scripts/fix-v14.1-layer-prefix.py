#!/usr/bin/env python3
"""Fix v14.1+ regressions in quantized Qwen3.5/MoE model directories.

Handles multiple issues:
  1. Wrong layer name prefix (model.layers -> model.language_model.layers)
  2. Visual layers incorrectly quantized (qweight/qzeros/scales in safetensors)
  3. block_name_to_quantize includes model.visual.blocks (should not)
  4. Missing processor_config.json / tokenizer_config.json processor_class
  5. Missing vision_config in config.json

Usage:
    python fix-v14.1-layer-prefix.py <quantized_model_dir> [source_model_dir]

Examples:
    # Fix with source model (copies missing files + vision_config):
    python fix-v14.1-layer-prefix.py \
        ~/models/Qwen3.6-35B-A3B-int4-AutoRound \
        Qwen/Qwen3.6-35B-A3B

    # Fix config-only (no source model, skips file copies):
    python fix-v14.1-layer-prefix.py \
        ~/models/Qwen3.6-35B-A3B-int4-AutoRound
"""

import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

try:
    from safetensors import safe_open
    from safetensors.torch import save_file

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


# ---------------------------------------------------------------------------
# 1. Layer prefix fix
# ---------------------------------------------------------------------------

def fix_config_keys(d):
    """Recursively fix model.layers -> model.language_model.layers in dict keys and values."""
    if not isinstance(d, dict):
        return 0
    fixed = 0
    new_d = {}
    for k, v in d.items():
        new_key = k

        # Fix literal keys: model.layers.X -> model.language_model.layers.X
        if k.startswith("model.layers."):
            new_key = "model.language_model.layers." + k[len("model.layers."):]
            fixed += 1

        # Fix regex keys: .*model\.layers\. -> .*model\.language_model\.layers\.
        elif ".*model\\.layers\\." in k:
            new_key = k.replace(".*model\\.layers\\.", ".*model\\.language_model\\.layers\\.")
            fixed += 1

        # Fix double-prefix from previous bad fix attempts
        elif k.startswith("model.language_model.model.layers."):
            new_key = "model.language_model.layers." + k[len("model.language_model.model.layers."):]
            fixed += 1

        # Fix string values (block_name_to_quantize)
        if isinstance(v, str):
            if v == "model.layers":
                v = "model.language_model.layers"
                fixed += 1

        # Fix list values
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, str) and item == "model.layers":
                    v[i] = "model.language_model.layers"
                    fixed += 1

        # Recurse into nested dicts
        if isinstance(v, dict):
            fixed += fix_config_keys(v)

        new_d[new_key] = v

    d.clear()
    d.update(new_d)
    return fixed


# ---------------------------------------------------------------------------
# 2. block_name_to_quantize: remove model.visual.blocks
# ---------------------------------------------------------------------------

def fix_block_name_to_quantize(qc):
    """Remove model.visual.blocks from block_name_to_quantize list."""
    if not isinstance(qc, dict):
        return False
    block_name = qc.get("block_name_to_quantize")
    if isinstance(block_name, list):
        before = list(block_name)
        qc["block_name_to_quantize"] = [
            b for b in block_name if b != "model.visual.blocks"
        ]
        return qc["block_name_to_quantize"] != before
    return False


# ---------------------------------------------------------------------------
# 3. Vision config restoration
# ---------------------------------------------------------------------------

def fix_vision_config(cfg, source_cfg=None):
    """Ensure vision_config block exists for multimodal Qwen3.5 models.

    If the model is qwen3_5 or qwen3_5_moe but vision_config is missing,
    copy it from source_cfg.  If it exists but dtype is missing, set it.
    """
    model_type = cfg.get("model_type", "")
    if model_type not in ("qwen3_5", "qwen3_5_moe"):
        return False

    fixed = False
    vision_config = cfg.get("vision_config")

    # Case A: vision_config missing entirely — copy from source
    if not vision_config and source_cfg:
        src_vc = source_cfg.get("vision_config")
        if src_vc:
            cfg["vision_config"] = src_vc
            vision_config = src_vc
            fixed = True

    # Case B: vision_config exists but dtype missing
    if vision_config and vision_config.get("dtype") is None:
        cfg["vision_config"]["dtype"] = "bfloat16"
        fixed = True

    # Case C: still missing and no source — add minimal block
    if not cfg.get("vision_config"):
        cfg["vision_config"] = {
            "dtype": "bfloat16",
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "merge_size": 2,
            "patch_size": 16,
            "temporal_patch_size": 2,
        }
        fixed = True

    return fixed


# ---------------------------------------------------------------------------
# 4. Safetensors: remove quantized visual layer keys
# ---------------------------------------------------------------------------

def fix_safetensors_index(index_path):
    """Fix weight_map keys in model.safetensors.index.json if needed."""
    with open(index_path) as f:
        idx = json.load(f)

    wm = idx.get("weight_map", {})
    bad_keys = [k for k in wm if k.startswith("model.layers.")]

    if not bad_keys:
        return 0

    for k in list(wm.keys()):
        if k.startswith("model.layers."):
            wm["model.language_model." + k] = wm.pop(k)

    with open(index_path, "w") as f:
        json.dump(idx, f, indent=2)

    return len(bad_keys)


def fix_visual_quantization(model_dir, dry_run=False):
    """Remove quantized visual layer keys from safetensors files.

    Some quantization runs incorrectly quantize visual layers, producing
    qweight/qzeros/scales tensors alongside the original weight/bias.  This
    function rebuilds affected safetensors files keeping only the original
    (non-quantized) visual layer tensors.

    Returns (files_modified, keys_removed, keys_kept).
    """
    if not HAS_SAFETENSORS:
        print("    WARNING: safetensors not installed — cannot fix visual quantization")
        return 0, 0, 0

    import torch

    idx_path = model_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        return 0, 0, 0

    with open(idx_path) as f:
        idx = json.load(f)

    wm = idx.get("weight_map", {})

    # Identify quantized visual layer keys
    quantized_visual_suffixes = {"qweight", "qzeros", "scales"}
    quantized_visual_keys = []
    for k in wm:
        parts = k.split(".")
        if "visual" in parts and parts[-1] in quantized_visual_suffixes:
            quantized_visual_keys.append(k)

    if not quantized_visual_keys:
        return 0, 0, 0

    # Group quantized keys by source safetensors file
    quantized_by_file = {}
    for k in quantized_visual_keys:
        fname = wm[k]
        quantized_by_file.setdefault(fname, []).append(k)

    files_modified = 0
    total_removed = 0
    total_kept = 0

    for fname, qkeys in quantized_by_file.items():
        fpath = model_dir / fname
        if not fpath.exists():
            print(f"    WARNING: {fname} not found, skipping")
            continue

        print(f"    Processing {fname} ({len(qkeys)} quantized visual keys to remove)")

        if dry_run:
            for k in qkeys:
                print(f"      would remove: {k}")
            total_removed += len(qkeys)
            continue

        # Load all tensors from this file
        tensors = {}
        with safe_open(fpath, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

        # Identify visual keys to keep (weight, bias, etc.) — not quantized
        visual_keep = []
        visual_remove = []
        for k in list(tensors.keys()):
            parts = k.split(".")
            if "visual" in parts:
                if parts[-1] in quantized_visual_suffixes:
                    visual_remove.append(k)
                else:
                    visual_keep.append(k)

        if not visual_remove:
            continue

        # Remove quantized visual tensors
        for k in visual_remove:
            del tensors[k]

        # Rewrite the safetensors file
        save_file(tensors, fpath)

        # Update the weight_map: remove quantized visual keys
        for k in visual_remove:
            wm.pop(k, None)

        total_removed += len(visual_remove)
        total_kept += len(visual_keep)
        files_modified += 1

        print(f"      removed {len(visual_remove)} quantized keys, kept {len(visual_keep)} original keys")

    if files_modified and not dry_run:
        # Update metadata if present
        if "metadata" in idx:
            idx["metadata"]["total_size"] = sum(
                tensors[k].numel() * tensors[k].element_size()
                for k in tensors
            )

        # Rewrite the index
        with open(idx_path, "w") as f:
            json.dump(idx, f, indent=2)

    return files_modified, total_removed, total_kept


# ---------------------------------------------------------------------------
# 5. Architectures normalization
# ---------------------------------------------------------------------------

def fix_architectures(cfg):
    """Fix architectures field in config.json for Qwen3.5 models."""
    if not isinstance(cfg, dict):
        return False

    fixed = False

    # Fix architectures for qwen3_5
    if cfg.get("model_type") == "qwen3_5":
        if cfg.get("architectures") == ["Qwen3_5ForCausalLM"]:
            cfg["architectures"] = ["Qwen3_5ForConditionalGeneration"]
            fixed = True

    # Fix architectures for qwen3_5_moe
    if cfg.get("model_type") == "qwen3_5_moe":
        if cfg.get("architectures") == ["Qwen3_5MoeForCausalLM"]:
            cfg["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
            fixed = True

    return fixed


# ---------------------------------------------------------------------------
# 6. Tokenizer processor_class
# ---------------------------------------------------------------------------

def fix_tokenizer_processor_class(model_dir, source_dir=None):
    """Add processor_class to tokenizer_config.json if missing."""
    tok_config_path = model_dir / "tokenizer_config.json"
    if not tok_config_path.exists():
        return False

    with open(tok_config_path) as f:
        tok_config = json.load(f)

    if "processor_class" in tok_config and tok_config["processor_class"] is not None:
        return False  # Already present

    # Determine correct processor_class based on model_type
    cfg_path = model_dir / "config.json"
    processor_class = None

    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)

        model_type = cfg.get("model_type", "")
        if model_type in ("qwen3_5", "qwen3_5_moe"):
            processor_class = "Qwen3VLProcessor"
        elif model_type == "qwen2_vl":
            processor_class = "Qwen2VLProcessor"
        elif model_type == "qwen2_5_vl":
            processor_class = "Qwen2VLProcessor"

    # Fallback: try to get from source model
    if processor_class is None and source_dir:
        src_tok_config = Path(source_dir) / "tokenizer_config.json"
        if src_tok_config.exists():
            with open(src_tok_config) as f:
                src_config = json.load(f)
            processor_class = src_config.get("processor_class")

    if processor_class:
        tok_config["processor_class"] = processor_class
        with open(tok_config_path, "w") as f:
            json.dump(tok_config, f, indent=2)
        return True

    return False


# ---------------------------------------------------------------------------
# 7. Copy missing files from source model
# ---------------------------------------------------------------------------

def copy_missing_files(source_dir, target_dir):
    """Copy processor/preprocessor configs from source model if missing."""
    files_to_copy = [
        "processor_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ]
    copied = []
    for fname in files_to_copy:
        src = Path(source_dir) / fname
        dst = Path(target_dir) / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied.append(fname)
    return copied


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2 or "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)

    dry_run = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--dry-run"]

    model_dir = Path(args[0]).expanduser()
    source_dir = Path(args[1]).expanduser() if len(args) > 1 else None

    if not model_dir.is_dir():
        print(f"Error: {model_dir} is not a directory")
        sys.exit(1)

    if dry_run:
        print("DRY RUN — no files will be modified\n")

    print(f"Fixing: {model_dir}")
    if source_dir:
        print(f"Source: {source_dir}")
    print()

    # Load source config if available
    source_cfg = None
    if source_dir and (source_dir / "config.json").exists():
        with open(source_dir / "config.json") as f:
            source_cfg = json.load(f)

    # -----------------------------------------------------------------------
    # Step 1: Fix layer prefix in quantization_config.json
    # -----------------------------------------------------------------------
    qc_path = model_dir / "quantization_config.json"
    if qc_path.exists():
        with open(qc_path) as f:
            qc = json.load(f)
        prefix_fixed = fix_config_keys(qc)
        visual_fixed = fix_block_name_to_quantize(qc)
        if (prefix_fixed or visual_fixed) and not dry_run:
            with open(qc_path, "w") as f:
                json.dump(qc, f, indent=2)
        print(f"  quantization_config.json:")
        print(f"    layer prefix fixes: {prefix_fixed}")
        print(f"    removed visual.blocks from block_name_to_quantize: {visual_fixed}")
        print(f"    block_name_to_quantize: {qc.get('block_name_to_quantize')}")
    else:
        print("  quantization_config.json: not found, skipping")

    # -----------------------------------------------------------------------
    # Step 2: Fix config.json (nested quantization_config + vision_config)
    # -----------------------------------------------------------------------
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        prefix_fixed = fix_config_keys(cfg)
        arch_fixed = fix_architectures(cfg)
        visual_bnq = False
        # Also fix block_name_to_quantize inside nested quantization_config
        nested_qc = cfg.get("quantization_config", {})
        if nested_qc:
            visual_bnq = fix_block_name_to_quantize(nested_qc)
        vision_fixed = fix_vision_config(cfg, source_cfg)
        if (prefix_fixed or arch_fixed or visual_bnq or vision_fixed) and not dry_run:
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
        print(f"  config.json:")
        print(f"    layer prefix fixes: {prefix_fixed}")
        print(f"    architecture normalization: {arch_fixed}")
        print(f"    removed visual.blocks from nested block_name_to_quantize: {visual_bnq}")
        print(f"    vision_config fix: {vision_fixed}")
    else:
        print("  config.json: not found, skipping")

    # -----------------------------------------------------------------------
    # Step 3: Fix model.safetensors.index.json (layer prefix)
    # -----------------------------------------------------------------------
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        prefix_fixed = fix_safetensors_index(idx_path)
        print(f"  model.safetensors.index.json: layer prefix fixes: {prefix_fixed}")
    else:
        print("  model.safetensors.index.json: not found, skipping")

    # -----------------------------------------------------------------------
    # Step 4: Fix visual layer quantization in safetensors
    # -----------------------------------------------------------------------
    if HAS_SAFETENSORS:
        print(f"  safetensors visual layer fix:")
        files_mod, keys_rm, keys_kept = fix_visual_quantization(model_dir, dry_run=dry_run)
        print(f"    files rebuilt: {files_mod}")
        print(f"    quantized keys removed: {keys_rm}")
        print(f"    original keys kept: {keys_kept}")
    else:
        print("  safetensors visual layer fix: SKIPPED (safetensors not installed)")

    # -----------------------------------------------------------------------
    # Step 5: Fix tokenizer_config.json processor_class
    # -----------------------------------------------------------------------
    tok_fixed = fix_tokenizer_processor_class(model_dir, source_dir)
    print(f"  tokenizer_config.json: processor_class added: {tok_fixed}")

    # -----------------------------------------------------------------------
    # Step 6: Copy missing processor files from source model
    # -----------------------------------------------------------------------
    if source_dir and source_dir.is_dir():
        copied = copy_missing_files(source_dir, model_dir)
        for f in copied:
            print(f"  Copied {f} from source model")
        if not copied:
            print("  No missing processor files to copy")
    elif source_dir:
        print(f"  Source model not found: {source_dir}")
        print("  (processor_config.json may be missing)")
    else:
        print("  No source model specified — skipping file copy")

    print("\nDone!")


if __name__ == "__main__":
    main()
