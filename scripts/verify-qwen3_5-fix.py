#!/usr/bin/env python3
"""Verify Qwen3.5 fix is working correctly.

Checks that a quantized Qwen3.5 model directory has all the correct
config files for vLLM compatibility.

Usage:
    python scripts/verify-qwen3_5-fix.py <model_dir> [model_dir2 ...]
"""

import json
import sys
from pathlib import Path


def verify_model(model_dir: str) -> bool:
    """Verify a quantized Qwen3.5 model has correct configs."""
    model_dir = Path(model_dir)
    errors = []

    # Check config.json
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

        # Check architectures
        archs = config.get("architectures", [])
        expected_archs = {
            "qwen3_5": ["Qwen3_5ForConditionalGeneration"],
            "qwen3_5_moe": ["Qwen3_5MoeForConditionalGeneration"],
        }
        model_type = config.get("model_type")
        if model_type in expected_archs:
            if archs != expected_archs[model_type]:
                errors.append(f"Wrong architectures: {archs} (expected {expected_archs[model_type]})")
        elif archs:
            # Non-Qwen3.5 model — just check it has architectures
            pass

        # Check vision_config.dtype for multimodal models
        if model_type in ("qwen3_5", "qwen3_5_moe"):
            vision_config = config.get("vision_config", {})
            if vision_config and vision_config.get("dtype") != "bfloat16":
                errors.append(f"Wrong vision_config.dtype: {vision_config.get('dtype')}")
    else:
        errors.append("config.json not found")

    # Check quantization_config.json
    qc_path = model_dir / "quantization_config.json"
    if qc_path.exists():
        with open(qc_path) as f:
            qc = json.load(f)

        block_name = qc.get("block_name_to_quantize")
        # For Qwen3.5 models, block name should use language_model.layers prefix
        model_type = None
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            model_type = config.get("model_type")

        if model_type in ("qwen3_5", "qwen3_5_moe"):
            if "model.language_model.layers" not in str(block_name):
                errors.append(f"Wrong block_name_to_quantize: {block_name}")
    else:
        errors.append("quantization_config.json not found")

    # Check model.safetensors.index.json — visual layers should NOT be quantized
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)

        weight_map = index.get("weight_map", {})
        visual_qweight = [k for k in weight_map if "visual" in k and "qweight" in k]
        if visual_qweight:
            errors.append(f"Visual layers quantized: {visual_qweight[:3]}")
    else:
        errors.append("model.safetensors.index.json not found")

    # Check processor_config.json exists for multimodal models
    model_type = None
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        model_type = config.get("model_type")

    if model_type in ("qwen3_5", "qwen3_5_moe"):
        processor_path = model_dir / "processor_config.json"
        if not processor_path.exists():
            errors.append("processor_config.json missing")

    # Check tokenizer_config.json has processor_class
    tok_config_path = model_dir / "tokenizer_config.json"
    if tok_config_path.exists():
        with open(tok_config_path) as f:
            tok_config = json.load(f)

        if model_type in ("qwen3_5", "qwen3_5_moe"):
            expected_processor = "Qwen3VLProcessor"
            actual = tok_config.get("processor_class")
            if actual != expected_processor:
                errors.append(f"Wrong processor_class: {actual} (expected {expected_processor})")

    # Report
    if errors:
        print(f"❌ {model_dir.name}: FAILED")
        for err in errors:
            print(f"   - {err}")
        return False
    else:
        print(f"✅ {model_dir.name}: PASSED")
        return True


def main():
    if len(sys.argv) < 2:
        print("Usage: verify-qwen3_5-fix.py <model_dir> [model_dir2 ...]")
        sys.exit(1)

    results = []
    for model_dir in sys.argv[1:]:
        results.append(verify_model(model_dir))

    if all(results):
        print("\n✅ All models verified successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some models failed verification")
        sys.exit(1)


if __name__ == "__main__":
    main()
