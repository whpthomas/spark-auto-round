# Fix for v14.1 Quantized Models (Wrong Layer Name Prefix)

## What happened

A bug in `revert_checkpoint_conversion_mapping()` stripped the `^` regex anchor from
checkpoint conversion patterns. This caused `model.language_model.layers` to be saved as
`model.layers` in the config files of quantized models.

**Affected versions:** v14.1  
**Affected models:** Any multimodal model using `model.language_model.layers`  
(e.g., Qwen3.5, Qwen3.6-35B-A3B, etc.)

## Symptoms

- vLLM fails to load the quantized model
- `quantization_config.json` has `block_name_to_quantize: model.layers` instead of `model.language_model.layers`
- `extra_config` keys use `model.layers.X` instead of `model.language_model.layers.X`
- `processor_config.json` may be missing

## Good news

The safetensors tensors have the correct names (`model.language_model.layers.X`).
Only the config files are wrong. **No re-quantization is needed.**

## Fix

### Option 1: Run the fix script

```bash
git clone https://github.com/whpthomas/spark-auto-round.git
cd spark-auto-round

# Fix the quantized model
# Pass the source model to also copy missing processor files
python scripts/fix-v14.1-layer-prefix.py \
    ~/models/YOUR-MODEL-int4-AutoRound \
    Qwen/YOUR-MODEL
```

### Option 2: Manual fix

Edit `quantization_config.json` and `config.json` in your quantized model directory:

1. Change `block_name_to_quantize` from `"model.layers"` to `"model.language_model.layers"`
2. Fix all `extra_config` keys: replace `model.layers.` with `model.language_model.layers.`
3. If `processor_config.json` is missing, copy it from the source model

## What the script fixes

| File | What it does |
|------|-------------|
| `quantization_config.json` | Fixes `block_name_to_quantize` and all `extra_config` keys |
| `config.json` | Same fixes (nested `quantization_config` section) |
| `model.safetensors.index.json` | Fixes `weight_map` keys (if affected) |
| `processor_config.json` | Copies from source model if missing |

## Verify after fix

```bash
python3 -c "
import json
with open('~/models/YOUR-MODEL-int4-AutoRound/quantization_config.json') as f:
    d = json.load(f)
print('block_name_to_quantize:', d['block_name_to_quantize'])
"
```

Should print: `model.language_model.layers`

## Technical details

The bug was in `auto_round/utils/common.py`:

```python
# BEFORE (broken): stripped ^ anchor, causing 'model' to match anywhere
source_pattern = source_pattern.lstrip("^")
name, n_replace = re.subn(source_pattern, target_pattern, name)

# AFTER (fixed): simple string prefix parsing with double-conversion guard
anchored = source_pattern.startswith("^")
prefix = re.sub(r"^\^|\(.*\)", "", source_pattern)
if anchored and name.startswith(prefix) and not name.startswith(target_clean):
    name = target_clean + name[len(prefix):]
```

For input `model.language_model.layers`:
- **Before:** regex `model` matched twice → `model.language_model.language_model.language_model.layers`
- **After:** prefix check skips because name already starts with target → `model.language_model.layers` (unchanged)
