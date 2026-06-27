#!/bin/bash
cd ~/spark-auto-round
source .venv/bin/activate
pip install -e .
rm -rf ./models/Qwen3.5-0.8B* 2>/dev/null
# Halt after layer 3
spark-auto-round --clear-cache --offload --disable_torch_compile --shakedown --halt-after 3 Qwen/Qwen3.5-0.8B
# Resume/quit
spark-auto-round --offload --disable_torch_compile --shakedown Qwen/Qwen3.5-0.8B

spark-auto-round --dataset cuad --disable_torch_compile --shakedown Qwen/Qwen3.5-0.8B
spark-auto-round --dataset finqa --disable_torch_compile --shakedown Qwen/Qwen3.5-0.8B
spark-auto-round --dataset legalbench --disable_torch_compile --shakedown Qwen/Qwen3.5-0.8B
spark-auto-round --dataset legalbench-instruct --disable_torch_compile --shakedown Qwen/Qwen3.5-0.8B
spark-auto-round --dataset github-code-clean --disable_torch_compile --shakedown Qwen/Qwen3.5-0.8B
