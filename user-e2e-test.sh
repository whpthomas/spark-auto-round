#!/bin/bash
cd ~/spark-auto-round
source .venv/bin/activate
pip install -e .
rm -rf ~/models/Qwen3.5-0.8B* 2>/dev/null
# Halt after layer 3
spark-auto-round --disable_torch_compile --shakedown --halt-after 3 Qwen/Qwen3.5-0.8B --output_dir ~/models
# Resume/quit
spark-auto-round --disable_torch_compile --shakedown Qwen/Qwen3.5-0.8B --output_dir ~/models
