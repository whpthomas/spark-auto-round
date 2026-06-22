#!/usr/bin/env python
"""Compare two AutoRound-quantized models layer-by-layer.

Why this is similarity-based, not an exact diff
-----------------------------------------------
FlashAttention's backward pass is non-deterministic (CUDA atomics, not RNG), so
two GPU runs — even two *clean* runs — are not bit-identical, and because each
block is tuned on the previous block's quantized output, the divergence
compounds across layers. Resume restores RNG state but cannot undo kernel-level
nondeterminism, so a resumed model will not byte-match a clean one. That is
expected and is NOT a resume bug.

How to interpret results
------------------------
Establish the nondeterminism floor, then check resume stays within it:

    # two clean runs (no --resume) into different output dirs -> floor
    compare_quantized_models.py clean_A clean_B

    # clean vs resumed
    compare_quantized_models.py clean_A resumed

Resume is faithful if clean-vs-resumed divergence is no worse than
clean-vs-clean. A per-layer trend (later layers diverge more) is normal and
should appear in BOTH comparisons.

Metrics per tensor
------------------
* float tensors (e.g. scales): cosine similarity, relative Frobenius error,
  max abs diff.
* integer tensors (e.g. packed qweight, zeros): fraction of exactly-equal
  elements (cosine is meaningless on packed nibbles).

Runs on torch + safetensors only (no auto_round import), so it is robust across
code versions. Run it in the container the same way as the tests.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import torch
from safetensors.torch import load_file


def load_all(folder: str) -> dict:
    """Merge every *.safetensors shard in a model folder into one flat dict."""
    if not os.path.isdir(folder):
        sys.exit(f"not a directory: {folder}")
    files = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    if not files:
        sys.exit(f"no .safetensors files in {folder}")
    weights: dict = {}
    for f in files:
        weights.update(load_file(f))
    return weights


def layer_of(key: str) -> str:
    """Group a parameter key by its transformer layer, else 'other'."""
    m = re.search(r"layers\.(\d+)", key)
    return f"layer {int(m.group(1)):>3}" if m else "other"


def tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    if a.shape != b.shape:
        return {"status": "SHAPE", "detail": f"{tuple(a.shape)} vs {tuple(b.shape)}"}
    if torch.equal(a, b):
        return {"status": "identical"}
    is_float = a.is_floating_point() and b.is_floating_point()
    eq_frac = (a == b).float().mean().item()
    if is_float:
        af, bf = a.float().flatten(), b.float().flatten()
        denom = af.norm().item() or 1.0
        return {
            "status": "diff",
            "is_float": True,
            "cos": torch.nn.functional.cosine_similarity(af, bf, dim=0).item(),
            "rel_fro": (af - bf).norm().item() / denom,
            "max_abs": (af - bf).abs().max().item(),
            "eq_frac": eq_frac,
        }
    return {"status": "diff", "is_float": False, "eq_frac": eq_frac}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_a")
    ap.add_argument("model_b")
    ap.add_argument(
        "--cos-threshold", type=float, default=0.9999,
        help="Flag float tensors whose cosine similarity is below this (default 0.9999).",
    )
    args = ap.parse_args()

    wa, wb = load_all(args.model_a), load_all(args.model_b)
    ka, kb = set(wa), set(wb)
    only_a, only_b = sorted(ka - kb), sorted(kb - ka)
    shared = sorted(ka & kb)

    print(f"A: {args.model_a}  ({len(wa)} tensors)")
    print(f"B: {args.model_b}  ({len(wb)} tensors)")
    if only_a or only_b:
        print(f"\n!! KEY MISMATCH — only in A: {len(only_a)}, only in B: {len(only_b)}")
        for k in only_a[:10]:
            print(f"     only A: {k}")
        for k in only_b[:10]:
            print(f"     only B: {k}")
    if not shared:
        print("\nNo shared tensors — cannot compare.")
        return 1

    # Aggregate per layer.
    by_layer: dict[str, list] = defaultdict(list)
    for k in shared:
        by_layer[layer_of(k)].append(tensor_metrics(wa[k], wb[k]))

    def layer_sort_key(name: str):
        m = re.search(r"(\d+)", name)
        return (0, int(m.group(1))) if m else (1, 0)

    n_identical = n_diff = 0
    global_min_cos = 1.0
    flagged = []
    header = f"{'layer':>9}  {'tensors':>7}  {'ident':>5}  {'min_cos':>8}  {'max_relfro':>10}  {'min_eqfrac':>10}"
    print("\n" + header)
    print("-" * len(header))
    for layer in sorted(by_layer, key=layer_sort_key):
        ms = by_layer[layer]
        ident = sum(1 for m in ms if m["status"] == "identical")
        n_identical += ident
        n_diff += sum(1 for m in ms if m["status"] == "diff")
        cos_vals = [m["cos"] for m in ms if m.get("is_float") and "cos" in m]
        relfro_vals = [m["rel_fro"] for m in ms if m.get("is_float")]
        eqfrac_vals = [m["eq_frac"] for m in ms if m["status"] == "diff" and not m.get("is_float")]
        min_cos = min(cos_vals) if cos_vals else float("nan")
        max_relfro = max(relfro_vals) if relfro_vals else float("nan")
        min_eqfrac = min(eqfrac_vals) if eqfrac_vals else float("nan")
        if cos_vals:
            global_min_cos = min(global_min_cos, min_cos)
        if cos_vals and min_cos < args.cos_threshold:
            flagged.append((layer, min_cos))
        shapes = [m for m in ms if m["status"] == "SHAPE"]
        flag = "  <-- SHAPE MISMATCH" if shapes else ""
        cos_s = f"{min_cos:>8.5f}" if min_cos == min_cos else f"{'—':>8}"
        relfro_s = f"{max_relfro:>10.2e}" if max_relfro == max_relfro else f"{'—':>10}"
        eqfrac_s = f"{min_eqfrac:>10.4f}" if min_eqfrac == min_eqfrac else f"{'—':>10}"
        print(f"{layer:>9}  {len(ms):>7}  {ident:>5}  {cos_s}  {relfro_s}  {eqfrac_s}{flag}")

    print("-" * len(header))
    print(f"\nShared tensors: {len(shared)}  |  identical: {n_identical}  |  differing: {n_diff}")
    print(f"Global min cosine (float tensors): {global_min_cos:.6f}")
    if flagged:
        print(f"\n{len(flagged)} layer(s) below cos threshold {args.cos_threshold}:")
        for layer, c in flagged[:20]:
            print(f"    {layer}: min_cos={c:.5f}")
    print(
        "\nReminder: compare this against a clean-vs-clean baseline. Divergence at "
        "or below that floor = faithful resume; markedly worse = investigate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
