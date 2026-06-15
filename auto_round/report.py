# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Quantization report — collects per-layer metrics and writes quantization-report.txt.

Simplified for W4A16-only workflow (no adaptive FP8/FP16 fallback).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


# Thresholds for pass/warn status
COSINE_SIM_THRESHOLD = 0.99
PSNR_THRESHOLD = 45.0


@dataclass
class LayerResult:
    """Sensitivity result for a single quantized block.

    Attributes:
        block_name: Full block name (e.g. ``"model.layers.0"``).
        cosine_sim: Cosine similarity between reference and quantized output.
        psnr_db: Peak Signal-to-Noise Ratio in dB.
        init_loss: Loss at the first tuning iteration (``None`` if no tuning).
        best_loss: Best loss achieved across all tuning iterations.
        best_iter: Iteration at which best loss was achieved.
        total_iters: Total iterations actually run.
        passed: Whether both metrics exceed thresholds.
    """

    block_name: str
    cosine_sim: float
    psnr_db: float
    init_loss: float | None
    best_loss: float | None
    best_iter: int
    total_iters: int
    passed: bool


class QuantizationReport:
    """Collects per-block sensitivity metrics and writes a formatted report.

    Usage::

        report = QuantizationReport(
            model_name="Qwen/Qwen3.5-0.8B",
            version="0.14.0",
            cli_args={"batch_size": 8, "iters": 1000},
        )
        # After each block:
        report.add_layer("model.layers.0", cosine_sim=0.9998, psnr_db=52.3, loss=5.2e-6)
        # At the end:
        report.set_memory_summary(peak_ram_gb=31.25, peak_vram_gb=40.93)
        report.save("./output/my-model")
    """

    def __init__(
        self,
        model_name: str,
        version: str,
        cli_args: dict[str, str | int | float] | None = None,
    ):
        self.model_name = model_name
        self.version = version
        self.cli_args = cli_args or {}
        self.layers: list[LayerResult] = []
        self.peak_ram_gb: float | None = None
        self.peak_vram_gb: float | None = None

    def add_layer(
        self,
        block_name: str,
        cosine_sim: float,
        psnr_db: float,
        init_loss: float | None = None,
        best_loss: float | None = None,
        best_iter: int = 0,
        total_iters: int = 0,
    ) -> None:
        """Record metrics for a completed block.

        Args:
            block_name: Full block name (e.g. ``"model.layers.0"``).
            cosine_sim: Cosine similarity between reference and quantized output.
            psnr_db: Peak Signal-to-Noise Ratio in dB.
            init_loss: Loss at the first tuning iteration.
            best_loss: Best loss achieved across all tuning iterations.
            best_iter: Iteration at which best loss was achieved.
            total_iters: Total iterations actually run.
        """
        passed = cosine_sim >= COSINE_SIM_THRESHOLD and psnr_db >= PSNR_THRESHOLD
        self.layers.append(
            LayerResult(
                block_name=block_name,
                cosine_sim=cosine_sim,
                psnr_db=psnr_db,
                init_loss=init_loss,
                best_loss=best_loss,
                best_iter=best_iter,
                total_iters=total_iters,
                passed=passed,
            )
        )

    def set_memory_summary(
        self, peak_ram_gb: float | None = None, peak_vram_gb: float | None = None
    ) -> None:
        """Set peak memory usage for the report header.

        Args:
            peak_ram_gb: Peak RAM usage in GB.
            peak_vram_gb: Peak VRAM usage in GB.
        """
        self.peak_ram_gb = peak_ram_gb
        self.peak_vram_gb = peak_vram_gb

    def save(self, output_dir: str) -> Path:
        """Write the formatted report to ``<output_dir>/quantization-report.txt``.

        Args:
            output_dir: Directory to write the report into.

        Returns:
            Path to the written report file.
        """
        report_path = Path(output_dir) / "quantization-report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        lines = self._format_report()
        report_path.write_text("\n".join(lines) + "\n")
        return report_path

    def get_summary(self) -> dict[str, int]:
        """Return pass/warn counts.

        Returns:
            Dict with ``"total"``, ``"passed"``, ``"warn"`` keys.
        """
        total = len(self.layers)
        passed = sum(1 for l in self.layers if l.passed)
        return {"total": total, "passed": passed, "warn": total - passed}

    # ── Internal ────────────────────────────────────────────────────────

    def _status_icon(self, passed: bool) -> str:
        """Return emoji icon for pass/warn status."""
        return "🟢" if passed else "🟠"

    @staticmethod
    def _fmt_loss(value: float) -> str:
        """Format a loss value for display."""
        if value == 0:
            return "0"
        return f"{value:.4f}"

    def _format_report(self) -> list[str]:
        """Build the full report as a list of lines."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary = self.get_summary()

        lines = [
            "=== Quantization Report ===",
            f"Model: {self.model_name}",
            f"Date: {now}",
            f"Version: {self.version}",
            "",
        ]

        # CLI Arguments section
        lines.append("CLI Arguments:")
        if self.cli_args:
            for key, value in self.cli_args.items():
                lines.append(f"  --{key} {value}")
        lines.append("")

        # Memory Summary section
        lines.append("Memory Summary:")
        if self.peak_ram_gb is not None:
            lines.append(f"  Peak RAM: {self.peak_ram_gb:.2f} GB")
        if self.peak_vram_gb is not None:
            lines.append(f"  Peak VRAM: {self.peak_vram_gb:.2f} GB")
        if self.peak_ram_gb is None and self.peak_vram_gb is None:
            lines.append("  (not available)")
        lines.append("")

        # Sensitivity Analysis table
        lines.append("Sensitivity Analysis:")
        header = (
            f"{'Layer':<42} {'Cosine Sim':>10} {'PSNR (dB)':>10} "
            f"{'Iters':>10} {'Loss':>18} {'Status':>8}"
        )
        separator = "─" * len(header)
        lines.append(separator)
        lines.append(header)
        lines.append(separator)

        for lr in self.layers:
            icon = self._status_icon(lr.passed)
            status = "PASS" if lr.passed else "WARN"
            psnr_str = f"{lr.psnr_db:.1f}" if lr.psnr_db != float("inf") else "∞"

            # Iters column
            if lr.total_iters > 0:
                iters_str = f"{lr.best_iter}/{lr.total_iters}"
            else:
                iters_str = "—"

            # Loss column: "init → best" or single value or dash
            if lr.init_loss is not None and lr.best_loss is not None:
                loss_str = f"{self._fmt_loss(lr.init_loss)} → {self._fmt_loss(lr.best_loss)}"
            elif lr.best_loss is not None:
                loss_str = self._fmt_loss(lr.best_loss)
            else:
                loss_str = "—"

            lines.append(
                f"{icon} {lr.block_name:<40} {lr.cosine_sim:>10.4f} "
                f"{psnr_str:>10} {iters_str:>10} {loss_str:>18} {status:>8}"
            )

        lines.append("")
        lines.append("Summary:")
        lines.append(f"  Total blocks: {summary['total']}")
        lines.append(f"  Passed (🟢): {summary['passed']}")
        lines.append(f"  Warning (🟠): {summary['warn']}")
        lines.append("")
        lines.append(
            f"Thresholds: Cosine Similarity < {COSINE_SIM_THRESHOLD}, "
            f"PSNR < {PSNR_THRESHOLD} dB"
        )

        return lines
