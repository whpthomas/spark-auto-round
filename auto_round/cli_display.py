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
"""Self-managed CLI display for spark-auto-round quantization.

Replaces tqdm with a self-managed progress bar and prints color-coded
sensitivity lines after each block completes.

Layout during quantization::

    🟢 model.language_model.layers.0 | Cosine similarity 0.9998 | Peak Signal-to-Noise Ratio 52.3
    Quantizing model.language_model.layers.1:   2%|████░░░░░░| 1/64 [01:23<02:00, 1.5s/it]
"""

from __future__ import annotations

import shutil
import sys
import time

# ANSI escape sequences
_NEW_LINE = "\n"

# Thresholds for status icons (must match report.py)
COSINE_SIM_THRESHOLD = 0.99
PSNR_THRESHOLD = 45.0


class Colors:
    """ANSI color codes with automatic tty detection."""

    ENABLED = sys.stdout.isatty()

    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    ORANGE = "\033[38;5;208m"
    GREY = "\033[90m"

    @classmethod
    def green(cls, text: str) -> str:
        return f"{cls.GREEN}{text}{cls.RESET}" if cls.ENABLED else text

    @classmethod
    def orange(cls, text: str) -> str:
        return f"{cls.ORANGE}{text}{cls.RESET}" if cls.ENABLED else text

    @classmethod
    def bold(cls, text: str) -> str:
        return f"{cls.BOLD}{text}{cls.RESET}" if cls.ENABLED else text

    @classmethod
    def grey(cls, text: str) -> str:
        return f"{cls.GREY}{text}{cls.RESET}" if cls.ENABLED else text


class CLIDisplay:
    """Manages CLI output during quantization.

    Usage::

        display = CLIDisplay(total_blocks=64)
        display.begin()
        for i, block_name in enumerate(block_names):
            # ... quantize block ...
            display.print_sensitivity(block_name, cos_sim=0.9998, psnr_db=52.3, loss=5.2e-6)
        display.end()
    """

    def __init__(self, total_blocks: int):
        """Initialize the display.

        Args:
            total_blocks: Total number of blocks to quantize.
        """
        self._total_blocks = total_blocks
        self._blocks_done: int = 0
        self._start_time: float | None = None
        self._progress_printed: bool = False

    def begin(self) -> None:
        """Start the display. Prints the initial progress bar."""
        self._start_time = time.time()
        self._blocks_done = 0
        self._print_progress("model.layers.0")

    def print_sensitivity(
        self,
        block_name: str,
        cos_sim: float,
        psnr_db: float,
        loss: float,
    ) -> None:
        """Print a color-coded sensitivity line for a completed block.

        Clears the current progress bar line, prints the sensitivity line,
        then redraws the progress bar below.

        Args:
            block_name: Full block name (e.g. ``"model.language_model.layers.0"``).
            cos_sim: Cosine similarity between reference and quantized output.
            psnr_db: Peak Signal-to-Noise Ratio in dB.
            loss: Final tuning loss.
        """
        self._blocks_done += 1

        # Determine status icon
        passed = cos_sim >= COSINE_SIM_THRESHOLD and psnr_db >= PSNR_THRESHOLD
        icon = "🟢" if passed else "🟠"

        # Format PSNR
        psnr_str = f"{psnr_db:.1f}" if psnr_db != float("inf") else "∞"

        # Build the sensitivity line
        sensitivity_line = (
            f"{icon} {block_name} | "
            f"Cosine similarity {cos_sim:.4f} | "
            f"Peak Signal-to-Noise Ratio {psnr_str}"
        )

        if Colors.ENABLED:
            # Print sensitivity line
            if passed:
                sys.stdout.write(sensitivity_line + _NEW_LINE)
            else:
                sys.stdout.write(Colors.orange(sensitivity_line) + _NEW_LINE)
            sys.stdout.flush()
        else:
            # Non-TTY: just print
            println(sensitivity_line)

        # Print next block's progress bar
        if self._blocks_done < self._total_blocks:
            next_block = f"model.layers.{self._blocks_done}"
            self._print_progress(next_block)
        else:
            self._print_done()

    def update_progress(self, block_name: str) -> None:
        """Update the progress bar description (for compatibility).

        In the new flow, ``print_sensitivity`` handles both the sensitivity
        line and the next progress bar. This method is kept for backward
        compatibility but is a no-op.

        Args:
            block_name: Block name (unused in new flow).
        """

    def end(
        self,
        peak_ram_gb: float | None = None,
        peak_vram_gb: float | None = None,
    ) -> None:
        """Finalize the display.

        Args:
            peak_ram_gb: Peak RAM usage in GB (for logging).
            peak_vram_gb: Peak VRAM usage in GB (for logging).
        """
        elapsed = time.time() - self._start_time if self._start_time else 0
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{hours}:{minutes:02d}:{seconds:02d}"

        mem_parts = []
        if peak_ram_gb is not None:
            mem_parts.append(f"{peak_ram_gb:.1f} GB RAM")
        if peak_vram_gb is not None:
            mem_parts.append(f"{peak_vram_gb:.1f} GB VRAM")
        mem_str = f" | Peak {', '.join(mem_parts)}" if mem_parts else ""

        print(
            f"Quantization complete: {self._total_blocks}/{self._total_blocks} "
            f"blocks{mem_str} | {time_str}"
        )

    # ── Internal ────────────────────────────────────────────────────────

    def _print_progress(self, next_block_name: str) -> None:
        """Print the progress bar line.

        Args:
            next_block_name: Name of the next block to quantize.
        """
        total = self._total_blocks
        idx = self._blocks_done
        pct = int(idx / total * 100) if total > 0 else 0

        # Adaptive bar width
        terminal_width = shutil.get_terminal_size().columns
        # Elapsed time
        elapsed = time.time() - self._start_time if self._start_time else 0
        if idx > 0:
            rate = elapsed / idx
            remaining = rate * (total - idx)
            elapsed_str = self._format_time(elapsed)
            remaining_str = self._format_time(remaining)
            time_info = f"[{elapsed_str}<{remaining_str}, {rate:.1f}s/it]"
        else:
            time_info = ""

        # Reserve space for: "Quantizing {name}: {pct}%|{bar}|{idx}/{total} [elapsed<remaining, rate]"
        prefix = f"Quantizing {next_block_name}:{pct:3d}%|"
        suffix = f"|{idx}/{total} {time_info}"
        fixed_overhead = len(prefix) + len(suffix)
        bar_width = max(10, terminal_width - fixed_overhead)
        filled = int(bar_width * idx / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)

        line = f"{prefix}{bar}{suffix}"

        if Colors.ENABLED:
            sys.stdout.write(line + _NEW_LINE)
            sys.stdout.flush()
        else:
            println(line)

        self._progress_printed = True

    def _print_done(self) -> None:
        """Print the 'done' progress bar."""
        total = self._total_blocks
        bar = "█" * 20
        line = f"Quantizing done: 100%|{bar}|{total}/{total}"

        if Colors.ENABLED:
            sys.stdout.write(line + _NEW_LINE)
            sys.stdout.flush()
        else:
            println(line)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as H:MM:SS."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"
