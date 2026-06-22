"""Resumable-quantization tests.

Two layers:

* Unit tests for :class:`BlockCheckpointer` — save/load round-trip, the
  contiguity rule in ``latest_completed_index``, and ``should_stop``.

* The **tier-1 seam test**: the only novel thing resume does is hand the
  block-boundary activations (and RNG state) to the first un-tuned block.  We
  run a tiny model to completion (clean), then run it again with a forced stop
  partway and a resume, and assert the resumed run reproduces the *same*
  downstream activation trajectory the clean run produced.  If that holds, every
  block from the seam onward is identical and the resume is faithful.

These run on CPU with a 4-layer slice of cached gpt2 — no GPU, no network.
"""
import os
import shutil

import pytest
import torch

from auto_round.utils.resume import (
    BlockCheckpointer,
    ResumeSignatureMismatch,
    StopAfterBlock,
    compute_signature,
)

from .helpers import get_model_path, save_tiny_model

NUM_LAYERS = 4
SEQLEN = 16
NSAMPLES = 4
ITERS = 3
GROUP_SIZE = 32


# ── Unit tests for the checkpointer primitive ──────────────────────────────
class TestBlockCheckpointer:
    def test_inactive_without_dir(self):
        ckpt = BlockCheckpointer(resume_dir=None)
        assert ckpt.active is False
        # save is a no-op and must not raise
        ckpt.save(0, input_ids=[torch.zeros(1)], q_input=None, input_others={})
        assert ckpt.latest_completed_index() is None

    def test_save_load_roundtrip(self, tmp_path):
        ckpt = BlockCheckpointer(resume_dir=str(tmp_path))
        input_ids = [torch.randn(1, 4), torch.randn(1, 4)]
        q_input = [torch.randn(1, 4), torch.randn(1, 4)]
        input_others = {"position_ids": torch.arange(4)}

        ckpt.save(2, input_ids=input_ids, q_input=q_input, input_others=input_others)
        payload = ckpt.load(2)

        assert payload["block_index"] == 2
        for a, b in zip(payload["input_ids"], input_ids):
            assert torch.equal(a, b)
        for a, b in zip(payload["q_input"], q_input):
            assert torch.equal(a, b)
        assert torch.equal(payload["input_others"]["position_ids"], input_others["position_ids"])
        assert "rng_state" in payload

    def test_latest_completed_index_contiguous(self, tmp_path):
        ckpt = BlockCheckpointer(resume_dir=str(tmp_path))
        for i in range(3):
            ckpt.save(i, input_ids=[torch.zeros(1)], q_input=None, input_others={})
        assert ckpt.latest_completed_index() == 2

    def test_latest_completed_index_stops_at_gap(self, tmp_path):
        # A missing earlier checkpoint must halt the scan: resuming past a gap
        # would feed a later block the wrong activations.
        ckpt = BlockCheckpointer(resume_dir=str(tmp_path))
        ckpt.save(0, input_ids=[torch.zeros(1)], q_input=None, input_others={})
        ckpt.save(2, input_ids=[torch.zeros(1)], q_input=None, input_others={})
        assert ckpt.latest_completed_index() == 0

    def test_should_stop(self):
        ckpt = BlockCheckpointer(resume_dir=None, stop_after_block=1)
        assert ckpt.should_stop(0) is False
        assert ckpt.should_stop(1) is True
        assert ckpt.should_stop(2) is True

    def test_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AR_RESUME_DIR", str(tmp_path))
        monkeypatch.setenv("AR_STOP_AFTER_BLOCK", "3")
        ckpt = BlockCheckpointer.from_env()
        assert ckpt.active is True
        assert ckpt.resume_dir == str(tmp_path)
        assert ckpt.stop_after_block == 3

    def test_from_env_bad_stop_value(self, monkeypatch):
        monkeypatch.delenv("AR_RESUME_DIR", raising=False)
        monkeypatch.setenv("AR_STOP_AFTER_BLOCK", "not-an-int")
        ckpt = BlockCheckpointer.from_env()
        assert ckpt.stop_after_block is None


# ── CLI flag wiring ────────────────────────────────────────────────────────
class TestResumeCLI:
    def test_parser_accepts_resume_flags(self):
        from auto_round.__main__ import BasicArgumentParser

        parser = BasicArgumentParser()
        args = parser.parse_args(["some-model", "--resume", "--resume-dir", "/tmp/ckpt"])
        assert args.resume is True
        assert args.resume_dir == "/tmp/ckpt"

    def test_parser_resume_defaults(self):
        from auto_round.__main__ import BasicArgumentParser

        parser = BasicArgumentParser()
        args = parser.parse_args(["some-model"])
        assert args.resume is False
        assert args.resume_dir is None

    def test_tune_sets_resume_env_namespaced_under_output_dir(self):
        # tune() defaults the checkpoint dir under output_dir (mounted volume),
        # namespaced by model name so different models don't collide.
        import inspect

        import auto_round.__main__ as main_mod

        source = inspect.getsource(main_mod.tune)
        assert 'os.path.join(args.output_dir, ".resume", model_stem)' in source
        assert 'os.environ["AR_RESUME_DIR"]' in source

    def test_tune_cleans_up_on_success(self):
        import inspect

        import auto_round.__main__ as main_mod

        source = inspect.getsource(main_mod.tune)
        # cleanup() is called after quantize_and_save returns (i.e. on success).
        assert ".cleanup()" in source
        qs = source.index("quantize_and_save(")
        assert source.index(".cleanup()") > qs, "cleanup must run after quantize_and_save"

    def test_tune_forces_immediate_saving_on_resume(self):
        # Resume needs the immediate-saving path even when the model fits, so
        # tune() must OR resume into low_cpu_mem_usage.
        import inspect

        import auto_round.__main__ as main_mod

        source = inspect.getsource(main_mod.tune)
        assert "low_cpu_mem_usage = use_offload or resume_requested" in source


class TestCheckpointerCleanup:
    def test_cleanup_removes_checkpoints(self, tmp_path):
        ckpt = BlockCheckpointer(resume_dir=str(tmp_path / "rd"))
        for i in range(3):
            ckpt.save(i, input_ids=[torch.zeros(1)], q_input=None, input_others={})
        assert ckpt.latest_completed_index() == 2
        ckpt.cleanup()
        assert ckpt.latest_completed_index() is None
        assert not os.path.isdir(str(tmp_path / "rd"))  # empty dir removed

    def test_cleanup_inactive_noop(self):
        BlockCheckpointer(resume_dir=None).cleanup()  # must not raise


class TestResumeSignature:
    def test_signature_stable_and_param_sensitive(self):
        a = {"iters": 1000, "group_size": 128, "seed": 42}
        assert compute_signature(a) == compute_signature(dict(a))  # order-independent
        assert compute_signature(a) != compute_signature({**a, "iters": 200})

    def test_init_then_matching_verify_ok(self, tmp_path):
        ckpt = BlockCheckpointer(resume_dir=str(tmp_path / "rd"))
        params = {"iters": 1000, "group_size": 128}
        ckpt.verify_or_init_signature(params)  # writes manifest
        # A fresh checkpointer (simulating a new process) with same params is OK.
        BlockCheckpointer(resume_dir=str(tmp_path / "rd")).verify_or_init_signature(dict(params))

    def test_mismatch_raises(self, tmp_path):
        ckpt = BlockCheckpointer(resume_dir=str(tmp_path / "rd"))
        ckpt.verify_or_init_signature({"iters": 1000, "group_size": 128})
        with pytest.raises(ResumeSignatureMismatch) as exc:
            BlockCheckpointer(resume_dir=str(tmp_path / "rd")).verify_or_init_signature(
                {"iters": 200, "group_size": 128}
            )
        # Error names the changed key.
        assert "iters" in str(exc.value)

    def test_cleanup_removes_manifest(self, tmp_path):
        ckpt = BlockCheckpointer(resume_dir=str(tmp_path / "rd"))
        ckpt.verify_or_init_signature({"iters": 1000})
        ckpt.save(0, input_ids=[torch.zeros(1)], q_input=None, input_others={})
        ckpt.cleanup()
        assert not os.path.isdir(str(tmp_path / "rd"))  # manifest + ckpt gone, dir removed


# ── Tier-1 seam test ───────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def tiny_model_path():
    path = "./tmp/tiny_gpt2_resume_model"
    # Random weights from config (no large weight download); fine for a
    # determinism/seam test where we only need a reproducible forward pass.
    saved = save_tiny_model(
        get_model_path("gpt2"), path, num_layers=NUM_LAYERS, use_config=True
    )
    yield saved
    shutil.rmtree(saved, ignore_errors=True)


def _make_dataset(vocab_size):
    g = torch.Generator().manual_seed(1234)
    return [torch.randint(0, vocab_size, (1, SEQLEN), generator=g) for _ in range(NSAMPLES)]


def _run_quantize(model_path, resume_dir, dataset, stop_after=None):
    """Run one quantization pass with resume env configured. Returns the block
    index the run stopped after, or None if it completed.

    Resume only engages under immediate-saving, so this runs the offload path
    (low_cpu_mem_usage=True) and saves to a per-resume-dir output folder."""
    from auto_round import AutoRound

    env = {"AR_RESUME_DIR": resume_dir}
    if stop_after is not None:
        env["AR_STOP_AFTER_BLOCK"] = str(stop_after)
    old = {k: os.environ.get(k) for k in ("AR_RESUME_DIR", "AR_STOP_AFTER_BLOCK")}
    os.environ.pop("AR_STOP_AFTER_BLOCK", None)
    os.environ.update(env)
    try:
        ar = AutoRound(
            model=model_path,
            scheme="W4A16",
            dataset=dataset,
            iters=ITERS,
            seqlen=SEQLEN,
            nsamples=NSAMPLES,
            batch_size=1,
            group_size=GROUP_SIZE,
            low_gpu_mem_usage=False,
            low_cpu_mem_usage=True,  # immediate-saving: resume's required path
            device_map="cpu",
            enable_torch_compile=False,
            seed=42,
        )
        ar.quantize_and_save(resume_dir + "_model", format="auto_round")
        return None
    except StopAfterBlock as e:
        return e.block_index
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _ckpt_files(d):
    return sorted(f for f in os.listdir(d) if f.startswith("block_") and f.endswith(".pt"))


def _assert_activations_equal(payload_a, payload_b, idx):
    for key in ("input_ids", "q_input"):
        a, b = payload_a[key], payload_b[key]
        if a is None and b is None:
            continue
        assert a is not None and b is not None, f"block {idx}: {key} None mismatch"
        assert len(a) == len(b), f"block {idx}: {key} length mismatch"
        for j, (ta, tb) in enumerate(zip(a, b)):
            assert torch.equal(ta, tb), (
                f"block {idx}: {key}[{j}] diverged between clean and resumed run"
            )


def _run_quantize_and_save(model_path, resume_dir, output_dir, dataset, stop_after=None):
    """Run a quantize+save pass with immediate-saving (offload) enabled.
    Returns the saved-model folder, or None if it stopped before finalizing."""
    from auto_round import AutoRound

    old = {k: os.environ.get(k) for k in ("AR_RESUME_DIR", "AR_STOP_AFTER_BLOCK")}
    os.environ.pop("AR_STOP_AFTER_BLOCK", None)
    if resume_dir is not None:
        os.environ["AR_RESUME_DIR"] = resume_dir
    else:
        os.environ.pop("AR_RESUME_DIR", None)
    if stop_after is not None:
        os.environ["AR_STOP_AFTER_BLOCK"] = str(stop_after)
    try:
        ar = AutoRound(
            model=model_path,
            scheme="W4A16",
            dataset=dataset,
            iters=ITERS,
            seqlen=SEQLEN,
            nsamples=NSAMPLES,
            batch_size=1,
            group_size=GROUP_SIZE,
            low_gpu_mem_usage=False,
            low_cpu_mem_usage=True,  # triggers is_immediate_saving (sharded write)
            device_map="cpu",
            enable_torch_compile=False,
            seed=42,
        )
        _, folders = ar.quantize_and_save(output_dir, format="auto_round")
        return folders[0] if isinstance(folders, (list, tuple)) else folders
    except StopAfterBlock:
        return None
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _load_all_weights(folder):
    import glob

    from safetensors.torch import load_file

    weights = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.safetensors"))):
        weights.update(load_file(f))
    return weights


def test_resume_assembles_identical_model(tiny_model_path, tmp_path):
    """A stop-then-resume run, with immediate-saving on, assembles a final
    model whose saved weights match an uninterrupted run bit-for-bit."""
    from transformers import AutoConfig

    vocab_size = AutoConfig.from_pretrained(tiny_model_path).vocab_size
    dataset = _make_dataset(vocab_size)

    # Reference: a normal, uninterrupted run (no resume env at all).
    ref_out = str(tmp_path / "ref_out")
    ref_folder = _run_quantize_and_save(tiny_model_path, None, ref_out, dataset)
    assert ref_folder is not None
    ref_weights = _load_all_weights(ref_folder)
    assert ref_weights, "reference run produced no safetensors weights"

    # Resume: interrupt after block 1, then finish in a fresh pass.
    scratch = str(tmp_path / "scratch")
    res_out = str(tmp_path / "res_out")
    stopped = _run_quantize_and_save(tiny_model_path, scratch, res_out, dataset, stop_after=1)
    assert stopped is None  # stopped before finalize
    res_folder = _run_quantize_and_save(tiny_model_path, scratch, res_out, dataset)
    assert res_folder is not None
    res_weights = _load_all_weights(res_folder)

    # The assembled weight sets must be identical.
    assert set(res_weights) == set(ref_weights), (
        f"key mismatch: only-in-ref={set(ref_weights) - set(res_weights)}, "
        f"only-in-resumed={set(res_weights) - set(ref_weights)}"
    )
    for key in ref_weights:
        a, b = ref_weights[key], res_weights[key]
        assert a.shape == b.shape, f"{key}: shape {a.shape} vs {b.shape}"
        assert torch.equal(a, b), f"{key}: weights diverged between reference and resumed run"

    # The resumed run's report must list every block, including those tuned
    # before the interrupt (restored from the checkpoint), not just the
    # post-resume ones.
    def _report_block_lines(folder):
        import glob

        txts = glob.glob(os.path.join(folder, "*report*.txt"))
        if not txts:
            return None
        with open(txts[0], encoding="utf-8") as f:
            return [ln for ln in f if "layers." in ln]

    ref_lines = _report_block_lines(ref_folder)
    res_lines = _report_block_lines(res_folder)
    if ref_lines is not None and res_lines is not None:
        assert len(res_lines) == len(ref_lines), (
            f"resumed report has {len(res_lines)} block rows, "
            f"reference has {len(ref_lines)} — pre-interrupt blocks missing"
        )


def test_resume_disabled_in_whole_model_mode(tiny_model_path, tmp_path):
    """When the model fits in memory (no immediate-saving), resume must disable
    itself: no checkpoints written, and the run completes + saves cleanly.
    Without the guard, skipped blocks crash the packer ('no attribute scale')."""
    from transformers import AutoConfig

    vocab_size = AutoConfig.from_pretrained(tiny_model_path).vocab_size
    dataset = _make_dataset(vocab_size)

    scratch = str(tmp_path / "scratch")
    out = str(tmp_path / "out")
    os.environ["AR_RESUME_DIR"] = scratch
    os.environ.pop("AR_STOP_AFTER_BLOCK", None)
    try:
        from auto_round import AutoRound

        ar = AutoRound(
            model=tiny_model_path, scheme="W4A16", dataset=dataset, iters=ITERS,
            seqlen=SEQLEN, nsamples=NSAMPLES, batch_size=1, group_size=GROUP_SIZE,
            low_gpu_mem_usage=False, low_cpu_mem_usage=False,  # whole-model mode
            device_map="cpu", enable_torch_compile=False, seed=42,
        )
        _, folders = ar.quantize_and_save(out, format="auto_round")
    finally:
        os.environ.pop("AR_RESUME_DIR", None)

    folder = folders[0] if isinstance(folders, (list, tuple)) else folders
    weights = _load_all_weights(folder)
    assert weights, "whole-model-mode run produced no weights"
    # Resume disabled → no checkpoints written.
    assert not (os.path.isdir(scratch) and any(
        f.startswith("block_") for f in os.listdir(scratch)
    )), "resume should not write checkpoints in whole-model mode"


def test_resume_reproduces_clean_trajectory(tiny_model_path, tmp_path):
    """Resuming from a mid-run checkpoint yields the same downstream block
    activations as an uninterrupted run."""
    from transformers import AutoConfig

    vocab_size = AutoConfig.from_pretrained(tiny_model_path).vocab_size
    dataset = _make_dataset(vocab_size)

    clean_dir = str(tmp_path / "clean")
    resume_dir = str(tmp_path / "resume")

    # 1. Clean run to completion — checkpoints every block boundary.
    stopped = _run_quantize(tiny_model_path, clean_dir, dataset, stop_after=None)
    assert stopped is None
    clean_files = _ckpt_files(clean_dir)
    assert clean_files == [f"block_{i:06d}.pt" for i in range(NUM_LAYERS)], clean_files

    # 2a. Interrupted run — stop cleanly after block 1.
    stopped = _run_quantize(tiny_model_path, resume_dir, dataset, stop_after=1)
    assert stopped == 1
    assert _ckpt_files(resume_dir) == ["block_000000.pt", "block_000001.pt"]

    # 2b. Fresh process resumes from the checkpoint and finishes.
    stopped = _run_quantize(tiny_model_path, resume_dir, dataset, stop_after=None)
    assert stopped is None
    assert _ckpt_files(resume_dir) == [f"block_{i:06d}.pt" for i in range(NUM_LAYERS)]

    # 3. The seam invariant: blocks tuned *after* the resume point (2, 3) must
    #    match the clean run bit-for-bit.
    clean_ckpt = BlockCheckpointer(resume_dir=clean_dir)
    resume_ckpt = BlockCheckpointer(resume_dir=resume_dir)
    for idx in range(2, NUM_LAYERS):
        _assert_activations_equal(clean_ckpt.load(idx), resume_ckpt.load(idx), idx)
