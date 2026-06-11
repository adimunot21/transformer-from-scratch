"""
End-to-end training loop tests on a micro model + synthetic shards (CPU).

The resume test encodes the Phase 1 gate: killing a run and resuming from
the latest checkpoint must reproduce the exact same weights as the
uninterrupted run.
"""

import numpy as np
import torch

from src.llm.config import ModelConfig, TrainConfig
from src.llm.data import write_shards
from src.llm.train import get_lr, load_model_from_checkpoint, train


def make_shards(tmp_path):
    rng = np.random.default_rng(0)
    docs = [rng.integers(0, 100, size=rng.integers(20, 200)).tolist() for _ in range(200)]
    write_shards(iter(docs), str(tmp_path), split="train", shard_tokens=8_000)
    write_shards(iter(docs[:50]), str(tmp_path), split="val", shard_tokens=8_000)


def micro_configs(tmp_path, out_dir, max_steps=6):
    mc = ModelConfig(vocab_size=128, d_model=32, n_layers=2, n_heads=2,
                     n_kv_heads=1, block_size=16, ffn_hidden=64)
    tc = TrainConfig(data_dir=str(tmp_path), micro_batch_size=2, grad_accum_steps=2,
                     max_lr=1e-3, min_lr=1e-4, warmup_steps=2, max_steps=max_steps,
                     device="cpu", compile=False, out_dir=str(out_dir),
                     eval_interval=3, eval_steps=2, sample_interval=10_000,
                     checkpoint_interval=3)
    return mc, tc


def test_train_runs_and_saves(tmp_path):
    make_shards(tmp_path)
    out = tmp_path / "ckpt"
    mc, tc = micro_configs(tmp_path, out)
    torch.manual_seed(0)
    train(mc, tc)
    assert (out / "final.pt").exists()
    assert (out / "best.pt").exists()

    model, ckpt = load_model_from_checkpoint(str(out / "final.pt"))
    assert model.cfg.d_model == 32
    x = torch.randint(0, 128, (1, 8))
    logits, _ = model(x)
    assert logits.shape == (1, 8, 128)


def test_resume_reproduces_uninterrupted_run(tmp_path):
    make_shards(tmp_path)

    # Run A: 6 steps straight through.
    out_a = tmp_path / "a"
    mc, tc_a = micro_configs(tmp_path, out_a, max_steps=6)
    torch.manual_seed(42)
    train(mc, tc_a)

    # Run B: same config, "killed" after step 3, then resumed to 6.
    out_b = tmp_path / "b"
    mc_b, tc_b = micro_configs(tmp_path, out_b, max_steps=6)
    torch.manual_seed(42)
    train(mc_b, tc_b, stop_at_step=3)
    train(mc_b, tc_b, resume=str(out_b / "latest.pt"))

    a = torch.load(out_a / "final.pt", weights_only=False)["model_state"]
    b = torch.load(out_b / "final.pt", weights_only=False)["model_state"]
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-6), f"mismatch in {k} after resume"


def test_lr_schedule():
    tc = TrainConfig(warmup_steps=10, max_steps=100, max_lr=1e-3, min_lr=1e-4)
    assert get_lr(0, tc) < get_lr(9, tc) <= 1e-3          # warming up
    assert abs(get_lr(9, tc) - 1e-3) < 1e-9               # peak at end of warmup
    assert get_lr(50, tc) < get_lr(10, tc)                 # decaying
    assert abs(get_lr(99, tc) - 1e-4) < 5e-6               # lands near min_lr
    assert get_lr(150, tc) == 1e-4                         # clamped past the end
