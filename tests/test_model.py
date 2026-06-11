"""
Correctness gates for the v2 model (run: pytest tests/ -v).

These encode the invariants that make the cheap TinyStories run and the
expensive FineWeb run trustworthy:
- causality: no information leaks from future tokens
- the from-scratch attention path matches PyTorch's SDPA kernel
- KV-cached generation is exactly equivalent to recomputing from scratch
- RoPE encodes relative position
- parameter count matches the closed-form formula
"""

import math

import pytest
import torch

from src.llm.config import ModelConfig
from src.llm.model import KVCache, Transformer, apply_rope, precompute_rope
from src.llm.sample import generate, sample_next

torch.manual_seed(0)


def tiny_cfg(**kw) -> ModelConfig:
    defaults = dict(vocab_size=128, d_model=64, n_layers=2, n_heads=4,
                    n_kv_heads=2, block_size=32, ffn_hidden=128, dropout=0.0)
    defaults.update(kw)
    return ModelConfig(**defaults)


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------

def test_causality():
    """Logits at position t must not change when tokens AFTER t change."""
    cfg = tiny_cfg()
    model = Transformer(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        logits_a, _ = model(x)
        x_perturbed = x.clone()
        x_perturbed[0, 10:] = torch.randint(0, cfg.vocab_size, (6,))
        logits_b, _ = model(x_perturbed)
    assert torch.allclose(logits_a[0, :10], logits_b[0, :10], atol=1e-5), \
        "future tokens leaked into past positions"
    assert not torch.allclose(logits_a[0, 10:], logits_b[0, 10:], atol=1e-3), \
        "perturbed positions should differ (sanity check)"


# ---------------------------------------------------------------------------
# Manual attention == SDPA
# ---------------------------------------------------------------------------

def test_manual_matches_sdpa():
    cfg_sdpa = tiny_cfg(attn_impl="sdpa")
    cfg_manual = tiny_cfg(attn_impl="manual")
    model_sdpa = Transformer(cfg_sdpa).eval()
    model_manual = Transformer(cfg_manual).eval()
    model_manual.load_state_dict(model_sdpa.state_dict())

    x = torch.randint(0, cfg_sdpa.vocab_size, (2, 24))
    with torch.no_grad():
        la, _ = model_sdpa(x)
        lb, _ = model_manual(x)
    assert torch.allclose(la, lb, atol=1e-5), \
        f"max diff {(la - lb).abs().max().item()}"


def test_mha_fallback():
    """n_kv_heads == n_heads must behave as plain multi-head attention."""
    cfg = tiny_cfg(n_kv_heads=4)  # == n_heads
    model = Transformer(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    logits, _ = model(x)
    assert logits.shape == (1, 8, cfg.vocab_size)


# ---------------------------------------------------------------------------
# KV-cache equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attn_impl", ["sdpa", "manual"])
def test_kv_cache_matches_full_forward(attn_impl):
    """Feeding tokens incrementally through the cache must reproduce the
    full-context logits for the last position."""
    cfg = tiny_cfg(attn_impl=attn_impl)
    model = Transformer(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 12))

    with torch.no_grad():
        full_logits, _ = model(x)

        cache = KVCache(cfg, batch_size=1, device="cpu")
        logits_prefill, _ = model(x[:, :8], kv_cache=cache)
        assert torch.allclose(logits_prefill[:, -1], full_logits[:, 7], atol=1e-5)
        for t in range(8, 12):
            step_logits, _ = model(x[:, t : t + 1], kv_cache=cache)
        assert torch.allclose(step_logits[:, -1], full_logits[:, -1], atol=1e-5), \
            f"max diff {(step_logits[:, -1] - full_logits[:, -1]).abs().max().item()}"


def test_cached_generation_matches_uncached():
    """Greedy generation must be token-identical with and without the cache."""
    cfg = tiny_cfg()
    model = Transformer(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (2, 5))
    out_cached = generate(model, prompt, 20, temperature=0.0, use_cache=True)
    out_uncached = generate(model, prompt, 20, temperature=0.0, use_cache=False)
    assert torch.equal(out_cached, out_uncached)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def test_rope_relative_property():
    """dot(rope(q, i), rope(k, j)) must depend only on i - j."""
    head_dim, max_seq = 16, 64
    cos, sin = precompute_rope(head_dim, max_seq, theta=10000.0)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    def score(i, j):
        qi = apply_rope(q, cos[i : i + 1], sin[i : i + 1])
        kj = apply_rope(k, cos[j : j + 1], sin[j : j + 1])
        return (qi * kj).sum().item()

    # same offset (5), different absolute positions
    assert math.isclose(score(7, 2), score(40, 35), rel_tol=1e-4)
    # different offset must differ
    assert not math.isclose(score(7, 2), score(7, 4), rel_tol=1e-3)


def test_rope_zero_position_is_identity():
    cos, sin = precompute_rope(8, 4, theta=10000.0)
    x = torch.randn(1, 2, 1, 8)
    out = apply_rope(x, cos[:1], sin[:1])
    assert torch.allclose(out, x, atol=1e-6)


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------

def test_param_count_formula():
    cfg = tiny_cfg()
    model = Transformer(cfg)
    d, h, kv, hd, f = cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.ffn_hidden
    per_block = (
        d * (h + 2 * kv) * hd   # qkv
        + d * d                 # attn out proj
        + 3 * d * f             # swiglu gate/up/down
        + 2 * d                 # two RMSNorm weights
    )
    expected = cfg.vocab_size * d + cfg.n_layers * per_block + d  # emb + blocks + final norm
    # lm_head is tied -> contributes no extra params
    assert model.num_params() == expected


# ---------------------------------------------------------------------------
# Sampling filters
# ---------------------------------------------------------------------------

def test_top_k_filter():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    counts = set()
    for _ in range(50):
        counts.add(sample_next(logits, temperature=1.0, top_k=2).item())
    assert counts <= {2, 3}, "top_k=2 must only sample the two best tokens"


def test_top_p_keeps_best_token():
    # Extremely peaked: top-p with tiny p must still keep the argmax.
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    for _ in range(10):
        assert sample_next(logits, temperature=1.0, top_p=0.01).item() == 0


def test_top_p_filters_tail():
    # Uniform 4-way distribution (0.25 each): p=0.45 keeps exactly the two
    # tokens needed to reach >= 0.45 cumulative mass.
    logits = torch.zeros(1, 4)
    seen = set()
    for _ in range(100):
        seen.add(sample_next(logits, temperature=1.0, top_p=0.45).item())
    assert len(seen) <= 2


def test_greedy_is_argmax():
    logits = torch.tensor([[0.1, 5.0, 0.2]])
    assert sample_next(logits, temperature=0.0).item() == 1
