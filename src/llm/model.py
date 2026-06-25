"""
Modern decoder-only transformer (Llama-3 style), from scratch in PyTorch.

What changed vs. the legacy src/model.py (GPT-2 style) and why:

  Legacy (v1)                      This (v2)
  ---------------------------     ------------------------------------------
  Learned positional embeddings → RoPE: rotate q/k by position-dependent
                                   angles; encodes RELATIVE position and adds
                                   zero parameters.
  LayerNorm (mean + var, bias)  →  RMSNorm: scale-only normalization — no
                                   mean subtraction, no bias. Fewer params,
                                   same stability.
  GELU MLP (4x)                 →  SwiGLU: gated MLP. Same FLOPs budget at
                                   hidden ≈ 8/3·d, better loss per FLOP.
  Per-head Q/K/V Linears in a   →  One fused QKV projection, heads split by
  Python loop                      reshape — one big GEMM instead of 3·n_heads
                                   small ones.
  Full multi-head attention     →  Grouped-Query Attention (GQA): n_kv_heads
                                   K/V heads shared across n_heads Q heads.
                                   Shrinks the KV-cache and projections.
  Materialized T×T attention    →  F.scaled_dot_product_attention (flash
  matrix                           kernel, no T×T tensor in memory). The
                                   from-scratch path is kept as
                                   attn_impl="manual" and unit-tested to
                                   match SDPA.
  No KV-cache (O(T^2)/token)    →  Static preallocated KV-cache: each new
                                   token attends to cached K/V, O(T)/token.
  Separate LM head              →  Weight tying: LM head shares the token
                                   embedding matrix.

Biases are omitted everywhere (Llama convention).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.llm.config import ModelConfig


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root-mean-square normalization: x / rms(x) * weight.

    LayerNorm subtracts the mean and divides by the standard deviation.
    RMSNorm skips the mean subtraction — empirically the re-centering
    doesn't matter, so we save the computation and the bias parameter.
    Computed in fp32 for numerical stability under bf16 autocast.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


# ---------------------------------------------------------------------------
# Rotary positional embeddings (RoPE)
# ---------------------------------------------------------------------------

def precompute_rope(head_dim: int, max_seq: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos/sin tables for RoPE.

    Each pair of channels (2i, 2i+1) is rotated by angle pos * theta^(-2i/d).
    Low channel pairs rotate fast (capture local order), high pairs rotate
    slowly (capture long-range position).

    Returns cos, sin of shape (max_seq, head_dim // 2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(max_seq).float()
    angles = torch.outer(pos, freqs)  # (max_seq, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Rotate pairs of channels of x by position-dependent angles.

    x: (B, n_heads, T, head_dim); cos/sin: (T, head_dim/2).

    The key property: dot(rope(q, i), rope(k, j)) depends only on (i - j),
    so attention scores see RELATIVE positions.
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]            # even / odd channels
    cos = cos.view(1, 1, *cos.shape)                # broadcast over B, heads
    sin = sin.view(1, 1, *sin.shape)
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


# ---------------------------------------------------------------------------
# KV-cache
# ---------------------------------------------------------------------------

class KVCache:
    """
    Static preallocated cache of K and V for every layer.

    During generation, each step only computes q/k/v for the NEW token,
    writes k/v into the cache at the current position, and attends over
    the filled prefix — O(T) per token instead of O(T^2).
    """

    def __init__(self, cfg: ModelConfig, batch_size: int, device, dtype=torch.float32):
        shape = (cfg.n_layers, 2, batch_size, cfg.n_kv_heads, cfg.block_size, cfg.head_dim)
        self.cache = torch.zeros(shape, device=device, dtype=dtype)
        self.pos = 0  # number of cached positions

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """Append k, v (B, n_kv_heads, T_new, head_dim) at the current position;
        return the full cached prefix including the new tokens."""
        T_new = k.shape[2]
        self.cache[layer, 0, :, :, self.pos : self.pos + T_new] = k
        self.cache[layer, 1, :, :, self.pos : self.pos + T_new] = v
        end = self.pos + T_new
        return self.cache[layer, 0, :, :, :end], self.cache[layer, 1, :, :, :end]

    def advance(self, n: int):
        self.pos += n


# ---------------------------------------------------------------------------
# Attention (GQA, fused QKV, SDPA or manual)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """
    Causal self-attention with grouped-query attention.

    GQA: n_heads query heads share n_kv_heads key/value heads
    (group size = n_heads / n_kv_heads). With n_kv_heads == n_heads this
    is exactly standard multi-head attention.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.group = cfg.n_heads // cfg.n_kv_heads

        qkv_dim = (cfg.n_heads + 2 * cfg.n_kv_heads) * cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, qkv_dim, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, layer_idx: int = 0, kv_cache: KVCache | None = None):
        B, T, _ = x.shape
        H, KV, D = self.n_heads, self.n_kv_heads, self.head_dim

        qkv = self.qkv(x)
        q, k, v = qkv.split([H * D, KV * D, KV * D], dim=-1)
        q = q.view(B, T, H, D).transpose(1, 2)    # (B, H, T, D)
        k = k.view(B, T, KV, D).transpose(1, 2)   # (B, KV, T, D)
        v = v.view(B, T, KV, D).transpose(1, 2)

        # RoPE on q and k (cos/sin already sliced to this chunk's positions)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)  # (B, KV, pos+T, D)

        # Expand KV heads to match query heads (no-op when group == 1)
        if self.group > 1:
            k = k.repeat_interleave(self.group, dim=1)
            v = v.repeat_interleave(self.group, dim=1)

        if self.cfg.attn_impl == "sdpa":
            # With a cache and T == 1 the single query may attend to the whole
            # prefix (no mask needed); otherwise apply causal masking.
            is_causal = kv_cache is None or T > 1
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=is_causal,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            out = self._manual_attention(q, k, v, causal=(kv_cache is None or T > 1))

        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.proj(out)

    def _manual_attention(self, q, k, v, causal: bool):
        """
        The from-scratch path: the same math SDPA's flash kernel computes,
        written out explicitly. Materializes the (T_q, T_k) score matrix, so
        use it for learning/verification, not for training big models.
        """
        T_q, T_k = q.shape[2], k.shape[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if causal:
            # Offset handles the cached case where T_k > T_q: query i sits at
            # absolute position (T_k - T_q + i) and may attend up to there.
            mask = torch.ones(T_q, T_k, dtype=torch.bool, device=q.device).tril(T_k - T_q)
            att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        if self.training and self.dropout > 0:
            att = F.dropout(att, p=self.dropout)
        return att @ v


# ---------------------------------------------------------------------------
# SwiGLU feed-forward
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """
    Gated MLP: down( silu(gate(x)) * up(x) ).

    The gate lets the network modulate each hidden unit multiplicatively —
    consistently better loss per FLOP than the plain GELU MLP. Hidden dim is
    ~8/3·d_model so total FLOPs match the legacy 4·d_model GELU MLP.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


# ---------------------------------------------------------------------------
# Transformer block and full model
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Pre-norm residual block: x += attn(norm(x)); x += ffn(norm(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, layer_idx: int = 0, kv_cache: KVCache | None = None):
        x = x + self.attn(self.attn_norm(x), cos, sin, layer_idx, kv_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope(cfg.head_dim, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 residual scaling: each block adds 2 residual contributions
        # (attn.proj, ffn.down); scale their init so the residual stream's
        # variance stays O(1) regardless of depth.
        scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(block.ffn.down.weight, mean=0.0, std=0.02 * scale)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(self, idx, targets=None, kv_cache: KVCache | None = None):
        """
        idx: (B, T) token ids. With a kv_cache, idx holds only the NEW tokens
        (positions kv_cache.pos .. kv_cache.pos+T-1) and the cache is advanced.
        Returns (logits, loss); loss is None without targets.
        """
        B, T = idx.shape
        start = kv_cache.pos if kv_cache is not None else 0
        assert start + T <= self.cfg.block_size, "sequence exceeds block_size"

        cos = self.rope_cos[start : start + T]
        sin = self.rope_sin[start : start + T]

        x = self.drop(self.tok_emb(idx))
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, layer_idx=i, kv_cache=kv_cache)
        x = self.norm_f(x)

        if kv_cache is not None:
            kv_cache.advance(T)

        loss = None
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        elif kv_cache is not None:
            # Generation: only the last position's logits are needed.
            logits = self.lm_head(x[:, -1:, :])
        else:
            logits = self.lm_head(x)

        return logits, loss

    def configure_optimizer(self, weight_decay: float, lr: float, betas: tuple, device_type: str = "cpu"):
        """
        AdamW with weight decay on 2D params only (matmul weights/embeddings);
        norms get no decay. Fused kernel on CUDA.
        """
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)


if __name__ == "__main__":
    cfg = ModelConfig()
    model = Transformer(cfg)
    print(f"params: {model.num_params():,} ({model.num_params(non_embedding=True):,} non-embedding)")
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss = model(x, targets=x)
    print(f"logits {tuple(logits.shape)}  loss {loss.item():.3f}  (random ≈ {math.log(cfg.vocab_size):.3f})")
