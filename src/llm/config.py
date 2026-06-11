"""
Configuration for the modern (v2) transformer.

Two dataclasses:
- ModelConfig: architecture hyperparameters. Saved into every checkpoint so a
  model can always be rebuilt from its .pt file alone.
- TrainConfig: optimization / data / logging settings for a training run.

Concrete presets live in configs/ (tinystories_25m.py, pretrain_110m.py, ...).
"""

from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4          # GQA; set equal to n_heads for plain MHA
    block_size: int = 1024       # max context length
    ffn_hidden: int = 2048       # SwiGLU hidden dim (~8/3 * d_model, multiple of 256)
    rope_theta: float = 10000.0
    dropout: float = 0.0         # 0 for single-epoch pretraining
    tie_weights: bool = True
    attn_impl: str = "sdpa"      # "sdpa" (flash) or "manual" (from-scratch path)

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must divide by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must divide by n_kv_heads"
        assert self.attn_impl in ("sdpa", "manual")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**d)


@dataclass
class TrainConfig:
    # Data
    data_dir: str = "data/shards"        # directory of uint16 .bin shards
    val_tokens: int = 100_000_000        # tokens held out for validation

    # Batching: micro_batch * grad_accum * block_size = tokens per optimizer step
    micro_batch_size: int = 32
    grad_accum_steps: int = 16

    # Optimization (124M-class recipe — copy, don't tune on rented hardware)
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 700
    max_steps: int = 19_000
    weight_decay: float = 0.1            # applied to 2D params only
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0

    # Runtime
    device: str = "auto"                 # "auto" | "cuda" | "cpu" | "mps"
    dtype: str = "bfloat16"              # autocast dtype on cuda
    compile: bool = True

    # Logging / checkpointing
    out_dir: str = "checkpoints/llm"
    eval_interval: int = 500
    eval_steps: int = 50
    sample_interval: int = 1000
    checkpoint_interval: int = 1000
    wandb_project: str = ""              # empty = wandb disabled
    run_name: str = "run"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["betas"] = list(self.betas)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        d = dict(d)
        if "betas" in d:
            d["betas"] = tuple(d["betas"])
        return cls(**d)
