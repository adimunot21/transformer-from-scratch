"""
Phase 1 validation run: ~25M params on TinyStories.

Purpose: prove the ENTIRE pipeline (tokenizer -> shards -> training ->
resumable checkpoints -> KV-cached sampling) before spending the Phase 2
budget. TinyStories is known to yield coherent English at this scale.

~500M tokens, 1 epoch ≈ 1-2 hr on an RTX 4090 / free Colab T4 (longer).

Gate to Phase 2: grammatical stories with consistent characters over 3+
sentences; kill-and-resume reproduces the loss curve; cached == uncached
generation.
"""

from src.llm.config import ModelConfig, TrainConfig

tokenizer_path = "data/tokenizer_32k.json"

model_config = ModelConfig(
    vocab_size=32768,
    d_model=512,
    n_layers=8,
    n_heads=8,
    n_kv_heads=4,
    block_size=1024,
    ffn_hidden=1536,
    dropout=0.0,
)

train_config = TrainConfig(
    data_dir="data/shards_tinystories",
    micro_batch_size=16,
    grad_accum_steps=8,          # 16*8*1024 ≈ 131K tokens/step
    max_lr=1e-3,                 # small models take a hotter lr
    min_lr=1e-4,
    warmup_steps=300,
    max_steps=4000,              # ≈ 0.5B tokens, one epoch
    out_dir="checkpoints/tinystories_25m",
    eval_interval=200,
    sample_interval=400,
    checkpoint_interval=500,
    run_name="tinystories-25m",
)
