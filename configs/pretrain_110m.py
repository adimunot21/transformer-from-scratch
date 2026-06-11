"""
Phase 2 main run: ~101M params on FineWeb-Edu sample-10BT (1x H100 80GB).

Recipe copied from the known-good GPT-2 124M class (llm.c / nanoGPT) —
do NOT tune hyperparameters on rented hardware; spend the budget on tokens.

micro_batch 32 x grad_accum 16 x seq 1024 = 524,288 tokens/step
19,000 steps ≈ 10.0B tokens ≈ 7-9 hr on H100 at 300-400K tok/s ≈ $15-20.

Success bar: val loss ≈ 3.0-3.3, HellaSwag acc_norm >= 29% (GPT-2 124M ≈ 31%).

Before committing to the run:
  1. nvidia-smi — check clocks, no throttling
  2. 10-minute smoke test — require >= 300K tok/s, else change host
  3. confirm checkpoints land on the persistent volume AND HF Hub
"""

from src.llm.config import ModelConfig, TrainConfig

tokenizer_path = "data/tokenizer_32k.json"

model_config = ModelConfig(
    vocab_size=32768,
    d_model=768,
    n_layers=12,
    n_heads=12,
    n_kv_heads=4,
    block_size=1024,
    ffn_hidden=2048,
    dropout=0.0,
)

train_config = TrainConfig(
    data_dir="data/shards_fineweb",
    micro_batch_size=32,         # tune upward until ~78GB used
    grad_accum_steps=16,
    max_lr=6e-4,
    min_lr=6e-5,
    warmup_steps=700,
    max_steps=19_000,
    weight_decay=0.1,
    out_dir="checkpoints/pretrain_110m",
    eval_interval=500,           # every ~260M tokens
    sample_interval=1000,
    checkpoint_interval=1000,    # every ~25-30 min
    wandb_project="transformer-from-scratch-v2",
    run_name="pretrain-110m-fineweb10b",
)
