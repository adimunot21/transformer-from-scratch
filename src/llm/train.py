"""
Pretraining loop for the v2 transformer.

What this adds over the legacy src/train.py:
- device auto-selection (cuda / mps / cpu) and bf16 autocast on CUDA
  (bf16 has fp32's exponent range, so unlike fp16 it needs NO GradScaler)
- gradient accumulation: micro_batch * grad_accum * block_size tokens per
  optimizer step, so the effective batch is independent of GPU memory
- torch.compile for kernel fusion (fixed batch shapes — see data.py)
- fully resumable checkpoints: model, optimizer, step, dataloader position,
  and RNG state — killing and restarting reproduces the loss curve
- throughput logging (tokens/sec + MFU) — on rented GPUs the tok/s meter,
  not the loss, is what tells you you're wasting money
- optional wandb logging (set TrainConfig.wandb_project)

Run via a config file:  python -m src.llm.train configs/tinystories_25m.py
"""

import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from src.llm.config import ModelConfig, TrainConfig
from src.llm.data import ShardedDataLoader
from src.llm.model import Transformer
from src.llm.sample import generate
from src.llm.tokenizer import Tokenizer


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_lr(step: int, cfg: TrainConfig) -> float:
    """Linear warmup -> cosine decay to min_lr."""
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.max_lr - cfg.min_lr)


def flops_per_token(model_cfg: ModelConfig) -> float:
    """~6N + attention term; good enough for an MFU estimate."""
    n = (model_cfg.d_model ** 2) * (4 + 3 * model_cfg.ffn_hidden / model_cfg.d_model) * model_cfg.n_layers
    attn = 2 * model_cfg.n_layers * model_cfg.block_size * model_cfg.d_model
    return 6 * (n + model_cfg.vocab_size * model_cfg.d_model) + 6 * attn


@torch.no_grad()
def estimate_val_loss(model, loader, eval_steps: int, autocast_ctx) -> float:
    model.eval()
    total = 0.0
    for _ in range(eval_steps):
        x, y = loader.next_batch()
        with autocast_ctx:
            _, loss = model(x, targets=y)
        total += loss.item()
    model.train()
    return total / eval_steps


def save_checkpoint(path, raw_model, optimizer, step, model_cfg, train_cfg,
                    train_loader, val_loss):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "model_config": model_cfg.to_dict(),
        "train_config": train_cfg.to_dict(),
        "loader_state": train_loader.state_dict(),
        "rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "val_loss": val_loss,
    }, path)


def load_model_from_checkpoint(path, device="cpu") -> tuple[Transformer, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_cfg = ModelConfig.from_dict(ckpt["model_config"])
    model = Transformer(model_cfg)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state)
    model.to(device)
    return model, ckpt


def train(model_cfg: ModelConfig, cfg: TrainConfig, tokenizer_path: str | None = None,
          resume: str | None = None, stop_at_step: int | None = None):
    """stop_at_step: exit cleanly after that step (checkpoint included) —
    for throughput smoke tests on rented GPUs and for testing resume."""
    device = pick_device(cfg.device)
    device_type = "cuda" if device.startswith("cuda") else device
    print(f"device: {device}")

    use_bf16 = device_type == "cuda" and cfg.dtype == "bfloat16"
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_bf16 else nullcontext())

    train_loader = ShardedDataLoader(cfg.data_dir, "train", cfg.micro_batch_size,
                                     model_cfg.block_size, device)
    val_loader = ShardedDataLoader(cfg.data_dir, "val", cfg.micro_batch_size,
                                   model_cfg.block_size, device)

    model = Transformer(model_cfg).to(device)
    optimizer = model.configure_optimizer(cfg.weight_decay, cfg.max_lr, cfg.betas, device_type)

    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state"].items()}
        model.load_state_dict(state)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        train_loader.load_state_dict(ckpt["loader_state"])
        torch.random.set_rng_state(ckpt["rng_state"].cpu())
        if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng_state"]])
        start_step = ckpt["step"] + 1
        print(f"resumed from {resume} at step {start_step}")

    raw_model = model
    if cfg.compile and device_type == "cuda":
        model = torch.compile(model)

    tok = Tokenizer.load(tokenizer_path) if tokenizer_path else None

    wandb = None
    if cfg.wandb_project:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=cfg.wandb_project, name=cfg.run_name,
                   config={**model_cfg.to_dict(), **cfg.to_dict()},
                   resume="allow")

    n_params = raw_model.num_params()
    tokens_per_step = cfg.micro_batch_size * cfg.grad_accum_steps * model_cfg.block_size
    print(f"params: {n_params:,} | tokens/step: {tokens_per_step:,} | "
          f"total tokens: {tokens_per_step * cfg.max_steps / 1e9:.2f}B")

    model.train()
    best_val = float("inf")
    t0 = time.time()

    for step in range(start_step, cfg.max_steps):
        lr = get_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = train_loader.next_batch()
            with autocast_ctx:
                _, loss = model(x, targets=y)
            loss = loss / cfg.grad_accum_steps
            loss_accum += loss.item()
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # ---- logging ----
        if step % 10 == 0:
            if device_type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            t0 = time.time()
            tok_per_sec = tokens_per_step * 10 / dt if step > start_step else 0.0
            print(f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e} | "
                  f"gnorm {grad_norm:.2f} | {tok_per_sec/1e3:.0f}K tok/s")
            if wandb:
                wandb.log({"train/loss": loss_accum, "lr": lr,
                           "grad_norm": grad_norm.item(), "tok_per_sec": tok_per_sec},
                          step=step)

        if step % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
            val_loss = estimate_val_loss(model, val_loader, cfg.eval_steps, autocast_ctx)
            print(f"step {step:6d} | VAL loss {val_loss:.4f}")
            if wandb:
                wandb.log({"val/loss": val_loss}, step=step)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(os.path.join(cfg.out_dir, "best.pt"), raw_model,
                                optimizer, step, model_cfg, cfg, train_loader, val_loss)

        if tok and step > 0 and step % cfg.sample_interval == 0:
            ctx = torch.tensor([[tok.eot_id]], dtype=torch.long, device=device)
            out = generate(raw_model, ctx, max_new_tokens=120, temperature=0.8, top_p=0.95)
            print(f"--- sample @ {step} ---\n{tok.decode(out[0].tolist())}\n---")
            model.train()

        if step > 0 and step % cfg.checkpoint_interval == 0:
            save_checkpoint(os.path.join(cfg.out_dir, "latest.pt"), raw_model,
                            optimizer, step, model_cfg, cfg, train_loader, None)

        if stop_at_step is not None and step >= stop_at_step:
            save_checkpoint(os.path.join(cfg.out_dir, "latest.pt"), raw_model,
                            optimizer, step, model_cfg, cfg, train_loader, None)
            print(f"stopped at step {step} (stop_at_step)")
            return

    save_checkpoint(os.path.join(cfg.out_dir, "final.pt"), raw_model, optimizer,
                    cfg.max_steps - 1, model_cfg, cfg, train_loader, best_val)
    print(f"done. best val loss {best_val:.4f}")


if __name__ == "__main__":
    import argparse
    import runpy

    p = argparse.ArgumentParser()
    p.add_argument("config", help="path to a config .py defining model_config/train_config/tokenizer_path")
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    args = p.parse_args()

    ns = runpy.run_path(args.config)
    train(ns["model_config"], ns["train_config"], ns.get("tokenizer_path"), resume=args.resume)
