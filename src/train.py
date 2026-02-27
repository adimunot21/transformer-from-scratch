"""
Training loop for the GPT model.

Key components:
- AdamW optimizer (Adam with proper weight decay)
- Cosine learning rate schedule with linear warmup
- Gradient clipping for stability
- Periodic validation loss and text generation samples
"""

import math
import time
import torch
from torch.utils.data import DataLoader
from src.dataset import create_datasets
from src.model import GPT


# ---------------------------------------------------------------------------
# Hyperparameters — all in one place
# ---------------------------------------------------------------------------
CONFIG = {
    # Model
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 4,
    "block_size": 256,
    "dropout": 0.1,

    # Training
    "batch_size": 64,
    "lr": 3e-4,
    "max_steps": 5000,
    "warmup_steps": 500,
    "grad_clip": 1.0,

    # Logging
    "eval_interval": 250,     # Evaluate val loss every N steps
    "eval_steps": 20,         # How many val batches to average over
    "sample_interval": 500,   # Generate a sample every N steps
    "checkpoint_interval": 1000,
}


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float) -> float:
    """
    Learning rate schedule: linear warmup → cosine decay.

    Why warmup? At the start, the model's gradients are essentially random.
    A high learning rate on random gradients = instability. Warmup lets the
    optimizer "find its footing" before taking big steps.

    Why cosine decay? Gradually reducing the learning rate lets the model
    fine-tune its weights as it converges, rather than overshooting.
    """
    # Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Cosine decay after warmup
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_steps):
    """
    Estimate train and val loss by averaging over a few batches.

    We use model.eval() to disable dropout during evaluation,
    then switch back to model.train() afterward.
    """
    model.eval()
    losses = {}

    for name, loader in [("train", train_loader), ("val", val_loader)]:
        total = 0.0
        loader_iter = iter(loader)
        for _ in range(eval_steps):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y = next(loader_iter)
            _, loss = model(x, y)
            total += loss.item()
        losses[name] = total / eval_steps

    model.train()
    return losses


def generate_sample(model, tokenizer, seed_text="\n", length=200, temperature=0.8):
    """Generate a text sample from the model."""
    model.eval()
    tokens = tokenizer.encode(seed_text)
    ctx = torch.tensor([tokens], dtype=torch.long)
    output = model.generate(ctx, max_new_tokens=length, temperature=temperature)
    model.train()
    return tokenizer.decode(output[0].tolist())


def train():
    cfg = CONFIG

    # ---- Data ----
    print("Loading data...")
    with open("data/input.txt", "r") as f:
        text = f.read()

    tok, train_ds, val_ds = create_datasets(text, cfg["block_size"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=True)

    # ---- Model ----
    model = GPT(
        vocab_size=tok.vocab_size,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        block_size=cfg["block_size"],
        dropout=cfg["dropout"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    # ---- Training loop ----
    print(f"\nStarting training for {cfg['max_steps']} steps...")
    print("-" * 60)

    train_iter = iter(train_loader)
    model.train()
    t0 = time.time()

    for step in range(cfg["max_steps"]):
        # Get next batch (restart iterator if exhausted)
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        # Set learning rate for this step
        lr = get_lr(step, cfg["warmup_steps"], cfg["max_steps"], cfg["lr"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward + backward
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])

        optimizer.step()

        # ---- Logging ----

        # Evaluate loss periodically
        if step % cfg["eval_interval"] == 0 or step == cfg["max_steps"] - 1:
            losses = estimate_loss(model, train_loader, val_loader, cfg["eval_steps"])
            elapsed = time.time() - t0
            print(
                f"Step {step:5d} | "
                f"train loss: {losses['train']:.4f} | "
                f"val loss: {losses['val']:.4f} | "
                f"lr: {lr:.2e} | "
                f"time: {elapsed:.1f}s"
            )

        # Generate a sample periodically
        if step > 0 and step % cfg["sample_interval"] == 0:
            sample = generate_sample(model, tok)
            print(f"\n--- Sample at step {step} ---")
            print(sample)
            print("---\n")

        # Save checkpoint periodically
        if step > 0 and step % cfg["checkpoint_interval"] == 0:
            path = f"checkpoints/model_step{step}.pt"
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg,
                "vocab_size": tok.vocab_size,
            }, path)
            print(f"Saved checkpoint: {path}")

    # ---- Final save ----
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")

    final_path = "checkpoints/model_final.pt"
    torch.save({
        "step": cfg["max_steps"],
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": cfg,
        "vocab_size": tok.vocab_size,
    }, final_path)
    print(f"Saved final model: {final_path}")

    # Final generation sample
    print("\n" + "=" * 60)
    print("FINAL GENERATION SAMPLE")
    print("=" * 60)
    sample = generate_sample(model, tok, length=500, temperature=0.8)
    print(sample)


if __name__ == "__main__":
    train()