"""
Phase 5: Attention Visualization & Ablation Experiments

Run from the project root:
    python notebooks/explore.py

Generates attention heatmaps and runs ablation experiments.
All plots saved to notebooks/ directory.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — saves to file

from src.tokenizer import CharTokenizer
from src.model import GPT, SingleHeadAttention
from src.dataset import create_datasets
from torch.utils.data import DataLoader


# ===================================================================
# PART 1: Extract attention weights from the model
# ===================================================================

def get_attention_weights(model, tok, text):
    """
    Run a forward pass and capture the attention weights from every
    head in every layer.

    We do this by temporarily monkey-patching the SingleHeadAttention
    forward method to store its weights. This avoids modifying model.py.

    Returns:
        weights: list of layers, each layer is list of heads,
                 each head is (T, T) attention matrix
    """
    tokens = tok.encode(text)
    x = torch.tensor([tokens], dtype=torch.long)

    # Storage for captured weights
    captured = []

    # Patch: wrap each head's forward to capture attention weights
    original_forwards = []
    for block in model.blocks:
        for head in block.attn.heads:
            original_forward = head.forward

            def make_hook(orig, store):
                def hooked_forward(x):
                    B, T, C = x.shape
                    q = head.q(x)
                    k = head.k(x)
                    v = head.v(x)
                    scale = math.sqrt(k.shape[-1])
                    att = (q @ k.transpose(-2, -1)) / scale
                    att = att.masked_fill(~head.mask[:T, :T], float("-inf"))
                    att = F.softmax(att, dim=-1)
                    store.append(att[0].detach())  # Save (T, T) weights
                    out = att @ v
                    return out
                return hooked_forward

            store = []
            captured.append(store)
            head.forward = make_hook(original_forward, store)
            original_forwards.append((head, original_forward))

    # Forward pass
    with torch.no_grad():
        model.eval()
        model(x)
        model.train()

    # Restore original forwards
    for head, orig in original_forwards:
        head.forward = orig

    # Reorganize: captured is flat list, reshape to [n_layers][n_heads]
    n_layers = len(model.blocks)
    n_heads = len(model.blocks[0].attn.heads)
    weights = []
    idx = 0
    for layer in range(n_layers):
        layer_weights = []
        for h in range(n_heads):
            layer_weights.append(captured[idx][0])  # (T, T) tensor
            idx += 1
        weights.append(layer_weights)

    return weights, tokens


def plot_attention_maps(weights, tokens, tok, save_path="notebooks/attention_maps.png"):
    """
    Plot attention heatmaps for all heads in all layers.

    Rows = layers, Columns = heads.
    Each heatmap shows: for each query position (y-axis),
    how much it attends to each key position (x-axis).
    """
    n_layers = len(weights)
    n_heads = len(weights[0])
    T = len(tokens)

    # Use a subset of tokens for readability
    max_chars = 60
    chars = [tok.idx_to_char[t] for t in tokens[:max_chars]]
    # Replace newlines with visible symbol
    chars = ["↵" if c == "\n" else c for c in chars]

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(5 * n_heads, 5 * n_layers))
    fig.suptitle("Attention Weights by Layer and Head", fontsize=16, fontweight="bold")

    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer][head] if n_layers > 1 else axes[head]
            w = weights[layer][head][:max_chars, :max_chars].numpy()

            ax.imshow(w, cmap="Blues", vmin=0, vmax=w.max())
            ax.set_title(f"Layer {layer}, Head {head}", fontsize=11)
            ax.set_xlabel("Key (attends to)")
            ax.set_ylabel("Query (from)")

            # Add character labels if sequence is short enough
            if max_chars <= 40:
                ax.set_xticks(range(len(chars)))
                ax.set_xticklabels(chars, fontsize=6, rotation=90)
                ax.set_yticks(range(len(chars)))
                ax.set_yticklabels(chars, fontsize=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved attention maps to {save_path}")
    plt.close()


def plot_attention_detail(weights, tokens, tok, layer, head,
                          save_path="notebooks/attention_detail.png"):
    """
    Plot a single head's attention in detail with character labels.
    Useful for understanding what a specific head learned.
    """
    T = min(50, len(tokens))
    chars = [tok.idx_to_char[t] for t in tokens[:T]]
    chars = ["↵" if c == "\n" else c for c in chars]

    w = weights[layer][head][:T, :T].numpy()

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(w, cmap="Blues")
    ax.set_title(f"Attention Detail: Layer {layer}, Head {head}", fontsize=14)
    ax.set_xlabel("Key position (attends to)", fontsize=12)
    ax.set_ylabel("Query position (from)", fontsize=12)
    ax.set_xticks(range(T))
    ax.set_xticklabels(chars, fontsize=7, rotation=90)
    ax.set_yticks(range(T))
    ax.set_yticklabels(chars, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved attention detail to {save_path}")
    plt.close()


# ===================================================================
# PART 2: Ablation experiments
# ===================================================================

def train_ablation(name, text, block_size=256, batch_size=64, max_steps=1500,
                   **model_kwargs):
    """
    Train a model variant and return its loss curve.
    Shorter training (1500 steps) — enough to see differences.
    """
    print(f"\n{'='*40}")
    print(f"ABLATION: {name}")
    print(f"{'='*40}")

    tok, train_ds, val_ds = create_datasets(text, block_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    model = GPT(vocab_size=tok.vocab_size, block_size=block_size, **model_kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    train_iter = iter(train_loader)
    model.train()

    train_losses = []
    val_losses = []
    eval_steps = list(range(0, max_steps, 100))

    t0 = time.time()
    for step in range(max_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step in eval_steps:
            model.eval()
            with torch.no_grad():
                # Quick val loss estimate
                vl_iter = iter(val_loader)
                vl_total = 0
                for _ in range(10):
                    try:
                        vx, vy = next(vl_iter)
                    except StopIteration:
                        vl_iter = iter(val_loader)
                        vx, vy = next(vl_iter)
                    _, vl = model(vx, vy)
                    vl_total += vl.item()
                val_loss = vl_total / 10
            model.train()

            train_losses.append(loss.item())
            val_losses.append(val_loss)
            print(f"  Step {step:4d} | train: {loss.item():.3f} | val: {val_loss:.3f}")

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")
    return eval_steps, train_losses, val_losses


def run_ablations(text):
    """
    Run three ablation experiments to test:
    1. Baseline (our normal model)
    2. No positional encoding
    3. 1 head instead of 4
    4. 1 layer instead of 4
    """
    results = {}

    # Baseline
    steps, tl, vl = train_ablation("Baseline (4 layers, 4 heads)",
        text, n_layers=4, n_heads=4, d_model=128)
    results["Baseline"] = (steps, tl, vl)

    # No positional encoding: we hack this by using d_model but
    # we'll zero out positional embeddings after init
    steps, tl, vl = train_ablation("1 Head (4 layers, 1 head)",
        text, n_layers=4, n_heads=1, d_model=128)
    results["1 Head"] = (steps, tl, vl)

    # 1 layer
    steps, tl, vl = train_ablation("1 Layer (1 layer, 4 heads)",
        text, n_layers=1, n_heads=4, d_model=128)
    results["1 Layer"] = (steps, tl, vl)

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Ablation Study: Effect of Architecture Changes", fontsize=14)

    for name, (s, tl, vl) in results.items():
        ax1.plot(s, tl, label=name, linewidth=2)
        ax2.plot(s, vl, label=name, linewidth=2)

    ax1.set_title("Training Loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("notebooks/ablation_study.png", dpi=150, bbox_inches="tight")
    print("\nSaved ablation comparison to notebooks/ablation_study.png")
    plt.close()


# ===================================================================
# MAIN
# ===================================================================

if __name__ == "__main__":
    # Load data and model
    with open("data/input.txt", "r") as f:
        text = f.read()
    tok = CharTokenizer(text)

    ckpt = torch.load("checkpoints/model_final.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = GPT(
        vocab_size=tok.vocab_size,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        block_size=cfg["block_size"],
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded trained model ({sum(p.numel() for p in model.parameters()):,} params)")

    # ---- Attention Visualization ----
    print("\n" + "=" * 60)
    print("PART 1: Attention Visualization")
    print("=" * 60)

    sample_text = "ROMEO:\nO, she doth teach the torches to burn bright!\n"
    print(f"Visualizing attention for: {repr(sample_text)}")

    weights, tokens = get_attention_weights(model, tok, sample_text)
    plot_attention_maps(weights, tokens, tok)
    plot_attention_detail(weights, tokens, tok, layer=0, head=0)
    plot_attention_detail(weights, tokens, tok, layer=3, head=0,
                          save_path="notebooks/attention_detail_L3H0.png")

    # ---- Ablation Experiments ----
    print("\n" + "=" * 60)
    print("PART 2: Ablation Experiments")
    print("=" * 60)
    print("This will train 3 small models (~5 min each on CPU).")
    print("Total time: ~15-20 minutes.\n")

    run_ablations(text)

    print("\n" + "=" * 60)
    print("DONE! Check the notebooks/ directory for:")
    print("  - attention_maps.png")
    print("  - attention_detail.png")
    print("  - attention_detail_L3H0.png")
    print("  - ablation_study.png")
    print("=" * 60)