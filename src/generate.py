"""
Interactive text generation with the trained model.

Three modes:
1. Generate from a prompt you type
2. Compare different temperatures side by side
3. Compare greedy vs top-k vs full sampling
"""

import torch
from src.tokenizer import CharTokenizer
from src.model import GPT


def load_model(checkpoint_path: str, data_path: str = "data/input.txt"):
    """Load a trained model from a checkpoint."""
    # Rebuild the tokenizer from the original data
    with open(data_path, "r") as f:
        text = f.read()
    tok = CharTokenizer(text)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    # Rebuild model with same config
    model = GPT(
        vocab_size=tok.vocab_size,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        block_size=cfg["block_size"],
        dropout=0.0,  # No dropout during inference
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model from step {ckpt['step']} ({n_params:,} params)")
    return model, tok, cfg


def generate(model, tok, prompt="\n", length=500, temperature=1.0, top_k=None):
    """Generate text from a prompt."""
    tokens = tok.encode(prompt)
    ctx = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(ctx, max_new_tokens=length, temperature=temperature, top_k=top_k)
    return tok.decode(out[0].tolist())


def demo_temperature(model, tok, prompt="KING"):
    """Show how temperature affects generation."""
    print("=" * 60)
    print(f"TEMPERATURE COMPARISON | Prompt: '{prompt}'")
    print("=" * 60)

    for temp in [0.3, 0.8, 1.0, 1.5]:
        text = generate(model, tok, prompt=prompt, length=300, temperature=temp)
        print(f"\n--- Temperature = {temp} ---")
        print(text)
        print()


def demo_sampling(model, tok, prompt="ROMEO:"):
    """Compare greedy, top-k, and full sampling."""
    print("=" * 60)
    print(f"SAMPLING STRATEGY COMPARISON | Prompt: '{prompt}'")
    print("=" * 60)

    # Greedy (temperature very low ≈ always pick the most likely)
    print("\n--- Greedy (temp=0.1) ---")
    print(generate(model, tok, prompt=prompt, length=300, temperature=0.1))

    # Top-k = 10
    print("\n--- Top-k = 10, temp=0.8 ---")
    print(generate(model, tok, prompt=prompt, length=300, temperature=0.8, top_k=10))

    # Full sampling
    print("\n--- Full sampling, temp=1.0 ---")
    print(generate(model, tok, prompt=prompt, length=300, temperature=1.0))


def interactive(model, tok, cfg):
    """Interactive prompt loop."""
    print("=" * 60)
    print("INTERACTIVE MODE")
    print("Type a prompt and hit Enter. Type 'quit' to exit.")
    print("Commands: /temp 0.5  /topk 10  /len 300")
    print("=" * 60)

    temperature = 0.8
    top_k = None
    length = 500

    while True:
        prompt = input("\nPrompt> ")
        if prompt.lower() == "quit":
            break
        if prompt.startswith("/temp"):
            temperature = float(prompt.split()[1])
            print(f"Temperature set to {temperature}")
            continue
        if prompt.startswith("/topk"):
            val = prompt.split()[1]
            top_k = None if val.lower() == "none" else int(val)
            print(f"Top-k set to {top_k}")
            continue
        if prompt.startswith("/len"):
            length = int(prompt.split()[1])
            print(f"Length set to {length}")
            continue

        text = generate(model, tok, prompt=prompt, length=length,
                        temperature=temperature, top_k=top_k)
        print(text)


if __name__ == "__main__":
    model, tok, cfg = load_model("checkpoints/model_final.pt")

    # Run the demos first
    demo_temperature(model, tok)
    demo_sampling(model, tok)

    # Then drop into interactive mode
    interactive(model, tok, cfg)