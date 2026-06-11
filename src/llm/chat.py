"""
Interactive multi-turn chat CLI for the SFT'd model.

Usage:
    python -m src.llm.chat --ckpt checkpoints/sft/final.pt --tokenizer data/tokenizer_32k.json

The conversation is re-rendered through the chat template each turn and
generation stops at <|im_end|>. Commands: /temp 0.7, /topp 0.9, /reset, quit.
"""

import argparse

import torch

from src.llm.sample import generate
from src.llm.sft import render_conversation
from src.llm.tokenizer import Tokenizer
from src.llm.train import load_model_from_checkpoint, pick_device


def chat_once(model, tok: Tokenizer, messages: list[dict], device: str,
              temperature: float = 0.7, top_p: float = 0.9,
              max_new_tokens: int = 256) -> str:
    """Render history + assistant header, generate until <|im_end|>."""
    ids, _ = render_conversation(messages, tok)
    ids = ids + [tok.im_start_id] + tok.encode("assistant\n")
    # keep the most recent context that fits, leaving room to generate
    ids = ids[-(model.cfg.block_size - max_new_tokens):]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(model, x, max_new_tokens, temperature=temperature,
                   top_p=top_p, stop_tokens={tok.im_end_id})
    reply_ids = out[0, len(ids):].tolist()
    if tok.im_end_id in reply_ids:
        reply_ids = reply_ids[: reply_ids.index(tok.im_end_id)]
    return tok.decode(reply_ids).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--system", default=None, help="optional system prompt")
    args = p.parse_args()

    device = pick_device()
    model, _ = load_model_from_checkpoint(args.ckpt, device)
    model.eval()
    tok = Tokenizer.load(args.tokenizer)
    print(f"loaded {model.num_params():,} param model on {device}. "
          "Type a message ('quit' to exit, /reset to clear history).")

    temperature, top_p = 0.7, 0.9
    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user.lower() == "quit":
            break
        if user == "/reset":
            messages = messages[:1] if args.system else []
            print("(history cleared)")
            continue
        if user.startswith("/temp"):
            temperature = float(user.split()[1])
            print(f"(temperature={temperature})")
            continue
        if user.startswith("/topp"):
            top_p = float(user.split()[1])
            print(f"(top_p={top_p})")
            continue

        messages.append({"role": "user", "content": user})
        reply = chat_once(model, tok, messages, device, temperature, top_p)
        messages.append({"role": "assistant", "content": reply})
        print(f"\nassistant> {reply}")


if __name__ == "__main__":
    main()
