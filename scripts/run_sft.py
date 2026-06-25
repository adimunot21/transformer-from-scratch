"""
Phase 3: SFT the pretrained model on smol-smoltalk.

    python scripts/run_sft.py --ckpt checkpoints/pretrain_110m/best.pt \
        --tokenizer data/tokenizer_32k.json --out checkpoints/sft

smol-smoltalk is the SFT mix used for SmolLM2-135M-Instruct — i.e. curated
for models of exactly this size. Conversations longer than the context
window are dropped by SFTDataset.

Expect < 1 hr on H100, 2-3 hr on a 4090. Watch val loss per epoch — small
models overfit SFT quickly; 2-3 epochs is plenty.
"""

import argparse
import os

import torch

from src.llm.sft import SFTDataset, debug_print_batch, sft_train
from src.llm.tokenizer import Tokenizer
from src.llm.train import load_model_from_checkpoint, pick_device


def load_conversations(limit: int | None = None) -> list[list[dict]]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [ex["messages"] for ex in ds]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", default="checkpoints/sft")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="cap conversations (debugging)")
    args = p.parse_args()

    device = pick_device()
    tok = Tokenizer.load(args.tokenizer)
    model, _ = load_model_from_checkpoint(args.ckpt, device)
    print(f"loaded {model.num_params():,} params on {device}")

    convs = load_conversations(args.limit)
    n_val = max(len(convs) // 100, 50)
    train_ds = SFTDataset(convs[n_val:], tok, model.cfg.block_size)
    val_ds = SFTDataset(convs[:n_val], tok, model.cfg.block_size)
    print(f"{len(train_ds)} train / {len(val_ds)} val conversations fit in context")

    # ALWAYS eyeball the mask before burning GPU time: »tokens« carry loss.
    from torch.utils.data import DataLoader
    x, y = next(iter(DataLoader(train_ds, batch_size=2)))
    debug_print_batch(x, y, tok)

    sft_train(model, train_ds, epochs=args.epochs, lr=args.lr,
              batch_size=args.batch_size, device=device, val_dataset=val_ds)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "final.pt")
    torch.save({"model_state": model.state_dict(),
                "model_config": model.cfg.to_dict()}, path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
