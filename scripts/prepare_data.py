"""
Tokenize a HF dataset into uint16 shards for pretraining.

RUN THIS OFFLINE (your own machine / cheap CPU box) — never on a rented GPU.
For the 10B-token FineWeb-Edu sample expect several hours of CPU time and
~19GB of output; upload the shard directory to a (free) HF dataset repo and
pull it onto the training box.

    # Phase 1 validation data (~30 min):
    python scripts/prepare_data.py --dataset roneneldan/TinyStories \
        --tokenizer data/tokenizer_32k.json --out data/shards_tinystories \
        --val-tokens 5000000

    # Phase 2 pretraining data:
    python scripts/prepare_data.py --dataset HuggingFaceFW/fineweb-edu \
        --config sample-10BT --tokenizer data/tokenizer_32k.json \
        --out data/shards_fineweb --val-tokens 100000000
"""

import argparse

from src.llm.data import write_meta, write_shards
from src.llm.tokenizer import Tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--val-tokens", type=int, default=100_000_000,
                   help="first N tokens become the validation split")
    p.add_argument("--max-docs", type=int, default=None)
    args = p.parse_args()

    from datasets import load_dataset

    tok = Tokenizer.load(args.tokenizer)
    ds = load_dataset(args.dataset, name=args.config, split=args.split, streaming=True)

    state = {"n": 0, "val_done": False, "val_count": 0}

    def tokens(target_split: str):
        """Yield per-document token lists (with <|endoftext|> separators).
        The first --val-tokens go to val, the rest to train."""
        from tqdm import tqdm
        for ex in tqdm(ds, desc=target_split):
            if args.max_docs and state["n"] >= args.max_docs:
                return
            state["n"] += 1
            ids = tok.encode(ex["text"], add_eot=True)
            if not state["val_done"]:
                state["val_count"] += len(ids)
                if state["val_count"] >= args.val_tokens:
                    state["val_done"] = True
                if target_split == "val":
                    yield ids
                    if state["val_done"]:
                        return
            else:
                if target_split == "train":
                    yield ids

    n_val = write_shards(tokens("val"), args.out, split="val")
    n_train = write_shards(tokens("train"), args.out, split="train")
    write_meta(args.out, args.tokenizer, {"train": n_train, "val": n_val})
    print(f"wrote {n_train/1e9:.2f}B train + {n_val/1e6:.0f}M val tokens to {args.out}")


if __name__ == "__main__":
    main()
