"""
Train the 32K byte-level BPE tokenizer on a FineWeb-Edu sample.

This is run ONCE, before any data preparation. The tokenizer is then frozen:
every shard and checkpoint depends on it, so retraining it invalidates
everything downstream.

    python scripts/train_tokenizer.py --out data/tokenizer_32k.json \
        --dataset HuggingFaceFW/fineweb-edu --config sample-10BT --docs 400000
"""

import argparse

from src.llm.tokenizer import train_tokenizer


def text_iterator(dataset: str, config: str, n_docs: int):
    from datasets import load_dataset
    ds = load_dataset(dataset, name=config, split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= n_docs:
            break
        yield ex["text"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/tokenizer_32k.json")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--config", default="sample-10BT")
    p.add_argument("--docs", type=int, default=400_000, help="~2-4GB of text")
    p.add_argument("--vocab-size", type=int, default=32768)
    args = p.parse_args()

    tok = train_tokenizer(text_iterator(args.dataset, args.config, args.docs),
                          vocab_size=args.vocab_size)
    tok.save(args.out)
    print(f"saved {tok.vocab_size}-token tokenizer to {args.out}")

    sample = "The transformer architecture has revolutionized NLP."
    ids = tok.encode(sample)
    print(f"sanity: {len(sample.split())} words -> {len(ids)} tokens")
    assert tok.decode(ids) == sample


if __name__ == "__main__":
    main()
