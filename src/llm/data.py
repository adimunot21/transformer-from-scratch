"""
Data pipeline: offline tokenization into uint16 shards + a packed loader.

Why this design (vs. the legacy CharDataset):
- Pretraining data (~10B tokens) must be tokenized ONCE, offline, on cheap
  hardware — never on-the-fly on a $2/hr GPU. Tokens are stored as raw
  uint16 (vocab 32768 fits) in .bin shards of ~100M tokens.
- The loader memory-maps shards (no RAM blowup) and yields fixed-shape
  (B, T) batches — fixed shapes are required for torch.compile to avoid
  recompilation stalls.
- Documents are joined with <|endoftext|> so the model learns where
  documents END — without separators it blends documents and never stops
  generating.

Shard layout: data_dir/{train,val}_{i:04d}.bin, plus meta.json recording the
tokenizer path and token counts.
"""

import json
from pathlib import Path

import numpy as np
import torch

SHARD_TOKENS = 100_000_000  # tokens per shard file


def write_shards(token_iterator, out_dir: str, split: str = "train",
                 shard_tokens: int = SHARD_TOKENS) -> int:
    """
    Stream token-id lists (one per document, WITH trailing <|endoftext|>)
    into uint16 .bin shards. Returns total tokens written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    buf = np.empty(shard_tokens, dtype=np.uint16)
    filled, shard_idx, total = 0, 0, 0

    def flush(n):
        nonlocal shard_idx
        path = out / f"{split}_{shard_idx:04d}.bin"
        buf[:n].tofile(path)
        shard_idx += 1

    for ids in token_iterator:
        arr = np.asarray(ids, dtype=np.uint16)
        total += len(arr)
        while len(arr) > 0:
            space = shard_tokens - filled
            take = min(space, len(arr))
            buf[filled : filled + take] = arr[:take]
            filled += take
            arr = arr[take:]
            if filled == shard_tokens:
                flush(filled)
                filled = 0
    if filled > 0:
        flush(filled)

    return total


class ShardedDataLoader:
    """
    Iterates fixed-shape (B, T) next-token batches over memmapped shards.

    Sequential within a shard with a per-epoch random starting offset —
    adjacent batches don't overlap, and state (shard index + position) is
    checkpointable so training can resume mid-epoch deterministically.
    """

    def __init__(self, data_dir: str, split: str, batch_size: int, block_size: int,
                 device: str = "cpu", seed: int = 0):
        self.paths = sorted(Path(data_dir).glob(f"{split}_*.bin"))
        assert self.paths, f"no {split}_*.bin shards in {data_dir}"
        self.B, self.T = batch_size, block_size
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.shard_idx = 0
        self.pos = 0
        self._load_shard()

    def _load_shard(self):
        self.tokens = np.memmap(self.paths[self.shard_idx], dtype=np.uint16, mode="r")
        # Random small offset decorrelates epochs without full shuffling.
        self.pos = int(self.rng.integers(0, self.T))

    def _advance_shard(self):
        self.shard_idx = (self.shard_idx + 1) % len(self.paths)
        self._load_shard()

    def next_batch(self):
        need = self.B * self.T + 1
        if self.pos + need > len(self.tokens):
            self._advance_shard()
        chunk = self.tokens[self.pos : self.pos + need].astype(np.int64)
        self.pos += self.B * self.T
        x = torch.from_numpy(chunk[:-1]).view(self.B, self.T)
        y = torch.from_numpy(chunk[1:]).view(self.B, self.T)
        if self.device.startswith("cuda"):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    def state_dict(self) -> dict:
        return {"shard_idx": self.shard_idx, "pos": self.pos,
                "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict):
        self.rng.bit_generator.state = state["rng"]
        self.shard_idx = state["shard_idx"]
        self.tokens = np.memmap(self.paths[self.shard_idx], dtype=np.uint16, mode="r")
        self.pos = state["pos"]


def write_meta(out_dir: str, tokenizer_path: str, counts: dict):
    meta = {"tokenizer": tokenizer_path, "tokens": counts}
    with open(Path(out_dir) / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
