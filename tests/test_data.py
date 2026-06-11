"""Tests for the shard writer and packed loader."""

import numpy as np
import torch

from src.llm.data import ShardedDataLoader, write_shards


def make_shards(tmp_path, n_tokens=10_000, shard_tokens=4_000):
    docs = []
    pos = 0
    while pos < n_tokens:
        length = min(np.random.randint(50, 500), n_tokens - pos)
        docs.append(list(range(pos, pos + length)))  # sequential ids: easy to verify
        pos += length
    # keep ids in uint16 range
    docs = [[t % 60000 for t in d] for d in docs]
    total = write_shards(iter(docs), str(tmp_path), split="train", shard_tokens=shard_tokens)
    return total


def test_shard_roundtrip(tmp_path):
    total = make_shards(tmp_path)
    files = sorted(tmp_path.glob("train_*.bin"))
    assert len(files) >= 2, "should have split into multiple shards"
    read_back = np.concatenate([np.fromfile(f, dtype=np.uint16) for f in files])
    assert len(read_back) == total
    # the stream is the concatenation of docs -> first tokens are 0,1,2,...
    assert list(read_back[:10]) == list(range(10))


def test_loader_next_token_shift(tmp_path):
    make_shards(tmp_path)
    loader = ShardedDataLoader(str(tmp_path), "train", batch_size=2, block_size=16)
    x, y = loader.next_batch()
    assert x.shape == (2, 16) and y.shape == (2, 16)
    assert x.dtype == torch.int64
    # y must be x shifted by one within the flat stream
    flat_x = x.view(-1)
    flat_y = y.view(-1)
    assert torch.equal(flat_x[1:], flat_y[:-1])


def test_loader_resume_is_deterministic(tmp_path):
    make_shards(tmp_path)
    loader = ShardedDataLoader(str(tmp_path), "train", batch_size=2, block_size=8, seed=3)
    for _ in range(5):
        loader.next_batch()
    state = loader.state_dict()
    expected = [loader.next_batch() for _ in range(3)]

    loader2 = ShardedDataLoader(str(tmp_path), "train", batch_size=2, block_size=8, seed=99)
    loader2.load_state_dict(state)
    actual = [loader2.next_batch() for _ in range(3)]

    for (xa, ya), (xb, yb) in zip(expected, actual):
        assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_loader_wraps_across_shards(tmp_path):
    make_shards(tmp_path, n_tokens=2_000, shard_tokens=1_000)
    loader = ShardedDataLoader(str(tmp_path), "train", batch_size=4, block_size=32)
    for _ in range(50):  # far more batches than one shard holds
        x, y = loader.next_batch()
        assert x.shape == (4, 32)
