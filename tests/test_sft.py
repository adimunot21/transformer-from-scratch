"""SFT template + loss-masking tests."""

import pytest
import torch

from src.llm.sft import IGNORE_INDEX, SFTDataset, render_conversation
from src.llm.tokenizer import Tokenizer, train_tokenizer

CORPUS = [
    "user assistant system hello how are you today fine thanks",
    "What is the capital of France? Paris is the capital.",
    "def add(a, b): return a + b",
] * 60


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return train_tokenizer(iter(CORPUS), vocab_size=500)


CONV = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "Thanks!"},
    {"role": "assistant", "content": "You're welcome."},
]


def test_only_assistant_tokens_have_labels(tok):
    ids, labels = render_conversation(CONV, tok)
    assert len(ids) == len(labels)
    learned = [i for i, l in zip(ids, labels) if l != IGNORE_INDEX]
    decoded = tok.decode(learned)
    # everything the model learns must come from assistant turns (+ im_end)
    assert "Paris." in decoded
    assert "welcome" in decoded
    assert "France" not in decoded, "user tokens must be masked"
    assert "user" not in decoded, "role headers must be masked"


def test_im_end_is_learned(tok):
    ids, labels = render_conversation(CONV, tok)
    # every assistant <|im_end|> must carry a label (model learns to stop)
    learned_im_ends = sum(1 for i, l in zip(ids, labels)
                          if i == tok.im_end_id and l != IGNORE_INDEX)
    assert learned_im_ends == 2  # two assistant turns


def test_dataset_shift_and_padding(tok):
    block = 128
    ds = SFTDataset([CONV], tok, block_size=block)
    assert len(ds) == 1
    x, y = ds[0]
    assert x.shape == (block,) and y.shape == (block,)

    ids, labels = render_conversation(CONV, tok)
    n = len(ids)
    # x holds the conversation then padding
    assert x[:n].tolist() == ids
    assert (x[n:] == tok.pad_id).all()
    # y is labels shifted left by one; padding region is ignored
    assert y[: n - 1].tolist() == labels[1:]
    assert (y[n - 1 :] == IGNORE_INDEX).all()
    # the shift means: where y[t] != ignore, model at position t predicts x[t+1]
    for t in range(n - 1):
        if y[t].item() != IGNORE_INDEX:
            assert y[t].item() == x[t + 1].item()


def test_too_long_conversations_dropped(tok):
    long_conv = [{"role": "user", "content": "word " * 500},
                 {"role": "assistant", "content": "ok"}]
    ds = SFTDataset([long_conv], tok, block_size=64)
    assert len(ds) == 0


def test_loss_ignores_masked_positions(tok):
    """cross_entropy with ignore_index must skip masked labels entirely."""
    import torch.nn.functional as F
    logits = torch.randn(6, 10)
    labels = torch.tensor([1, IGNORE_INDEX, 3, IGNORE_INDEX, IGNORE_INDEX, 2])
    loss = F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX)
    manual = F.cross_entropy(logits[[0, 2, 5]], labels[[0, 2, 5]])
    assert torch.allclose(loss, manual)
