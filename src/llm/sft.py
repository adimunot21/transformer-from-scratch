"""
Supervised fine-tuning (SFT): turn the pretrained LM into a chat model.

Two pieces:
1. A ChatML-style template using the special tokens reserved when the
   tokenizer was trained:
       <|im_start|>user\n{message}<|im_end|>\n
       <|im_start|>assistant\n{reply}<|im_end|>\n
2. Loss masking: the model is trained ONLY on assistant tokens. Every
   prompt/template/padding position gets label -100 (cross_entropy's
   ignore_index), so the model learns to WRITE replies, not to predict
   user messages.

The classic silent failure here is a masking bug — the model still trains,
loss still falls, but it learns the wrong objective. Hence
`debug_print_batch`, which decodes exactly which tokens carry loss; run it
on one batch before every SFT launch.
"""

import torch
from torch.utils.data import DataLoader, Dataset

from src.llm.tokenizer import Tokenizer

IGNORE_INDEX = -100


def render_conversation(messages: list[dict], tok: Tokenizer) -> tuple[list[int], list[int]]:
    """
    Render [{"role": "user"|"assistant"|"system", "content": str}, ...] into
    (input_ids, labels). Labels are IGNORE_INDEX everywhere except assistant
    content tokens and the <|im_end|> that closes each assistant turn (the
    model must learn to STOP).
    """
    input_ids: list[int] = []
    labels: list[int] = []

    def add(ids: list[int], learn: bool):
        input_ids.extend(ids)
        labels.extend(ids if learn else [IGNORE_INDEX] * len(ids))

    for msg in messages:
        role, content = msg["role"], msg["content"]
        header = [tok.im_start_id] + tok.encode(f"{role}\n")
        add(header, learn=False)
        body = tok.encode(content)
        closer = [tok.im_end_id] + tok.encode("\n")
        if role == "assistant":
            add(body, learn=True)
            add(closer[:1], learn=True)    # learn to emit <|im_end|>
            add(closer[1:], learn=False)   # trailing newline is template
        else:
            add(body, learn=False)
            add(closer, learn=False)
    return input_ids, labels


class SFTDataset(Dataset):
    """
    Conversations rendered to fixed-length (input, target) pairs.

    Targets are inputs shifted one position left (next-token prediction),
    with the label mask shifted accordingly; sequences are truncated/padded
    to block_size. Conversations longer than block_size are dropped at
    construction (the model can't see them anyway).
    """

    def __init__(self, conversations: list[list[dict]], tok: Tokenizer, block_size: int):
        self.tok = tok
        self.block_size = block_size
        self.examples = []
        for messages in conversations:
            ids, labels = render_conversation(messages, tok)
            if len(ids) <= 1 or len(ids) > block_size:
                continue
            self.examples.append((ids, labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ids, labels = self.examples[i]
        x = torch.full((self.block_size,), self.tok.pad_id, dtype=torch.long)
        y = torch.full((self.block_size,), IGNORE_INDEX, dtype=torch.long)
        n = len(ids)
        x[: n] = torch.tensor(ids, dtype=torch.long)
        # next-token shift: position t predicts token t+1
        y[: n - 1] = torch.tensor(labels[1:], dtype=torch.long)
        return x, y


def debug_print_batch(x: torch.Tensor, y: torch.Tensor, tok: Tokenizer, row: int = 0):
    """Decode one row, marking the tokens that carry loss with »«.
    Run this before launching any SFT run."""
    pieces = []
    for t in range(x.shape[1] - 1):
        token = tok.decode([x[row, t + 1].item()])
        pieces.append(f"»{token}«" if y[row, t].item() != IGNORE_INDEX else token)
    print("".join(pieces))


def sft_train(model, dataset: SFTDataset, *, epochs: int = 2, lr: float = 1e-4,
              min_lr: float = 1e-5, batch_size: int = 16, grad_clip: float = 1.0,
              device: str = "cpu", weight_decay: float = 0.1,
              log_every: int = 20, val_dataset: SFTDataset | None = None):
    """Simple SFT loop: cosine lr, no warmup (weights are already good)."""
    import math

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = model.configure_optimizer(weight_decay, lr, (0.9, 0.95),
                                          "cuda" if device.startswith("cuda") else "cpu")
    total_steps = epochs * len(loader)
    model.train()
    step = 0
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            cur_lr = min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * step / total_steps))
            for g in optimizer.param_groups:
                g["lr"] = cur_lr
            _, loss = model(x, targets=y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if step % log_every == 0:
                print(f"epoch {epoch} step {step}/{total_steps} | loss {loss.item():.4f} | lr {cur_lr:.2e}")
            step += 1
        if val_dataset is not None:
            val_loss = evaluate_sft(model, val_dataset, batch_size, device)
            print(f"epoch {epoch} | val loss {val_loss:.4f}")
    return model


@torch.no_grad()
def evaluate_sft(model, dataset: SFTDataset, batch_size: int, device: str) -> float:
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)
