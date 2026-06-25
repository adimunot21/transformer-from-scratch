"""
KV-cached autoregressive generation with temperature, top-k, and top-p.

The legacy generate() re-ran the full forward pass over the whole context for
every new token (O(T^2) per token). Here we pay the full pass once for the
prompt ("prefill"), then each subsequent step feeds ONE token through the model
against the cached K/V ("decode") — O(T) per token.
"""

import torch
import torch.nn.functional as F

from src.llm.model import KVCache, Transformer


def sample_next(logits: torch.Tensor, temperature: float = 1.0,
                top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
    """
    Sample one token id per batch row from (B, vocab) logits.

    Filter order matters: top-k first (keep k most likely), then top-p
    (keep the smallest set whose cumulative probability >= p), then sample.
    temperature == 0 means greedy argmax.
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, [-1]]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(dim=-1)
        # Drop a token if the cumulative probability BEFORE it already
        # reached top_p — keeps the smallest prefix with mass >= top_p
        # (the most likely token always survives).
        drop = probs.cumsum(dim=-1) - probs > top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(model: Transformer, idx: torch.Tensor, max_new_tokens: int,
             temperature: float = 0.8, top_k: int | None = None,
             top_p: float | None = None, stop_tokens: set[int] | None = None,
             use_cache: bool = True) -> torch.Tensor:
    """
    Generate up to max_new_tokens continuations of idx (B, T).

    stop_tokens: generation halts when every sequence has emitted a stop
    token (e.g. <|im_end|> in chat). Capped at the model's block_size.
    use_cache=False falls back to recomputing the full context each step —
    kept for the cache-equivalence unit test.
    """
    model.eval()
    cfg = model.cfg
    B, T = idx.shape
    max_new_tokens = min(max_new_tokens, cfg.block_size - T)
    if max_new_tokens <= 0:
        return idx

    device = idx.device
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    if use_cache:
        dtype = model.tok_emb.weight.dtype
        cache = KVCache(cfg, B, device, dtype=dtype)
        logits, _ = model(idx, kv_cache=cache)         # prefill
        for _ in range(max_new_tokens):
            next_tok = sample_next(logits[:, -1, :], temperature, top_k, top_p)
            idx = torch.cat([idx, next_tok], dim=1)
            if stop_tokens:
                finished |= torch.isin(next_tok.squeeze(1),
                                       torch.tensor(list(stop_tokens), device=device))
                if finished.all():
                    break
            if idx.shape[1] >= cfg.block_size:
                break
            logits, _ = model(next_tok, kv_cache=cache)  # decode one token
    else:
        for _ in range(max_new_tokens):
            logits, _ = model(idx[:, -cfg.block_size:])
            next_tok = sample_next(logits[:, -1, :], temperature, top_k, top_p)
            idx = torch.cat([idx, next_tok], dim=1)
            if stop_tokens:
                finished |= torch.isin(next_tok.squeeze(1),
                                       torch.tensor(list(stop_tokens), device=device))
                if finished.all():
                    break

    return idx
