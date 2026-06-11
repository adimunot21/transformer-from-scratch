"""
Hand-rolled zero-shot evals: HellaSwag and ARC-Easy.

Method (the standard loglikelihood approach, written out from scratch):
for each multiple-choice question, score every candidate completion by the
model's average per-token log-probability of the completion GIVEN the
context, and pick the best-scoring candidate. Length normalization matters:
without it, shorter completions win by default.

HellaSwag random baseline = 25%. GPT-2 124M scores ~31% acc_norm — that is
the bar for the Phase 2 pretrain (target: >= 29%).

Datasets are loaded from HF (`Rowan/hellaswag`, `allenai/ai2_arc`); run on
the training box, not in CI.
"""

import torch
import torch.nn.functional as F

from src.llm.model import Transformer
from src.llm.tokenizer import Tokenizer


@torch.no_grad()
def completion_logprob(model: Transformer, ctx_ids: list[int], completion_ids: list[int],
                       device: str = "cpu") -> tuple[float, float]:
    """
    Return (sum_logprob, mean_logprob) of completion_ids given ctx_ids.

    One forward pass over [ctx + completion]; the logits at positions
    len(ctx)-1 .. end-1 predict the completion tokens.
    """
    ids = (ctx_ids + completion_ids)[-model.cfg.block_size :]
    n_completion = min(len(completion_ids), len(ids) - 1)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = model(x)
    logprobs = F.log_softmax(logits[0].float(), dim=-1)
    # logits[t] predicts ids[t+1]; completion starts at len(ids)-n_completion
    start = len(ids) - n_completion
    token_lps = logprobs[start - 1 : -1].gather(
        1, x[0, start:].unsqueeze(1)
    ).squeeze(1)
    total = token_lps.sum().item()
    return total, total / n_completion


@torch.no_grad()
def score_multiple_choice(model: Transformer, tok: Tokenizer, context: str,
                          choices: list[str], device: str = "cpu") -> int:
    """Pick the choice with the highest length-normalized logprob."""
    model.eval()
    ctx_ids = tok.encode(context)
    scores = []
    for choice in choices:
        # leading space: completions continue the context mid-sentence
        comp_ids = tok.encode(" " + choice.lstrip())
        _, mean_lp = completion_logprob(model, ctx_ids, comp_ids, device)
        scores.append(mean_lp)
    return max(range(len(scores)), key=scores.__getitem__)


def run_hellaswag(model, tok, device="cpu", limit: int | None = 1000) -> float:
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    correct = 0
    for ex in ds:
        pred = score_multiple_choice(model, tok, ex["ctx"], ex["endings"], device)
        correct += int(pred == int(ex["label"]))
    acc = correct / len(ds)
    print(f"HellaSwag acc_norm: {acc:.4f} ({correct}/{len(ds)})  [random=0.25]")
    return acc


def run_arc_easy(model, tok, device="cpu", limit: int | None = None) -> float:
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    correct = 0
    for ex in ds:
        choices = ex["choices"]["text"]
        labels = ex["choices"]["label"]
        ctx = f"Question: {ex['question']}\nAnswer:"
        pred = score_multiple_choice(model, tok, ctx, choices, device)
        correct += int(labels[pred] == ex["answerKey"])
    acc = correct / len(ds)
    print(f"ARC-Easy acc: {acc:.4f} ({correct}/{len(ds)})  [random≈0.25]")
    return acc


if __name__ == "__main__":
    import argparse

    from src.llm.train import load_model_from_checkpoint, pick_device

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--limit", type=int, default=1000)
    args = p.parse_args()

    device = pick_device()
    model, _ = load_model_from_checkpoint(args.ckpt, device)
    tok = Tokenizer.load(args.tokenizer)
    run_hellaswag(model, tok, device, args.limit)
    run_arc_easy(model, tok, device, args.limit)
