# Transformer From Scratch

A complete implementation of the Transformer architecture in PyTorch — no `nn.Transformer`, no HuggingFace, no shortcuts. Every component (multi-head attention, positional encoding, layer norm, residual connections, training loop) built by hand.

Extends beyond standard language modeling into **BPE tokenization** and a **Decision Transformer** for reinforcement learning, demonstrating that the same attention mechanism powers both text generation and robot control.

## v2: From Shakespeare Toy to a Chatbot (`src/llm/`)

The original project (below) proves the mechanics; **v2 builds a model you can actually talk to**: a ~101M-parameter Llama-style LM, pretrained on 10B tokens of FineWeb-Edu and instruction-tuned on smol-smoltalk — for a total GPU budget of roughly £30–45 on a single rented H100.

What's modern about the v2 architecture (all hand-written, unit-tested):

| Legacy (v1, GPT-2 style) | v2 (Llama-3 style) |
|---|---|
| Learned positional embeddings | **RoPE** (rotary embeddings — relative positions, zero params) |
| LayerNorm with biases | **RMSNorm**, no biases anywhere |
| GELU MLP (4×) | **SwiGLU** gated MLP |
| Per-head Q/K/V linears in a Python loop | **Fused QKV** projection, one GEMM |
| Full multi-head attention | **Grouped-Query Attention** (4 KV heads shared by 12 Q heads) |
| Materialized T×T attention matrix | `F.scaled_dot_product_attention` (flash), with the from-scratch path kept as `attn_impl="manual"` and tested to match |
| O(T²)-per-token generation | **Static KV-cache** — O(T) per token, tested token-identical to the uncached path |
| Separate LM head | **Weight tying** with the token embedding |
| Char-level / pure-Python BPE | **32K byte-level BPE** trained on FineWeb-Edu (HF `tokenizers`), chat special tokens reserved from day one |
| CPU-only loop | bf16 autocast, grad accumulation, `torch.compile`, **fully resumable checkpoints** (model+optimizer+data position+RNG), wandb, tok/s + MFU metering |

### The three phases

1. **Phase 1 — pipeline proof (≈£0–5):** train the 25M config on TinyStories (`configs/tinystories_25m.py`) on a cheap GPU/Colab until it writes coherent stories. Gates: kill-and-resume reproduces the loss curve (tested in `tests/test_train.py`), cached == uncached generation.
2. **Phase 2 — pretraining (≈£16–32):** 101M params × 10B FineWeb-Edu tokens on 1× H100 (`configs/pretrain_110m.py`), ~7–9 hrs. Success bar: val loss ≈ 3.0–3.3, HellaSwag acc_norm ≥ 29% (`src/llm/evals.py`; GPT-2 124M ≈ 31%).
3. **Phase 3 — chatbot (≈£3–8):** SFT on smol-smoltalk with ChatML template and prompt-masked loss (`scripts/run_sft.py`), then chat via `python -m src.llm.chat --ckpt ... --tokenizer ...`.

### v2 quickstart

```bash
pip install -r requirements-v2.txt
pytest tests/ -v                       # 31 correctness gates, CPU, ~4s

# Phase 1 (run on any GPU box / Colab):
python scripts/train_tokenizer.py --out data/tokenizer_32k.json
python scripts/prepare_data.py --dataset roneneldan/TinyStories \
    --tokenizer data/tokenizer_32k.json --out data/shards_tinystories --val-tokens 5000000
python -m src.llm.train configs/tinystories_25m.py

# Phase 2 (tokenize offline first, then on the rented H100):
python scripts/prepare_data.py --dataset HuggingFaceFW/fineweb-edu --config sample-10BT \
    --tokenizer data/tokenizer_32k.json --out data/shards_fineweb
python -m src.llm.train configs/pretrain_110m.py            # add --resume checkpoints/.../latest.pt after a kill

# Phase 3:
python scripts/run_sft.py --ckpt checkpoints/pretrain_110m/best.pt --tokenizer data/tokenizer_32k.json
python -m src.llm.chat --ckpt checkpoints/sft/final.pt --tokenizer data/tokenizer_32k.json
```

Status: code + tests complete; training runs pending. Results will be reported here honestly (loss curves, evals, cost ledger, limitations).

---

## v1: The original educational project

## What's Inside

### Core Transformer (Phases 1–5)
- **Character-level GPT** trained on Shakespeare (~841K parameters)
- Implements: multi-head causal self-attention, feed-forward network, pre-norm residual blocks, learnable positional embeddings, cosine LR schedule with warmup
- Trains to **1.43 train / 1.62 val loss** — generates coherent Shakespeare-style dialogue
- Attention visualizations showing what each head learned
- Ablation study: depth vs width, confirming layers matter more than heads at this scale

### BPE Tokenizer (Phase 6)
- **Byte-Pair Encoding** implemented from scratch — the same algorithm used by GPT-2/3/4
- Learns 512 merge operations, achieving **2x compression** over character-level
- Full train/encode/decode/save/load pipeline
- Trained a larger BPE-based model (2.1M params) on GPU with regularization analysis

### Decision Transformer (Phase 6)
- **Reinforcement learning via sequence modeling** — same Transformer blocks, different domain
- Trained offline on CartPole episodes, conditions on desired return-to-go
- Achieves **3.5x random baseline** performance, with return controllability:
  - Target 10 → achieves 10.1 (precise low-return control)
  - Target 500 → achieves 76 mean, 491 max
- Demonstrates that attention is a general-purpose sequence processor

## Architecture

```
Token/State Embedding + Positional/Timestep Embedding
  → Dropout
  → N × TransformerBlock
      → LayerNorm → Multi-Head Self-Attention → Residual
      → LayerNorm → Feed-Forward (GELU) → Residual
  → Final LayerNorm
  → Linear Projection → Logits
```

Every component from scratch using only: `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout`, `F.cross_entropy`, `F.softmax`, and autograd.

## Results

### Language Model — Character Level
| Metric | Value |
|--------|-------|
| Parameters | 841K |
| Train loss | 1.43 |
| Val loss | 1.62 |
| Training time | ~2.8 hrs (CPU) |

### Language Model — BPE
| Metric | Before Regularization | After Regularization |
|--------|----------------------|---------------------|
| Parameters | 3M | 2.1M |
| Train loss | 2.11 | 2.92 |
| Val loss | 3.51 | 3.54 |
| Overfit gap | 1.41 | 0.62 |

### Ablation Study
| Config | Val Loss (1500 steps) | Insight |
|--------|----------------------|---------|
| Baseline (4L, 4H) | 1.85 | Full model |
| 1 Head (4L, 1H) | 1.81 | Multiple heads matter less at small scale |
| 1 Layer (1L, 4H) | 1.97 | Depth matters more than head count |

### Decision Transformer — CartPole
| Target Return | Achieved (mean ± std) | Max |
|--------------|----------------------|-----|
| 10 | 10.1 ± 0.3 | 11 |
| 50 | 45.3 ± 9.0 | 66 |
| 200 | 71.5 ± 39.1 | 198 |
| 500 | 76.0 ± 70.2 | 491 |
| Random baseline | 22.2 ± 14.1 | — |

## Project Structure

```
transformer-from-scratch/
├── src/
│   ├── model.py               ← Full GPT: attention, FFN, blocks, generation
│   ├── tokenizer.py           ← Character-level tokenizer
│   ├── bpe_tokenizer.py       ← Byte-Pair Encoding from scratch
│   ├── dataset.py             ← PyTorch Dataset + DataLoader
│   ├── train.py               ← Training loop with LR schedule
│   ├── generate.py            ← Interactive text generation
│   └── decision_transformer.py ← Decision Transformer for RL
├── notebooks/
│   ├── explore.py             ← Attention visualization + ablations
│   ├── attention_maps.png     ← Attention heatmaps (all layers/heads)
│   ├── attention_detail.png   ← Detailed single-head attention
│   ├── ablation_study.png     ← Loss curves: baseline vs 1-head vs 1-layer
│   ├── bpe_training_loss.png  ← BPE model training curve
│   └── bpe_regularized_loss.png ← After overfitting fix
├── data/
│   └── bpe_tokenizer_512.json ← Trained BPE vocabulary
└── checkpoints/               ← Model weights (local only, not in repo)
```

## Setup

```bash
conda create -n transformer python=3.11 -y
conda activate transformer
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib tqdm

# Download training data
curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Usage

```bash
# Train the character-level model (~2.8 hrs on CPU, ~15 min on GPU)
python -m src.train

# Interactive text generation
python -m src.generate

# Train BPE tokenizer
python -m src.bpe_tokenizer

# Attention visualization + ablation experiments
python notebooks/explore.py
```

## Key Takeaways

1. **Attention is general.** The same mechanism that predicts Shakespeare also controls a CartPole agent. The architecture doesn't care about the domain — it processes sequences.

2. **Depth > width at small scale.** Removing layers hurts more than removing heads, because depth enables feature composition across levels of abstraction.

3. **Regularization is about the data-to-parameter ratio.** The BPE model overfitting (gap 1.41) was fixed by dropout + weight decay + fewer layers (gap 0.62), not by changing the architecture.

4. **BPE compression matters.** 2x compression means the same context window sees twice as much text, enabling longer-range pattern learning.

## Course: Learn Transformers From Scratch

This project includes a full 8-chapter written course explaining every concept, every line of code, and every design decision. Written for someone who knows Python but is new to ML/deep learning.

| Chapter | Topic | What You'll Learn |
|---------|-------|-------------------|
| [0: Introduction](course/00_introduction.md) | Big picture & setup | What a language model is, the Transformer architecture overview, environment setup |
| [1: Data Pipeline](course/01_data_pipeline.md) | Tokenization & datasets | Character tokenizer, next-character prediction, PyTorch Dataset/DataLoader |
| [2: Embeddings](course/02_embeddings.md) | Turning numbers into meaning | Token embeddings, positional embeddings, broadcasting, dropout |
| [3: Attention](course/03_attention.md) | The heart of the Transformer | Q/K/V projections, scaled dot-product, causal masking, multi-head attention |
| [4: Transformer Block](course/04_transformer_block.md) | Assembling the full model | Feed-forward network, GELU, layer norm, residual connections, full GPT |
| [5: Training](course/05_training.md) | Teaching the model | AdamW optimizer, LR scheduling, gradient clipping, loss curves, perplexity |
| [6: Generation](course/06_generation.md) | Making the model write | Autoregressive decoding, temperature, top-k sampling, sampling strategies |
| [7: BPE Tokenizer](course/07_bpe_tokenizer.md) | The tokenizer behind GPT | Byte-Pair Encoding algorithm, byte-level encoding, compression, subword discovery |
| [8: Decision Transformer](course/08_decision_transformer.md) | RL via sequence modeling | Return conditioning, interleaved sequences, CartPole control, robotics connections |

## Built With
- PyTorch (CPU + CUDA)
- No high-level Transformer libraries — everything from scratch
- Google Colab (T4 GPU) for BPE and Decision Transformer training