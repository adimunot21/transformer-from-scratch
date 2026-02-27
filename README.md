# Transformer From Scratch

A complete implementation of the Transformer architecture in PyTorch — no `nn.Transformer`, no HuggingFace, no shortcuts. Every component (multi-head attention, positional encoding, layer norm, residual connections, training loop) built by hand.

Extends beyond standard language modeling into **BPE tokenization** and a **Decision Transformer** for reinforcement learning, demonstrating that the same attention mechanism powers both text generation and robot control.

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

## Built With
- PyTorch (CPU + CUDA)
- No high-level Transformer libraries — everything from scratch
- Google Colab (T4 GPU) for BPE and Decision Transformer training