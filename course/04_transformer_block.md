# Chapter 4: The Transformer Block — Assembling the Full Model

In Chapter 3, we built the attention mechanism — the part that gathers information from across the sequence. But attention alone isn't enough. The Transformer block has three more crucial components:

1. **Feed-Forward Network** — processes the gathered information
2. **Layer Normalization** — stabilizes training
3. **Residual Connections** — makes deep networks trainable

We'll build each one, combine them into a Transformer Block, then stack everything into the full GPT model.

## Part 1: The Feed-Forward Network

### What It Does

Attention gathers information. The feed-forward network (FFN) **processes** it.

Think of it this way: attention is like looking around a room and noting who's there. The FFN is like thinking about what you've seen and drawing conclusions.

Mechanically, the FFN is two linear layers with an activation function in between:

```
FFN(x) = Linear₂(GELU(Linear₁(x)))
```

That's it. No attention, no masking, no interaction between positions. The FFN is applied **independently** to each position — position 5's FFN computation doesn't see position 3's data at all. All the cross-position communication happened in attention; the FFN processes each position's gathered information on its own.

### The Expansion-Compression Pattern

The FFN has a distinctive shape — it expands the dimension by 4×, then compresses it back:

```
Input:    (B, T, 128)     ← d_model = 128
    │
    ▼  Linear₁: 128 → 512     ← expand 4×
    ▼  GELU activation
    ▼  Linear₂: 512 → 128     ← compress back
    │
Output:   (B, T, 128)     ← same shape as input
```

**Why expand then compress?** The expansion into a higher-dimensional space gives the network room to represent complex transformations. Think of it as "spreading out" the data into a larger workspace where it's easier to manipulate, then compressing the result back to the original size.

The 4× ratio is convention from the original Transformer paper. It works well in practice and has been kept by virtually all subsequent models.

### The GELU Activation Function

An **activation function** introduces non-linearity into the network. Without it, stacking linear layers would just produce another linear transformation — no matter how many layers you add, `Linear(Linear(x))` is equivalent to a single `Linear(x)`.

**GELU** (Gaussian Error Linear Unit) is the activation function used by GPT-2 and most modern Transformers. It's shaped like a smoother version of ReLU:

```
ReLU(x):   max(0, x)          ← hard cutoff at 0
GELU(x):   x × Φ(x)           ← smooth transition around 0
           where Φ is the standard normal CDF

Roughly:
  x = -2   → GELU(-2) ≈ -0.04    (almost zeroed out)
  x = -1   → GELU(-1) ≈ -0.16    (mostly suppressed)
  x =  0   → GELU(0)  =  0.00
  x =  1   → GELU(1)  ≈  0.84    (mostly passed through)
  x =  2   → GELU(2)  ≈  1.96    (almost unchanged)
```

The key property: large positive values pass through almost unchanged, large negative values get squashed to near zero, and there's a smooth transition around zero. This lets the network "gate" information — passing useful signals and suppressing irrelevant ones.

**Why GELU over ReLU?** ReLU has a hard cutoff at zero (`max(0, x)`), which means its gradient is exactly 0 for negative inputs. This can cause "dead neurons" — neurons that stop learning because they always output 0. GELU's smooth curve avoids this problem.

### The Code

```python
class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

### Line-by-Line

```python
self.net = nn.Sequential(
    nn.Linear(d_model, 4 * d_model),    # 128 → 512
    nn.GELU(),                            # activation
    nn.Linear(4 * d_model, d_model),     # 512 → 128
    nn.Dropout(dropout),                  # regularization
)
```

`nn.Sequential` chains layers together — the output of each layer feeds into the next. It's equivalent to:

```python
def forward(self, x):
    x = self.linear1(x)      # (B, T, 128) → (B, T, 512)
    x = F.gelu(x)            # (B, T, 512) — non-linearity
    x = self.linear2(x)      # (B, T, 512) → (B, T, 128)
    x = self.dropout(x)      # (B, T, 128) — regularization
    return x
```

### Parameter Count

```
Linear₁: 128 × 512 + 512 (bias) = 66,048
Linear₂: 512 × 128 + 128 (bias) = 65,664
Total: 131,712 parameters

For comparison:
  Multi-Head Attention: 65,536 parameters
  Feed-Forward Network: 131,712 parameters
```

The FFN has **twice** as many parameters as attention. This is true across virtually all Transformer models. The FFN is where most of the model's "knowledge" is stored — attention routes information, but the FFN processes it.

---

## Part 2: Layer Normalization

### The Problem

Neural networks learn by adjusting weights based on gradients. But if the values flowing through the network are wildly different scales — some activations at 0.001, others at 1000 — the gradients become unstable. The network oscillates instead of converging.

**Layer normalization** solves this by normalizing the values at each position to have mean 0 and standard deviation 1.

### The Math

Given a vector `x = [x₁, x₂, ..., xₙ]` (one position's embedding, so n = d_model = 128):

```
Step 1: Compute mean
  μ = (x₁ + x₂ + ... + xₙ) / n

Step 2: Compute variance
  σ² = ((x₁-μ)² + (x₂-μ)² + ... + (xₙ-μ)²) / n

Step 3: Normalize
  x̂ᵢ = (xᵢ - μ) / √(σ² + ε)

  where ε = 0.00001 (tiny constant to prevent division by zero)

Step 4: Scale and shift (learnable)
  outᵢ = γᵢ × x̂ᵢ + βᵢ

  where γ (scale) and β (shift) are learnable parameters
```

### A Concrete Example

```
Input vector: x = [3.0, 1.0, 5.0, -1.0]

Step 1: mean μ = (3 + 1 + 5 + (-1)) / 4 = 8/4 = 2.0

Step 2: variance σ² = ((3-2)² + (1-2)² + (5-2)² + (-1-2)²) / 4
                     = (1 + 1 + 9 + 9) / 4
                     = 20/4 = 5.0

Step 3: normalize (using √5 ≈ 2.236)
  x̂ = [(3-2)/2.236, (1-2)/2.236, (5-2)/2.236, (-1-2)/2.236]
  x̂ = [0.447, -0.447, 1.342, -1.342]

  Now: mean ≈ 0, std ≈ 1  ✓

Step 4: scale and shift (γ and β start at [1,1,1,1] and [0,0,0,0])
  At initialization, this step does nothing (multiply by 1, add 0)
  During training, the network can learn to adjust the scale/shift
```

### Why "Layer" Norm and Not "Batch" Norm?

There are different types of normalization. The key difference is **what you average over**:

```
Input shape: (B, T, C) = (64, 256, 128)

Batch Norm:  normalize across the batch dimension (B)
             For each of the 128 features, compute mean/var across all 64×256 positions
             Problem: depends on batch size, doesn't work well with variable-length sequences

Layer Norm:  normalize across the feature dimension (C)
             For each individual position, compute mean/var across its 128 features
             Works independently per position — no batch dependency
```

Layer Norm is preferred in Transformers because:
1. It works the same at batch size 1 (generation) and batch size 64 (training)
2. Each position is normalized independently — no interaction between sequences in a batch
3. It handles variable-length sequences naturally

### The Code

We use PyTorch's built-in `nn.LayerNorm`:

```python
ln = nn.LayerNorm(128)    # normalize vectors of dimension 128

x = torch.randn(64, 256, 128)    # random input
out = ln(x)                       # normalized output
# out has the same shape (64, 256, 128)
# but each of the 64×256 vectors now has mean ≈ 0, std ≈ 1
```

Parameters: just `γ` (128 values) and `β` (128 values) = **256 parameters**. Tiny compared to attention and FFN.

---

## Part 3: Residual Connections

### The Vanishing Gradient Problem

Deep networks (many layers) face a fundamental problem during training. Gradients flow backward through the network during backpropagation. Each layer the gradient passes through multiplies it by that layer's weights. After many multiplications:

- If weights are < 1 on average: gradients shrink exponentially → **vanishing gradients**
- If weights are > 1 on average: gradients grow exponentially → **exploding gradients**

Vanishing gradients mean early layers stop learning — they get near-zero gradient signals and can't update their weights. A 4-layer network effectively becomes a 1-2 layer network.

### The Solution: Skip Connections

A **residual connection** (also called a skip connection) adds the input directly to the output:

```
Without residual:
  output = F(x)                    ← gradient must flow through F

With residual:
  output = x + F(x)               ← gradient flows BOTH through F AND directly
```

The `+ x` creates a "highway" for the gradient to flow through. Even if `F` produces near-zero gradients, the gradient can still flow through the identity path (`x`).

Visually:

```
     x ─────────────────────┐
     │                       │
     ▼                       │  (skip connection)
  ┌──────┐                   │
  │  F   │  (attention       │
  │      │   or FFN)         │
  └──┬───┘                   │
     │                       │
     ▼                       ▼
     F(x)        +          x
                 │
                 ▼
            x + F(x)           ← output
```

### The Intuition

Think of `F(x)` as a **refinement** rather than a complete transformation. The residual connection says: "Start with what you already have (`x`), then add whatever new information the layer discovered (`F(x)`)."

This has a beautiful interpretation:
- If the layer learns something useful: `F(x)` is non-zero, it adds new information
- If the layer has nothing to add: `F(x)` ≈ 0, the input passes through unchanged
- The network can learn to "skip" layers that aren't needed for a particular input

In practice, this means a 4-layer network with residual connections is **at least as powerful** as a 1-layer network, because the other layers can learn to be identity functions. Without residuals, deeper doesn't always mean better.

### Pre-Norm vs. Post-Norm

There are two ways to combine LayerNorm and residual connections:

```
Post-Norm (original 2017 Transformer):
  output = LayerNorm(x + F(x))

Pre-Norm (GPT-2 and most modern models):
  output = x + F(LayerNorm(x))
```

We use **pre-norm** because:
1. It's more stable during training — the normalization happens before the layer, so the layer always receives well-scaled inputs
2. No careful learning rate tuning needed
3. Virtually all modern Transformers use it (GPT-2, GPT-3, LLaMA, etc.)

The difference is subtle but matters. Post-norm normalizes after the residual addition, which means the residual path can have unnormalized values. Pre-norm normalizes before the layer, giving the layer clean inputs, and the residual path stays unnormalized (which is fine — it's just accumulating refinements).

---

## Part 4: The Transformer Block

Now we combine everything into one block:

```
Input x: (B, T, d_model)
    │
    ├───────────────────┐
    │                   │ (residual connection)
    ▼                   │
  LayerNorm             │
    ▼                   │
  Multi-Head Attention  │
    ▼                   │
    + ◄─────────────────┘  x = x + Attention(LayerNorm(x))
    │
    ├───────────────────┐
    │                   │ (residual connection)
    ▼                   │
  LayerNorm             │
    ▼                   │
  Feed-Forward          │
    ▼                   │
    + ◄─────────────────┘  x = x + FFN(LayerNorm(x))
    │
Output x: (B, T, d_model)     ← same shape as input!
```

Two sub-layers, each with its own LayerNorm and residual connection:
1. **Attention sub-layer**: gather information from other positions
2. **FFN sub-layer**: process the gathered information

### The Code

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))    # Residual + pre-norm attention
        x = x + self.ffn(self.ln2(x))     # Residual + pre-norm FFN
        return x
```

### Line-by-Line

```python
self.ln1 = nn.LayerNorm(d_model)
self.attn = MultiHeadAttention(d_model, n_heads, block_size, dropout)
```
- `ln1`: LayerNorm for the attention sub-layer (256 parameters)
- `attn`: The multi-head attention from Chapter 3 (65,536 parameters)

```python
self.ln2 = nn.LayerNorm(d_model)
self.ffn = FeedForward(d_model, dropout)
```
- `ln2`: LayerNorm for the FFN sub-layer (256 parameters)
- `ffn`: The feed-forward network from Part 1 (131,712 parameters)

```python
x = x + self.attn(self.ln1(x))
```
This single line does three things:
1. `self.ln1(x)` — normalize x (pre-norm)
2. `self.attn(...)` — apply multi-head attention
3. `x + ...` — add the residual connection

Read it as: "Start with x, add whatever attention learned from the normalized input."

```python
x = x + self.ffn(self.ln2(x))
```
Same pattern: normalize → FFN → add to residual. The input to this line is already enriched by attention (from the previous line). Now the FFN processes that enriched representation.

### Parameter Count per Block

```
LayerNorm 1:           256
Multi-Head Attention:  65,536
LayerNorm 2:           256
Feed-Forward:          131,712
─────────────────────────────
Total per block:       197,760

× 4 blocks = 791,040 parameters in Transformer blocks
```

---

## Part 5: The Full GPT Model

Now we stack everything into the complete model:

```
Input: (B, T) — batch of token indices
    │
    ▼
Token Embedding     (65 × 128 = 8,320 params)
  + Position Embedding  (256 × 128 = 32,768 params)
    │
    ▼
Dropout
    │
    ▼
Transformer Block 0    (197,760 params)
    │
    ▼
Transformer Block 1    (197,760 params)
    │
    ▼
Transformer Block 2    (197,760 params)
    │
    ▼
Transformer Block 3    (197,760 params)
    │
    ▼
Final LayerNorm        (256 params)
    │
    ▼
Linear Projection      (128 × 65 + 65 = 8,385 params)
    │
    ▼
Output: (B, T, 65) — logits (raw scores) for each character
```

### The Code

```python
class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        block_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size

        # Embedding layers
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, block_size, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm + output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        pos = torch.arange(T, device=idx.device)

        tok = self.tok_emb(idx)
        pos = self.pos_emb(pos)
        x = self.drop(tok + pos)

        x = self.blocks(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss
```

### Line-by-Line Walkthrough

```python
self.blocks = nn.Sequential(*[
    TransformerBlock(d_model, n_heads, block_size, dropout)
    for _ in range(n_layers)
])
```
- Creates 4 Transformer blocks and chains them with `nn.Sequential`.
- The `*` unpacks the list into arguments: `nn.Sequential(block0, block1, block2, block3)`.
- `nn.Sequential` passes the output of each block to the next — exactly the stacked architecture we want.

```python
self.ln_f = nn.LayerNorm(d_model)
```
- Final LayerNorm applied after all Transformer blocks, before the output projection.
- This is a pre-norm convention: since each block applies LayerNorm *before* its sub-layers, the output of the last block hasn't been normalized. This final LayerNorm ensures the output projection receives clean, normalized inputs.

```python
self.head = nn.Linear(d_model, vocab_size)
```
- The output projection: converts from the model's internal dimension (128) to the vocabulary size (65).
- For each position, this produces 65 numbers — one for each character. These are called **logits** — raw, unnormalized scores.

```python
self.apply(self._init_weights)
```
- `self.apply(fn)` calls `fn` on every sub-module in the model.
- `_init_weights` initializes all Linear and Embedding layers with small random values (std=0.02).
- **Why does initialization matter?** Bad initialization can make training fail entirely. If weights are too large, activations explode. If too small, gradients vanish. The std=0.02 value comes from GPT-2 and works well in practice.

Now the forward pass:

```python
def forward(self, idx, targets=None):
    B, T = idx.shape
```
- `idx` is the input: shape `(B, T)` — batch of token index sequences.
- `targets` is optional. During training, we pass targets and compute the loss. During generation, we just want logits.

```python
tok = self.tok_emb(idx)     # (B, T) → (B, T, 128)
pos = self.pos_emb(pos)     # (T,) → (T, 128)
x = self.drop(tok + pos)    # (B, T, 128)
```
- Token embedding + positional embedding + dropout. Covered in Chapter 2.

```python
x = self.blocks(x)          # (B, T, 128) → (B, T, 128)
```
- Pass through all 4 Transformer blocks sequentially. Each block does attention + FFN with residual connections. The shape stays `(B, T, 128)` throughout.

```python
x = self.ln_f(x)            # (B, T, 128)
logits = self.head(x)       # (B, T, 128) → (B, T, 65)
```
- Final normalization, then project to vocabulary size.
- `logits[b, t, c]` = "how likely is character c to be the next character after position t in sequence b?"
- These are raw scores, not probabilities. To get probabilities, you'd apply softmax. But for the loss function, we use raw logits directly (more numerically stable).

### The Loss Function

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),    # (B*T, 65)
    targets.view(-1),                     # (B*T,)
)
```

**Cross-entropy loss** measures how wrong the model's predictions are.

First, the reshaping:
- `logits.view(-1, logits.size(-1))` reshapes from `(64, 256, 65)` to `(16384, 65)` — flatten the batch and sequence dimensions
- `targets.view(-1)` reshapes from `(64, 256)` to `(16384,)` — matching flatten
- Now each of the 16,384 positions has 65 logits and 1 correct answer

**What cross-entropy does** for each position:
1. Apply softmax to the 65 logits → get probabilities for each character
2. Look at the probability assigned to the **correct** character
3. Compute `-log(p_correct)` — the loss for this position
4. Average across all 16,384 positions

```
Example for one position:
  Logits: [0.1, 0.5, -0.3, 2.1, 0.8, ...]   (65 values)
  After softmax: [0.03, 0.05, 0.02, 0.25, 0.07, ...]   (sum to 1)
  Correct character: index 3
  Probability of correct: 0.25
  Loss: -log(0.25) = 1.39

If the model were perfect:
  Probability of correct: 1.0
  Loss: -log(1.0) = 0.0

If the model is random (uniform over 65 chars):
  Probability of correct: 1/65 ≈ 0.0154
  Loss: -log(1/65) = log(65) ≈ 4.17
```

That's why an untrained model gives a loss of ~4.17 — it's assigning roughly equal probability to all 65 characters. As training progresses, the loss drops because the model assigns higher probability to the correct next character.

---

## Part 6: Total Parameter Count

```
Token Embedding:        65 × 128          =     8,320
Position Embedding:     256 × 128         =    32,768
Dropout:                                        0

Transformer Block × 4:
  LayerNorm 1:          128 + 128         =       256  ]
  Multi-Head Attention:                      65,536  ] × 4 blocks
  LayerNorm 2:          128 + 128         =       256  ] = 791,040
  Feed-Forward:                             131,712  ]

Final LayerNorm:        128 + 128         =       256
Output Projection:      128 × 65 + 65    =     8,385
                                          ──────────
Total:                                      841,281 parameters
```

841,281 learnable numbers. The network will adjust all of them during training to minimize the loss — to get better at predicting the next character.

For perspective: GPT-2 has 117 million parameters. GPT-3 has 175 billion. Our model is ~140× smaller than GPT-2, but the architecture is identical.

---

## Part 7: Weight Initialization — Why It Matters

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

When the model is first created, its parameters need starting values. PyTorch has defaults, but the Transformer benefits from careful initialization.

**Why std=0.02?** Each layer multiplies its input by weights. If weights are too large (std=1.0), values grow exponentially through layers:

```
Layer 1 output: ~1.0
Layer 2 output: ~10.0
Layer 3 output: ~100.0
Layer 4 output: ~1000.0     ← exploded!
```

With std=0.02, the layers barely change the scale:

```
Layer 1 output: ~1.0
Layer 2 output: ~1.0
Layer 3 output: ~1.0
Layer 4 output: ~1.0         ← stable
```

**Why zero biases?** Starting with zero bias means the layers initially compute roughly `weight × input + 0`. No offset. The biases will adjust during training.

**Why initialize embeddings the same way?** Embeddings are just lookup tables, but they participate in the same gradient flow. Small initial values keep the whole network in a stable regime from the start.

---

## Part 8: Verifying the Model

Before training, always verify the shapes and initial loss:

```python
if __name__ == "__main__":
    import math

    vocab_size = 65
    block_size = 256
    batch_size = 4

    model = GPT(vocab_size=vocab_size, block_size=block_size)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    x = torch.randint(0, vocab_size, (batch_size, block_size))
    y = torch.randint(0, vocab_size, (batch_size, block_size))
    logits, loss = model(x, y)

    print(f"Logits shape: {logits.shape}")
    print(f"Expected:     ({batch_size}, {block_size}, {vocab_size})")
    print(f"Loss: {loss.item():.4f}")
    print(f"Expected loss (random): ~{math.log(vocab_size):.4f}")
```

**What to check:**
1. **Parameter count**: ~841K. If it's wildly different, something is wrong.
2. **Logits shape**: `(4, 256, 65)`. Batch size × sequence length × vocab size.
3. **Initial loss**: ~4.17 (which is `ln(65)`). An untrained model should perform like random guessing. If the initial loss is much higher or lower, something is wrong with the architecture.

---

## Summary

The complete Transformer architecture, layer by layer:

```
┌──────────────────────────────────────────┐
│          Token Embedding (65 → 128)       │
│        + Position Embedding (256 → 128)   │
│        + Dropout(0.1)                     │
├──────────────────────────────────────────┤
│                                          │
│  ┌── Transformer Block ──────────────┐   │
│  │                                    │   │
│  │  x ─────────────────┐             │   │
│  │  │ LayerNorm         │ (residual)  │   │
│  │  │ Multi-Head Attn   │             │   │  × 4
│  │  └──────── + ────────┘             │   │
│  │  x ─────────────────┐             │   │
│  │  │ LayerNorm         │ (residual)  │   │
│  │  │ Feed-Forward      │             │   │
│  │  └──────── + ────────┘             │   │
│  │                                    │   │
│  └────────────────────────────────────┘   │
│                                          │
├──────────────────────────────────────────┤
│           Final LayerNorm                 │
│           Linear (128 → 65)               │
│           → Logits                        │
└──────────────────────────────────────────┘
```

Each component's role:

| Component | What It Does | Why |
|-----------|-------------|-----|
| Token Embedding | Character → vector | Networks need numbers |
| Position Embedding | Add position info | Transformer has no built-in order |
| Multi-Head Attention | Gather info from other positions | Context is needed for prediction |
| Feed-Forward Network | Process gathered info | Non-linear transformation |
| Layer Normalization | Stabilize activations | Prevents training instability |
| Residual Connections | Skip connections | Makes deep networks trainable |
| Output Projection | Vector → character scores | Convert back to predictions |

## What's Next

In [Chapter 5](05_training.md), we bring the model to life. We'll build the training loop — optimizer, learning rate schedule, gradient clipping — and watch the loss drop from 4.17 (random) down to ~1.4 as the model learns to write Shakespeare.