# Chapter 3: Self-Attention — The Heart of the Transformer

This is the most important chapter. Attention is the mechanism that makes Transformers work, and understanding it deeply is the single most valuable thing you can get from this project.

Take your time with this one.

## The Problem Attention Solves

After Chapter 2, each position has a 128-dimensional vector that encodes "what character am I?" and "where am I in the sequence?" But each position only knows about **itself**. Position 5 has no idea what's at position 3 or position 0.

For language modeling, this is useless. To predict the next character, you need **context**:

```
"ROMEO:\nO, she doth teach the torches to burn bri___"
```

To predict `g` (completing "bright"), the model needs to:
- Know the sentence started with "O, she doth" (poetic style → "bright" is likely)
- Know "burn bri" is in progress (→ completing a word starting with "bri")
- Know "ROMEO:" appeared earlier (→ this is Romeo speaking)

Each position needs to **gather information** from other positions. That's what attention does.

## Part 1: The Intuition — A Library Analogy

Imagine you're in a library doing research. You have a **question** (what you're looking for), and each book on the shelf has a **label** (what it contains) and **content** (the actual information).

Your process:
1. Compare your **question** to each book's **label** to find relevant books
2. Books with labels that match your question well get your **attention**
3. You read the **content** of the relevant books, weighted by how relevant they are

Self-attention works the same way:
- Each position creates a **Query** (Q): "What am I looking for?"
- Each position creates a **Key** (K): "What do I contain?"
- Each position creates a **Value** (V): "What information can I provide?"

The process:
1. Compare each Query against every Key → get **attention scores** (relevance)
2. Normalize scores to probabilities (softmax) → get **attention weights**
3. Use the weights to take a weighted sum of Values → get **output**

Let's make this concrete with a tiny example.

## Part 2: Attention by Example

Consider a 4-token sequence: `"the "` → tokens at positions 0, 1, 2, 3.

After embedding, each position has a vector. Let's pretend `d_model = 4` (tiny, for illustration):

```
Position 0 ("t"): x₀ = [1.0, 0.5, -0.3, 0.8]
Position 1 ("h"): x₁ = [0.2, 0.9, 0.1, -0.4]
Position 2 ("e"): x₂ = [0.7, -0.2, 0.6, 0.3]
Position 3 (" "): x₃ = [-0.1, 0.4, 0.8, 0.5]
```

### Step 1: Create Q, K, V

Each position's vector gets transformed into three different vectors through **linear projections** — that is, multiplication by three different weight matrices:

```
Q = x × W_Q       "What am I looking for?"
K = x × W_K       "What do I contain?"
V = x × W_V       "What information do I provide?"
```

These weight matrices are **learnable** — the network learns what to ask for, what to advertise, and what to share. This is the key to attention's power: the network learns *what* to pay attention *to*.

For all four positions at once:

```
X = [x₀]      Q = X × W_Q      K = X × W_K      V = X × W_V
    [x₁]
    [x₂]
    [x₃]
shape: (4, 4)   (4, 4)×(4,4)    (4, 4)×(4,4)    (4, 4)×(4,4)
                = (4, 4)         = (4, 4)         = (4, 4)
```

Each position now has its own Q, K, and V vectors. Let's say after the projection:

```
Q₀ = [0.3, 0.7]    K₀ = [0.5, 0.1]    V₀ = [1.0, 0.2]
Q₁ = [0.1, 0.9]    K₁ = [0.8, 0.6]    V₁ = [0.3, 0.8]
Q₂ = [0.6, 0.2]    K₂ = [0.2, 0.9]    V₂ = [0.7, 0.5]
Q₃ = [0.4, 0.5]    K₃ = [0.7, 0.3]    V₃ = [0.4, 0.6]
```

(I'm using 2D vectors here to keep the example manageable. In our model they're 32D.)

### Step 2: Compute Attention Scores

For each position, we compute how well its Query matches every Key. The match is computed using a **dot product** — the standard measure of similarity between two vectors.

The dot product of two vectors `a = [a₁, a₂]` and `b = [b₁, b₂]` is:

```
a · b = a₁×b₁ + a₂×b₂
```

If `a` and `b` point in the same direction, the dot product is large and positive. If they point in opposite directions, it's large and negative. If they're perpendicular, it's zero.

Let's compute the scores for position 2 (Q₂ looking at all Keys):

```
score(Q₂, K₀) = Q₂ · K₀ = 0.6×0.5 + 0.2×0.1 = 0.30 + 0.02 = 0.32
score(Q₂, K₁) = Q₂ · K₁ = 0.6×0.8 + 0.2×0.6 = 0.48 + 0.12 = 0.60
score(Q₂, K₂) = Q₂ · K₂ = 0.6×0.2 + 0.2×0.9 = 0.12 + 0.18 = 0.30
score(Q₂, K₃) = Q₂ · K₃ = 0.6×0.7 + 0.2×0.3 = 0.42 + 0.06 = 0.48
```

Position 2's query matches best with Key 1 (score 0.60), meaning position 2 "wants to look at" position 1.

In matrix form, we compute ALL scores for ALL positions simultaneously:

```
Scores = Q × K^T

         K₀^T  K₁^T  K₂^T  K₃^T
Q₀  [   0.42   0.62   0.32   0.52  ]
Q₁  [   0.14   0.62   0.83   0.34  ]
Q₂  [   0.32   0.60   0.30   0.48  ]
Q₃  [   0.25   0.62   0.53   0.43  ]

Shape: (4, 4) — score[i][j] = "how much should position i attend to position j"
```

### Step 3: Scale

Before applying softmax, we divide by `√(d_k)`, where `d_k` is the dimension of the key vectors:

```
Scaled Scores = Scores / √(d_k)
```

**Why scale?** Without scaling, when `d_k` is large, the dot products become large in magnitude. Large values pushed through softmax become very close to 0 or 1 — the attention becomes "hard" (looking at only one position) rather than "soft" (blending information from multiple positions). Dividing by `√(d_k)` keeps the values in a range where softmax produces useful gradients.

With our example (`d_k = 2`, so `√2 ≈ 1.41`):

```
Scaled = Scores / 1.41

         K₀     K₁     K₂     K₃
Q₀  [   0.30   0.44   0.23   0.37  ]
Q₁  [   0.10   0.44   0.59   0.24  ]
Q₂  [   0.23   0.43   0.21   0.34  ]
Q₃  [   0.18   0.44   0.38   0.30  ]
```

### Step 4: Causal Mask

This is critical for language modeling. When the model is predicting the next character at position 2, it should NOT be able to see positions 3, 4, 5, etc. — that's the future! If it could see the future, it would just copy the answer instead of learning to predict.

The **causal mask** sets all "future" scores to negative infinity:

```
Mask (lower triangular):
[  1,  0,  0,  0 ]     "Position 0 can only see position 0"
[  1,  1,  0,  0 ]     "Position 1 can see positions 0-1"
[  1,  1,  1,  0 ]     "Position 2 can see positions 0-2"
[  1,  1,  1,  1 ]     "Position 3 can see positions 0-3"

After masking (0 → -inf):
         K₀     K₁     K₂     K₃
Q₀  [   0.30,  -inf,  -inf,  -inf  ]
Q₁  [   0.10,  0.44,  -inf,  -inf  ]
Q₂  [   0.23,  0.43,  0.21,  -inf  ]
Q₃  [   0.18,  0.44,  0.38,  0.30  ]
```

Position 0 can ONLY attend to itself. Position 2 can attend to positions 0, 1, and 2 — but not 3.

### Step 5: Softmax

**Softmax** converts arbitrary numbers into a probability distribution — all values become positive and sum to 1:

```
softmax([a, b, c]) = [e^a / (e^a + e^b + e^c),
                      e^b / (e^a + e^b + e^c),
                      e^c / (e^a + e^b + e^c)]
```

Key properties:
- All outputs are between 0 and 1
- All outputs sum to 1
- Larger inputs get larger outputs (exponential magnification)
- `e^(-inf) = 0`, so masked positions become exactly 0

After softmax:

```
Attention Weights:
         K₀     K₁     K₂     K₃
Q₀  [   1.00,  0.00,  0.00,  0.00  ]   ← only sees itself
Q₁  [   0.41,  0.59,  0.00,  0.00  ]   ← looks at pos 0 and 1
Q₂  [   0.31,  0.38,  0.31,  0.00  ]   ← looks at pos 0, 1, and 2
Q₃  [   0.21,  0.28,  0.27,  0.24  ]   ← looks at all 4
```

Each row sums to 1. These are the **attention weights** — how much each position "looks at" each other position.

### Step 6: Weighted Sum of Values

Finally, we use the attention weights to combine the Value vectors:

```
Output₂ = 0.31 × V₀ + 0.38 × V₁ + 0.31 × V₂ + 0.00 × V₃

         = 0.31 × [1.0, 0.2]
         + 0.38 × [0.3, 0.8]
         + 0.31 × [0.7, 0.5]
         + 0.00 × [0.4, 0.6]

         = [0.31, 0.06] + [0.11, 0.30] + [0.22, 0.16] + [0, 0]

         = [0.64, 0.52]
```

Position 2's output is a blend of information from positions 0, 1, and 2, weighted by how relevant each one is. Position 3 contributes nothing (weight = 0) because of the causal mask.

In matrix form:

```
Output = Attention_Weights × V

Shape: (4, 4) × (4, 2) = (4, 2)
```

Each position now has an output vector that contains information gathered from all the positions it was allowed to see.

---

## Part 3: The Full Attention Formula

The entire attention mechanism in one equation:

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k + mask) × V
```

Where:
- `Q = X × W_Q` — queries, shape `(T, d_k)`
- `K = X × W_K` — keys, shape `(T, d_k)`
- `V = X × W_V` — values, shape `(T, d_k)`
- `K^T` — K transposed, shape `(d_k, T)`
- `Q × K^T` — attention scores, shape `(T, T)`
- `√d_k` — scaling factor (scalar)
- `mask` — causal mask (0 for visible, -inf for hidden)
- `softmax(...)` — applied row-wise, shape `(T, T)`
- `... × V` — weighted sum, shape `(T, d_k)`

With batch dimension, all shapes get a `B` prefix: `(B, T, d_k)`, etc.

---

## Part 4: Single-Head Attention — The Code

```python
class SingleHeadAttention(nn.Module):
    def __init__(self, d_model: int, head_dim: int, block_size: int, dropout: float):
        super().__init__()
        self.q = nn.Linear(d_model, head_dim, bias=False)
        self.k = nn.Linear(d_model, head_dim, bias=False)
        self.v = nn.Linear(d_model, head_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).bool()
        )

    def forward(self, x):
        B, T, C = x.shape

        q = self.q(x)    # (B, T, head_dim)
        k = self.k(x)    # (B, T, head_dim)
        v = self.v(x)    # (B, T, head_dim)

        scale = math.sqrt(k.shape[-1])
        att = (q @ k.transpose(-2, -1)) / scale

        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v
        return out
```

### Line-by-Line Walkthrough

```python
self.q = nn.Linear(d_model, head_dim, bias=False)
self.k = nn.Linear(d_model, head_dim, bias=False)
self.v = nn.Linear(d_model, head_dim, bias=False)
```

Three separate linear projections. `nn.Linear(128, 32, bias=False)` creates a weight matrix of shape `(128, 32)` with no bias term. When applied to an input of shape `(B, T, 128)`, it produces output of shape `(B, T, 32)`.

Why `head_dim` and not `d_model`? In multi-head attention (Part 5), we split the model dimension across multiple heads. With `d_model=128` and 4 heads, each head operates on `128/4 = 32` dimensions. This keeps the total computation the same as a single head with the full dimension.

`bias=False` — the original Transformer paper doesn't use biases in the Q, K, V projections. It works fine either way, but `False` is conventional.

```python
self.register_buffer(
    "mask",
    torch.tril(torch.ones(block_size, block_size)).bool()
)
```

This creates the causal mask. Let's unpack it:

```python
torch.ones(block_size, block_size)
# A 256×256 matrix of all 1s

torch.tril(...)
# tril = "triangular lower" — zeros out everything above the diagonal
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]

.bool()
# Convert to True/False

register_buffer(...)
# Store as part of the model (moves to GPU with the model,
# saved in checkpoints) but NOT a learnable parameter (no gradients)
```

Now the forward pass:

```python
B, T, C = x.shape
```
- Unpack the input dimensions. `B=64`, `T=256`, `C=128` (or `head_dim` for a single head).

```python
q = self.q(x)    # (B, T, head_dim)
k = self.k(x)    # (B, T, head_dim)
v = self.v(x)    # (B, T, head_dim)
```
- Apply the three linear projections. Each transforms the 128-dim input into a 32-dim vector.
- The same input `x` goes into all three projections, but the weight matrices are different, so Q, K, V are different.

```python
scale = math.sqrt(k.shape[-1])
```
- `k.shape[-1]` is `head_dim` (32). `sqrt(32) ≈ 5.66`.

```python
att = (q @ k.transpose(-2, -1)) / scale
```
- `@` is matrix multiplication in PyTorch.
- `k.transpose(-2, -1)` swaps the last two dimensions: `(B, T, head_dim)` → `(B, head_dim, T)`.
- The multiplication: `(B, T, head_dim) @ (B, head_dim, T)` → `(B, T, T)`.
- Each element `att[b, i, j]` = "how much should position i in sequence b attend to position j?"
- Divide by `scale` to prevent softmax saturation.

```python
att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
```
- `self.mask[:T, :T]` — slice the mask to the actual sequence length (might be shorter than `block_size`).
- `~` inverts the mask: `True` becomes `False` and vice versa. So the upper triangle (future positions) becomes `True`.
- `masked_fill(condition, value)` — wherever the condition is `True`, fill with the value.
- Result: future positions get `-inf`. After softmax, `e^(-inf) = 0` — future positions contribute nothing.

```python
att = F.softmax(att, dim=-1)
```
- Apply softmax along the last dimension (each row independently).
- Each row becomes a probability distribution that sums to 1.
- `dim=-1` means "apply softmax across the last axis" — i.e., for each query position, normalize across all key positions.

```python
att = self.dropout(att)
```
- Randomly zero out some attention weights during training.
- This prevents the model from relying too heavily on any single position.

```python
out = att @ v
```
- `(B, T, T) @ (B, T, head_dim)` → `(B, T, head_dim)`
- For each position, compute a weighted sum of all value vectors.
- Positions with zero attention weight (future positions) contribute nothing.

### Shape Trace

```
Input:      x shape (B, T, d_model)         = (64, 256, 128)

Q, K, V projections:
            q shape (B, T, head_dim)         = (64, 256, 32)
            k shape (B, T, head_dim)         = (64, 256, 32)
            v shape (B, T, head_dim)         = (64, 256, 32)

Attention scores:
            q @ k^T shape (B, T, T)          = (64, 256, 256)
            (each position's compatibility with every other position)

After mask + softmax:
            att shape (B, T, T)              = (64, 256, 256)
            (probabilities — each row sums to 1)

Output:
            att @ v shape (B, T, head_dim)   = (64, 256, 32)
            (weighted blend of values)
```

---

## Part 5: Multi-Head Attention — Why One Head Isn't Enough

A single attention head learns **one** pattern of what to attend to. But language has many types of relationships:

- The character right before me (local pattern)
- The start of the current word (word boundary)
- The character name at the start of this speech (long-range structure)
- Matching brackets or punctuation (syntax)

One head can't learn all of these simultaneously. Multi-head attention runs **multiple heads in parallel**, each with its own Q, K, V projections, each free to learn a different attention pattern.

### How It Works

```
Input x: (B, T, 128)
         │
    ┌────┴────┬────────┬────────┐
    │         │        │        │
  Head 0    Head 1   Head 2   Head 3
  Q,K,V     Q,K,V    Q,K,V    Q,K,V
  dim=32    dim=32   dim=32   dim=32
    │         │        │        │
  att→out   att→out  att→out  att→out
  (B,T,32)  (B,T,32) (B,T,32) (B,T,32)
    │         │        │        │
    └────┬────┴────────┴────────┘
         │
    Concatenate → (B, T, 128)
         │
    Output Projection (Linear 128→128)
         │
    Output: (B, T, 128)
```

Each head operates on `d_model / n_heads = 128 / 4 = 32` dimensions. After attention, we concatenate all heads back to 128 dimensions, then apply one more linear projection to mix the information across heads.

**The total computation is the same** as one big head with 128 dimensions, but now the model can learn 4 different attention patterns.

### The Code

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        head_dim = d_model // n_heads

        self.heads = nn.ModuleList([
            SingleHeadAttention(d_model, head_dim, block_size, dropout)
            for _ in range(n_heads)
        ])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
```

### Line-by-Line

```python
assert d_model % n_heads == 0
head_dim = d_model // n_heads
```
- We need to split the embedding dimension evenly across heads.
- 128 / 4 = 32 dimensions per head.

```python
self.heads = nn.ModuleList([
    SingleHeadAttention(d_model, head_dim, block_size, dropout)
    for _ in range(n_heads)
])
```
- Create 4 independent attention heads.
- `nn.ModuleList` (not a regular Python list) ensures PyTorch tracks these as model parameters.
- Each head has its own Q, K, V weight matrices — they learn independently.

```python
self.proj = nn.Linear(d_model, d_model)
```
- Output projection: `(128 → 128)`. After concatenating the 4 heads (each 32-dim → total 128-dim), this linear layer lets the model mix information across heads.
- Without this projection, the heads would be completely independent — the output at dimensions 0-31 would come purely from head 0, dimensions 32-63 from head 1, etc. The projection allows cross-head interaction.

```python
out = torch.cat([h(x) for h in self.heads], dim=-1)
```
- Run all 4 heads, concatenate along the last dimension.
- Each head outputs `(B, T, 32)`, concatenation gives `(B, T, 128)`.

```python
out = self.proj(out)
out = self.dropout(out)
```
- Apply the output projection and dropout.
- Final shape: `(B, T, 128)` — same as the input. This is important — it means attention can be stacked (the output of one attention layer can be the input to the next).

---

## Part 6: What Attention Learns — Real Results

After training our model, each head develops a distinct attention pattern. Here's what we observed in our trained model's attention maps:

### Layer 0, Head 0 — The "Local Context" Head
```
This head shows a strong diagonal pattern — each position primarily
attends to nearby characters, with attention fading as distance increases.

  "b u r n   b r i g h t"
   ↑
   This position mainly looks at the few characters right before it.
```

This head helps the model complete words. If it sees "bri", attending to the most recent characters tells it to predict "g" (completing "bright").

### Layer 3, Head 0 — The "Structural" Head
```
This head shows bright vertical stripes at specific positions —
the character name ("ROMEO") and the colon.

  "R O M E O : \n O ,   s h e   d o t h"
   ▲ ▲ ▲ ▲ ▲ ▲
   These positions get high attention from EVERYWHERE in the sequence.
```

This head helps the model maintain dialogue structure. Every position in Romeo's speech "looks back" at "ROMEO:" to know who's speaking. This is how the model knows to generate words in Romeo's style.

### The Key Insight

No one programmed these patterns. We just gave the model Q, K, V projections and trained it to predict the next character. The model **discovered** that some heads should focus on local context and others should focus on global structure, because that's what's useful for the task.

---

## Part 7: Attention — The Parameter Count

Let's count how many learnable parameters are in multi-head attention:

```
Per head:
  W_Q: d_model × head_dim = 128 × 32 = 4,096
  W_K: d_model × head_dim = 128 × 32 = 4,096
  W_V: d_model × head_dim = 128 × 32 = 4,096
  Subtotal per head: 12,288

4 heads: 12,288 × 4 = 49,152

Output projection:
  W_proj: d_model × d_model = 128 × 128 = 16,384

Total for MultiHeadAttention: 49,152 + 16,384 = 65,536
```

For comparison, the token embedding table has `65 × 128 = 8,320` parameters. Attention is where most of the model's capacity lives.

---

## Summary

The complete attention mechanism:

```
Input x: (B, T, d_model)
    │
    │  Split into n_heads parallel heads
    ▼
For each head:
    │
    ├── Q = x × W_Q        "What am I looking for?"
    ├── K = x × W_K        "What do I contain?"
    └── V = x × W_V        "What can I provide?"
    │
    │  Compute scores: Q × K^T / √d_k
    │  Apply causal mask (future → -inf)
    │  Softmax (normalize to probabilities)
    │  Weighted sum: weights × V
    │
    └── head output: (B, T, head_dim)

Concatenate all heads → (B, T, d_model)
    │
    ▼ Output projection (Linear)
    │
Output: (B, T, d_model)    ← same shape as input
```

The formula one more time:

```
MultiHead(X) = Concat(head₁, ..., headₙ) × W_O

where headᵢ = softmax(Q_i × K_i^T / √d_k + mask) × V_i
      Q_i = X × W_Qi,  K_i = X × W_Ki,  V_i = X × W_Vi
```

Key takeaways:
1. **Attention lets each position gather information from other positions**
2. **Q, K, V projections are learned** — the model discovers what to attend to
3. **Causal masking** prevents looking into the future
4. **Multiple heads** let the model learn multiple attention patterns simultaneously
5. **Scaling by √d_k** keeps softmax gradients healthy

## What's Next

In [Chapter 4](04_transformer_block.md), we combine attention with two more essential components — the **feed-forward network** (where the model "thinks" about the gathered information) and **residual connections with layer normalization** (which make deep networks possible to train). Together, these form the **Transformer Block** — the repeating unit we stack to build the full model.