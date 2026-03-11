# Chapter 2: Embeddings — Giving Meaning to Numbers

## The Problem

After Chapter 1, our text looks like this:

```
"ROMEO:" → [44, 35, 33, 17, 35, 10]
```

These numbers are arbitrary labels. The number 44 doesn't "mean" anything — it's just the index we assigned to "R". The number 45 (which might be "S") is not "closer" to "R" in any meaningful sense. The network can't do useful math on arbitrary indices.

We need to convert each index into a **vector** — a list of numbers that the network can manipulate mathematically. This is called an **embedding**.

## Part 1: What Is an Embedding?

An embedding maps each token to a point in a high-dimensional space. Instead of "R" being the bare number 44, it becomes a vector of 128 numbers:

```
"R" (index 44) → [0.23, -0.15, 0.82, 0.04, -0.67, 0.31, ...]   (128 numbers)
"O" (index 35) → [0.11,  0.45, 0.03, 0.91, -0.22, 0.56, ...]   (128 numbers)
"M" (index 33) → [-0.08, 0.33, 0.71, 0.18,  0.42, 0.09, ...]   (128 numbers)
```

These vectors start random and are **learned during training**. The network discovers that certain characters should have similar vectors. For example, after training:

- Vowels (`a`, `e`, `i`, `o`, `u`) might cluster together in embedding space
- Uppercase letters might share certain dimensions
- Characters that appear in similar contexts (like `t` and `s`, which both commonly follow vowels) might have similar vectors

### Why 128 Dimensions?

Each dimension gives the network one "axis" to encode information. With 128 dimensions, the network has 128 independent channels to represent properties of each character. Think of it like describing a person: you could use height, weight, age, hair color, etc. — each dimension captures one aspect. The network decides what each dimension represents during training.

128 is our choice for this project — small enough to train on a CPU, large enough to capture useful patterns. GPT-2 uses 768 dimensions. GPT-3 uses 12,288.

### The Embedding as a Lookup Table

Mechanically, an embedding is just a table (matrix) of shape `(vocab_size, d_model)`:

```
Embedding table (65 × 128):

         dim 0    dim 1    dim 2    dim 3    ...    dim 127
token 0  [ 0.12,  -0.34,   0.56,   0.01,   ...,   0.23  ]   ← "\n"
token 1  [ 0.45,   0.11,  -0.78,   0.33,   ...,  -0.15  ]   ← " "
token 2  [ 0.08,  -0.92,   0.44,   0.67,   ...,   0.51  ]   ← "!"
  ...
token 64 [ 0.31,   0.05,  -0.13,   0.89,   ...,  -0.42  ]   ← "z"
```

To "embed" token 44, you just look up row 44 of this table. That's it. No multiplication, no computation — just a table lookup.

In PyTorch:

```python
import torch
import torch.nn as nn

# Create an embedding table: 65 tokens, each gets a 128-dim vector
embedding = nn.Embedding(65, 128)

# Look up tokens
tokens = torch.tensor([44, 35, 33, 17, 35, 10])  # "ROMEO:"
vectors = embedding(tokens)
print(vectors.shape)  # torch.Size([6, 128])
```

Each of the 6 tokens became a 128-dimensional vector. The output shape went from `(6,)` to `(6, 128)`.

### With Batches

In training, we process 64 sequences at once, each 256 tokens long:

```python
batch = torch.randint(0, 65, (64, 256))   # (batch=64, seq_len=256)
vectors = embedding(batch)
print(vectors.shape)  # torch.Size([64, 256, 128])
```

Shape `(64, 256, 128)` means: 64 sequences, 256 positions each, 128-dimensional embedding at each position. This `(B, T, C)` shape is the standard tensor shape throughout our entire model.

---

## Part 2: The Position Problem

Embeddings solve the "meaning" problem, but they create a new one: **the model has no idea about order**.

Consider these two inputs:

```
"cat sat"  → embeddings: [vec_c, vec_a, vec_t, vec_space, vec_s, vec_a, vec_t]
"sat cat"  → embeddings: [vec_s, vec_a, vec_t, vec_space, vec_c, vec_a, vec_t]
```

The individual vectors are the same — `vec_a` is always the same vector regardless of position. But the meaning is completely different! "cat sat" and "sat cat" have different structures.

In a recurrent neural network (like LSTM), position is implicit — the network processes tokens one at a time in order. But the Transformer processes all positions **simultaneously** (that's what makes it fast). So we need to explicitly tell it where each token sits in the sequence.

### Positional Embedding

The solution: add a **position embedding** to each token embedding. Just like we have a table that maps "character → vector," we have a second table that maps "position → vector."

```
Position embedding table (256 × 128):

           dim 0    dim 1    dim 2    ...    dim 127
pos   0  [ 0.05,   0.12,  -0.03,   ...,    0.08  ]   ← "I'm at position 0"
pos   1  [-0.11,   0.08,   0.15,   ...,   -0.03  ]   ← "I'm at position 1"
pos   2  [ 0.03,  -0.22,   0.07,   ...,    0.14  ]   ← "I'm at position 2"
  ...
pos 255  [ 0.17,   0.01,   0.33,   ...,   -0.09  ]   ← "I'm at position 255"
```

The final representation of each token is the **sum** of its token embedding and its position embedding:

```
final[i] = token_embedding(token_id[i]) + position_embedding(i)
```

Concretely, for the input "ROMEO:" at positions 0-5:

```
Position 0: embedding("R") + embedding(pos 0) = vec_R + vec_pos0
Position 1: embedding("O") + embedding(pos 1) = vec_O + vec_pos1
Position 2: embedding("M") + embedding(pos 2) = vec_M + vec_pos2
Position 3: embedding("E") + embedding(pos 3) = vec_E + vec_pos3
Position 4: embedding("O") + embedding(pos 4) = vec_O + vec_pos4
Position 5: embedding(":") + embedding(pos 5) = vec_colon + vec_pos5
```

Notice that the two "O"s (positions 1 and 4) start with the same token embedding but get *different* position embeddings, so they end up as different vectors. The network can now distinguish them.

### Why Addition, Not Concatenation?

You might think: why not concatenate the two vectors (`[token_vec, pos_vec]`) to get a 256-dimensional vector? You could! But addition is more parameter-efficient — it keeps the dimension at 128 instead of doubling it. The network can learn to "share" the 128 dimensions between token identity and position information. In practice, addition works just as well.

### Learnable vs. Sinusoidal Positional Encoding

The original Transformer paper (2017) used a fixed mathematical formula (sine and cosine waves of different frequencies) for positional encoding. GPT-2 switched to **learnable** positional embeddings — a plain `nn.Embedding` that's trained alongside the rest of the model.

We use learnable embeddings because:
1. Simpler to implement (just another `nn.Embedding`)
2. The model can learn whatever positional patterns are most useful
3. At our scale, both approaches perform similarly

The tradeoff: learnable embeddings can only handle positions up to `block_size` (256 in our case). Sinusoidal encoding can theoretically extrapolate to longer sequences. This doesn't matter for us — we'll never see sequences longer than 256.

---

## Part 3: The Code

Here's how embeddings appear in our model:

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, block_size=256, ...):
        super().__init__()

        # Token embedding: vocab_size → d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # Position embedding: block_size → d_model
        self.pos_emb = nn.Embedding(block_size, d_model)

        # Dropout for regularization (explained in Chapter 5)
        self.drop = nn.Dropout(0.1)

    def forward(self, idx):
        B, T = idx.shape    # batch size, sequence length

        # Create position indices: [0, 1, 2, ..., T-1]
        pos = torch.arange(T, device=idx.device)

        # Look up embeddings
        tok = self.tok_emb(idx)    # (B, T) → (B, T, d_model)
        pos = self.pos_emb(pos)    # (T,)   → (T, d_model)

        # Add them together
        x = self.drop(tok + pos)   # (B, T, d_model)

        return x
```

### Line-by-Line Walkthrough

```python
self.tok_emb = nn.Embedding(vocab_size, d_model)
```
- Creates the token embedding table: 65 rows (one per character), 128 columns (embedding dimension).
- Total parameters: 65 × 128 = **8,320**

```python
self.pos_emb = nn.Embedding(block_size, d_model)
```
- Creates the position embedding table: 256 rows (one per position), 128 columns.
- Total parameters: 256 × 128 = **32,768**

```python
B, T = idx.shape
```
- `idx` has shape `(64, 256)` — 64 sequences of 256 tokens each.
- `B = 64` (batch size), `T = 256` (sequence length).

```python
pos = torch.arange(T, device=idx.device)
```
- Creates `[0, 1, 2, 3, ..., 255]` — the position indices.
- `device=idx.device` ensures the positions are on the same device (CPU or GPU) as the input.

```python
tok = self.tok_emb(idx)    # (B, T) → (B, T, d_model)
```
- Looks up each token in the embedding table.
- Shape: `(64, 256)` → `(64, 256, 128)`
- Each of the 64×256 = 16,384 integers became a 128-dimensional vector.

```python
pos = self.pos_emb(pos)    # (T,) → (T, d_model)
```
- Looks up each position in the position table.
- Shape: `(256,)` → `(256, 128)`
- Note this is NOT batched — the same position embeddings are used for every sequence in the batch.

```python
x = self.drop(tok + pos)   # (B, T, d_model)
```
- **Addition with broadcasting**: `tok` is `(64, 256, 128)` and `pos` is `(256, 128)`. PyTorch automatically "broadcasts" `pos` across the batch dimension — it adds the same position vectors to every sequence. The result is `(64, 256, 128)`.
- `self.drop` applies dropout — randomly zeros out some values during training to prevent overfitting. More on this in Chapter 5.

### Shape Trace

```
Input:
  idx shape: (64, 256)           ← 64 sequences of 256 token indices

After token embedding:
  tok shape: (64, 256, 128)      ← each index became a 128-dim vector

After position embedding:
  pos shape: (256, 128)          ← position vectors (shared across batch)

After addition + dropout:
  x shape:   (64, 256, 128)      ← final input to the Transformer blocks
```

---

## Part 4: Understanding Broadcasting

The addition `tok + pos` deserves more explanation, because **broadcasting** is a concept you'll use constantly in deep learning.

`tok` has shape `(64, 256, 128)` and `pos` has shape `(256, 128)`. How can you add a 3D tensor and a 2D tensor?

PyTorch (and NumPy) have a rule: when dimensions don't match, the smaller tensor is "stretched" (broadcast) across the missing dimension. Here's how it works:

```
tok shape: (64, 256, 128)
pos shape:      (256, 128)

Step 1: Align from the right
  tok: (64, 256, 128)
  pos: (  , 256, 128)     ← missing the first dimension

Step 2: Broadcast — repeat pos 64 times along the first dimension
  pos effectively becomes: (64, 256, 128)
  (but PyTorch doesn't actually copy the data — it's done efficiently)

Step 3: Add element-wise
  result: (64, 256, 128)
```

This means every sequence in the batch gets the **same** position embeddings added. Position 0 always gets `pos_emb(0)`, position 1 always gets `pos_emb(1)`, regardless of which sequence it's in. This makes sense — position 5 means "fifth character" no matter what the actual characters are.

---

## Part 5: What the Network Will Learn

At initialization, both embedding tables contain random numbers. During training, the network adjusts them to be useful.

After training, you'd find patterns like:

**Token embeddings:**
- Characters that appear in similar contexts get similar vectors
- Vowels cluster together; consonants cluster together
- Uppercase and lowercase versions of the same letter are related
- Punctuation characters (`.`, `,`, `:`, `!`) form their own cluster

**Position embeddings:**
- Nearby positions have similar vectors (position 5 is similar to position 6)
- Certain positions have special properties (position 0 often starts a character name; the position after `\n` often starts a new speaker)

These patterns emerge automatically from the training data — we never tell the network about vowels or line structure. It discovers them because they're useful for predicting the next character.

---

## Part 6: Dropout — A Preview

You noticed `self.drop = nn.Dropout(0.1)` in the code. Dropout is a **regularization technique** that randomly sets 10% of the values to zero during training.

```
Before dropout: [0.23, -0.15, 0.82, 0.04, -0.67, 0.31, 0.44, -0.08, 0.71, 0.18]
After dropout:  [0.23,  0.00, 0.82, 0.04,  0.00, 0.31, 0.44, -0.08, 0.00, 0.18]
                         ↑                   ↑                         ↑
                     randomly zeroed        zeroed                   zeroed
```

Why is this useful? It forces the network to be **redundant** — it can't rely on any single dimension because that dimension might be dropped. This prevents the network from memorizing specific training examples and encourages it to learn general patterns.

Dropout is only active during training. During generation (inference), all values are kept. We'll discuss this more in Chapter 5.

---

## Summary

| Concept | What It Does | Shape Change |
|---------|-------------|--------------|
| Token Embedding | Index → meaningful vector | `(B, T)` → `(B, T, d_model)` |
| Position Embedding | Position index → position vector | `(T,)` → `(T, d_model)` |
| Addition | Combine token + position info | `(B,T,d)` + `(T,d)` → `(B,T,d)` |
| Dropout | Randomly zero values (regularization) | shape unchanged |

The complete flow so far:

```
Raw text:    "ROMEO:\nO, she..."
                │
                ▼   Tokenizer
Token IDs:   [44, 35, 33, 17, 35, 10, 0, 35, 6, 1, 57, 46, 43, ...]
                │
                ▼   Batching (DataLoader)
Batch:       shape (64, 256)  — 64 sequences of 256 tokens
                │
                ▼   Token Embedding
Token vecs:  shape (64, 256, 128)  — each token is now a 128-dim vector
                │
                ▼   + Position Embedding
Combined:    shape (64, 256, 128)  — now contains BOTH identity and position
                │
                ▼   Dropout
Final input: shape (64, 256, 128)  — ready for the Transformer blocks
```

We now have a rich numerical representation of our text. Each position contains a 128-dimensional vector that encodes both *what* character it is and *where* it sits in the sequence.

## What's Next

In [Chapter 3](03_attention.md), we build the core of the Transformer: **self-attention**. This is the mechanism that lets each position look at every other position and decide what information to gather. It's the single most important idea in modern AI, and we're going to implement it from scratch.