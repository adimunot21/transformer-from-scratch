# Chapter 1: Data Pipeline — Turning Text into Training Data

## The Goal

Before we can train a neural network, we need to solve two problems:

1. **Representation**: Neural networks operate on numbers, not text. We need to convert characters into integers.
2. **Structure**: We need to organize the data into (input, target) pairs that teach the network "given this sequence, the next character is that."

By the end of this chapter you'll have:
- A character-level tokenizer that converts text ↔ numbers
- A PyTorch Dataset that produces training examples
- A DataLoader that feeds batches to the model

## Part 1: The Character Tokenizer

### What Is Tokenization?

Tokenization is the process of breaking text into discrete units (tokens) and assigning each one a number. There are many ways to tokenize:

| Method | Example | Vocab Size |
|--------|---------|-----------|
| Character-level | `"Hello"` → `["H","e","l","l","o"]` | ~65-100 |
| Word-level | `"Hello world"` → `["Hello", "world"]` | ~50,000+ |
| Subword (BPE) | `"unhappiness"` → `["un","happi","ness"]` | ~30,000-50,000 |

We start with **character-level** because:
- The vocabulary is tiny (~65 unique characters in Shakespeare)
- The tokenizer is trivial to implement — no complex algorithms
- 100% of our mental energy goes toward understanding the Transformer

We'll build BPE (what real LLMs use) in Chapter 7.

### Building the Tokenizer

The entire tokenizer is ~20 lines of code. Here's what it does:

1. Find every unique character in the text
2. Sort them (so the mapping is deterministic — same text always gives same mapping)
3. Build two lookup tables: character → integer, and integer → character

```python
# src/tokenizer.py

class CharTokenizer:
    def __init__(self, text: str):
        # sorted() gives us a deterministic ordering
        chars = sorted(set(text))
        self.vocab_size = len(chars)

        # The two lookup tables — this IS the tokenizer
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of integers."""
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices: list[int]) -> str:
        """Convert a list of integers back to a string."""
        return "".join(self.idx_to_char[i] for i in indices)
```

### Line-by-Line Walkthrough

```python
chars = sorted(set(text))
```
- `set(text)` finds every unique character. For Shakespeare, this gives us characters like `\n`, ` `, `!`, `'`, `,`, `.`, `A`-`Z`, `a`-`z`, etc.
- `sorted()` puts them in a consistent order (by ASCII value). This ensures that every time you create a tokenizer from the same text, `"a"` always maps to the same number.

```python
self.vocab_size = len(chars)
```
- For Shakespeare, this is **65** — the 65 unique characters that appear in the text. This number matters later: it determines the size of our embedding table and our output layer.

```python
self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
```
- This is a dictionary comprehension. `enumerate(chars)` produces `(0, '\n'), (1, ' '), (2, '!'), ...`. We flip it to `{'\n': 0, ' ': 1, '!': 2, ...}`.
- This is the **encoding** table — given a character, what's its number?

```python
self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
```
- The reverse: `{0: '\n', 1: ' ', 2: '!', ...}`.
- This is the **decoding** table — given a number, what's its character?

```python
def encode(self, text: str) -> list[int]:
    return [self.char_to_idx[ch] for ch in text]
```
- Walk through each character in the string, look up its number.
- `"Hello"` → `[20, 43, 50, 50, 53]` (exact numbers depend on the vocabulary)

```python
def decode(self, indices: list[int]) -> str:
    return "".join(self.idx_to_char[i] for i in indices)
```
- Walk through each number, look up its character, join them into a string.
- `[20, 43, 50, 50, 53]` → `"Hello"`

### Testing the Tokenizer

Add this to the bottom of `src/tokenizer.py`:

```python
if __name__ == "__main__":
    with open("data/input.txt", "r") as f:
        text = f.read()

    tok = CharTokenizer(text)
    print(f"Vocab size: {tok.vocab_size}")
    print(f"Characters: {''.join(tok.idx_to_char[i] for i in range(tok.vocab_size))}")

    # Round-trip test: encode then decode should give back the original
    sample = text[:100]
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded)
    assert sample == decoded, "Round-trip failed!"
    print(f"\nSample: {repr(sample)}")
    print(f"Encoded: {encoded[:50]}...")
    print(f"Decoded: {repr(decoded[:50])}...")
    print("\nRound-trip test passed!")
```

Run it:
```bash
python -m src.tokenizer
```

Expected output:
```
Vocab size: 65
Characters:
 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
```

Notice that the first character is `\n` (newline) and the second is ` ` (space). They're sorted by ASCII value. The vocabulary is 65 characters — this is our entire "language."

### Why This Matters

This simple mapping — character to integer — is the bridge between human-readable text and the mathematical world of neural networks. Every large language model has a tokenizer at its front door. Ours just happens to be the simplest possible version.

The **round-trip test** (`encode` then `decode` gives back the original) is critical. If this fails, everything downstream is broken. Always test your tokenizer.

---

## Part 2: Understanding the Training Task

### Next-Character Prediction

Our model will learn by predicting the next character given a context. Let's make this concrete.

Take the text: `"First Citizen:\nBefore we proceed"`

If our **context window** (called `block_size`) is 8 characters, then one training example is:

```
Input (x):  "First Ci"     →  [18, 47, 56, 57, 58, 1, 15, 47]
Target (y): "irst Cit"     →  [47, 56, 57, 58, 1, 15, 47, 58]
```

Notice: **y is just x shifted right by one character.** The target for position 0 is position 1's character. The target for position 1 is position 2's character. And so on.

But here's the subtle part: this single example actually contains **8 training signals**, not just 1:

```
Given "F"           → predict "i"      (position 0 → target at position 0 of y)
Given "Fi"          → predict "r"      (position 1 → target at position 1 of y)
Given "Fir"         → predict "s"      (position 2 → target at position 2 of y)
Given "Firs"        → predict "t"      (position 3 → target at position 3 of y)
Given "First"       → predict " "      (position 4 → target at position 4 of y)
Given "First "      → predict "C"      (position 5 → target at position 5 of y)
Given "First C"     → predict "i"      (position 6 → target at position 6 of y)
Given "First Ci"    → predict "t"      (position 7 → target at position 7 of y)
```

The model processes all 8 positions simultaneously and makes predictions at all of them. This is efficient — one forward pass gives us 8 learning signals.

**How does the model at position 3 not "cheat" by looking at position 4?** Through a **causal mask** — a mechanism that blocks each position from seeing future positions. We'll build this in Chapter 3. For now, just know that position 3 can only see positions 0, 1, 2, and 3.

### Visualizing the Training Data

```
Full text:  F i r s t   C i t i z e n : \n B e f o r e ...
            │ │ │ │ │ │ │ │
            ▼ ▼ ▼ ▼ ▼ ▼ ▼ ▼
Input  x:  [F,i,r,s,t, ,C,i]     ← 8 characters of context
Target y:  [i,r,s,t, ,C,i,t]     ← each shifted by 1 (the "answer")
```

This sliding window moves through the entire text. Position 0 gives one training example, position 1 gives another, position 2 gives another, and so on. With ~1.1 million characters and a block size of 256, we get about 1 million training examples.

---

## Part 3: PyTorch Basics You Need

Before we build the Dataset, here's the minimum PyTorch you need to know.

### Tensors

A tensor is like a NumPy array but can run on GPUs and supports automatic gradient computation.

```python
import torch

# Create a tensor from a list
x = torch.tensor([1, 2, 3, 4, 5])
print(x)          # tensor([1, 2, 3, 4, 5])
print(x.shape)    # torch.Size([5])

# 2D tensor (matrix)
m = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(m.shape)    # torch.Size([3, 2]) — 3 rows, 2 columns

# Random tensor
r = torch.randn(2, 3)   # 2×3, values from normal distribution
print(r.shape)    # torch.Size([2, 3])
```

### Tensor Shapes in Our Model

Throughout this project, you'll see tensors with these shapes:

```
(B, T)       — a batch of token sequences
                B = batch size (e.g., 64)
                T = sequence length (e.g., 256)

(B, T, C)    — a batch of embedded sequences
                C = embedding dimension (e.g., 128)

(B, T, V)    — a batch of predictions
                V = vocabulary size (e.g., 65)
```

For example, `(64, 256, 128)` means: "64 sequences in this batch, each 256 tokens long, each token represented by a 128-dimensional vector."

### Slicing Tensors

```python
data = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80])

# Slice: start:end (end is exclusive)
print(data[2:5])      # tensor([30, 40, 50])
print(data[:3])        # tensor([10, 20, 30])  — first 3
print(data[-2:])       # tensor([70, 80])      — last 2
```

This is how we'll extract input (`data[idx:idx+block_size]`) and target (`data[idx+1:idx+block_size+1]`) — two slices offset by one position.

### dtype: Data Types

```python
# Integers (for token indices)
tokens = torch.tensor([1, 2, 3], dtype=torch.long)

# Floating point (for embeddings, computations)
embeddings = torch.randn(3, 4)  # float32 by default
```

Token indices must be `torch.long` (64-bit integers) because PyTorch's embedding layers require it. Neural network computations use `float32` by default.

---

## Part 4: Building the Dataset

### What Is a PyTorch Dataset?

A `Dataset` is a class that tells PyTorch:
- How many examples do you have? (`__len__`)
- Give me example number `i`. (`__getitem__`)

PyTorch's `DataLoader` then handles shuffling, batching, and feeding these examples to the model efficiently.

### The CharDataset Class

```python
# src/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from src.tokenizer import CharTokenizer


class CharDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        """
        Args:
            data: 1D tensor of encoded characters (the full text as integers)
            block_size: number of characters the model sees at once (context window)
        """
        self.data = data
        self.block_size = block_size

    def __len__(self):
        # Every position (except the last block_size chars) is a valid start
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y
```

### Line-by-Line Walkthrough

```python
def __init__(self, data: torch.Tensor, block_size: int):
    self.data = data
    self.block_size = block_size
```
- `data` is the entire text encoded as a single long tensor of integers. For Shakespeare, this is ~1.1 million integers.
- `block_size` is how many characters the model sees at once — its "context window." We use 256.

```python
def __len__(self):
    return len(self.data) - self.block_size
```
- If `data` has 1,000,000 elements and `block_size` is 256, we have 999,744 valid starting positions.
- Why subtract? Because each example needs `block_size` characters for input AND one more for the last target. Starting at position `len(data) - block_size` would mean `y` extends beyond the data.

Actually, let's trace through this carefully:
```
data = [a, b, c, d, e, f, g, h]     (length 8)
block_size = 3

idx=0:  x = [a, b, c],  y = [b, c, d]    ✓
idx=1:  x = [b, c, d],  y = [c, d, e]    ✓
idx=2:  x = [c, d, e],  y = [d, e, f]    ✓
idx=3:  x = [d, e, f],  y = [e, f, g]    ✓
idx=4:  x = [e, f, g],  y = [f, g, h]    ✓
idx=5:  x = [f, g, h],  y = [g, h, ???]  ✗ out of bounds!

Valid indices: 0 through 4 → that's 5 = len(data) - block_size = 8 - 3
```

```python
def __getitem__(self, idx):
    x = self.data[idx : idx + self.block_size]
    y = self.data[idx + 1 : idx + self.block_size + 1]
    return x, y
```
- `x`: the input sequence, `block_size` characters starting at `idx`
- `y`: the target sequence, `block_size` characters starting at `idx + 1`
- `y` is simply `x` shifted right by one position. The model's job is to predict `y` from `x`.

### The Train/Val Split

```python
def create_datasets(text: str, block_size: int, train_split: float = 0.9):
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)

    # Simple split — first 90% train, last 10% val
    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_dataset = CharDataset(train_data, block_size)
    val_dataset = CharDataset(val_data, block_size)

    return tok, train_dataset, val_dataset
```

**Why split?** We need to know if the model is actually learning generalizable patterns or just memorizing the training data. The **validation set** is text the model never sees during training — if it performs well on the val set, it's truly learning the structure of English/Shakespeare, not just memorizing specific passages.

**Why 90/10?** This is a common ratio. 90% is enough data to learn from; 10% is enough to get a reliable estimate of generalization performance.

**Why split sequentially (first 90%, not random)?** For text data, random splitting would create examples where the model trains on "The cat sat on the" and validates on "mat" — leaking information across the split. Sequential splitting ensures the val set is an entirely separate section of text.

### The DataLoader

```python
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
```

The DataLoader does three things:
1. **Batching**: Groups 64 individual examples into one batch (more efficient than processing one at a time)
2. **Shuffling**: Randomizes the order each epoch (prevents the model from learning the sequence order of the training data)
3. **Iteration**: Lets us loop with `for x, y in train_loader:`

**Why batch_size=64?** Each example is `(256,)` integers — tiny. A batch of 64 is `(64, 256)` — still small enough to fit in memory easily. Larger batches give more stable gradient estimates but use more memory. 64 is a good default.

**Why shuffle?** Without shuffling, the model would see the data in the same order every epoch. It might learn patterns from the ordering (e.g., "after this scene comes that scene") rather than the actual language. Shuffling forces it to learn general patterns.

---

## Part 5: Putting It All Together

The complete `src/dataset.py` file:

```python
"""
Dataset for character-level language modeling.

The key idea: for next-character prediction, every position
in the text gives us a training example.

If block_size=8 and the text is "First Citizen", then one
training sample might be:

  x = "First Ci"  (input)
  y = "irst Cit"  (target — shifted by one)

Each character in x should predict the next character in y.
The model learns from ALL 8 predictions simultaneously.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from src.tokenizer import CharTokenizer


class CharDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def create_datasets(text: str, block_size: int, train_split: float = 0.9):
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)

    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_dataset = CharDataset(train_data, block_size)
    val_dataset = CharDataset(val_data, block_size)

    print(f"Total characters: {len(data):,}")
    print(f"Train: {len(train_data):,} chars → {len(train_dataset):,} samples")
    print(f"Val:   {len(val_data):,} chars → {len(val_dataset):,} samples")
    print(f"Vocab size: {tok.vocab_size}")

    return tok, train_dataset, val_dataset


if __name__ == "__main__":
    with open("data/input.txt", "r") as f:
        text = f.read()

    block_size = 256
    batch_size = 64

    tok, train_ds, val_ds = create_datasets(text, block_size)

    # Create a DataLoader and inspect one batch
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    x_batch, y_batch = next(iter(loader))

    print(f"\nBatch shapes: x={x_batch.shape}, y={y_batch.shape}")
    print(f"Expected: x=({batch_size}, {block_size}), y=({batch_size}, {block_size})")

    # Decode the first sample to verify it makes sense
    print(f"\n--- First sample in batch (x) ---")
    print(tok.decode(x_batch[0].tolist()))
    print(f"\n--- First sample in batch (y) ---")
    print(tok.decode(y_batch[0].tolist()))
    print(f"\n(y should be x shifted by one character)")
```

### Running It

```bash
python -m src.dataset
```

Expected output (the specific text will vary due to shuffling):
```
Total characters: 1,115,394
Train: 1,003,854 chars → 1,003,598 samples
Val:   111,540 chars → 111,284 samples
Vocab size: 65

Batch shapes: x=torch.Size([64, 256]), y=torch.Size([64, 256])
Expected: x=(64, 256), y=(64, 256)

--- First sample in batch (x) ---
[some Shakespeare text...]

--- First sample in batch (y) ---
[same text, shifted by one character...]

(y should be x shifted by one character)
```

### Verifying the Shift

Look at the last few characters of `x` and `y`. If `x` ends with `"the ba"`, `y` should end with `"he bas"` — shifted by exactly one character. This is the fundamental structure of our training data: input and target are offset by one.

---

## Summary

Here's what we built and why:

| Component | What It Does | Why We Need It |
|-----------|-------------|----------------|
| `CharTokenizer` | Converts characters ↔ integers | Neural networks need numbers, not text |
| `CharDataset` | Produces (input, target) pairs | Defines what the model learns: next-char prediction |
| `DataLoader` | Batches and shuffles examples | Efficient training, prevents order memorization |
| `create_datasets` | Splits data into train/val | Measures if the model truly generalizes |

### The Data Flow

```
"First Citizen:\nBefore we..."
          │
          ▼  CharTokenizer.encode()
[18, 47, 56, 57, 58, 1, 15, 47, 58, 47, 64, 43, 52, ...]
          │
          ▼  torch.tensor(..., dtype=torch.long)
tensor([18, 47, 56, 57, 58, 1, 15, 47, 58, 47, 64, 43, 52, ...])
          │
          ▼  CharDataset.__getitem__(idx)
x = tensor([18, 47, 56, ...])    (256 tokens)
y = tensor([47, 56, 57, ...])    (256 tokens, shifted by 1)
          │
          ▼  DataLoader(batch_size=64)
x_batch = tensor of shape (64, 256)    ← 64 examples
y_batch = tensor of shape (64, 256)    ← 64 targets
```

This is the complete pipeline from raw text to training batches. In the next chapter, we'll build the neural network that takes these `(64, 256)` batches and learns to predict the next character.

## What's Next

In [Chapter 2](02_embeddings.md), we start building the model — beginning with how the network turns those integer tokens into rich vector representations that capture meaning.