# Chapter 7: Byte-Pair Encoding — The Tokenizer Behind GPT

In Chapters 1–6, we used a character-level tokenizer: each character is a token. This is simple but wasteful — the model spends enormous effort learning to spell words, character by character. Every time it generates "the", it has to predict "t", then "h", then "e" — three separate decisions for one of the most common words in English.

What if "the" were a single token? The model would predict it in one step and could use its context window for higher-level patterns instead of low-level spelling.

This is exactly what **Byte-Pair Encoding (BPE)** does. It's the tokenization algorithm used by GPT-2, GPT-3, GPT-4, and most modern language models. In this chapter, we build it from scratch.

## Part 1: The Problem with Character-Level Tokenization

### Wasted Capacity

With character-level tokenization and a context window of 256 tokens, the model sees 256 characters — roughly 40-50 words. That's a short paragraph.

With BPE tokenization (vocab size ~500-50,000), the same 256 tokens might represent 500+ characters — two or three paragraphs. The model sees more context with the same computational cost.

### Wasted Learning

A character-level model must learn:
- "t" followed by "h" is common
- "th" followed by "e" is common
- "the" followed by " " is common

A BPE model with "the " as a single token already "knows" this. Its learning capacity goes toward understanding grammar, meaning, and structure — not spelling.

### The Numbers from Our Project

```
Character-level: 1,115,394 tokens for the full Shakespeare text
BPE (512 merges): 557,011 tokens — 2× compression

Same 256-token context window sees:
  Character: ~256 characters (~50 words)
  BPE:       ~512 characters (~100 words)
```

Twice the context for the same computation. This directly translates to better long-range coherence.

---

## Part 2: The BPE Algorithm — Intuition

BPE starts with the smallest possible tokens (individual bytes) and iteratively merges the most frequent adjacent pair into a new token. Each merge creates a slightly larger vocabulary with slightly better compression.

### A Toy Example

Let's run BPE on the tiny text: `"aaabdaaabac"`

**Initial vocabulary**: every unique character
```
Vocabulary: {a, b, c, d}
Tokens: [a, a, a, b, d, a, a, a, b, a, c]
```

**Merge 1**: Find the most frequent adjacent pair

```
Pairs and their counts:
  (a, a) → 4 times    ← WINNER
  (a, b) → 2 times
  (b, d) → 1 time
  (d, a) → 1 time
  (b, a) → 1 time
  (a, c) → 1 time

Merge (a, a) into new token "Z":
Vocabulary: {a, b, c, d, Z}    (Z represents "aa")
Tokens: [Z, a, b, d, Z, a, b, a, c]
```

**Merge 2**: Find the most frequent pair again

```
Pairs and their counts:
  (Z, a) → 2 times    ← WINNER
  (a, b) → 2 times    ← (tie — pick either one)
  (b, d) → 1 time
  (d, Z) → 1 time
  (b, a) → 1 time
  (a, c) → 1 time

Merge (Z, a) into new token "Y":
Vocabulary: {a, b, c, d, Z, Y}    (Y represents "aaa")
Tokens: [Y, b, d, Y, b, a, c]
```

**Merge 3**: Again

```
Pairs and their counts:
  (Y, b) → 2 times    ← WINNER
  (b, d) → 1 time
  (d, Y) → 1 time
  (b, a) → 1 time
  (a, c) → 1 time

Merge (Y, b) into new token "X":
Vocabulary: {a, b, c, d, Z, Y, X}    (X represents "aaab")
Tokens: [X, d, X, a, c]
```

After 3 merges:
- Original: 11 tokens → Now: 5 tokens (2.2× compression)
- The vocabulary grew from 4 to 7 tokens
- "aaab" became a single token because it was a frequent pattern

### The Key Insight

BPE discovers **frequent subword units** purely from statistics. On English text, it naturally learns:
- Common words: "the", "and", "for"
- Common suffixes: "ing", "tion", "ed"
- Common prefixes: "un", "re", "pre"
- Common bigrams: "th", "he", "in"

Nobody tells BPE about English morphology. It discovers these patterns because they're frequent.

---

## Part 3: Bytes, Not Characters

Our BPE implementation works at the **byte level**, not the character level. This is what GPT-2 does.

### Why Bytes?

Characters are tricky. Unicode has over 150,000 characters — Chinese, Arabic, emoji, mathematical symbols, etc. If we started with a character-level vocabulary, we'd need 150,000+ base tokens before any merges.

Bytes are simple: there are exactly 256 possible byte values (0–255). ANY text — English, Chinese, code, binary — can be represented as a sequence of bytes via UTF-8 encoding.

```python
"Hello".encode("utf-8")       → b'\x48\x65\x6c\x6c\x6f'    → [72, 101, 108, 108, 111]
"你好".encode("utf-8")         → b'\xe4\xbd\xa0\xe5\xa5\xbd' → [228, 189, 160, 229, 165, 189]
"🎉".encode("utf-8")          → b'\xf0\x9f\x8e\x89'          → [240, 159, 142, 137]
```

Starting with 256 byte-level tokens means:
- We can tokenize ANY text (no "unknown token" errors)
- The base vocabulary is small and fixed
- Merges build up from bytes to characters to subwords to words

For ASCII text (English, most code), one byte = one character. So for our Shakespeare project, byte-level and character-level are nearly identical. But the byte-level approach generalizes to any language.

### Our Vocabulary Structure

```
Token IDs 0–255:    Individual bytes (the base vocabulary)
Token IDs 256–767:  Merged tokens (512 merges → 512 new tokens)

Total vocabulary: 256 + 512 = 768 tokens
```

---

## Part 4: The Training Code

```python
class BPETokenizer:
    def __init__(self):
        self.merges = {}          # (tok_a, tok_b) → new_token_id
        self.vocab = {}           # token_id → bytes
        self.merge_list = []      # ordered list of merges for encoding

    def train(self, text, num_merges=256, verbose=True):
        # Step 1: Convert text to bytes
        text_bytes = text.encode("utf-8")

        # Split into chunks at word boundaries
        chunks = []
        current = []
        for b in text_bytes:
            current.append(b)
            if b in (ord(" "), ord("\n")):
                chunks.append(current)
                current = []
        if current:
            chunks.append(current)

        # Initialize vocabulary with all 256 bytes
        self.vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256

        # Step 2: Iteratively merge
        for i in range(num_merges):
            # Count all adjacent pairs
            pair_counts = self._get_pair_counts(chunks)

            if not pair_counts:
                break

            # Find most frequent pair
            pair, count = pair_counts.most_common(1)[0]

            if count < 2:
                break

            # Create new token
            new_bytes = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.vocab[next_id] = new_bytes

            # Record merge
            self.merges[pair] = next_id
            self.merge_list.append(pair)

            # Apply merge to all chunks
            chunks = self._merge_pair(chunks, pair, next_id)

            next_id += 1
```

### Line-by-Line Walkthrough

```python
text_bytes = text.encode("utf-8")
```
- Convert the entire text to a sequence of bytes. For ASCII text (Shakespeare), each character becomes one byte. The string "Hello" becomes `[72, 101, 108, 108, 111]`.

```python
chunks = []
current = []
for b in text_bytes:
    current.append(b)
    if b in (ord(" "), ord("\n")):
        chunks.append(current)
        current = []
```
- Split the text into chunks at spaces and newlines.
- **Why split?** This prevents merges across word boundaries. Without splitting, the tokenizer might merge the "d" at the end of "and" with the "t" at the start of "the" into a "dt" token. This is meaningless and wasteful. Splitting at spaces ensures merges only happen within words or at word edges (like "the " becoming one token, which includes the trailing space).
- `ord(" ")` gives the byte value of space (32). `ord("\n")` gives newline (10).

```python
self.vocab = {i: bytes([i]) for i in range(256)}
next_id = 256
```
- Initialize the vocabulary with all 256 possible bytes.
- `bytes([72])` creates a single-byte object `b'H'`.
- New merged tokens will start at ID 256.

```python
pair_counts = self._get_pair_counts(chunks)
```
- Count every adjacent pair across all chunks. This is the core statistic that drives BPE — which two adjacent tokens appear together most often?

```python
pair, count = pair_counts.most_common(1)[0]
```
- Find the single most frequent pair. On Shakespeare, the first merge is typically `(32, 116)` which is `(" ", "t")` — the space before "t" is extremely common because of words like "the", "to", "that", "this".

```python
if count < 2:
    break
```
- If no pair appears more than once, further merging is useless. Every merge would save exactly one token at most. In practice, this never triggers with enough data.

```python
new_bytes = self.vocab[pair[0]] + self.vocab[pair[1]]
self.vocab[next_id] = new_bytes
```
- Create the new token by concatenating the bytes of the two tokens being merged.
- If we're merging token 32 (`b' '`) and token 116 (`b't'`), the new token's bytes are `b' t'`.
- Store it in the vocabulary at the next available ID (256, 257, 258, ...).

```python
self.merges[pair] = next_id
self.merge_list.append(pair)
```
- Record this merge in two places:
  - `merges` dict: used for fast lookup during encoding
  - `merge_list`: preserves the ORDER of merges, which is critical for encoding (Part 5)

```python
chunks = self._merge_pair(chunks, pair, next_id)
```
- Apply the merge to all chunks. Every occurrence of `(pair[0], pair[1])` adjacent in any chunk gets replaced with `next_id`.

### The Merge Function

```python
def _merge_pair(self, token_lists, pair, new_id):
    result = []
    for tokens in token_lists:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(new_id)
                i += 2          # Skip both tokens
            else:
                new_tokens.append(tokens[i])
                i += 1
        result.append(new_tokens)
    return result
```

This walks through each chunk, scanning for the pair. When found, it replaces both tokens with the new merged token and skips ahead by 2. Otherwise, it keeps the token and advances by 1.

```
Before merge (pair = (97, 97), new_id = 256):
  [97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99]
   ^   ^              ^   ^
   merge              merge

After merge:
  [256, 97, 98, 100, 256, 97, 98, 97, 99]
```

Notice the first `(97, 97, 97)` becomes `(256, 97)` not `(256, 256)` — the scan goes left to right and doesn't re-examine merged tokens. This is intentional and matches the standard BPE algorithm.

---

## Part 5: Encoding — Applying Merges to New Text

Training learns the merge rules. **Encoding** applies them to new text.

The critical rule: **apply merges in the same order they were learned**.

```python
def encode(self, text):
    # Start with raw bytes
    tokens = list(text.encode("utf-8"))

    # Apply each merge in order
    for pair in self.merge_list:
        new_id = self.merges[pair]
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return tokens
```

### Why Order Matters

Suppose our merges were:
1. Merge `(a, b)` → token 256
2. Merge `(256, c)` → token 257    (i.e., merge "abc" → single token)

For the text "abc":

**Correct order (merge 1 then 2):**
```
Start:    [a, b, c]
Merge 1:  [256, c]       ← "ab" merged first
Merge 2:  [257]          ← "ab" + "c" merged into "abc"
Result:   1 token ✓
```

**Wrong order (merge 2 first):**
```
Start:    [a, b, c]
Merge 2:  Looking for (256, c)... but 256 doesn't exist yet!
          [a, b, c]      ← nothing happens
Merge 1:  [256, c]       ← "ab" merged
Result:   2 tokens ✗     ← missed the second merge
```

Merge 2 references token 256, which only exists after merge 1. Applying merges out of order breaks the encoding.

This is why we store `self.merge_list` as an ordered list, not just a dictionary.

### Encoding Example

```python
tok.encode("the torches")
```

Step through (simplified):

```
Start: [116, 104, 101, 32, 116, 111, 114, 99, 104, 101, 115]
        t    h    e    _   t    o    r    c    h    e    s

After early merges (e.g., "th" → 258, "e " → 260):
       [258, 260, 116, 111, 114, 99, 104, 101, 115]
        th   e_   t    o    r    c    h    e    s

After more merges (e.g., "or" → 280, "ch" → 275):
       [258, 260, 116, 280, 275, 101, 115]
        th   e_   t    or   ch   e    s

After "the " merge (if it exists as a single token):
       [300, 116, 280, 275, 101, 115]
        the_ t    or   ch   e    s
```

11 bytes → 6 tokens. The common word "the " (with space) is one token. Frequent substrings like "th", "or", "ch" are single tokens. Rare characters like "s" stay as individual bytes.

---

## Part 6: Decoding — Tokens Back to Text

Decoding is simple — look up each token's bytes and concatenate:

```python
def decode(self, tokens):
    byte_list = b"".join(self.vocab[t] for t in tokens)
    return byte_list.decode("utf-8", errors="replace")
```

### Line-by-Line

```python
byte_list = b"".join(self.vocab[t] for t in tokens)
```
- For each token ID, look up its bytes in the vocabulary.
- Token 300 → `b'the '` (4 bytes)
- Token 116 → `b't'` (1 byte)
- Token 280 → `b'or'` (2 bytes)
- Join all bytes into one long byte string.

```python
return byte_list.decode("utf-8", errors="replace")
```
- Convert bytes back to a Python string using UTF-8 encoding.
- `errors="replace"` means if there's an invalid byte sequence (shouldn't happen with correct tokenization), replace it with `�` instead of crashing.

Decoding always produces the original text — this is guaranteed by the design. Each token maps to a specific byte sequence, and concatenating them in order reproduces the original byte sequence.

---

## Part 7: What Our BPE Learned

When we trained BPE with 512 merges on Shakespeare, here's what it discovered:

### Early Merges (most frequent pairs)

```
Merge   1: " t"  → token 256    (space before "t" — "the", "to", "that")
Merge   2: "e "  → token 257    (end of word patterns — "the ", "be ", "me ")
Merge   3: ", "  → token 258    (comma-space — ubiquitous in Shakespeare)
Merge   4: "th"  → token 259    ("the", "that", "this", "thou", "thee")
Merge   5: "he"  → token 260    ("the", "he", "her", "here", "heaven")
```

These are the most common adjacent byte pairs in English text. The algorithm discovered them automatically.

### Middle Merges (common subwords)

```
Merge  50: "e, "  → token 305   (word ending before comma)
Merge 100: "di"   → token 355   ("did", "die", "disdain")
Merge 150: "de"   → token 405   ("death", "deed", "desire")
Merge 200: "him " → token 455   (complete common word with space)
```

By merge 200, the tokenizer is discovering complete short words and word fragments that align with English morphology.

### Late Merges (longer subwords)

```
Merge 250: "so"   → token 505
Merge 300: "for"  → token 556   (complete word)
Merge 400: "ther" → token 656   (suffix — "other", "whether", "father")
Merge 500: "ear " → token 756   ("hear", "dear", "fear", "near")
```

By the end, the tokenizer has discovered common English words and meaningful suffixes — purely from frequency statistics on Shakespeare.

### Special Structure Tokens

One of the most interesting merges:

```
":\n" → token 267    (colon followed by newline)
```

In Shakespeare's plays, `:\n` appears after every character name in the dialogue:

```
ROMEO:\n
JULIET:\n
KING HENRY:\n
```

The tokenizer discovered that this two-byte sequence is so frequent it deserves its own token. This means the model can represent "end of character name, start of speech" as a single token — a structural unit specific to the training data.

---

## Part 8: Saving and Loading

The tokenizer needs to be saved so we can use it later (for generation, for loading the trained model):

```python
def save(self, path):
    data = {
        "merge_list": self.merge_list,
        "vocab": {str(k): list(v) for k, v in self.vocab.items()},
    }
    with open(path, "w") as f:
        json.dump(data, f)
```

```python
@classmethod
def load(cls, path):
    with open(path, "r") as f:
        data = json.load(f)
    tok = cls()
    tok.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
    tok.merge_list = [tuple(p) for p in data["merge_list"]]
    tok.merges = {}
    next_id = 256
    for pair in tok.merge_list:
        tok.merges[pair] = next_id
        next_id += 1
    return tok
```

### Why Save As JSON?

- Human-readable — you can open the file and inspect the vocabulary
- Portable — works across Python versions and platforms
- The vocabulary is small (~500 entries) so file size doesn't matter

### A Note on vocab Serialization

```python
"vocab": {str(k): list(v) for k, v in self.vocab.items()}
```

JSON doesn't support integer keys or bytes objects. So we convert:
- Keys: `int → str` (e.g., `256 → "256"`)
- Values: `bytes → list of ints` (e.g., `b'the' → [116, 104, 101]`)

When loading, we reverse these conversions.

---

## Part 9: Using BPE with Our Model

The model architecture doesn't change at all. The only differences are:

### 1. Vocabulary Size

```python
# Character-level
model = GPT(vocab_size=65, ...)

# BPE with 512 merges
model = GPT(vocab_size=768, ...)    # 256 bytes + 512 merges
```

The embedding table grows from `65 × 128 = 8,320` parameters to `768 × 128 = 98,304` parameters. The output projection grows similarly. Everything else stays the same.

### 2. Dataset

The Dataset class is identical — it still returns `(x, y)` pairs where `y` is `x` shifted by one. The only change is that `x` and `y` contain BPE token IDs instead of character indices.

```python
# Character-level encoding
data = torch.tensor(char_tok.encode(text), dtype=torch.long)
# 1,115,394 tokens

# BPE encoding
data = torch.tensor(bpe_tok.encode(text), dtype=torch.long)
# 557,011 tokens — half as many!
```

### 3. Context Coverage

With `block_size=256`:
- Character-level: each training example covers 256 characters
- BPE: each training example covers ~512 characters (2× compression)

The model sees twice as much context, which helps it learn longer-range patterns like dialogue coherence and consistent character voices.

---

## Part 10: BPE vs. Character-Level — Results Comparison

From our experiments:

### Generation Quality

**Character-level model:**
```
"you have be heaven the world is a unto either,
But battless fearful newll: but be more"
```
Occasional nonsense words ("battless", "newll") because the model has to spell every word character by character. One mistake in the middle of a word produces gibberish.

**BPE model:**
```
"Please you to hear you shall believe your brother's love?
And now the day was the majesty of your daughter."
```
Nearly all real words. Because common words are single tokens, the model can't misspell them. Its errors are grammatical (wrong word choice) rather than orthographic (wrong spelling).

### The Overfitting Challenge

The BPE model overfits more because:
- Larger vocabulary → more parameters (embedding + output layers grow)
- Fewer training tokens (557K vs 1.1M) → less data per parameter
- We compensated with more dropout (0.25 vs 0.1) and weight decay (0.1)

This is a general pattern in ML: better representations (BPE) require either more data or more regularization.

---

## Summary

The BPE algorithm:

```
Input text → UTF-8 bytes → Split at word boundaries

Repeat num_merges times:
  1. Count all adjacent token pairs
  2. Find the most frequent pair
  3. Create a new token = concatenation of the pair
  4. Replace all occurrences of the pair with the new token

Result: a vocabulary of 256 + num_merges tokens
        an ordered list of merge rules for encoding
```

Key concepts:

| Concept | What It Means |
|---------|--------------|
| Byte-level | Start with 256 base tokens (all possible bytes) — handles any text |
| Merge | Combine two adjacent tokens into one new token |
| Merge order | Merges must be applied in training order during encoding |
| Compression | Common patterns become single tokens → shorter sequences |
| Vocab size | 256 + num_merges. More merges = better compression but larger embedding table |

The BPE tokenizer is the bridge between raw text and the model's input. It determines what the model's "atoms" are — the smallest units it thinks in. Character-level models think in letters. BPE models think in subwords. GPT-4 might think in units like "the ", "tion", "ing", "however", making it far more efficient at representing language.

## What's Next

In [Chapter 8](08_decision_transformer.md), we take the exact same Transformer architecture and apply it to a completely different domain: **reinforcement learning**. The Decision Transformer treats RL as sequence prediction — instead of predicting the next character, it predicts the next action given a desired reward. Same attention, same feed-forward layers, different input and output. This demonstrates that the Transformer is a general-purpose sequence processor, not just a language tool.