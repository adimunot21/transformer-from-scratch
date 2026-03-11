# Chapter 6: Generation — Making the Model Write

The model is trained. It can look at any sequence of characters and predict a probability distribution over the next character. But how do we turn that into actual generated text?

This chapter covers **autoregressive generation** — producing text one character at a time — and the sampling strategies that control whether the output is conservative or creative.

## Part 1: Autoregressive Generation

### The Core Loop

Generation works by repeating one simple step:

```
1. Feed the current text to the model
2. Get the predicted probabilities for the next character
3. Pick a character from those probabilities
4. Append it to the text
5. Go to step 1
```

"Autoregressive" means each new prediction depends on all previous predictions. The model generates one token, then uses that token as part of the input for generating the next token.

### A Concrete Example

Starting with the prompt "ROMEO:":

```
Step 1: Feed "ROMEO:" to the model
        → Model outputs probabilities for the next character
        → "\n" has probability 0.72 (most likely after a colon in dialogue)
        → Pick "\n"
        → Text is now "ROMEO:\n"

Step 2: Feed "ROMEO:\n" to the model
        → "O" has probability 0.08, "T" has 0.11, "A" has 0.06, ...
        → Pick "T" (sampled from the distribution)
        → Text is now "ROMEO:\nT"

Step 3: Feed "ROMEO:\nT" to the model
        → "h" has probability 0.45 (common after "T" at start of speech)
        → Pick "h"
        → Text is now "ROMEO:\nTh"

...continue for hundreds of characters...
```

Each step, the model sees everything generated so far and predicts the next character. The predictions build on each other — the model knows it's generating Romeo's speech because "ROMEO:\n" is in the context.

### Context Window Limitation

Our model has a `block_size` of 256. It can only see the most recent 256 characters. If we've generated 300 characters, we crop to the last 256:

```
Full generated text: [c₁, c₂, c₃, ..., c₄₄, c₄₅, ..., c₃₀₀]
                      ↑ forgotten                  ↑──── model sees these 256 ────↑

Model input: [c₄₅, c₄₆, ..., c₃₀₀]   (last 256 characters)
```

Characters beyond the window are "forgotten." This is a fundamental limitation of fixed-context Transformers. GPT-2 has a window of 1024 tokens. GPT-4 has windows up to 128K tokens. Bigger windows = more context = more coherent long-form text.

---

## Part 2: From Logits to Characters

The model outputs **logits** — raw, unnormalized scores for each character. To turn these into a character, we need to:

1. Convert logits to probabilities (softmax)
2. Choose a character from the probability distribution

### Softmax Refresher

```
logits = [2.1, 0.5, -0.3, 1.8, 0.1, ...]     (65 values, one per character)

probabilities = softmax(logits)
             = [e^2.1, e^0.5, e^-0.3, e^1.8, e^0.1, ...] / sum
             = [0.28, 0.06, 0.03, 0.21, 0.04, ...]        (sum to 1.0)
```

Higher logits → higher probabilities. The character with logit 2.1 gets probability 0.28 — the model's best guess. But other characters still have non-zero probability.

### Sampling vs. Argmax

Now we need to pick a character. There are two fundamental approaches:

**Greedy (argmax)**: Always pick the character with the highest probability.
```
probabilities: [0.28, 0.06, 0.03, 0.21, 0.04, ...]
greedy choice: index 0 (probability 0.28)
```

**Sampling**: Randomly pick a character, where each character's chance of being picked equals its probability.
```
probabilities: [0.28, 0.06, 0.03, 0.21, 0.04, ...]
sampled choice: usually index 0 or 3 (highest probs), but occasionally index 1, 4, etc.
```

Greedy decoding sounds like it should produce the "best" output, but it doesn't. It produces **repetitive, boring text**:

```
Greedy (temp=0.1):
"ROMEO:\nThe shall be the so so so so so man the soul
That the shall be the shall be the stands of the strange"
```

The model gets stuck in loops because the highest-probability next token is often the same common word. Sampling introduces variety — occasionally picking a less likely but more interesting word, which leads the model down different paths.

---

## Part 3: Temperature

Temperature is the single most important parameter for controlling generation. It modifies the probability distribution before sampling.

### The Math

Before applying softmax, divide all logits by the temperature T:

```
adjusted_logits = logits / T
probabilities = softmax(adjusted_logits)
```

That's it — one division. But the effect is dramatic.

### How Temperature Changes the Distribution

Let's trace through a concrete example. Suppose the model outputs these logits for four characters:

```
Original logits: [2.0, 1.0, 0.5, 0.1]
```

**Temperature = 1.0 (default — no change):**
```
logits / 1.0 = [2.0, 1.0, 0.5, 0.1]
softmax       = [0.45, 0.17, 0.10, 0.07]

Character 0 is most likely (0.45) but others have reasonable chances.
```

**Temperature = 0.3 (low — more confident):**
```
logits / 0.3 = [6.67, 3.33, 1.67, 0.33]
softmax       = [0.91, 0.03, 0.01, 0.00]

Character 0 dominates (0.91). Almost always picked. Very repetitive output.
```

**Temperature = 1.5 (high — more random):**
```
logits / 1.5 = [1.33, 0.67, 0.33, 0.07]
softmax       = [0.35, 0.18, 0.13, 0.10]

Distribution is flatter. Character 0 is still most likely but others have
substantial probability. More variety, but also more nonsense.
```

### The Intuition

Dividing by a small number makes large logits even larger relative to small ones — **sharpening** the distribution. The model becomes more "confident" (or more accurately, we amplify its existing preferences).

Dividing by a large number makes all logits closer to each other — **flattening** the distribution. The model becomes more "random" — even unlikely characters get a reasonable chance.

### Visual Intuition

```
Temperature = 0.3 (sharp):
  ████████████████████  char A (0.91)
  █                     char B (0.03)
                        char C (0.01)
                        char D (0.00)

Temperature = 1.0 (normal):
  █████████            char A (0.45)
  ███                  char B (0.17)
  ██                   char C (0.10)
  █                    char D (0.07)

Temperature = 1.5 (flat):
  ███████              char A (0.35)
  ████                 char B (0.18)
  ███                  char C (0.13)
  ██                   char D (0.10)
```

### Temperature = 0 (Limit Case)

As temperature approaches 0, the distribution becomes infinitely sharp — all probability concentrates on the single highest logit. This is equivalent to greedy/argmax decoding. In practice, we use T=0.1 instead of T=0 to avoid division by zero.

### What We Observed in Our Model

```
Temperature 0.3:
  "KING HENRY VI:\nThe would that have make the world of the see,
   That with the country of the son and of the such"
  → Grammatical structure, but repetitive ("the...of the...of the...")

Temperature 0.8 (sweet spot):
  "HENRY BOLINGBROKE:\nAnd spight I delive your greatene you
   What their time intents a more of fear"
  → Good variety, mostly real words, creative but coherent

Temperature 1.0:
  "Romeo lifter up it: the on: therefore of him,
   Will be unruther? Thou little thee and grave!"
  → More creativity, occasional nonsense words ("unruther")

Temperature 1.5:
  "But shult proveier hobid, if can Lovere fir: fore dwhat pntuche-"
  → Creative chaos. Many nonsense words. Structure breaks down.
```

**Temperature 0.8** is the standard "sweet spot" for most language models. It's the default we use.

---

## Part 4: Top-k Sampling

### The Problem with Full Sampling

Even with a good temperature, the model might occasionally sample a very unlikely character. With 65 characters, the bottom 50 might each have tiny probabilities (0.001), but collectively they add up. Every few characters, you might sample one of these unlikely options, producing a typo or nonsense.

### The Fix: Only Sample from the Top k

**Top-k sampling** restricts the sampling pool to only the k most probable characters. All other characters get their probability set to zero:

```
Full distribution (65 chars):
  "e": 0.25, " ": 0.18, "t": 0.12, "a": 0.08, "h": 0.06,
  "n": 0.05, "o": 0.04, "r": 0.03, "s": 0.03, "i": 0.02,
  ... 55 more characters with tiny probabilities ...

Top-k = 10:
  "e": 0.29, " ": 0.21, "t": 0.14, "a": 0.09, "h": 0.07,
  "n": 0.06, "o": 0.05, "r": 0.04, "s": 0.03, "i": 0.02
  (renormalized to sum to 1)
```

The probabilities are renormalized so they sum to 1 after removing the low-probability characters. Now the model can only produce one of the 10 most likely characters — no accidental nonsense from the tail of the distribution.

### The Implementation Trick

Rather than explicitly selecting the top k and renormalizing, we use a cleaner approach:

```python
if top_k is not None:
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = float("-inf")
```

1. `torch.topk(logits, k)` returns the k largest values
2. `v[:, [-1]]` is the smallest of the top-k values (the threshold)
3. Any logit below this threshold is set to `-inf`
4. After softmax, `e^(-inf) = 0` — these characters get zero probability

This is the same masking trick we used for the causal mask in attention. `-inf` before softmax = 0 probability after softmax.

### Top-k vs. Temperature

They solve different problems and work well together:

```
Temperature:  Controls the SHAPE of the distribution (sharp vs flat)
Top-k:        Controls the SIZE of the sampling pool (few vs many options)
```

A good combination: `temperature=0.8, top_k=10`:
- Temperature 0.8 slightly sharpens the distribution (reduces randomness)
- Top-k 10 eliminates the long tail (prevents rare nonsense)
- Result: creative but clean text

---

## Part 5: The Complete Generate Function

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        # Crop context to block_size
        ctx = idx[:, -self.block_size:]

        # Forward pass
        logits, _ = self(ctx)

        # Only care about the last position's prediction
        logits = logits[:, -1, :]

        # Apply temperature
        logits = logits / temperature

        # Optional top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append to the sequence
        idx = torch.cat([idx, next_token], dim=1)

    return idx
```

### Line-by-Line

```python
@torch.no_grad()
```
- Disable gradient tracking. Generation doesn't need gradients, and disabling them saves memory and speeds up computation. Without this, PyTorch would build a computation graph for every forward pass — after 500 generation steps, that graph would be enormous.

```python
ctx = idx[:, -self.block_size:]
```
- Crop the context to the maximum length the model can handle (256 characters).
- At step 1 of generation, `idx` might be 5 tokens (the prompt). No cropping needed.
- At step 300, `idx` is 305 tokens. We take the last 256.
- The `-self.block_size:` slice always takes the most recent tokens.

```python
logits, _ = self(ctx)
```
- Full forward pass through the model. We pass `targets=None` (no loss needed), so the second return value is `None`.
- `logits` has shape `(B, T, vocab_size)` — predictions for every position.

```python
logits = logits[:, -1, :]
```
- We only need the prediction at the **last** position — that's where the model predicts the next new character.
- Shape goes from `(B, T, 65)` to `(B, 65)` — one prediction per sequence in the batch.
- All the other positions' predictions were useful during training but are irrelevant during generation. The model already "saw" those positions' answers in the input.

```python
logits = logits / temperature
```
- Scale the logits by temperature. See Part 3 for the full explanation.

```python
if top_k is not None:
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = float("-inf")
```
- Top-k filtering. See Part 4.
- `min(top_k, logits.size(-1))` prevents requesting more values than exist (if k > vocab_size).

```python
probs = F.softmax(logits, dim=-1)
```
- Convert logits to probabilities. Each of the 65 values becomes a probability between 0 and 1, summing to 1.

```python
next_token = torch.multinomial(probs, num_samples=1)
```
- **Multinomial sampling**: randomly pick one index, where each index's probability of being picked equals its value in `probs`.
- If `probs = [0.4, 0.35, 0.2, 0.05]`, index 0 is picked 40% of the time, index 1 is picked 35% of the time, etc.
- `num_samples=1` — pick exactly one token.
- Shape: `(B, 1)` — one new token per sequence in the batch.

```python
idx = torch.cat([idx, next_token], dim=1)
```
- Append the new token to the sequence.
- `torch.cat` concatenates along dimension 1 (the sequence length dimension).
- Shape: `(B, T)` → `(B, T+1)` — the sequence grew by one token.

### Shape Trace Through One Generation Step

```
Starting: idx shape (1, 10)          ← prompt "ROMEO:\n..." encoded

Crop:     ctx shape (1, 10)          ← no crop needed (10 < 256)

Forward:  logits shape (1, 10, 65)   ← predictions at all 10 positions

Last pos: logits shape (1, 65)       ← prediction for position 10 only

Temperature + Top-k: shape unchanged  (1, 65)

Softmax:  probs shape (1, 65)        ← probabilities

Sample:   next_token shape (1, 1)    ← one sampled character

Concat:   idx shape (1, 11)          ← sequence grew by 1
```

After 200 generation steps, `idx` has shape `(1, 210)` — the original 10-token prompt plus 200 generated characters.

---

## Part 6: The Interactive Generation Script

The standalone script wraps the generate function with user-friendly features:

```python
def load_model(checkpoint_path, data_path="data/input.txt"):
    # Rebuild tokenizer from original data
    with open(data_path, "r") as f:
        text = f.read()
    tok = CharTokenizer(text)

    # Load checkpoint and rebuild model
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = GPT(
        vocab_size=tok.vocab_size,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        block_size=cfg["block_size"],
        dropout=0.0,          # No dropout during inference!
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()              # Switch to evaluation mode

    return model, tok, cfg
```

### Two Important Details

**`dropout=0.0` during inference.** During training, dropout randomly zeroes values to prevent overfitting. During generation, we want the model's full capacity — all neurons active. Setting dropout to 0.0 in the constructor (rather than relying on `model.eval()`) makes this explicit and foolproof.

Actually, `model.eval()` also disables dropout. So why set it to 0.0? Belt and suspenders — if someone forgets `model.eval()`, the model still works correctly.

**`map_location="cpu"`.** When loading a checkpoint, this ensures it loads onto CPU regardless of where it was saved. If the checkpoint was saved on GPU (e.g., in Colab), this prevents errors on a CPU-only machine.

---

## Part 7: Other Sampling Strategies

Temperature and top-k are the two strategies we implemented. For completeness, here are others used in production LLMs:

### Top-p (Nucleus Sampling)

Instead of keeping a fixed number of top tokens (top-k), keep the smallest set of tokens whose cumulative probability exceeds a threshold p:

```
Probabilities (sorted): [0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03, ...]

Top-p = 0.9:
  0.30 + 0.25 = 0.55   (keep going)
  + 0.15 = 0.70         (keep going)
  + 0.10 = 0.80         (keep going)
  + 0.08 = 0.88         (keep going)
  + 0.05 = 0.93         (stop! exceeded 0.9)

Keep first 6 tokens, zero out the rest.
```

The advantage over top-k: the pool size adapts. When the model is confident (one token has 0.95 probability), only 1 token is kept. When the model is uncertain (flat distribution), many tokens are kept. This is more adaptive than a fixed k.

### Beam Search

Instead of sampling one token at a time, track the top B (beam width) most probable sequences simultaneously:

```
"ROMEO:\n" → track top 3 continuations:
  Beam 1: "ROMEO:\nT"  (prob 0.15)
  Beam 2: "ROMEO:\nO"  (prob 0.12)
  Beam 3: "ROMEO:\nA"  (prob 0.09)

Next step: expand each beam, keep top 3 overall:
  Beam 1: "ROMEO:\nTh" (prob 0.15 × 0.40 = 0.060)
  Beam 2: "ROMEO:\nO," (prob 0.12 × 0.30 = 0.036)
  Beam 3: "ROMEO:\nTe" (prob 0.15 × 0.20 = 0.030)
```

Beam search finds higher-probability sequences than greedy decoding, but produces very "safe," often boring text. It's used more in machine translation than in creative text generation.

### Repetition Penalty

Reduce the probability of tokens that have already appeared recently. This combats the repetition problem ("the shall be the shall be the shall be") by making the model less likely to reuse recent tokens.

We didn't implement these in our model, but they're all modifications to the same step — adjusting the logits or probabilities before sampling. The generation loop itself stays the same.

---

## Part 8: Understanding What Generation Reveals

Generation isn't just a party trick — it's a diagnostic tool. The quality of generated text tells you exactly what the model has and hasn't learned:

### Diagnosis Table

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Random characters, no words | Model barely trained | Train longer |
| Real words but no grammar | Model learned frequencies, not structure | More layers/training |
| Repetitive loops ("the the the") | Temperature too low, or model underfit | Raise temp, train longer |
| Correct format but nonsense content | Model learned surface structure | More data, bigger model |
| Occasional nonsense words | Character-level model's inherent limit | Switch to BPE tokenization |
| Perfect Shakespeare | You've recreated GPT-4 (unlikely at 841K params) | Celebrate |

### The "First Citizen" Test

A good diagnostic prompt is "First Citizen:\n" — the very first line of the training data. If the model generates Coriolanus-style dialogue (which it did in our experiments), it learned which characters belong in which plays. If it generates random Shakespeare, it learned general style but not specific context.

---

## Summary

The generation pipeline:

```
Prompt: "ROMEO:"
    │
    ▼  Encode
[44, 35, 33, 17, 35, 10]
    │
    ▼  Loop for max_new_tokens:
    │
    │    ┌──────────────────────────────────────────┐
    │    │ 1. Crop to last 256 tokens               │
    │    │ 2. Forward pass → logits (B, T, 65)      │
    │    │ 3. Take last position → (B, 65)          │
    │    │ 4. Divide by temperature                 │
    │    │ 5. Top-k filter (optional)               │
    │    │ 6. Softmax → probabilities               │
    │    │ 7. Sample one token                      │
    │    │ 8. Append to sequence                    │
    │    └──────────────────────────────────────────┘
    │
    ▼  Decode
"ROMEO:\nO, she doth teach the torches to burn bright!..."
```

Three knobs that control generation:

| Parameter | Low Value | High Value | Sweet Spot |
|-----------|-----------|------------|------------|
| Temperature | Repetitive, safe | Random, nonsensical | 0.7–0.9 |
| Top-k | Very restricted (k=3) | Unrestricted (k=65) | 5–15 |
| Length | Short snippet | Long text (may lose coherence) | 200–500 |

## What's Next

In [Chapter 7](07_bpe_tokenizer.md), we replace our simple character-level tokenizer with **Byte-Pair Encoding (BPE)** — the same tokenization algorithm used by GPT-2, GPT-3, and GPT-4. This gives the model a smarter vocabulary where common subwords like "the", "ing", and "tion" are single tokens, dramatically improving compression and the model's ability to learn language patterns.