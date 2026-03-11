# Chapter 5: Training — Teaching the Model to Write Shakespeare

We have a model with 841,281 parameters, all initialized to small random values. Right now it's useless — it assigns roughly equal probability to all 65 characters regardless of context. Training is the process of adjusting those 841,281 numbers so the model gets better at predicting the next character.

This chapter covers everything that happens during training: the optimizer, learning rate scheduling, the training loop, and how to interpret what's happening.

## Part 1: How Neural Networks Learn

### The Training Loop (Big Picture)

Every step of training follows the same four steps:

```
Step 1: FORWARD PASS
  Feed a batch of data through the model → get predictions

Step 2: COMPUTE LOSS
  Compare predictions to actual answers → get a single number (the loss)

Step 3: BACKWARD PASS
  Compute gradients: "how should each parameter change to reduce the loss?"

Step 4: UPDATE
  Adjust each parameter slightly in the direction that reduces the loss
```

Repeat this thousands of times. Each repetition is called a **training step** (or iteration). Let's unpack each step.

### Step 1: Forward Pass

We already built this in Chapter 4. Feed token indices in, get logits out:

```python
logits, loss = model(x_batch, y_batch)
```

The model processes the input through embeddings → attention → FFN → output projection, producing a prediction for every position.

### Step 2: Loss

Also built in Chapter 4. Cross-entropy loss measures how wrong the predictions are:

```
loss ≈ 4.17 → model is guessing randomly (1/65 chance for each char)
loss ≈ 2.5  → model learned common character frequencies
loss ≈ 1.5  → model learned word patterns and structure
loss ≈ 1.0  → model learned longer-range patterns
loss = 0.0  → model predicts perfectly (never happens in practice)
```

### Step 3: Backward Pass (Backpropagation)

This is where PyTorch's autograd shines. When we call:

```python
loss.backward()
```

PyTorch computes the **gradient** of the loss with respect to every single parameter in the model. A gradient tells you: "if I increase this parameter by a tiny amount, how much does the loss change?"

```
Parameter: W = 0.05
Gradient:  ∂loss/∂W = 0.3

Interpretation: increasing W by 0.001 would increase loss by ~0.0003
                Therefore: DECREASE W to reduce loss
```

The gradient is computed using the **chain rule** from calculus, applied automatically through the computation graph. You don't need to implement this — PyTorch tracks every operation during the forward pass and automatically computes all gradients during the backward pass.

For our model, `loss.backward()` computes 841,281 gradients — one for every parameter.

### Step 4: Update (The Optimizer)

Given the gradients, the simplest update rule is **gradient descent**:

```
new_weight = old_weight - learning_rate × gradient
```

If the gradient is positive (increasing the weight increases loss), we decrease the weight. If the gradient is negative (increasing the weight decreases loss), we increase the weight. The **learning rate** controls how big a step we take.

But plain gradient descent has problems. Modern deep learning uses smarter optimizers.

---

## Part 2: The AdamW Optimizer

### Why Not Plain Gradient Descent?

Gradient descent treats all parameters the same — same learning rate, same step size. But in a neural network:

- Some parameters get large, frequent gradients (common patterns)
- Others get small, rare gradients (rare patterns)
- Gradient direction can oscillate (noisy)

Using the same learning rate for all of them is inefficient.

### What Adam Does

**Adam** (Adaptive Moment Estimation) maintains two additional numbers for each parameter:

1. **First moment (m)**: Running average of the gradient. Like a "momentum" — if the gradient has been pointing in the same direction for many steps, the momentum builds up, allowing bigger steps. If the gradient keeps flipping, the momentum stays small.

2. **Second moment (v)**: Running average of the gradient squared. This captures the *magnitude* of typical gradients for this parameter. Parameters with consistently large gradients get their learning rate reduced. Parameters with small gradients get their learning rate increased.

The update becomes:

```
m = 0.9 × m + 0.1 × gradient           ← smooth the gradient direction
v = 0.999 × v + 0.001 × gradient²       ← track gradient magnitude

update = m / (√v + ε)                    ← adaptive step size

new_weight = old_weight - lr × update
```

The division by `√v` is the key: it normalizes the step size by the typical gradient magnitude. Parameters with noisy, large gradients take smaller steps. Parameters with consistent, small gradients take larger steps. This **adapts** the effective learning rate per parameter.

### The "W" in AdamW — Weight Decay

**AdamW** adds one thing to Adam: **weight decay**. After each update, it slightly shrinks all weights toward zero:

```
new_weight = old_weight - lr × update - lr × weight_decay × old_weight
```

Weight decay is a form of **regularization** — it discourages the model from using very large weights. This prevents overfitting because large weights mean the model is "trying too hard" to fit specific training examples.

Think of it as a force pulling all weights toward zero. The model has to "justify" each weight being non-zero by showing it helps reduce the loss. Weights that don't help much get pulled to zero.

### In Code

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
```

That's it. PyTorch handles all the momentum, adaptive rates, and weight decay internally. The key parameter is `lr=3e-4` (0.0003) — the base learning rate.

**Why 3e-4?** This is a common default for Transformer training, popularized by the original BERT paper. It's small enough to not overshoot but large enough to make progress. Too large (1e-2) and training explodes. Too small (1e-6) and training takes forever.

---

## Part 3: Learning Rate Scheduling

### Why Not a Constant Learning Rate?

Using `lr=3e-4` for every step is suboptimal:

- **Early in training**: The model is far from a good solution. Larger steps are beneficial — they explore the loss landscape quickly.
- **Late in training**: The model is near a good solution. Large steps overshoot. Smaller steps let it settle into a good minimum.

A **learning rate schedule** varies the learning rate over the course of training.

### Warmup + Cosine Decay

We use the most common schedule in modern Transformer training:

```
Phase 1: Linear Warmup (first 500 steps)
  LR ramps from ~0 up to max_lr (3e-4)

Phase 2: Cosine Decay (remaining steps)
  LR smoothly decreases from max_lr down to ~0
```

Visually:

```
Learning Rate
  3e-4 │          ╭──────╮
       │         ╱        ╲
       │        ╱          ╲
       │       ╱            ╲
       │      ╱              ╲
       │     ╱                ╲
  0    │────╱                  ╲────
       └─────────────────────────────
       0   500               5000
           warmup            Steps
```

### Why Warmup?

At step 0, all parameters are random. The gradients are essentially noise. If you hit the model with a large learning rate on noise gradients, it can jump to a terrible region of the loss landscape and never recover.

Warmup starts with a tiny learning rate and ramps up. This gives the model a few hundred steps to "get its bearings" — the Adam optimizer's momentum estimates (m and v) accumulate some useful statistics, and the gradients become more meaningful. By step 500, the optimizer has good estimates and can handle the full learning rate.

### Why Cosine Decay?

As training progresses, the model gets closer to a good solution. The cosine curve provides a smooth, gradual reduction in learning rate. The cosine shape has a nice property: it decreases slowly at first (near the top of the curve) then more quickly, then slowly again (near the bottom). This spends more time at moderate learning rates, which is where the most useful learning happens.

### The Code

```python
def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float) -> float:
    # Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Cosine decay after warmup
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
```

### Line-by-Line

```python
if step < warmup_steps:
    return max_lr * (step + 1) / warmup_steps
```
- During warmup, LR increases linearly.
- Step 0: `3e-4 × 1/500 = 6e-7` (very small)
- Step 250: `3e-4 × 251/500 = 1.5e-4` (halfway)
- Step 499: `3e-4 × 500/500 = 3e-4` (full LR)

```python
progress = (step - warmup_steps) / (max_steps - warmup_steps)
```
- `progress` goes from 0.0 (right after warmup) to 1.0 (last step).
- This normalizes the step number into a 0-to-1 range for the cosine function.

```python
return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
```
- `cos(0) = 1.0` → at start of decay: `0.5 × (1 + 1) = 1.0 × max_lr`
- `cos(π/2) = 0.0` → at midpoint: `0.5 × (1 + 0) = 0.5 × max_lr`
- `cos(π) = -1.0` → at end: `0.5 × (1 + (-1)) = 0.0`

The LR smoothly goes from `max_lr` to 0.

### Applying the Schedule

PyTorch's optimizer doesn't apply the schedule automatically. We manually set the learning rate each step:

```python
lr = get_lr(step, warmup_steps=500, max_steps=5000, max_lr=3e-4)
for param_group in optimizer.param_groups:
    param_group["lr"] = lr
```

`optimizer.param_groups` is a list of dictionaries, each containing a `"lr"` key. For our setup there's just one group (all parameters), so we set the same LR for everything.

---

## Part 4: Gradient Clipping

### The Exploding Gradient Problem

Sometimes, a single training step produces extremely large gradients. This can happen when:
- The model sees an unusual input
- The loss function has a steep region
- Numerical instability in the computation

Large gradients → large parameter updates → the model jumps to a bad region → even larger gradients next step → training explodes:

```
Step 100: loss = 2.5
Step 101: loss = 2.4
Step 102: loss = 47.3      ← gradient explosion
Step 103: loss = NaN        ← model is destroyed
```

### The Fix: Clip the Gradients

**Gradient clipping** limits the total magnitude of all gradients. If the combined "length" (norm) of all gradient vectors exceeds a threshold, scale them all down proportionally:

```
If ‖gradients‖ > max_norm:
    gradients = gradients × (max_norm / ‖gradients‖)
```

This preserves the **direction** of the gradients (which parameters should increase vs decrease) but caps the **magnitude** (how much they change).

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### What This Means Practically

In normal training, the gradient norm is typically 0.1–1.0. Clipping at 1.0 means:
- Most steps: gradients are < 1.0, clipping does nothing
- Rare bad steps: gradients are > 1.0, they get scaled down to 1.0
- The model continues training stably instead of exploding

This is a safety net, not a performance enhancer. You could train without it and often be fine, but the one time you get an unlucky gradient spike, it saves you.

---

## Part 5: Evaluation — Measuring Progress

### Why Not Just Track Training Loss?

Training loss tells you how well the model fits the **training data**. But we care about how well it handles **new text it hasn't seen**. A model that memorizes the training data (low training loss) but fails on new text (high val loss) is useless — this is **overfitting**.

### Validation Loss

We periodically pause training and measure loss on the validation set (the 10% of text we held out):

```python
@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_steps):
    model.eval()
    losses = {}

    for name, loader in [("train", train_loader), ("val", val_loader)]:
        total = 0.0
        loader_iter = iter(loader)
        for _ in range(eval_steps):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y = next(loader_iter)
            _, loss = model(x, y)
            total += loss.item()
        losses[name] = total / eval_steps

    model.train()
    return losses
```

### Line-by-Line

```python
@torch.no_grad()
```
- **Decorator that disables gradient computation.** During evaluation, we don't need gradients (we're not updating parameters). Disabling them saves memory and computation — about 2× faster.

```python
model.eval()
```
- Switches the model to **evaluation mode**. This changes the behavior of dropout (disabled — all neurons active) and layer norm (uses stored statistics). During training, dropout randomly zeros values. During evaluation, we want deterministic, complete predictions.

```python
_, loss = model(x, y)
total += loss.item()
```
- `loss.item()` extracts the loss as a plain Python float. Without `.item()`, the loss tensor stays in the computation graph, consuming memory.

```python
losses[name] = total / eval_steps
```
- Average over `eval_steps` (20) batches for a stable estimate. A single batch's loss is noisy; averaging gives a better picture.

```python
model.train()
```
- Switch back to training mode (re-enables dropout).

### Reading the Loss Gap

```
Step 1000: train=2.10, val=2.25, gap=0.15   ← healthy
Step 3000: train=1.50, val=1.65, gap=0.15   ← still healthy
Step 5000: train=1.43, val=1.62, gap=0.19   ← slight overfitting, acceptable
```

vs.

```
Step 1000: train=3.20, val=3.45, gap=0.25   ← healthy
Step 3000: train=2.10, val=3.50, gap=1.40   ← severe overfitting!
Step 5000: train=2.05, val=3.51, gap=1.46   ← model is memorizing
```

**Healthy training**: both losses decrease together, small gap.
**Overfitting**: train loss keeps dropping but val loss stalls or increases.

When you see overfitting, the fixes include: more data, more dropout, weight decay, or a smaller model. We explored this in our BPE model experiments.

---

## Part 6: Generating Samples During Training

One of the most fun parts — watching the model's output improve as it trains:

```python
def generate_sample(model, tokenizer, seed_text="\n", length=200, temperature=0.8):
    model.eval()
    tokens = tokenizer.encode(seed_text)
    ctx = torch.tensor([tokens], dtype=torch.long)
    output = model.generate(ctx, max_new_tokens=length, temperature=temperature)
    model.train()
    return tokenizer.decode(output[0].tolist())
```

We generate a sample every 500 steps. Here's a rough timeline of what you'll see:

### Step 0 (Random)
```
xq$P-ZK'FjYBw:iMl&VfRN.EGQmU;T!cpzs
```
Complete noise. Every character is equally likely.

### Step 500 (Learning Character Frequencies)
```
the the ath and the sore to the har
the dore ond fir t the
```
The model learned that spaces follow words, that "the" and "and" are common, and that lines break at certain points. But it can't spell or form sentences.

### Step 2000 (Learning Words and Structure)
```
KING RICHARD:
The world is a the more of the such
That shall be the prove of the death,
And the shall the common of the part
```
Now it generates real words, character names in caps with colons, dialogue structure. It's clearly "Shakespeare-like" even if the sentences don't make sense.

### Step 5000 (Final)
```
QUEEN ELIZABETH:
Please you to hear you shall believe your brother's love?
And now the day was the majesty of your daughter.
```
Near-grammatical sentences. Proper dialogue format. Real character names. Occasionally produces coherent phrases.

This progression — from noise to structure to plausible text — is deeply satisfying to watch, and each stage corresponds to specific things the model is learning.

---

## Part 7: The Complete Training Script

```python
def train():
    cfg = CONFIG

    # ---- Data ----
    with open("data/input.txt", "r") as f:
        text = f.read()
    tok, train_ds, val_ds = create_datasets(text, cfg["block_size"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=True)

    # ---- Model ----
    model = GPT(
        vocab_size=tok.vocab_size,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        block_size=cfg["block_size"],
        dropout=cfg["dropout"],
    )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    # ---- Training loop ----
    train_iter = iter(train_loader)
    model.train()

    for step in range(cfg["max_steps"]):
        # Get next batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        # Set learning rate for this step
        lr = get_lr(step, cfg["warmup_steps"], cfg["max_steps"], cfg["lr"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward pass
        logits, loss = model(x, y)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])

        # Update parameters
        optimizer.step()
```

### Line-by-Line: The Inner Loop

```python
try:
    x, y = next(train_iter)
except StopIteration:
    train_iter = iter(train_loader)
    x, y = next(train_iter)
```
- Get the next batch from the DataLoader.
- When we've exhausted all batches (one **epoch**), restart the iterator.
- An epoch = one full pass through the training data. With ~1M samples and batch size 64, one epoch is ~15,600 steps. With 5,000 max steps, we don't even complete one epoch.

```python
lr = get_lr(step, cfg["warmup_steps"], cfg["max_steps"], cfg["lr"])
for param_group in optimizer.param_groups:
    param_group["lr"] = lr
```
- Compute the learning rate for this step (warmup + cosine decay).
- Manually set it in the optimizer.

```python
logits, loss = model(x, y)
```
- **Forward pass.** The input `x` flows through the entire model (embeddings → 4 Transformer blocks → output projection) and produces logits. The loss is computed against targets `y`.

```python
optimizer.zero_grad(set_to_none=True)
```
- **Clear previous gradients.** PyTorch accumulates gradients by default (they add up across calls to `.backward()`). We need to zero them before computing new ones.
- `set_to_none=True` is slightly faster than setting to zero — it frees the gradient memory instead of writing zeros.

```python
loss.backward()
```
- **Backward pass.** PyTorch computes the gradient of `loss` with respect to every parameter. After this call, every parameter's `.grad` attribute contains its gradient.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
```
- **Gradient clipping.** If the total gradient norm exceeds 1.0, scale all gradients down. Safety net against exploding gradients.

```python
optimizer.step()
```
- **Parameter update.** Adam uses the gradients (plus its stored momentum and variance) to update every parameter. After this call, the model is slightly better at predicting the next character.

### The Order Matters

```
zero_grad → forward → backward → clip → step
```

This order is important:
1. `zero_grad` first — clear old gradients
2. `forward` — compute predictions and loss
3. `backward` — compute gradients from the loss
4. `clip` — cap gradient magnitude (must happen after backward, before step)
5. `step` — use the gradients to update parameters

If you swap backward and step, you update with old (zero) gradients. If you forget zero_grad, gradients accumulate across steps.

---

## Part 8: Saving Checkpoints

```python
if step > 0 and step % checkpoint_interval == 0:
    path = f"checkpoints/model_step{step}.pt"
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": cfg,
        "vocab_size": tok.vocab_size,
    }, path)
```

### What Gets Saved

- `model.state_dict()` — all 841,281 parameters. This is the model itself.
- `optimizer.state_dict()` — Adam's internal state (momentum and variance for each parameter). Needed to resume training without losing the optimizer's accumulated knowledge.
- `config` — hyperparameters. Needed to recreate the model architecture when loading.
- `vocab_size` — needed to rebuild the tokenizer.

### Why Save Regularly

Training on CPU takes hours. If your laptop crashes at step 4000, you don't want to start over. Checkpoints let you resume from the last save.

Also, the best model might not be the final one. If the model overfits, the model at step 3000 might perform better on validation than the model at step 5000. Having checkpoints lets you go back.

---

## Part 9: Hyperparameters — All the Choices We Made

```python
CONFIG = {
    # Model
    "d_model": 128,        # Embedding dimension
    "n_heads": 4,          # Number of attention heads
    "n_layers": 4,         # Number of Transformer blocks
    "block_size": 256,     # Context window (characters)
    "dropout": 0.1,        # Dropout rate

    # Training
    "batch_size": 64,      # Sequences per training step
    "lr": 3e-4,            # Peak learning rate
    "max_steps": 5000,     # Total training steps
    "warmup_steps": 500,   # LR warmup duration
    "grad_clip": 1.0,      # Gradient clipping threshold

    # Logging
    "eval_interval": 250,
    "eval_steps": 20,
    "sample_interval": 500,
    "checkpoint_interval": 1000,
}
```

### Why These Specific Values?

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `d_model=128` | Small. GPT-2 uses 768. We're on CPU with limited time. |
| `n_heads=4` | d_model must be divisible by n_heads. 128/4=32 per head is reasonable. |
| `n_layers=4` | Enough depth to compose features. Our ablation showed 1 layer isn't enough. |
| `block_size=256` | 256 characters of context. Enough for a full speech. |
| `dropout=0.1` | Light regularization. Our data is small, don't want too much. |
| `batch_size=64` | Fits in 16GB RAM. Larger = more stable gradients but slower per step. |
| `lr=3e-4` | Standard Transformer default. |
| `max_steps=5000` | Enough to see convergence on CPU in ~3 hours. |
| `warmup_steps=500` | 10% of training. Standard ratio. |
| `grad_clip=1.0` | Standard value. Rarely triggers but prevents explosions. |

These are not sacred values. Part of becoming an ML engineer is developing intuition for tuning them.

---

## Part 10: Reading the Training Output

Here's what actual training output looks like and how to read it:

```
Step     0 | train loss: 4.1956 | val loss: 4.1900 | lr: 6.00e-07 | time: 0.0s
```
- **Step 0**: First step. Loss ≈ 4.17 = ln(65). Model is randomly guessing. Correct.
- **lr: 6e-7**: Warmup just started. LR is near zero.

```
Step   500 | train loss: 2.4012 | val loss: 2.4156 | lr: 3.00e-04 | time: 900.0s
```
- **Loss dropped to ~2.4**: Model learned character frequencies and common pairs.
- **lr: 3e-4**: Warmup complete. Full learning rate.
- **Gap (0.014)**: Very small. No overfitting yet.

```
Step  2500 | train loss: 1.6234 | val loss: 1.7012 | lr: 1.76e-04 | time: 5400.0s
```
- **Loss ~1.6-1.7**: Model learned words and dialogue structure.
- **lr: 1.76e-4**: Cosine decay has reduced the LR to about half.
- **Gap (0.08)**: Still small. Healthy training.

```
Step  5000 | train loss: 1.4302 | val loss: 1.6210 | lr: 3.66e-11 | time: 10124.0s
```
- **Final loss**: 1.43 train, 1.62 val.
- **lr: near zero**: Cosine decay finished.
- **Gap (0.19)**: Some overfitting, but acceptable.
- **time: ~2.8 hours**: Complete training time on CPU.

### What the Loss Numbers Mean in Practice

```
Loss 4.17 → perplexity 65    → "I'm choosing between 65 characters randomly"
Loss 2.50 → perplexity 12.2  → "I've narrowed it down to ~12 likely characters"
Loss 1.62 → perplexity 5.1   → "I've narrowed it down to ~5 likely characters"
Loss 1.43 → perplexity 4.2   → "I've narrowed it down to ~4 likely characters"
```

**Perplexity** = e^loss. It represents "how many options the model is effectively choosing between." A perplexity of 4.2 means the model is, on average, as uncertain as if it were choosing between about 4 equally likely characters at each position. This is remarkably good for a tiny character-level model.

---

## Summary

The complete training pipeline:

```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                         │
│                                                         │
│  for step in range(5000):                               │
│                                                         │
│    1. Get batch        (x, y) from DataLoader           │
│    2. Set learning rate  warmup → cosine decay          │
│    3. Forward pass      logits, loss = model(x, y)      │
│    4. Zero gradients    optimizer.zero_grad()            │
│    5. Backward pass     loss.backward()                 │
│    6. Clip gradients    clip_grad_norm_(1.0)            │
│    7. Update weights    optimizer.step()                │
│                                                         │
│    Every 250 steps:  evaluate on train + val set        │
│    Every 500 steps:  generate a text sample             │
│    Every 1000 steps: save a checkpoint                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

| Component | What It Does | Why |
|-----------|-------------|-----|
| AdamW | Adaptive parameter updates | Per-parameter learning rates + momentum |
| LR Schedule | Warmup then decay | Stable start, fine-grained convergence |
| Gradient Clipping | Cap gradient magnitude | Prevents training explosions |
| Validation Loss | Measure generalization | Detect overfitting |
| Checkpoints | Save model periodically | Resume training, compare stages |
| Sample Generation | Generate text during training | Visual feedback on progress |

After 5000 steps: loss drops from 4.17 to 1.43, the model generates recognizable Shakespeare, and you have checkpoints saved for experimentation.

## What's Next

In [Chapter 6](06_generation.md), we explore **text generation** — how the model uses its learned probabilities to produce text one character at a time, and how temperature and top-k sampling control the output's creativity and coherence.