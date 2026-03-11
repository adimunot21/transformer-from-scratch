# Chapter 8: Decision Transformer — Reinforcement Learning via Sequence Modeling

This is the final chapter, and it's the most surprising one. We're going to take the same Transformer blocks we built for Shakespeare — the same multi-head attention, the same feed-forward network, the same residual connections — and use them to control a robot.

Well, a simulated robot. But the principle is identical to what's used in real robotics research.

## Part 1: What Is Reinforcement Learning?

### The Setup

In reinforcement learning (RL), an **agent** interacts with an **environment**:

```
┌─────────┐    action     ┌─────────────┐
│         │──────────────▶│             │
│  Agent  │               │ Environment │
│         │◀──────────────│             │
└─────────┘  state, reward └─────────────┘
```

Each time step:
1. The agent observes the **state** (what's happening in the environment)
2. The agent chooses an **action** (what to do)
3. The environment returns a **reward** (how good was that action?) and a new **state**

The goal: choose actions that maximize total reward over time.

### CartPole — Our Environment

CartPole is a classic RL problem. A pole is balanced on top of a cart that can move left or right. The goal is to keep the pole upright:

```
        ╱
       ╱    ← pole (must stay upright)
      ╱
   ┌──┴──┐
   │ cart │
   └──┬──┘
───────────────── track ─────────────────
     ◄──── can move left or right ────►
```

**State**: 4 numbers
- Cart position (where on the track)
- Cart velocity (how fast it's moving)
- Pole angle (how far from vertical)
- Pole angular velocity (how fast it's falling)

**Actions**: 2 choices
- 0 = push cart left
- 1 = push cart right

**Reward**: +1 for every time step the pole stays upright

**Episode ends when**: the pole falls past 15° or the cart goes off-screen. Maximum episode length is 500 steps (score of 500 = perfect).

**Random policy** (choosing left/right randomly) averages about 20-25 steps. The pole falls almost immediately. A good policy can reach 200-500 steps.

### Traditional RL vs. What We're Doing

Traditional RL algorithms (like PPO, DQN, SAC) learn by **trial and error** — the agent acts in the environment, gets rewards, and updates its policy to get better rewards. This requires millions of interactions with the environment.

The Decision Transformer takes a completely different approach: **offline learning from a dataset of pre-collected episodes**. No interaction with the environment during training. The model reads transcripts of past games and learns to predict what actions lead to high rewards.

This is exactly like how our Shakespeare model works — it never "writes Shakespeare" during training. It reads existing Shakespeare and learns to predict the next token. The Decision Transformer reads existing CartPole episodes and learns to predict the next action.

---

## Part 2: The Key Insight — RL as Sequence Prediction

Here's the central idea of the Decision Transformer paper (Chen et al., 2021):

**A Shakespeare episode:**
```
Sequence: [char₁, char₂, char₃, char₄, ...]
Task:     predict char₂ given char₁
          predict char₃ given char₁, char₂
          predict char₄ given char₁, char₂, char₃
```

**A CartPole episode:**
```
Sequence: [R̂₁, s₁, a₁, R̂₂, s₂, a₂, R̂₃, s₃, a₃, ...]
Task:     predict a₁ given R̂₁, s₁
          predict a₂ given R̂₁, s₁, a₁, R̂₂, s₂
          predict a₃ given R̂₁, s₁, a₁, R̂₂, s₂, a₂, R̂₃, s₃
```

Where:
- `sₜ` = state at time t (4 numbers for CartPole)
- `aₜ` = action at time t (0 or 1)
- `R̂ₜ` = **return-to-go** at time t (explained below)

Both are sequence prediction problems. The Transformer doesn't know or care whether it's processing characters or (return, state, action) triples. It just sees a sequence of vectors and learns patterns.

### What Is Return-to-Go?

Return-to-go (RTG) at time t is the **total future reward from time t onward**:

```
R̂ₜ = rₜ + rₜ₊₁ + rₜ₊₂ + ... + rₜ (final)
```

For a CartPole episode that lasts 50 steps (each step gives reward 1):

```
Time step:      0     1     2    ...   48    49
Reward:         1     1     1    ...    1     1
Return-to-go:  50    49    48   ...    2     1
```

At step 0, the return-to-go is 50 (50 more rewards to come). At step 48, the return-to-go is 2 (only 2 rewards left).

**Why include return-to-go?** This is the magic trick. During generation (test time), we can set the return-to-go to a HIGH value — telling the model "I want 500 total reward." The model then outputs actions that it learned are associated with high returns. Set RTG low, and it outputs actions associated with low returns.

We're **conditioning** the model on the desired outcome. This is exactly like prompting our Shakespeare model with "ROMEO:" vs "JULIET:" — the prompt controls what comes out.

---

## Part 3: The Training Data

### Collecting Episodes

We need a dataset of CartPole episodes — sequences of (state, action, reward). We collect these using two policies:

**Random policy** (500 episodes): Chooses left or right randomly. These episodes are short (average ~20 steps) and low-return. They show the model what "bad" play looks like.

**Heuristic policy** (500 episodes): A simple rule — push in the direction the pole is leaning. If the pole leans right, push right. This achieves average ~100-200 steps. It shows the model what "decent" play looks like.

```python
# Random policy
action = env.action_space.sample()    # random 0 or 1

# Heuristic policy
action = 1 if obs[2] > 0 else 0      # push toward lean direction
if random.random() < 0.1:
    action = 1 - action               # 10% noise for diversity
```

**Why both?** The model needs to see the contrast. If it only saw good episodes, it couldn't distinguish "actions that work" from "actions that don't" — all actions in the dataset would be good. By seeing both good (RTG=200) and bad (RTG=20) episodes, it learns: "when the RTG is high, take THESE actions; when RTG is low, THOSE actions were taken."

### Episode Structure

One episode becomes:

```
states:  [s₀, s₁, s₂, ..., sₜ]     each sᵢ is 4 numbers
actions: [a₀, a₁, a₂, ..., aₜ]     each aᵢ is 0 or 1
rewards: [1,  1,  1,  ...,  1]      CartPole gives +1 per step

Computed return-to-go:
  R̂₀ = T, R̂₁ = T-1, ..., R̂ₜ₋₁ = 1
```

### Sliding Windows

Just like our Shakespeare dataset creates overlapping windows of text, we create overlapping windows of (R̂, s, a) triples:

```
Episode: [R̂₀,s₀,a₀, R̂₁,s₁,a₁, R̂₂,s₂,a₂, ..., R̂₄₉,s₄₉,a₄₉]

Window starting at t=0:  [R̂₀,s₀,a₀, R̂₁,s₁,a₁, ..., R̂₁₉,s₁₉,a₁₉]  (K=20 steps)
Window starting at t=1:  [R̂₁,s₁,a₁, R̂₂,s₂,a₂, ..., R̂₂₀,s₂₀,a₂₀]
Window starting at t=2:  [R̂₂,s₂,a₂, R̂₃,s₃,a₃, ..., R̂₂₁,s₂₁,a₂₁]
...
```

Each window is one training sample. With 1000 episodes of varying length, we get tens of thousands of training samples.

**Context length K=20** means the model sees the last 20 time steps when predicting the next action. This is equivalent to `block_size=256` in our language model — it's how much history the model can use.

### Padding and Masking

Episodes have different lengths. Short episodes (random policy, ~20 steps) might be shorter than K=20. We pad shorter sequences with zeros and use a **mask** to tell the loss function "don't compute loss on padded positions":

```
Real data:  [R̂₀,s₀,a₀, R̂₁,s₁,a₁, R̂₂,s₂,a₂]     (3 real steps)
Padded:     [R̂₀,s₀,a₀, R̂₁,s₁,a₁, R̂₂,s₂,a₂, 0,0,0, 0,0,0, ...]   (padded to K=20)
Mask:       [1,         1,         1,         0,     0,     ...]

Loss = Σ(loss × mask) / Σ(mask)     ← only real positions contribute
```

---

## Part 4: The Architecture

### What Changes from Our GPT

The core Transformer blocks are **identical** — we literally import `TransformerBlock` from `model.py`. What changes is the input and output:

```
Shakespeare GPT:
  Input:  token indices → Token Embedding + Position Embedding
  Output: Linear → vocab_size logits

Decision Transformer:
  Input:  (R̂, state, action) triples → Three separate embeddings + Timestep Embedding
  Output: Linear → action logits (from state positions only)
```

### The Interleaved Sequence

This is the key architectural design. At each time step t, we have three elements: R̂ₜ, sₜ, aₜ. We interleave them into a single sequence:

```
Time step:    0           1           2          ...
Sequence: [R̂₀, s₀, a₀, R̂₁, s₁, a₁, R̂₂, s₂, a₂, ...]
Position:   0   1   2    3   4   5    6   7   8

With K=20 time steps, the sequence length is 3 × 20 = 60 tokens.
```

The Transformer processes this as a single sequence, just like it processes "ROMEO:\nO, she doth" as a sequence of characters. Attention can look at any previous position — an action token can attend to previous states, returns, and actions.

### Three Embedding Heads

In our Shakespeare model, we have one embedding: token index → vector.

The Decision Transformer needs three separate embeddings because the three types have very different formats:

```python
# Returns: a single number → d_model vector
self.embed_return = nn.Sequential(
    nn.Linear(1, d_model),
    nn.Tanh(),
)

# States: 4 numbers (CartPole) → d_model vector
self.embed_state = nn.Sequential(
    nn.Linear(state_dim, d_model),
    nn.Tanh(),
)

# Actions: discrete index → d_model vector
self.embed_action = nn.Sequential(
    nn.Embedding(act_dim, d_model),
    nn.Tanh(),
)
```

**Why `nn.Linear` for returns and states?** Returns are continuous numbers (like 50.0, 200.0). States are continuous vectors (like [0.02, 0.15, -0.03, 0.08]). We can't use `nn.Embedding` for continuous values — that only works for discrete indices. `nn.Linear` projects continuous inputs into the embedding space.

**Why `nn.Embedding` for actions?** Actions are discrete (0 or 1 in CartPole). Just like our character tokens, they're indices into a lookup table.

**Why `nn.Tanh()`?** The Tanh activation squashes values to the range [-1, 1]. This normalizes the embeddings, preventing any one modality from dominating. Without it, returns (which can be large numbers like 500) would produce much larger embeddings than states (which are small numbers like 0.02).

### Timestep Embedding (Not Position Embedding)

In our Shakespeare model, position embedding encodes "where in the sequence." In the Decision Transformer, **timestep embedding** encodes "when in the episode":

```python
self.embed_timestep = nn.Embedding(max_timestep, d_model)
```

The same timestep embedding is added to R̂ₜ, sₜ, and aₜ — all three elements at time t get the same timestep vector. This tells the model "these three things happened at the same moment."

This is different from position in the sequence. Position 0, 1, 2 are R̂₀, s₀, a₀ — three different positions but the same timestep. The timestep embedding captures the temporal structure.

```
Sequence: [R̂₀, s₀, a₀, R̂₁, s₁, a₁, R̂₂, s₂, a₂]
Timestep:   0    0    0    1    1    1    2    2    2
Position:   0    1    2    3    4    5    6    7    8

Timestep embedding: same for R̂₀, s₀, a₀ (all timestep 0)
Position:           handled implicitly by the causal mask
```

### The Forward Pass

```python
def forward(self, returns_to_go, states, actions, timesteps):
    B, K = states.shape[0], states.shape[1]

    # Embed each modality
    r_emb = self.ln_r(self.embed_return(returns_to_go))    # (B, K, d)
    s_emb = self.ln_s(self.embed_state(states))            # (B, K, d)
    a_emb = self.ln_a(self.embed_action(actions))          # (B, K, d)

    # Add timestep embeddings
    t_emb = self.embed_timestep(timesteps)                 # (B, K, d)
    r_emb = r_emb + t_emb
    s_emb = s_emb + t_emb
    a_emb = a_emb + t_emb

    # Interleave into sequence: [R̂₁, s₁, a₁, R̂₂, s₂, a₂, ...]
    seq = torch.stack([r_emb, s_emb, a_emb], dim=2)       # (B, K, 3, d)
    seq = seq.reshape(B, 3 * K, self.d_model)              # (B, 3K, d)

    # Through the Transformer
    seq = self.drop(seq)
    seq = self.blocks(seq)
    seq = self.ln_f(seq)

    # Extract state positions and predict actions
    s_repr = seq[:, 1::3, :]                               # (B, K, d)
    action_logits = self.action_head(s_repr)               # (B, K, act_dim)
    return action_logits
```

### Line-by-Line Walkthrough

```python
r_emb = self.ln_r(self.embed_return(returns_to_go))    # (B, K, d)
s_emb = self.ln_s(self.embed_state(states))            # (B, K, d)
a_emb = self.ln_a(self.embed_action(actions))          # (B, K, d)
```
- Embed each modality separately. After this, all three types are in the same d_model-dimensional space.
- Layer norm (`ln_r`, `ln_s`, `ln_a`) normalizes each stream independently, ensuring they're on similar scales.
- Shapes: returns `(B, K, 1)` → `(B, K, d)`, states `(B, K, 4)` → `(B, K, d)`, actions `(B, K)` → `(B, K, d)`.

```python
t_emb = self.embed_timestep(timesteps)
r_emb = r_emb + t_emb
s_emb = s_emb + t_emb
a_emb = a_emb + t_emb
```
- Add the same timestep embedding to all three elements at each timestep. Identical to how our Shakespeare model adds position embeddings to token embeddings.

```python
seq = torch.stack([r_emb, s_emb, a_emb], dim=2)    # (B, K, 3, d)
seq = seq.reshape(B, 3 * K, self.d_model)           # (B, 3K, d)
```
This is the interleaving step. Let's trace it carefully:

```
Before stack (K=3 timesteps for illustration):
  r_emb: [r₀, r₁, r₂]       shape (B, 3, d)
  s_emb: [s₀, s₁, s₂]       shape (B, 3, d)
  a_emb: [a₀, a₁, a₂]       shape (B, 3, d)

After stack (dim=2):
  [[r₀, s₀, a₀],
   [r₁, s₁, a₁],             shape (B, 3, 3, d)
   [r₂, s₂, a₂]]

After reshape:
  [r₀, s₀, a₀, r₁, s₁, a₁, r₂, s₂, a₂]    shape (B, 9, d)
```

The reshape interleaves the three streams into one sequence. This is now a standard sequence that our Transformer blocks can process — exactly like a sequence of character embeddings.

```python
seq = self.blocks(seq)
```
- **The SAME Transformer blocks from Chapter 4.** Same multi-head attention, same feed-forward network, same residual connections. The blocks don't know they're processing RL data instead of text. They just see a sequence of vectors and apply attention.

```python
s_repr = seq[:, 1::3, :]    # (B, K, d)
```
- Extract only the **state positions** from the output sequence.
- `1::3` means "start at index 1, take every 3rd element": positions 1, 4, 7, 10, ...
- These are the positions where states were placed during interleaving.
- Remember the interleaved order: `[R̂₀, s₀, a₀, R̂₁, s₁, a₁, ...]` — states are at positions 1, 4, 7, ...

**Why predict from state positions?** The action at time t should be predicted from the context up to and including state t. In the interleaved sequence, the state token at position 1 has attended to position 0 (R̂₀) and position 1 (s₀ itself). This gives it the return-to-go and state — exactly the information needed to choose an action.

```python
action_logits = self.action_head(s_repr)    # (B, K, act_dim)
```
- Project each state representation to action logits.
- For CartPole: `act_dim=2`, so the output is 2 numbers per timestep — logits for "push left" vs "push right."
- This is exactly like our Shakespeare model's `self.head = nn.Linear(d_model, vocab_size)` — project from the internal representation to the output space.

---

## Part 5: Training

Training is similar to our Shakespeare model, with one key difference: the loss function.

### The Loss

```python
# Forward pass
action_logits = model(rtg, states, actions, timesteps)    # (B, K, 2)

# Flatten for cross-entropy
logits_flat = action_logits.reshape(-1, 2)     # (B*K, 2)
actions_flat = actions.reshape(-1)              # (B*K,)
mask_flat = mask.reshape(-1)                    # (B*K,)

# Masked cross-entropy
loss_all = F.cross_entropy(logits_flat, actions_flat, reduction="none")
loss = (loss_all * mask_flat).sum() / mask_flat.sum()
```

This is the same cross-entropy loss from Chapter 5, but with masking:

1. Compute loss at every position: "how wrong was the predicted action vs. the actual action?"
2. Multiply by the mask: positions with real data (mask=1) keep their loss, padded positions (mask=0) contribute zero
3. Average over only the real positions

**The training signal**: "Given this sequence of (return, state, action) history, predict the action that was actually taken." The model learns to associate high-return contexts with the actions that produced those high returns.

### What the Model Learns

After training, the model has internalized patterns like:

```
"When the return-to-go is high (200+) and the pole is leaning right...
 → push right (action 1)"

"When the return-to-go is high and the pole is nearly vertical...
 → the action that keeps it balanced (depends on velocity)"

"When the return-to-go is low (10-20) and the pole is leaning...
 → either action is fine, the episode is ending soon anyway"
```

The model doesn't "understand" physics. It learned statistical patterns: "in episodes where the total future reward was high, and the state looked like this, action 1 was taken." This is pure pattern matching — the same thing our Shakespeare model does with text.

---

## Part 6: Evaluation — The Magic of Return Conditioning

This is where the Decision Transformer shines. During evaluation, we control the model's behavior by setting the return-to-go:

```python
def evaluate(model, target_return, n_episodes=50):
    for each episode:
        # Start with desired return
        rtg[0] = target_return

        for each step:
            # Get action from model
            action = model.get_action(rtg, states, actions, timesteps)

            # Take action in environment
            next_state, reward = env.step(action)

            # Update return-to-go: subtract the reward we just got
            rtg[next_step] = rtg[current_step] - reward
```

### How Return-to-Go Decreases

```
Target: 200

Step 0:  RTG = 200   "I want 200 more reward"
         (get reward 1)
Step 1:  RTG = 199   "I want 199 more reward"
         (get reward 1)
Step 2:  RTG = 198   "I want 198 more reward"
         ...
Step 50: RTG = 150   "I want 150 more reward"
```

The RTG naturally decreases as the agent collects rewards. This is correct — at each step, the remaining desired reward should decrease by the reward just received.

### The Context Window

Just like our Shakespeare model uses a sliding window of 256 characters, the Decision Transformer uses a sliding window of K=20 timesteps:

```
Early in episode (step 5):
  Model sees: steps 0-5      (fits in window)

Late in episode (step 50):
  Model sees: steps 31-50    (sliding window of last 20 steps)
  Steps 0-30 are forgotten
```

For CartPole, 20 steps of history is usually enough — the physics only depends on the current state and recent trajectory. For more complex environments, a larger context window would help.

### Our Results

```
Target Return:  10  →  Achieved: 10.1 ± 0.3   (max: 11)
Target Return:  50  →  Achieved: 45.3 ± 9.0   (max: 66)
Target Return: 100  →  Achieved: 62.7 ± 23.9  (max: 110)
Target Return: 200  →  Achieved: 71.5 ± 39.1  (max: 198)
Target Return: 500  →  Achieved: 76.0 ± 70.2  (max: 491)
Random baseline:       22.2 ± 14.1
```

### Interpreting the Results

**Target 10 → Achieved 10.1**: This is the most remarkable result. The model learned to be precisely bad on purpose. A random policy averages 22 — it's actually hard to score exactly 10. The model has to take actions that cause the pole to fall at just the right time. This proves it genuinely understands the relationship between desired returns and actions.

**Target 500 → Achieved 76 mean, 491 max**: The mean is low because CartPole is unstable — one wrong move ends the episode. But the max of 491 (near-perfect) shows the model CAN achieve very high returns. The variance is high because the model sometimes makes a mistake early and can't recover.

**Higher target → higher achievement**: The clear upward trend confirms the model learned return-conditioned behavior. It's not just memorizing one strategy — it has a spectrum of behaviors from "fail quickly" to "balance perfectly."

**The gap between target and achieved**: The model was trained on heuristic episodes that averaged 100-200 reward. For targets above 200, the model is being asked to generalize beyond its training data — extrapolate to behavior better than anything it's seen. That it achieves 491 at all is impressive.

---

## Part 7: The Connection to Robotics

The Decision Transformer is directly relevant to real robotics. Here's why:

### Offline Learning

Robots are expensive. Physical robots break. Simulation time costs money. Traditional RL requires millions of environment interactions to learn. The Decision Transformer learns from a fixed dataset of past experiences — no online interaction needed.

In a factory setting: collect a dataset of human operators performing a task, train a Decision Transformer, deploy it. No trial-and-error on the physical robot.

### Goal Conditioning

By changing the return-to-go, you change the robot's behavior. This is a form of **goal conditioning** — telling the robot "how well" to perform rather than programming specific behavior:

```
RTG = 100: Robot completes the task carefully but slowly
RTG = 200: Robot completes the task quickly and efficiently
RTG = 500: Robot attempts the most aggressive, optimal strategy
```

This is more flexible than a single fixed policy.

### Sequence Modeling as a Unified Framework

The Decision Transformer shows that the same architecture can handle text AND control. Modern robotics research is increasingly using Transformer-based architectures for:

- **Trajectory prediction**: Given past robot states, predict future states
- **Multi-task learning**: One model that handles multiple robot tasks
- **Vision-Language-Action models**: Combine camera input, language instructions, and motor output in one Transformer

The attention mechanism's ability to process variable-length sequences and capture long-range dependencies makes it naturally suited for these tasks.

---

## Part 8: Comparing Shakespeare GPT and Decision Transformer

Let's put the two models side by side to see how similar they really are:

| Aspect | Shakespeare GPT | Decision Transformer |
|--------|----------------|---------------------|
| Input tokens | Characters (65 unique) | (R̂, state, action) triples |
| Embedding | 1 embedding table | 3 embedding projections |
| Position info | Position embedding (0, 1, 2, ...) | Timestep embedding (shared across R̂,s,a) |
| Transformer blocks | Multi-head attention + FFN | **Identical** — same code |
| Output | Predict next character (65-way) | Predict next action (2-way) |
| Context | Last 256 characters | Last 20 timesteps (60 tokens) |
| Training data | 1MB of Shakespeare text | 1000 CartPole episodes |
| Loss function | Cross-entropy on characters | Cross-entropy on actions |
| Generation | Autoregressive (one char at a time) | Autoregressive (one action at a time) |
| Conditioning | Prompt ("ROMEO:") | Return-to-go (target=200) |

The Transformer blocks row is the key: **same code**. We literally `from src.model import TransformerBlock`. The attention mechanism, feed-forward network, layer norm, and residual connections are unchanged. The only differences are in how we prepare the input and interpret the output.

This is the fundamental insight: **attention is a general-purpose sequence processor**. It doesn't "know" about language or control or music or proteins. It learns patterns in sequences. Different inputs and outputs adapt it to different domains.

---

## Part 9: The Complete Architecture Diagram

```
Inputs at each timestep t:
  R̂ₜ (return-to-go, 1 number)
  sₜ (state, 4 numbers for CartPole)
  aₜ (action, discrete: 0 or 1)

┌──────────────────────────────────────────────────────┐
│                                                      │
│  R̂ₜ ──→ Linear(1,64) ──→ Tanh ──→ LayerNorm ──┐    │
│                                                  │    │
│  sₜ ──→ Linear(4,64) ──→ Tanh ──→ LayerNorm ──┤    │
│                                                  │    │
│  aₜ ──→ Embedding(2,64) → Tanh ──→ LayerNorm ──┤    │
│                                                  │    │
│  timestep ──→ Embedding(500,64) ────────────────┤    │
│                                                  │    │
│              ┌──── Add timestep to each ────┘    │    │
│              │                                        │
│              ▼                                        │
│  Interleave: [R̂₀,s₀,a₀, R̂₁,s₁,a₁, ...]           │
│  Shape: (B, 3K, 64)                                  │
│              │                                        │
│              ▼                                        │
│  ┌────────────────────────────┐                      │
│  │   TransformerBlock × 3     │  ← SAME as Chapters  │
│  │   (attention + FFN +       │     3 and 4!          │
│  │    residual + layernorm)   │                      │
│  └────────────┬───────────────┘                      │
│               │                                       │
│               ▼                                       │
│  Extract state positions (indices 1, 4, 7, ...)      │
│  Shape: (B, K, 64)                                   │
│               │                                       │
│               ▼                                       │
│  Linear(64, 2) → action logits                       │
│  Shape: (B, K, 2)                                    │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## Summary

The Decision Transformer demonstrates three powerful ideas:

### 1. RL as Sequence Modeling
Traditional RL uses complex algorithms (value functions, policy gradients, temporal difference learning). The Decision Transformer replaces all of that with sequence prediction — the same task our Shakespeare model does. The algorithmic simplicity is remarkable.

### 2. Return Conditioning
By including the desired return-to-go in the input, the model learns a **spectrum of behaviors**. A single trained model can act conservatively (low target) or aggressively (high target), controlled at test time. This is more flexible than traditional RL, which learns a single fixed policy.

### 3. Attention is General-Purpose
The same 3-layer, 4-head Transformer that generates Shakespeare also controls a CartPole agent. The architecture doesn't change — only the input representation and output interpretation change. This generality is why Transformers have taken over virtually every field in AI: natural language processing, computer vision, speech recognition, protein folding, weather prediction, robotics, and more.

---

## Course Complete

Over these 8 chapters, you've built:

1. **A character-level tokenizer** — converting text to numbers and back
2. **A PyTorch Dataset** — structuring data for next-token prediction
3. **Token and positional embeddings** — giving meaning to indices
4. **Self-attention from scratch** — Q, K, V projections, scaling, causal masking, multi-head
5. **Feed-forward network** — processing gathered information
6. **Transformer blocks** — combining attention + FFN with residual connections and layer norm
7. **A complete GPT model** — stacking blocks into a language model
8. **A training loop** — AdamW, learning rate scheduling, gradient clipping
9. **Autoregressive generation** — temperature, top-k sampling
10. **Attention visualization** — seeing what each head learned
11. **Ablation studies** — proving depth matters more than width
12. **A BPE tokenizer** — the same algorithm GPT-2/3/4 uses
13. **A Decision Transformer** — RL via the same architecture

Every component was written from scratch using only PyTorch's basic building blocks. You understand not just what a Transformer does, but **why** each piece exists, **how** it works mathematically, and **what happens** when you change it.

This is the foundation for everything in modern AI. Whether you go into robotics, NLP, computer vision, or any other field — the Transformer architecture and the training techniques you've learned here are the common language.