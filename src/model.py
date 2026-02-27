"""
GPT-style Transformer, built from scratch.

Architecture (decoder-only, like GPT-2):
  Token Embedding + Positional Embedding
  → Dropout
  → N × TransformerBlock
      → LayerNorm → MultiHeadAttention → Residual
      → LayerNorm → FeedForward → Residual
  → Final LayerNorm
  → Linear → logits

We use PRE-NORM (LayerNorm before attention/FFN), not post-norm.
This is what GPT-2 and most modern models use — trains more stably.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Phase 2B: Single-Head Self-Attention
# ---------------------------------------------------------------------------
# We build single-head first so you can understand the core mechanism,
# then multi-head wraps multiple of these in parallel.

class SingleHeadAttention(nn.Module):
    """
    One head of self-attention.

    The core idea:
    - Each token produces a Query ("what am I looking for?"),
      a Key ("what do I contain?"), and a Value ("what do I give?")
    - Attention score = how well Query matches each Key
    - Output = weighted sum of Values, weighted by attention scores

    The causal mask ensures token i can only attend to tokens 0..i
    (can't look into the future during generation).
    """

    def __init__(self, d_model: int, head_dim: int, block_size: int, dropout: float):
        super().__init__()
        self.q = nn.Linear(d_model, head_dim, bias=False)
        self.k = nn.Linear(d_model, head_dim, bias=False)
        self.v = nn.Linear(d_model, head_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Causal mask: lower-triangular matrix
        # register_buffer = part of the model state but not a parameter (no gradients)
        # Shape: (block_size, block_size), True on lower triangle
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).bool()
        )

    def forward(self, x):
        B, T, C = x.shape  # batch, seq_len, d_model

        q = self.q(x)  # (B, T, head_dim)
        k = self.k(x)  # (B, T, head_dim)
        v = self.v(x)  # (B, T, head_dim)

        # Attention scores: (B, T, head_dim) @ (B, head_dim, T) → (B, T, T)
        # Each element [i,j] = "how much should position i attend to position j?"
        scale = math.sqrt(k.shape[-1])
        att = (q @ k.transpose(-2, -1)) / scale

        # Apply causal mask: set future positions to -inf
        # After softmax, -inf becomes 0 — so future tokens contribute nothing
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))

        att = F.softmax(att, dim=-1)  # Normalize to probabilities
        att = self.dropout(att)

        # Weighted sum of values
        out = att @ v  # (B, T, T) @ (B, T, head_dim) → (B, T, head_dim)
        return out


# ---------------------------------------------------------------------------
# Phase 2C: Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads running in parallel, then concatenated.

    Why multiple heads? Each head can learn a DIFFERENT attention pattern:
    - Head 1 might learn "attend to the previous character"
    - Head 2 might learn "attend to the start of the current word"
    - Head 3 might learn "attend to matching brackets"
    - Head 4 might learn "attend to the same character elsewhere"

    The concatenation + output projection lets the model combine
    these different perspectives.
    """

    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        head_dim = d_model // n_heads

        self.heads = nn.ModuleList([
            SingleHeadAttention(d_model, head_dim, block_size, dropout)
            for _ in range(n_heads)
        ])
        # Output projection: combines the concatenated head outputs
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run all heads in parallel, concatenate along the last dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


# ---------------------------------------------------------------------------
# Phase 2D: Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Applied independently to each position (each token gets the same
    transformation, but different input). This is where the model does
    its "thinking" — attention gathers information, FFN processes it.

    Architecture: Linear → GELU → Linear → Dropout
    The inner dimension is 4× the model dimension (standard ratio).

    Why GELU over ReLU? Smoother activation, slightly better training.
    GPT-2 and most modern transformers use GELU.
    """

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


# ---------------------------------------------------------------------------
# Phase 2E: Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    One Transformer block = attention + feed-forward, with
    layer norm and residual connections.

    The residual connection (x + sublayer(x)) is critical:
    - Allows gradients to flow directly through the network
    - Without it, deep networks are nearly impossible to train
    - Think of it as "start with what you had, then add new info"

    Pre-norm means we normalize BEFORE each sublayer:
      x = x + Attention(LayerNorm(x))
      x = x + FFN(LayerNorm(x))

    Post-norm (original paper) does it after:
      x = LayerNorm(x + Attention(x))

    Pre-norm trains more stably — that's why GPT-2 switched to it.
    """

    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # Residual + pre-norm attention
        x = x + self.ffn(self.ln2(x))   # Residual + pre-norm FFN
        return x


# ---------------------------------------------------------------------------
# Phase 2F: Full GPT Model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """
    The full model. Stacks everything together.

    Forward pass:
    1. Look up token embeddings (what is each character?)
    2. Add positional embeddings (where is each character?)
    3. Pass through N transformer blocks (process the information)
    4. Final layer norm
    5. Project to vocabulary size (predict next character probabilities)

    We use learnable positional embeddings (like GPT-2) rather than
    sinusoidal (like the original Transformer paper). Both work fine
    at this scale, but learnable is simpler to implement.
    """

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

        # Weight initialization (important for stable training)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights using the scheme from GPT-2:
        - Linear layers: normal distribution with std=0.02
        - Embeddings: normal distribution with std=0.02
        - LayerNorm: bias=0, weight=1 (these are the defaults, but explicit is good)
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Args:
            idx: (batch, seq_len) tensor of token indices
            targets: (batch, seq_len) tensor of target indices, or None

        Returns:
            logits: (batch, seq_len, vocab_size) — raw predictions
            loss: scalar loss if targets provided, else None
        """
        B, T = idx.shape

        # Create position indices: [0, 1, 2, ..., T-1]
        pos = torch.arange(T, device=idx.device)  # (T,)

        # Embeddings
        tok = self.tok_emb(idx)   # (B, T, d_model)
        pos = self.pos_emb(pos)   # (T, d_model) — broadcasts across batch
        x = self.drop(tok + pos)  # (B, T, d_model)

        # Transformer blocks
        x = self.blocks(x)        # (B, T, d_model)

        # Output
        x = self.ln_f(x)          # (B, T, d_model)
        logits = self.head(x)     # (B, T, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # cross_entropy expects (N, C) so we flatten
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B*T, vocab_size)
                targets.view(-1),                   # (B*T,)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive generation.

        Given a context (idx), predict one character at a time,
        append it, and repeat.

        Args:
            idx: (B, T) starting context
            max_new_tokens: how many characters to generate
            temperature: >1 = more random, <1 = more focused
            top_k: if set, only sample from the top k most likely characters
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size (model can't see more than this)
            ctx = idx[:, -self.block_size:]

            # Forward pass
            logits, _ = self(ctx)

            # Only care about the last position's prediction
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply temperature
            logits = logits / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to the sequence
            idx = torch.cat([idx, next_token], dim=1)

        return idx


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test with dummy data
    vocab_size = 65
    block_size = 256
    batch_size = 4

    model = GPT(vocab_size=vocab_size, block_size=block_size)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Test forward pass
    x = torch.randint(0, vocab_size, (batch_size, block_size))
    y = torch.randint(0, vocab_size, (batch_size, block_size))
    logits, loss = model(x, y)
    print(f"Logits shape: {logits.shape}")
    print(f"Expected:     ({batch_size}, {block_size}, {vocab_size})")
    print(f"Loss: {loss.item():.4f}")
    print(f"Expected loss (random): ~{math.log(vocab_size):.4f}")

    # Test generation
    context = torch.zeros((1, 1), dtype=torch.long)  # Start with token 0
    generated = model.generate(context, max_new_tokens=50)
    print(f"\nGenerated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")

    print("\nAll shape checks passed!")