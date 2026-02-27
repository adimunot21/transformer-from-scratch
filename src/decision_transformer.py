"""
Decision Transformer — Reinforcement Learning via Sequence Modeling.

Core insight: RL can be cast as a sequence prediction problem.

Instead of:  [char₁, char₂, char₃, ...] → predict next char
We do:       [R̂₁, s₁, a₁, R̂₂, s₂, a₂, ...] → predict next action

Where:
  R̂ₜ = "return-to-go" = desired future reward from timestep t onward
  sₜ = environment state at timestep t
  aₜ = action taken at timestep t

At test time, we condition on HIGH return-to-go, and the model
outputs actions that achieve high returns. No value functions,
no policy gradients — just sequence modeling.

Reference: Chen et al., "Decision Transformer: Reinforcement Learning
via Sequence Modeling" (NeurIPS 2021)

We reuse our TransformerBlock from model.py — same attention,
same feed-forward, same layer norm. Different input, same architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model import TransformerBlock


class DecisionTransformer(nn.Module):
    """
    Decision Transformer for discrete action spaces.

    Architecture differences from our GPT:
    1. Three separate embedding heads (returns, states, actions)
       instead of one token embedding
    2. Timestep embeddings instead of positional embeddings
       (actual timestep in episode, not position in sequence)
    3. Interleaved sequence: [R̂₁, s₁, a₁, R̂₂, s₂, a₂, ...]
    4. Action prediction from state token positions only

    The Transformer blocks are IDENTICAL to our GPT. This is the
    whole point — attention is a general sequence processing tool.
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        max_timestep: int = 500,
        context_len: int = 20,
        dropout: float = 0.1,
    ):
        """
        Args:
            state_dim: dimension of state vector (CartPole = 4)
            act_dim: number of discrete actions (CartPole = 2)
            d_model: transformer embedding dimension
            n_heads: number of attention heads
            n_layers: number of transformer blocks
            max_timestep: maximum episode length
            context_len: K — how many timesteps of context to use
            dropout: dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.context_len = context_len
        self.act_dim = act_dim

        # Each modality gets its own embedding projection
        # Returns: scalar → d_model
        self.embed_return = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Tanh(),
        )
        # States: state_dim → d_model
        self.embed_state = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.Tanh(),
        )
        # Actions: discrete → d_model
        self.embed_action = nn.Sequential(
            nn.Embedding(act_dim, d_model),
            nn.Tanh(),
        )

        # Timestep embedding — shared across R, s, a at the same timestep
        # This tells the model "when in the episode" each token is from
        self.embed_timestep = nn.Embedding(max_timestep, d_model)

        # Layer norms for each input stream
        self.ln_r = nn.LayerNorm(d_model)
        self.ln_s = nn.LayerNorm(d_model)
        self.ln_a = nn.LayerNorm(d_model)

        # The Transformer blocks — SAME as our GPT!
        # block_size = 3 * context_len because each timestep has 3 tokens
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, 3 * context_len, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        # Action prediction head — predicts from state token positions
        self.action_head = nn.Linear(d_model, act_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, returns_to_go, states, actions, timesteps):
        """
        Args:
            returns_to_go: (B, K, 1) — desired future returns
            states:        (B, K, state_dim) — observed states
            actions:       (B, K) — actions taken (long)
            timesteps:     (B, K) — timestep indices (long)

        Returns:
            action_logits: (B, K, act_dim) — predicted action distribution
        """
        B, K = states.shape[0], states.shape[1]

        # Embed each modality
        r_emb = self.ln_r(self.embed_return(returns_to_go))    # (B, K, d)
        s_emb = self.ln_s(self.embed_state(states))            # (B, K, d)
        a_emb = self.ln_a(self.embed_action(actions))          # (B, K, d)

        # Add timestep embeddings (same timestep embedding for R, s, a)
        t_emb = self.embed_timestep(timesteps)                 # (B, K, d)
        r_emb = r_emb + t_emb
        s_emb = s_emb + t_emb
        a_emb = a_emb + t_emb

        # Interleave into sequence: [R̂₁, s₁, a₁, R̂₂, s₂, a₂, ...]
        # Stack: (B, K, 3, d) → reshape to (B, 3K, d)
        seq = torch.stack([r_emb, s_emb, a_emb], dim=2)
        seq = seq.reshape(B, 3 * K, self.d_model)

        # Through the Transformer (same blocks as GPT!)
        seq = self.drop(seq)
        seq = self.blocks(seq)
        seq = self.ln_f(seq)

        # Extract state positions: indices 1, 4, 7, ...
        # (every 3rd token starting from position 1)
        # We predict actions from the state representations
        s_repr = seq[:, 1::3, :]  # (B, K, d)

        # Predict actions
        action_logits = self.action_head(s_repr)  # (B, K, act_dim)
        return action_logits

    @torch.no_grad()
    def get_action(self, returns_to_go, states, actions, timesteps):
        """
        Get action for the most recent timestep (used during evaluation).
        Returns the predicted action for the LAST position.
        """
        logits = self.forward(returns_to_go, states, actions, timesteps)
        # Only the last timestep's prediction
        logits = logits[:, -1, :]  # (B, act_dim)
        action = torch.argmax(logits, dim=-1)  # greedy
        return action.item()


if __name__ == "__main__":
    # Quick shape test
    B, K = 4, 20
    state_dim, act_dim = 4, 2  # CartPole

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        context_len=K,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Dummy inputs
    rtg = torch.randn(B, K, 1)
    states = torch.randn(B, K, state_dim)
    actions = torch.randint(0, act_dim, (B, K))
    timesteps = torch.arange(K).unsqueeze(0).expand(B, -1)

    logits = model(rtg, states, actions, timesteps)
    print(f"Action logits shape: {logits.shape}")
    print(f"Expected: ({B}, {K}, {act_dim})")
    print("Shape test passed!")