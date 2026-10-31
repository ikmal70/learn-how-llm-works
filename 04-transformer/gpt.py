"""Step 4: a full GPT (decoder-only transformer).

Step 03 gave us one attention layer wired to the output. A real transformer adds:
  - FeedForward: an MLP so each token can "think" about what attention gathered
  - Residual connections: x = x + f(x), so information and gradients flow through deep stacks
  - LayerNorm: keeps the numbers at a stable scale
  - Many Blocks stacked, so later layers build on earlier ones

Shape letters: B = batch, T = sequence length, C = embedding size, V = vocab size.

Run the demo:  uv run python 04-transformer/gpt.py
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# reuse step 03's attention and (through it) step 02's data helpers and training loop
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "03-attention"))
from attention import (  # noqa: E402,F401
    MultiHeadAttention, VOCAB_SIZE, get_device, load_data, split_data, decode, get_batch, train,
)


class FeedForward(nn.Module):
    """Per-token MLP: expand to 4x wider, nonlinearity, shrink back.

    Attention moves information BETWEEN tokens. This layer processes each token on its own.
    """

    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, T, C) -> (B, T, C)


class Block(nn.Module):
    """One transformer block: attention then MLP, each with LayerNorm first and a residual around it."""

    def __init__(self, n_embd: int, n_heads: int, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_heads, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # "pre-norm": normalize BEFORE each sub-layer. The "x +" is the residual connection:
        # the block only learns a correction to x instead of rebuilding it from scratch.
        x = x + self.attn(self.ln1(x))  # (B, T, C)
        x = x + self.ffn(self.ln2(x))  # (B, T, C)
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, n_heads: int, n_layers: int,
                 block_size: int, tie_weights: bool = False):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads, block_size) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)  # final LayerNorm before the output layer
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        if tie_weights:
            # input embedding and output layer share one matrix (GPT-2 does this).
            # Saves vocab_size * n_embd parameters, which matters for big vocabularies.
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        # small random start (GPT-2 style); the default init is a bit too large for deep stacks
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) token ids. Returns (logits (B, T, V), loss or None)."""
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))  # (B, T, C)
        x = self.blocks(x)  # (B, T, C) through every block
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, V)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Extend idx (B, T) by max_new_tokens tokens (step 06 adds temperature, top-k and a KV cache)."""
        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size :]  # position table only has block_size entries
            logits, _ = self(context)
            probs = F.softmax(logits[:, -1, :], dim=-1)  # (B, V)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # tied weights are counted once


if __name__ == "__main__":
    device = get_device()
    data = load_data("02-bigram/data/input.txt")
    train_data, val_data = split_data(data)

    torch.manual_seed(0)
    model = GPT(VOCAB_SIZE, n_embd=128, n_heads=4, n_layers=4, block_size=64).to(device)
    print(f"parameters: {model.num_params():,}")

    losses = train(model, train_data, steps=3000, batch_size=64, block_size=64,
                   lr=2e-3, device=device)
    print(f"final loss (avg of last 100 steps): {sum(losses[-100:]) / 100:.3f}"
          f"  (bigram ~2.47, one attention layer ~2.05)")

    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(start, max_new_tokens=400)[0].tolist()))
