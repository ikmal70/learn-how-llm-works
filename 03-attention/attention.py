"""Step 3: self-attention.

The bigram (step 02) only sees the previous token. Attention lets every token
look at ALL earlier tokens and decide how much each one matters.

Build-up:
  1. causal_mean_*      : "average the past" three ways, ending in the softmax-mask trick
  2. Head               : one attention head (queries, keys, values)
  3. MultiHeadAttention : several heads in parallel, outputs concatenated
  4. AttentionLanguageModel : embeddings + positions + attention + output layer

Shape letters: B = batch, T = sequence length (time), C = embedding size,
V = vocab size, hs = head size.

Run the demo:  uv run python 03-attention/attention.py
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# reuse the data helpers and training loop from step 02
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02-bigram"))
from bigram import VOCAB_SIZE, get_device, load_data, split_data, decode, get_batch, train  # noqa: E402,F401


# ---------- 1. averaging the past, three equivalent ways ----------
# Goal: out[b, t] = mean of x[b, 0..t]. A token may look at itself and earlier tokens, never later ones.


def causal_mean_loop(x: torch.Tensor) -> torch.Tensor:
    """Slow and obvious. x: (B, T, C) -> (B, T, C)."""
    B, T, C = x.shape
    out = torch.zeros(B, T, C, device=x.device)
    for b in range(B):
        for t in range(T):
            out[b, t] = x[b, : t + 1].mean(dim=0)
    return out


def causal_mean_matmul(x: torch.Tensor) -> torch.Tensor:
    """Same result with ONE matrix multiply. A lower-triangular matrix of weights does the averaging."""
    B, T, C = x.shape
    w = torch.tril(torch.ones(T, T, device=x.device))  # (T, T), 1s on and below the diagonal
    w = w / w.sum(dim=1, keepdim=True)  # each row sums to 1, e.g. row 2 = [1/3, 1/3, 1/3, 0]
    return w @ x  # (T, T) @ (B, T, C) -> (B, T, C)


def causal_mean_softmax(x: torch.Tensor) -> torch.Tensor:
    """Same result again, via softmax. This is the form attention uses.

    Start with all-equal scores, set the future to -inf, softmax. exp(-inf) = 0,
    so the future gets weight 0 and the rest share the weight equally.
    In real attention the scores are not all zero: they come from query-key matching.
    """
    B, T, C = x.shape
    mask = torch.tril(torch.ones(T, T, device=x.device))
    scores = torch.zeros(T, T, device=x.device).masked_fill(mask == 0, float("-inf"))
    w = F.softmax(scores, dim=-1)  # (T, T)
    return w @ x  # (B, T, C)


# ---------- 2. one attention head ----------


class Head(nn.Module):
    """One head of causal self-attention.

    Each token makes three vectors from its embedding:
      query: "what am I looking for?"
      key:   "what do I contain?"
      value: "what do I hand over if someone attends to me?"
    score(t, s) = query_t . key_s. High score = token t pays more attention to token s.
    """

    def __init__(self, n_embd: int, head_size: int, block_size: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # not a parameter (never trained), but saved with the model and moved to the GPU with it
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        self.last_attn = None  # kept after each forward so we can plot it

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, T, hs)."""
        B, T, C = x.shape
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        # (B, T, hs) @ (B, hs, T) -> (B, T, T): a score for every (token, earlier token) pair
        scores = q @ k.transpose(-2, -1)
        scores = scores * k.shape[-1] ** -0.5  # scale by 1/sqrt(hs) so softmax doesn't saturate
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))  # no peeking ahead
        weights = F.softmax(scores, dim=-1)  # (B, T, T), each row sums to 1
        self.last_attn = weights.detach()

        return weights @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)


# ---------- 3. several heads ----------


class MultiHeadAttention(nn.Module):
    """n_heads independent heads, each of size C // n_heads, concatenated back to size C.

    Different heads can learn to look for different things (e.g. the previous letter, or the last space).
    """

    def __init__(self, n_embd: int, n_heads: int, block_size: int):
        super().__init__()
        assert n_embd % n_heads == 0, "n_embd must be divisible by n_heads"
        head_size = n_embd // n_heads
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # lets the heads' outputs mix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, T, C)."""
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # n_heads x (B, T, hs) -> (B, T, C)
        return self.proj(out)


# ---------- 4. a language model that uses attention ----------


class AttentionLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, n_heads: int, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)  # token id -> vector
        self.pos_emb = nn.Embedding(block_size, n_embd)  # position -> vector (attention itself has no sense of order)
        self.attn = MultiHeadAttention(n_embd, n_heads, block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)  # vector -> a score per possible next token

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) token ids. Returns (logits (B, T, V), loss or None)."""
        B, T = idx.shape
        tok = self.tok_emb(idx)  # (B, T, C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T, C)
        x = tok + pos  # (B, T, C), position added to every item in the batch
        x = self.attn(x)  # (B, T, C): each token now carries information from earlier tokens
        logits = self.lm_head(x)  # (B, T, V)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Extend idx (B, T) by max_new_tokens tokens."""
        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size :]  # position table only has block_size entries
            logits, _ = self(context)
            probs = F.softmax(logits[:, -1, :], dim=-1)  # (B, V)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


if __name__ == "__main__":
    device = get_device()
    data = load_data("02-bigram/data/input.txt")
    train_data, val_data = split_data(data)

    torch.manual_seed(0)
    model = AttentionLanguageModel(VOCAB_SIZE, n_embd=64, n_heads=4, block_size=32).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    losses = train(model, train_data, steps=3000, batch_size=32, block_size=32,
                   lr=3e-3, device=device)
    print(f"final loss (avg of last 100 steps): {sum(losses[-100:]) / 100:.3f}  (bigram was ~2.47)")

    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(start, max_new_tokens=300)[0].tolist()))
