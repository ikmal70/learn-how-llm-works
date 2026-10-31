"""Step 2: a bigram language model in PyTorch.

Idea: predict the next token using ONLY the current token. The whole model is
one lookup table: row = current token, columns = scores for every possible next token.

Tokens here are raw bytes (vocab 256), same starting point as step 01.

Run the demo:  uv run python 02-bigram/bigram.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 256  # one token per byte


def get_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ---------- data ----------


def load_data(path: str) -> torch.Tensor:
    """Read a text file -> 1-D tensor of byte values, shape (N,), dtype long."""
    with open(path, "rb") as f:
        return torch.tensor(list(f.read()), dtype=torch.long)


def split_data(data: torch.Tensor, val_frac: float = 0.1):
    """First 90% for training, last 10% for validation."""
    n = int(len(data) * (1 - val_frac))
    return data[:n], data[n:]


def decode(ids) -> str:
    """Token ids (tensor or list) -> text."""
    return bytes(int(i) for i in ids).decode("utf-8", errors="replace")


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str = "cpu"):
    """Sample random chunks of text for training.

    Returns x, y, both shape (batch_size, block_size).
    y is x shifted ONE token to the right: y[b, t] is the token that follows x[b, t].

    Tiny example: if data = [10, 11, 12, 13, 14] and the chosen start is 1, block_size=3:
        x = [11, 12, 13]
        y = [12, 13, 14]
    """
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


# ---------- model ----------


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # the entire model: row i = scores (logits) for "which token comes after token i"
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) token ids. targets: (B, T) or None.

        Returns (logits, loss):
            logits: (B, T, vocab_size)
            loss:   scalar tensor, or None if targets is None
        """
        logits = self.table(idx)  # (B, T, V)
        if targets is None:
            return logits, None

        B, T, V = logits.shape
        # cross_entropy wants (N, V) scores and (N,) targets, so flatten batch and time
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Extend idx (B, T) by max_new_tokens tokens -> (B, T + max_new_tokens)."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)  # (B, T, V)
            logits = logits[:, -1, :]  # last position only -> (B, V)
            probs = F.softmax(logits, dim=-1)  # (B, V)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)  # (B, T + 1)
        return idx


# ---------- training ----------


def train(model, train_data, steps: int, batch_size: int, block_size: int,
          lr: float, device: str = "cpu") -> list[float]:
    """Train the model, return the loss at every step."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        x, y = get_batch(train_data, block_size, batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


if __name__ == "__main__":
    device = get_device()
    data = load_data("02-bigram/data/input.txt")
    train_data, val_data = split_data(data)

    model = BigramLanguageModel(VOCAB_SIZE).to(device)
    losses = train(model, train_data, steps=3000, batch_size=32, block_size=8,
                   lr=1e-2, device=device)
    print(f"final loss (avg of last 100 steps): {sum(losses[-100:]) / 100:.3f}")

    start = torch.zeros((1, 1), dtype=torch.long, device=device)  # token 0 as a seed
    print(decode(model.generate(start, max_new_tokens=300)[0].tolist()))
