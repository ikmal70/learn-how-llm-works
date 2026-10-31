"""Step 6: generating text well.

Two ideas:
  1. Sampling: how to pick the next token from the model's probabilities
     (greedy, temperature, top-k, top-p).
  2. KV cache: stop recomputing the same keys and values for every new token.

Shape letters: B = batch, T = sequence length, V = vocab size, hs = head size.

Run the demo:  uv run python 06-inference/inference.py
(needs a trained checkpoint: uv run python 05-training/train.py)
"""

import os
import sys
import time

import tiktoken
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "04-transformer"))
from gpt import GPT, get_device  # noqa: E402

CKPT_PATH = os.path.join(HERE, "..", "05-training", "out", "ckpt.pt")


# ---------- 1. sampling ----------
# The model gives logits (B, V): one raw score per possible next token. We pick one token from them.


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k highest scores, push the rest to -inf (probability 0 after softmax)."""
    kth_best = torch.topk(logits, k, dim=-1).values[:, -1:]  # (B, 1) the k-th highest score
    return logits.masked_fill(logits < kth_best, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of top tokens whose probabilities add up to at least p."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    # drop a token if the tokens ranked above it already cover p (so the top token is always kept)
    drop = (cumulative - probs) > p
    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
    # put the filtered scores back in their original vocabulary order
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
                top_p: float | None = None) -> torch.Tensor:
    """logits (B, V) -> next token ids (B, 1).

    temperature: below 1 sharpens the distribution (safer, repetitive), above 1 flattens it
                 (more random). 0 means greedy: always take the single most likely token.
    """
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    probs = F.softmax(logits, dim=-1)  # (B, V)
    return torch.multinomial(probs, num_samples=1)  # (B, 1)


# ---------- 2. generation without a cache (what step 04 did) ----------


@torch.no_grad()
def generate(model: GPT, idx: torch.Tensor, max_new_tokens: int, **sampling) -> torch.Tensor:
    """Every step re-runs the model on the whole sequence so far. Simple, but it repeats work."""
    for _ in range(max_new_tokens):
        context = idx[:, -model.block_size :]
        logits, _ = model(context)  # (B, T, V): computes scores for EVERY position, uses only the last
        next_id = sample_next(logits[:, -1, :], **sampling)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


# ---------- 3. generation with a KV cache ----------
# In attention, token t's key and value depend only on token t (and the layers below it), and the
# causal mask means old tokens never see new ones, so their keys and values NEVER change.
# So we compute them once, keep them, and each new token only computes its own q, k, v.
#   without cache: step n processes n tokens        -> total work grows like n^2
#   with cache:    step n processes 1 new token    -> total work grows like n


def new_cache(model: GPT):
    """One slot per (layer, head); each will hold (keys, values), each of shape (B, T_so_far, hs)."""
    return [[None] * len(block.attn.heads) for block in model.blocks]


@torch.no_grad()
def forward_cached(model: GPT, idx: torch.Tensor, cache) -> torch.Tensor:
    """Run the model on NEW tokens only, reusing and extending the cache. idx: (B, T_new) -> logits (B, T_new, V).

    Does the same math as GPT.forward, but unrolls the attention so we can read and write the cache.
    """
    B, T_new = idx.shape
    past = 0 if cache[0][0] is None else cache[0][0][0].shape[1]  # tokens already in the cache
    pos = torch.arange(past, past + T_new, device=idx.device)  # new tokens sit at positions past...
    x = model.tok_emb(idx) + model.pos_emb(pos)  # (B, T_new, C)

    for layer, block in enumerate(model.blocks):
        h = block.ln1(x)
        head_outputs = []
        for i, head in enumerate(block.attn.heads):
            q, k, v = head.query(h), head.key(h), head.value(h)  # each (B, T_new, hs)
            if cache[layer][i] is not None:  # glue the new k, v after the cached ones
                k = torch.cat([cache[layer][i][0], k], dim=1)  # (B, T_total, hs)
                v = torch.cat([cache[layer][i][1], v], dim=1)
            cache[layer][i] = (k, v)

            scores = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T_new, T_total)
            # new token j sits at absolute position past + j and may see keys 0 ... past + j
            visible = torch.tril(torch.ones(T_new, k.shape[1], device=idx.device), diagonal=past)
            scores = scores.masked_fill(visible == 0, float("-inf"))
            head_outputs.append(F.softmax(scores, dim=-1) @ v)  # (B, T_new, hs)

        x = x + block.attn.proj(torch.cat(head_outputs, dim=-1))  # residual around attention
        x = x + block.ffn(block.ln2(x))  # residual around the MLP

    return model.lm_head(model.ln_f(x))  # (B, T_new, V)


@torch.no_grad()
def generate_cached(model: GPT, idx: torch.Tensor, max_new_tokens: int, **sampling) -> torch.Tensor:
    """Same output as `generate`, much less work. Prompt plus new tokens must fit in block_size."""
    assert idx.shape[1] + max_new_tokens <= model.block_size, \
        "learned position embeddings stop at block_size, so the cache cannot slide like `generate` does"
    cache = new_cache(model)
    logits = forward_cached(model, idx, cache)  # "prefill": process the whole prompt once
    out = idx
    for _ in range(max_new_tokens):
        next_id = sample_next(logits[:, -1, :], **sampling)
        out = torch.cat([out, next_id], dim=1)
        logits = forward_cached(model, next_id, cache)  # "decode": ONE new token per step
    return out


# ---------- loading and text helpers ----------


def load_model(path: str = CKPT_PATH, device: str | None = None):
    """Rebuild the GPT from a step 05 checkpoint. Returns (model, checkpoint dict)."""
    device = device or get_device()
    ckpt = torch.load(path, map_location=device)
    model = GPT(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def complete(model: GPT, prompt: str, max_new_tokens: int = 150, use_cache: bool = True, **sampling) -> str:
    """Text in, text out. Uses GPT-2's tokenizer, same as training."""
    enc = tiktoken.get_encoding("gpt2")
    device = next(model.parameters()).device
    idx = torch.tensor([enc.encode_ordinary(prompt)], device=device)
    max_new_tokens = min(max_new_tokens, model.block_size - idx.shape[1]) if use_cache else max_new_tokens
    gen = generate_cached if use_cache else generate
    out = gen(model, idx, max_new_tokens, **sampling)
    return enc.decode(out[0].tolist())


if __name__ == "__main__":
    model, ckpt = load_model()
    device = next(model.parameters()).device
    print(f"loaded step {ckpt['step'] + 1}, lowest val loss seen {ckpt['best_val']:.3f}, device {device}\n")

    prompt = "Once upon a time"
    torch.manual_seed(0)
    for name, cfg in [("greedy", dict(temperature=0)),
                      ("temp 0.8, top-k 40", dict(temperature=0.8, top_k=40)),
                      ("temp 1.5", dict(temperature=1.5))]:
        print(f"--- {name} ---\n{complete(model, prompt, 100, **cfg)}\n")

    enc = tiktoken.get_encoding("gpt2")
    idx = torch.tensor([enc.encode_ordinary(prompt)], device=device)
    for use_cache, gen in [(False, generate), (True, generate_cached)]:
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        gen(model, idx, 200, temperature=0)
        if device == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0
        print(f"{'with' if use_cache else 'without'} KV cache: {200 / dt:.1f} tokens/s")
