"""Step 7b: supervised fine-tuning (SFT). Turn a text continuer into something that follows a request.

The pretrained model only knows "what text usually comes next". SFT keeps training it on
(prompt, response) examples in one fixed format, so it learns "after the prompt comes an answer to it".

Two things are new compared with pretraining (step 05):
  1. The training text is structured:  <prompt> + <response> + <end-of-text>
  2. The loss is counted on the response tokens only. The model should learn to ANSWER,
     not to predict the prompt (the prompt is given to it at use time).

Everything else (model, AdamW, LR schedule, loop) is reused from earlier steps.

Run:  uv run python 07-sft/prepare_sft.py     (once, needs the step 05 checkpoint too)
      uv run python 07-sft/sft.py             (about 15 minutes on an M5 Air)
Quick test:  uv run python 07-sft/sft.py --steps 20 --eval_interval 10 --n_prompts 5 --out_dir /tmp/sft_test

Shape letters: B = batch, T = longest example in the batch, V = vocab size.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import tiktoken
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for step_dir in ("04-transformer", "05-training", "06-inference"):
    sys.path.insert(0, os.path.join(HERE, "..", step_dir))
from gpt import get_device  # noqa: E402
from inference import load_model, sample_next  # noqa: E402
from train import configure_optimizer, get_lr  # noqa: E402

DATA_DIR = os.path.join(HERE, "data")
BASE_CKPT = os.path.join(HERE, "..", "05-training", "out", "ckpt.pt")
IGNORE = -100  # F.cross_entropy skips targets with this value
EOT_ID = 50256  # GPT-2 <|endoftext|>, also used as padding (padded positions are ignored anyway)


# ---------- the prompt format ----------


def format_prompt(fields: dict) -> str:
    """{'Words': 'quit, oak', 'Features': 'Dialogue'} -> the text the model sees before it answers.

    The fixed layout is what the model learns to recognise: '### Response:' means "your turn".
    """
    lines = [f"{key}: {fields[key]}" for key in ("Words", "Features", "Summary") if key in fields]
    return "### Instruction:\nWrite a short story.\n" + "\n".join(lines) + "\n\n### Response:\n"


# ---------- data ----------


def load_examples(split: str) -> list[tuple[np.ndarray, int]]:
    """-> list of (token ids, prompt length). The ids hold prompt + response + end-of-text."""
    d = np.load(os.path.join(DATA_DIR, f"{split}.npz"))
    ends = np.cumsum(d["lens"])
    starts = ends - d["lens"]
    return [(d["ids"][s:e].astype(np.int64), int(p)) for s, e, p in zip(starts, ends, d["prompt_lens"])]


def make_batch(batch: list[tuple[np.ndarray, int]], device: str):
    """Pad examples to equal length and build inputs x and masked targets y, both (B, T).

    y is x shifted by one, as in pretraining. The difference: targets that are prompt tokens (or padding)
    are set to IGNORE, so they add nothing to the loss.

    Tiny example, prompt length 2:  ids = [P1 P2 R1 R2 EOT]
        x = [P1 P2 R1 R2]      y = [IGNORE, R1, R2, EOT]
    At P1 the model is not asked to predict P2 (prompt). At P2 it must predict R1: the first answer token.
    """
    B = len(batch)
    T = max(len(ids) for ids, _ in batch) - 1
    x = np.full((B, T), EOT_ID, dtype=np.int64)
    y = np.full((B, T), IGNORE, dtype=np.int64)
    for b, (ids, prompt_len) in enumerate(batch):
        n = len(ids) - 1
        x[b, :n] = ids[:-1]
        y[b, :n] = ids[1:]
        y[b, : prompt_len - 1] = IGNORE  # target j is token j+1: ignore while that token is still prompt
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def sample_batch(examples, batch_size: int, device: str):
    picks = np.random.randint(0, len(examples), size=batch_size)
    return make_batch([examples[i] for i in picks], device)


# ---------- evaluation ----------


@torch.no_grad()
def response_loss(model, examples, batch_size: int, n_batches: int, device: str) -> float:
    """Average loss on response tokens only (how well the model predicts the answer given the prompt)."""
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        x, y = sample_batch(examples, batch_size, device)
        _, loss = model(x, y)  # cross_entropy ignores IGNORE targets
        total += loss.item()
    model.train()
    return total / n_batches


@torch.no_grad()
def respond(model, prompt: str, max_new_tokens: int = 230, temperature: float = 0.7, top_k: int | None = 40) -> str:
    """Prompt text -> the model's answer, stopping at end-of-text. Uses the step 06 sampler."""
    enc = tiktoken.get_encoding("gpt2")
    device = next(model.parameters()).device
    model.eval()
    ids = torch.tensor([enc.encode_ordinary(prompt)], device=device)  # (1, T0)
    start = ids.shape[1]
    for _ in range(min(max_new_tokens, model.block_size - start)):
        logits, _ = model(ids)  # (1, T, V)
        nxt = sample_next(logits[:, -1, :], temperature, top_k)  # (1, 1)
        if nxt.item() == EOT_ID:
            break
        ids = torch.cat([ids, nxt], dim=1)
    return enc.decode(ids[0, start:].tolist()).strip()


def word_use_rate(model, val_prompts: list[dict], n: int, **sampling) -> dict:
    """Does the story contain the words the prompt asked for? Cheap, honest proxy for 'followed the request'.

    Returns the average share of requested words used, and the share of prompts where ALL were used.
    """
    shares, all_used = [], 0
    for item in val_prompts[:n]:
        if not item["words"]:
            continue
        story = respond(model, item["prompt"], **sampling).lower()
        hits = [w.lower() in story for w in item["words"]]
        shares.append(sum(hits) / len(hits))
        all_used += all(hits)
    return {"avg_share": float(np.mean(shares)), "all_words": all_used / len(shares), "n": len(shares)}


# ---------- main ----------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)  # 10x lower than pretraining: adjust the model, don't rewrite it
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=30)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--n_prompts", type=int, default=40, help="prompts used for the word-use test")
    p.add_argument("--base_ckpt", default=BASE_CKPT)
    p.add_argument("--out_dir", default=os.path.join(HERE, "out"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(1337)
    np.random.seed(1337)
    device = get_device()

    train_ex, val_ex = load_examples("train"), load_examples("val")
    val_prompts = json.load(open(os.path.join(DATA_DIR, "val_prompts.json")))
    model, ckpt = load_model(args.base_ckpt, device)
    model.train()
    print(f"device: {device} | start from the step 05 model ({model.num_params():,} params) | "
          f"{len(train_ex):,} train / {len(val_ex):,} val examples")

    def report(tag: str) -> None:
        loss = response_loss(model, val_ex, args.batch_size, 20, device)
        words = word_use_rate(model, val_prompts, args.n_prompts, temperature=0.7, top_k=40)
        print(f"[{tag}] response loss {loss:.3f} | requested words used {words['avg_share']:.0%} "
              f"| all words used in {words['all_words']:.0%} of {words['n']} stories")
        model.train()

    report("before SFT")

    optimizer = configure_optimizer(model, args.lr, args.weight_decay)
    t0, running = time.time(), 0.0
    for step in range(args.steps):
        lr = get_lr(step, args.lr, args.min_lr, args.warmup_steps, args.steps)
        for g in optimizer.param_groups:
            g["lr"] = lr
        x, y = sample_batch(train_ex, args.batch_size, device)  # (B, T), (B, T)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        running += loss.item()

        if (step + 1) % 50 == 0:
            print(f"step {step + 1:4d} | train response loss {running / 50:.3f} | lr {lr:.1e} | "
                  f"{(time.time() - t0) / 60:.1f} min")
            running = 0.0
        if (step + 1) % args.eval_interval == 0 and step + 1 < args.steps:
            print(f"  val response loss {response_loss(model, val_ex, args.batch_size, 20, device):.3f}")

    report("after SFT ")
    torch.save({"model": model.state_dict(), "config": ckpt["config"], "step": args.steps - 1,
                "best_val": float("nan")}, os.path.join(args.out_dir, "sft_ckpt.pt"))
    print(f"saved {os.path.join(args.out_dir, 'sft_ckpt.pt')}")


if __name__ == "__main__":
    main()
