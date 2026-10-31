"""Step 5b: train a GPT properly on TinyStories.

What is new compared with step 04's `train()`:
  - real tokenizer + dataset on disk (memory-mapped, never fully loaded into RAM)
  - learning-rate warmup, then cosine decay
  - gradient clipping
  - AdamW with weight decay only on weight matrices
  - periodic evaluation on held-out text, a CSV log, and checkpoints you can resume from

Run:  uv run python 05-training/prepare_data.py     (once)
      uv run python 05-training/train.py            (about 30 minutes on an M5 Air)
Quick test:  uv run python 05-training/train.py --max_steps 50 --eval_interval 25 --out_dir /tmp/test_run

Shape letters: B = batch, T = block size (context length), V = vocab size.
"""

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "04-transformer"))
from gpt import GPT, get_device  # noqa: E402

DATA_DIR = os.path.join(HERE, "data")


# ---------- data ----------


def load_tokens(split: str) -> np.ndarray:
    """Memory-map a .bin file: behaves like an array, but data is read from disk on demand."""
    return np.memmap(os.path.join(DATA_DIR, f"{split}.bin"), dtype=np.uint16, mode="r")


def get_token_batch(data: np.ndarray, block_size: int, batch_size: int, device: str = "cpu"):
    """Same idea as step 02's get_batch, on a numpy array. Returns x, y of shape (B, T); y is x shifted by one."""
    starts = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in starts]).astype(np.int64)
    y = np.stack([data[i + 1 : i + block_size + 1] for i in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


# ---------- learning rate ----------


def get_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Linear warmup to max_lr, then cosine decay down to min_lr.

    Warmup: at step 0 the weights are random and gradients are noisy, so big steps can wreck training.
    Decay: late in training we want small steps to settle into a good minimum.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)  # 0 -> 1
    cosine = 0.5 * (1 + math.cos(math.pi * progress))  # 1 -> 0
    return min_lr + cosine * (max_lr - min_lr)


# ---------- optimizer ----------


def configure_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    """AdamW, with weight decay only on weight matrices (2-D). Biases and LayerNorm gains are 1-D: no decay.

    Weight decay gently shrinks weights toward zero each step, which discourages memorizing.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


# ---------- evaluation ----------


@torch.no_grad()
def estimate_loss(model, datasets: dict, block_size: int, batch_size: int, n_batches: int, device: str) -> dict:
    """Average loss over several random batches for each split. Returns {'train': ..., 'val': ...}."""
    model.eval()
    out = {}
    for name, data in datasets.items():
        total = 0.0
        for _ in range(n_batches):
            x, y = get_token_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            total += loss.item()
        out[name] = total / n_batches
    model.train()
    return out


# ---------- main ----------


def parse_args():
    p = argparse.ArgumentParser()
    # model
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--block_size", type=int, default=256)
    # optimization
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # bookkeeping
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--out_dir", default=os.path.join(HERE, "out"))
    p.add_argument("--resume", action="store_true", help="continue from out_dir/ckpt.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(1337)
    np.random.seed(1337)
    device = get_device()

    datasets = {"train": load_tokens("train"), "val": load_tokens("val")}
    vocab_size = 50257  # GPT-2 tokenizer

    config = dict(vocab_size=vocab_size, n_embd=args.n_embd, n_heads=args.n_heads,
                  n_layers=args.n_layers, block_size=args.block_size, tie_weights=True)
    model = GPT(**config).to(device)
    optimizer = configure_optimizer(model, args.lr, args.weight_decay)
    start_step, best_val = 0, float("inf")

    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step, best_val = ckpt["step"] + 1, ckpt["best_val"]
        print(f"resumed from step {ckpt['step']}")

    tokens_per_step = args.batch_size * args.block_size
    print(f"device: {device} | parameters: {model.num_params():,} | tokens/step: {tokens_per_step:,}")
    print(f"total training tokens: {tokens_per_step * args.max_steps / 1e6:.1f}M "
          f"({tokens_per_step * args.max_steps / len(datasets['train']):.2f} passes over the data)")

    log_path = os.path.join(args.out_dir, "log.csv")
    new_log = not (args.resume and os.path.exists(log_path))
    log_file = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["step", "lr", "train_loss", "val_loss", "elapsed_s"])

    t0 = time.time()
    running = 0.0
    model.train()
    for step in range(start_step, args.max_steps):
        lr = get_lr(step, args.lr, args.min_lr, args.warmup_steps, args.max_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_token_batch(datasets["train"], args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # cap the size of one update
        optimizer.step()
        running += loss.item()

        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            done = step + 1 - start_step
            print(f"step {step + 1:5d} | loss {running / args.log_interval:.3f} | lr {lr:.2e} | "
                  f"{done * tokens_per_step / elapsed:,.0f} tok/s | {elapsed / 60:.1f} min")
            running = 0.0

        if (step + 1) % args.eval_interval == 0 or step + 1 == args.max_steps:
            losses = estimate_loss(model, datasets, args.block_size, args.batch_size, args.eval_batches, device)
            print(f"  eval step {step + 1}: train {losses['train']:.3f} | val {losses['val']:.3f}")
            log.writerow([step + 1, f"{lr:.6f}", f"{losses['train']:.4f}", f"{losses['val']:.4f}",
                          f"{time.time() - t0:.0f}"])
            log_file.flush()
            if losses["val"] < best_val:
                best_val = losses["val"]
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "config": config, "step": step, "best_val": best_val}, ckpt_path)

    log_file.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min. lowest val loss seen {best_val:.3f} "
          f"(the checkpoint holds the final weights). checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
