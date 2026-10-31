"""Step 5a: turn TinyStories text into token ids on disk.

TinyStories: short, simple stories written by an LLM, made for training tiny models.
We download the first ~150MB, split it into stories, tokenize with GPT-2's BPE
(the same algorithm you built in step 01, but already trained and written in Rust,
so it is fast), and save everything as flat uint16 arrays.

Why uint16: GPT-2's vocab is 50,257 tokens, which fits in 16 bits (max 65,535).

Run:  uv run python 05-training/prepare_data.py
Creates (inside 05-training/data/): train.bin, val.bin
"""

import os

import numpy as np
import requests
import tiktoken

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
N_BYTES = 150_000_000  # the full file is 1.9GB, plenty for a small model in minutes
EOT = "<|endoftext|>"  # TinyStories separates stories with this marker


def download(path: str) -> None:
    print(f"downloading first {N_BYTES / 1e6:.0f}MB of TinyStories...")
    headers = {"Range": f"bytes=0-{N_BYTES - 1}"}
    with requests.get(URL, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_path = os.path.join(DATA_DIR, "tinystories_150mb.txt")
    if not os.path.exists(raw_path):
        download(raw_path)

    text = open(raw_path, encoding="utf-8").read()
    stories = text.split(EOT)[:-1]  # the range cut leaves a partial story at the end: drop it
    stories = [s.strip() for s in stories if s.strip()]
    print(f"{len(stories):,} stories")

    enc = tiktoken.get_encoding("gpt2")
    eot_id = enc.eot_token  # 50256, marks the end of a story
    tokenized = enc.encode_ordinary_batch(stories)  # list of token-id lists, multithreaded
    ids = []
    for t in tokenized:
        ids.extend(t)
        ids.append(eot_id)
    ids = np.array(ids, dtype=np.uint16)
    print(f"{len(ids):,} tokens ({len(text) / len(ids):.2f} characters per token)")

    # hold out the last 1% for validation. Cut at a story boundary: the arrays end on an EOT.
    n_val = len(ids) // 100
    cut = len(ids) - n_val
    cut += int(np.argmax(ids[cut:] == eot_id)) + 1
    ids[:cut].tofile(os.path.join(DATA_DIR, "train.bin"))
    ids[cut:].tofile(os.path.join(DATA_DIR, "val.bin"))
    print(f"train: {cut:,} tokens, val: {len(ids) - cut:,} tokens")


if __name__ == "__main__":
    main()
