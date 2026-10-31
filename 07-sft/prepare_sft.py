"""Step 7a: build the instruction dataset for supervised fine-tuning (SFT).

TinyStoriesInstruct = TinyStories where each story comes with a request:
    Words: quit, oak, gloomy            <- the story must use these words
    Features: Dialogue                  <- and have these features
    Summary: Sara and Ben ...           <- and match this summary
    Story: <the story>
We turn each record into (prompt, response): the request is the prompt, the story is the response.

For every example we save the token ids and how many of them belong to the prompt.
SFT uses that number to compute the loss on the response only.

Run:  uv run python 07-sft/prepare_sft.py
Creates (inside 07-sft/data/): train.npz, val.npz, val_prompts.json
"""

import json
import os
import sys

import numpy as np
import requests
import tiktoken

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sft import format_prompt  # noqa: E402

DATA_DIR = os.path.join(HERE, "data")
URL = "https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/resolve/main/TinyStories-Instruct-train.txt"
N_BYTES = 40_000_000  # ~25k examples: plenty to teach a format
EOT = "<|endoftext|>"
BLOCK_SIZE = 256  # the model's context length: longer examples are dropped
N_VAL = 500


def download(path: str) -> None:
    print(f"downloading first {N_BYTES / 1e6:.0f}MB of TinyStoriesInstruct...")
    with requests.get(URL, headers={"Range": f"bytes=0-{N_BYTES - 1}"}, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def parse_record(record: str) -> tuple[dict, str] | None:
    """One raw record -> ({'Words': ..., 'Features': ..., 'Summary': ...}, story). None if malformed."""
    if "Story:" not in record:
        return None
    head, story = record.split("Story:", 1)
    fields = {}
    for line in head.strip().split("\n"):
        key, _, value = line.partition(":")
        if key in ("Words", "Features", "Summary") and value.strip():
            fields[key] = value.strip()
    story = story.strip()
    return (fields, story) if fields and story else None


def save_split(path: str, examples: list[tuple[list[int], int]]) -> None:
    """Pack variable-length examples into flat arrays: all ids, each length, each prompt length."""
    ids = np.concatenate([np.array(e[0], dtype=np.uint16) for e in examples])
    lens = np.array([len(e[0]) for e in examples])
    prompt_lens = np.array([e[1] for e in examples])
    np.savez(path, ids=ids, lens=lens, prompt_lens=prompt_lens)


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_path = os.path.join(DATA_DIR, "instruct_40mb.txt")
    if not os.path.exists(raw_path):
        download(raw_path)

    records = open(raw_path, encoding="utf-8").read().split(EOT)[:-1]  # last record is cut off: drop it
    enc = tiktoken.get_encoding("gpt2")

    examples, val_info, dropped = [], [], 0
    for rec in records:
        parsed = parse_record(rec)
        if parsed is None:
            dropped += 1
            continue
        fields, story = parsed
        prompt = format_prompt(fields)
        prompt_ids = enc.encode_ordinary(prompt)
        # response = story + end-of-text, so the model also learns WHEN to stop
        ids = prompt_ids + enc.encode_ordinary(story) + [enc.eot_token]
        if len(ids) > BLOCK_SIZE:
            dropped += 1
            continue
        examples.append((ids, len(prompt_ids)))
        val_info.append({"prompt": prompt, "words": [w.strip() for w in fields.get("Words", "").split(",") if w.strip()]})

    print(f"{len(examples):,} examples kept, {dropped:,} dropped (too long or malformed)")
    resp_tokens = sum(len(i) - p for i, p in examples)
    print(f"avg {sum(len(i) for i, _ in examples) / len(examples):.0f} tokens per example, "
          f"{resp_tokens / sum(len(i) for i, _ in examples):.0%} of tokens are response")

    train, val = examples[:-N_VAL], examples[-N_VAL:]
    save_split(os.path.join(DATA_DIR, "train.npz"), train)
    save_split(os.path.join(DATA_DIR, "val.npz"), val)
    with open(os.path.join(DATA_DIR, "val_prompts.json"), "w") as f:
        json.dump(val_info[-N_VAL:], f)
    print(f"train: {len(train):,} examples, val: {len(val):,} examples")


if __name__ == "__main__":
    main()
