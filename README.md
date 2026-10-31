# LLM From Scratch

Learn how LLMs work by building a small GPT in PyTorch, one step at a time:
tokenizer, bigram model, self-attention, transformer, training, and generation.
Built and trained on a MacBook Air M5 (24GB), using the Apple GPU through PyTorch's `mps` backend.

## The steps

Each step has a folder with the code and a notebook of the same name that tests it piece by piece.

| # | Topic | Code | Notebook | What you get |
|---|-------|------|----------|--------------|
| 01 | Tokenization | `01-tokenization/bpe.py` | `01-tokenization.ipynb` | Byte-level BPE tokenizer written from scratch |
| 02 | Bigram model | `02-bigram/bigram.py` | `02-bigram.ipynb` | Simplest next-token predictor, the training loop |
| 03 | Self-attention | `03-attention/attention.py` | `03-attention.ipynb` | Causal attention head, multi-head attention |
| 04 | Transformer | `04-transformer/gpt.py` | `04-transformer.ipynb` | Full GPT: MLP, residuals, LayerNorm, stacked blocks |
| 05 | Training | `05-training/train.py` | `05-training.ipynb` | A 30M-parameter GPT trained on TinyStories |
| 06 | Inference | `06-inference/inference.py` | `06-inference.ipynb` | Temperature, top-k, top-p, KV cache |

Code lives in the `.py` files. Notebooks only import it and test it, so run them top to bottom.
See `COURSE.md` for the outline and progress, and `SETUP.md` for how the environment was built.

## Quick start

Needs [uv](https://docs.astral.sh/uv/) (`brew install uv`).

```bash
uv sync                                          # creates .venv with Python 3.12, torch, etc.
uv run python 01-tokenization/bpe.py             # tokenizer demo, prints "round trip OK"
```

In VS Code, open any notebook and pick the `.venv` Python as the kernel.

Data is not stored in git. Recreate it (commands in `SETUP.md`):

```bash
# step 02: tiny Shakespeare (~1MB)
mkdir -p 02-bigram/data && curl -L -o 02-bigram/data/input.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# step 05: download and tokenize ~150MB of TinyStories, then train (~40 min on an M5 Air)
uv run python 05-training/prepare_data.py
uv run python 05-training/train.py
```

Steps 02 to 04 run in seconds to a couple of minutes. Step 06 needs the checkpoint from step 05.

## Results

Same training loop, different architectures, on tiny Shakespeare (byte-level, lower loss is better):

| Model | Train loss | Val loss |
|-------|-----------:|---------:|
| Bigram (step 02) | 2.49 | 2.49 |
| One attention layer (step 03) | 2.04 | 2.17 |
| 4-layer GPT, 0.87M params (step 04) | 1.42 | 1.65 |

Step 05, a 30M-parameter GPT (6 layers, 6 heads, 384 wide, 256 context) on 36M tokens of TinyStories with GPT-2's tokenizer:

- 4,000 steps, 16.4M tokens seen, 40 minutes on the M5 Air
- final val loss 2.13 (perplexity about 8.4; random guessing would be 50,257)
- it writes simple, mostly coherent children's stories

Losses are only comparable within a dataset and tokenizer, not across the two tables.

Step 06: the KV cache gives identical output to plain generation. Speed-up measured on this model: about 2x to 4x on the CPU, and a modest 1.1x to 1.8x on the GPU, where overhead from many small operations dominates.

## Layout

```
01-tokenization/  02-bigram/  03-attention/  04-transformer/  05-training/  06-inference/
NN-name.ipynb     one notebook per step, at the project root
COURSE.md         course outline and progress
SETUP.md          environment setup notes
```

`05-training/data/` and `05-training/out/` (datasets, checkpoint, log) are git-ignored.

## Notes

- Everything is written from scratch except the tokenizer in step 05, which uses `tiktoken`'s GPT-2 BPE. It is the same algorithm as step 01, already trained and fast enough for 36M tokens.
- The model is a decoder-only transformer as in "Attention Is All You Need" (Vaswani et al., 2017), with the GPT-2 style changes: pre-norm, GELU, learned positions, tied embeddings.
