# Environment Setup Notes

How this project's Python environment was set up, and how to reproduce or use it.

Machine: MacBook Air M5, 24GB RAM, macOS. Shell: zsh.

## What was used

| Tool | Role |
|------|------|
| `uv` | Package and environment manager (replaces pip + venv + pyenv). Installed via Homebrew at `/opt/homebrew/bin/uv`. |
| `.venv/` | Project-local virtual environment, created and managed by uv. Python 3.12. |
| `pyproject.toml` | Lists the project's dependencies. |
| `uv.lock` | Exact pinned versions of everything installed, for reproducibility. |
| `ipykernel` | Lets VS Code run `.ipynb` notebooks using the `.venv`. |

The macOS system Python is 3.9 and was **not** used. uv downloads and manages its own Python 3.12 for this project.

## Commands that were run

All from the project root (`study/ai`).

```bash
# 1. Create the project (bare = just pyproject.toml, no sample files), pinned to Python 3.12
uv init --bare --name llm-from-scratch --python 3.12

# 2. Add runtime dependencies (creates .venv and uv.lock on first use)
uv add torch numpy matplotlib tiktoken regex

# 3. Verify PyTorch and the Apple GPU backend (MPS)
uv run python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
# expected: 2.15.0 True

# 4. Add the notebook kernel as a dev-only dependency
uv add --dev ipykernel
```

## Packages and why

| Package | Used for |
|---------|----------|
| `torch` | Tensors, autograd, `nn.Module`, training. Runs on the M-series GPU via MPS. |
| `numpy` | Array work outside PyTorch. |
| `matplotlib` | Loss curves, attention heatmaps. |
| `tiktoken` | Reference BPE tokenizer to compare our own against. |
| `regex` | Advanced regex (used for GPT-style tokenizer pre-splitting). |
| `ipykernel` (dev) | Notebook support in VS Code. |

## Daily usage

```bash
# Run a script in the project environment (no manual activation needed)
uv run python 01-tokenization/bpe.py

# Add a new package
uv add <package>

# Add a dev-only package
uv add --dev <package>

# Remove a package
uv remove <package>

# Recreate the environment from scratch on a new machine (reads uv.lock)
uv sync

# Optional: activate the venv in your shell instead of using `uv run`
source .venv/bin/activate
deactivate   # to leave
```

`uv run` is the preferred way: it always uses the right environment and keeps it in sync with `pyproject.toml`.

## Using the notebook in VS Code

1. Open or create a `.ipynb` file.
2. Click **Select Kernel** (top right) -> **Python Environments**.
3. Choose `llm-from-scratch (3.12.x)` pointing at `.venv/bin/python`.
4. If it doesn't appear, run **Developer: Reload Window** from the command palette.

Verify in the first cell:

```python
import torch
print(torch.__version__, torch.backends.mps.is_available())  # 2.15.0 True
```

Auto-reload edited modules (so changes to `.py` files apply without restarting the kernel):

```python
%load_ext autoreload
%autoreload 2
```

## Using the GPU (MPS)

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
x = torch.randn(4, 4, device=device)
model = model.to(device)
```

Notes:
- Data and model must be on the same device, or PyTorch raises an error.
- Some rare ops are not implemented on MPS. If you hit that, set `PYTORCH_ENABLE_MPS_FALLBACK=1` to let those ops run on CPU.
- Unified memory means the GPU shares the 24GB RAM with everything else. If training runs out of memory, reduce the batch size.

## Project layout

```
ai/
  README.md              # project overview
  COURSE.md              # course outline and progress
  SETUP.md               # this file
  pyproject.toml         # dependencies
  uv.lock                # pinned versions
  .venv/                 # environment (not committed)
  01-tokenization.ipynb  # one notebook per step, named after its folder
  01-tokenization/       # bpe.py
  02-bigram.ipynb
  02-bigram/             # bigram.py, data/ (not committed)
  03-attention.ipynb
  03-attention/          # attention.py
  04-transformer.ipynb
  04-transformer/        # gpt.py
  05-training.ipynb
  05-training/           # prepare_data.py, train.py, data/ and out/ (not committed)
  06-inference.ipynb
  06-inference/          # inference.py
```

## Data (not in git, recreate it)

```bash
# step 02: tiny Shakespeare, ~1MB
mkdir -p 02-bigram/data
curl -L -o 02-bigram/data/input.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# step 05: first 150MB of TinyStories, tokenized (downloads, then writes train.bin and val.bin)
uv run python 05-training/prepare_data.py

# step 05: train (about 30 min on an M5 Air), writes 05-training/out/ckpt.pt and log.csv
uv run python 05-training/train.py
```

No packages were added after the first setup: `numpy`, `requests` (installed with `tiktoken`), `matplotlib` and `tiktoken` cover steps 02 to 06.

## Troubleshooting

- **`ModuleNotFoundError` in a notebook**: the wrong kernel is selected. Switch to the `.venv` kernel.
- **`ModuleNotFoundError` in a script**: run it with `uv run python ...` instead of plain `python3` (which is the system 3.9).
- **`.venv` is broken or out of sync**: delete it and run `uv sync`.
- **Kernel doesn't show up**: reload the VS Code window, then reselect the interpreter.
