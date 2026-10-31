# Course Outline: Build a Small GPT From Scratch

**Goal:** understand how LLMs work by building, training and running your own small GPT-style model.
**Machine:** MacBook Air M5, 24GB RAM, PyTorch on `mps`.
**Final result:** a ~10-30M parameter GPT trained on TinyStories that generates short, coherent text.

## How each step works

1. Short concept explanation (intuition first, then the math you need).
2. You write the core code in a `.py` file. Hints first, full answers only if stuck.
3. Test in a notebook: run it, check shapes and outputs.
4. A few check questions before moving on.

Rule of thumb: model code lives in `.py` files, notebooks only import and test, training runs as scripts.

## Big picture

```
text -> tokens (ints) -> model -> P(next token | previous tokens) -> sample -> repeat
```

An LLM is a function that predicts the next token. Every step below builds one piece of that pipeline.

## Core path

| # | Folder | Topic | What you build | You will understand |
|---|--------|-------|----------------|---------------------|
| 01 | `01-tokenization/` | Tokenization (BPE) | Byte-level BPE tokenizer: `get_pair_counts`, `merge`, `train`, `encode`, `decode` | How text becomes integers, why subwords, why merge order matters |
| 02 | `02-bigram/` | Bigram model | Simplest next-token predictor in PyTorch | Embeddings, logits, cross-entropy loss, the training loop |
| 03 | `03-attention/` | Self-attention | Single attention head with causal mask, then multi-head | Queries, keys, values, how tokens look at earlier tokens |
| 04 | `04-transformer/` | Transformer and GPT | Residual connections, LayerNorm, MLP, positional encoding, full GPT class | How the pieces combine into one model |
| 05 | `05-training/` | Training | Data loader, AdamW, LR schedule, train on TinyStories | How the model learns, reading loss curves, overfitting |
| 06 | `06-inference/` | Generation | Sampling (temperature, top-k, top-p), KV cache | How the model produces text, and why it is fast or slow |

After step 05 you have a working GPT you wrote and trained yourself. Step 06 improves how it generates.

## Step details

### 01 Tokenization
- Why characters and words fall short, and why subwords (BPE) are the compromise.
- Algorithm: start from bytes (vocab 256), count adjacent pairs, merge the most frequent into a new ID, repeat.
- Check: round trip `decode(encode(text)) == text`, then compare with `tiktoken`.

### 02 Bigram model
- Next-token prediction as the training objective.
- `nn.Embedding` as a lookup table, logits shape `(B, T, vocab)`.
- Cross-entropy loss, backprop, optimizer step, first generated (nonsense) text.
- Gives a baseline loss to beat in later steps.

### 03 Self-attention
- Why tokens need to communicate, and how it replaced RNNs.
- Q, K, V projections, scaled dot-product, softmax, causal mask.
- Multi-head attention: several heads in parallel, then concat.
- Tensor shapes at every step, e.g. `(B, T, C)` and `(B, heads, T, T)`.

### 04 Transformer block and GPT
- Residual connections and LayerNorm (why training needs them).
- MLP (feed-forward) layer, and what it contributes next to attention.
- Positional information (learned embeddings first).
- Stack N blocks, final linear head to vocab logits. Count parameters.

### 05 Training
- Data: first 150MB of TinyStories, tokenized with GPT-2's BPE via `tiktoken` (the step 01 algorithm, pre-trained and fast; our pure-Python tokenizer would take too long on 36M tokens), stored as `uint16` arrays and memory-mapped.
- Batching text into `(input, target)` pairs shifted by one token.
- AdamW with weight decay on matrices only, learning-rate warmup then cosine decay, gradient clipping.
- Train/validation split, periodic evaluation, CSV log, checkpoints with resume.
- A 30M-parameter GPT (6 layers, 6 heads, 384 wide, 256 context, tied embeddings) trained as a script on `mps`.
- Reading loss curves, and what loss and perplexity mean.

### 06 Inference
- Logits to probabilities with softmax, then sampling.
- Temperature, top-k, top-p, and how each changes output (greedy as the extreme).
- KV cache: why generation can reuse past keys and values, and what it costs in memory.
- Check that the cache gives the same logits and the same greedy output, then measure the speed-up.

## Read the paper: "Attention Is All You Need" (Vaswani et al., Google, 2017)

Read it right after step 04, when Q, K, V, multi-head and residual connections are already familiar. Link: https://arxiv.org/abs/1706.03762

| Paper concept | Where we build it |
|---------------|-------------------|
| Scaled dot-product attention, `softmax(QK^T / sqrt(d)) V` | Step 03 |
| Multi-head attention | Step 03 |
| Causal masking (paper's decoder) | Step 03 |
| Residual connections, layer normalization | Step 04 |
| Position-wise feed-forward network | Step 04 |
| Positional encoding | Step 04 |
| Stack of N identical blocks | Step 04 |

How our GPT differs from the paper:
- **Decoder-only.** The paper has an encoder and a decoder (built for translation). GPT keeps only the decoder-style stack and predicts the next token, so there is no cross-attention.
- **Positional encoding.** Paper: fixed sinusoidal. Ours: learned embeddings first (like GPT-2). RoPE is an optional extension.
- **Pre-norm vs post-norm.** Paper: LayerNorm after each sub-layer. GPT-style: before, which trains more stably.
- **Activation.** Paper: ReLU in the feed-forward layer. GPT-style: GELU.
- **Tokenization.** We build BPE from scratch in step 01.

Reading task: compare Figure 1 (the full architecture) and Figure 2 (attention) with the code you wrote, and note each difference above.

## Optional extensions (after the core path)

- **RoPE:** the positional method modern models use.
- **LoRA fine-tuning:** adapt a pretrained model cheaply.
- **Post-training concepts:** SFT, RLHF, DPO.
- **Efficiency:** quantization, Flash Attention.
- **Architecture variants:** grouped-query attention, Mixture of Experts.
- **GGUF export:** convert for llama.cpp or Ollama (needs a compatible architecture).

## What the model outputs

| Level | Output |
|-------|--------|
| Each forward pass | Logits tensor, shape `(B, T, vocab_size)` |
| Saved after training | PyTorch checkpoint `.pt` (`state_dict`) plus the tokenizer merges |
| Generation loop | Token IDs, then `tok.decode(ids)` gives text |
| GGUF | Not produced by training. An optional conversion step afterwards |

## What this course does not cover

- Large-scale training (distributed training, trillions of tokens, data pipelines).
- Frontier research (reasoning models, interpretability).

Those are ongoing learning after the foundation here.

## Progress

- [x] 01 Tokenization (all 5 functions done, tested in `01-tokenization.ipynb`; check questions still open)
- [x] 02 Bigram model (ran `02-bigram.ipynb`: loss 5.89 -> 2.47, train 2.486 / val 2.492; check questions still open)
- [x] 03 Self-attention (ran `03-attention.ipynb`: loss 2.055 train / 2.165 val vs bigram 2.47; check questions still open)
- [x] 04 Transformer and GPT (ran `04-transformer.ipynb`: 1.43 train / 1.65 val, 865k params; check questions still open, paper not read yet)
- [x] 05 Training (ran `05-training.ipynb`: 30M params, 40 min, final val loss 2.126, perplexity 8.4; check questions still open)
- [x] 06 Inference (ran `06-inference.ipynb`: cache gives identical output, 1.2x on GPU, 2.5x to 5.7x on CPU; check questions still open)
- [ ] Read "Attention Is All You Need" (after step 04)
- [ ] Check questions for steps 01 to 06 (none answered yet)
- [ ] Extensions (RoPE, LoRA, SFT/RLHF concepts, quantization)

Ticked means the code is built and you ran its notebook. Understanding is confirmed by the check questions, tracked separately above.
