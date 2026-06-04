# Reproducing GPT-2 — Learning Edition (~10M params, Mac Mini M4)

This is a hands-on curriculum to deeply understand transformer training by building a small GPT from scratch — following the structure of Andrej Karpathy's "Let's Reproduce GPT-2 (124M)" video, but **sized to actually train on your Mac Mini M4 (16GB)**.

Every concept from the video applies identically. We just use smaller dimensions so you get fast iteration loops (seconds per experiment, not hours).

---

## Model Size: Why ~10M Parameters

| Config | GPT-2 (124M) | **Ours (~10M)** |
|---|---|---|
| Layers (`n_layer`) | 12 | 6 |
| Heads (`n_head`) | 12 | 6 |
| Embedding dim (`n_embd`) | 768 | 384 |
| Context length (`block_size`) | 1024 | 256 |
| Vocab size | 50257 | 50257 |
| **Total params** | **~124M** | **~10M** |

> [!NOTE]
> The architecture is **identical** to GPT-2 — same attention, same MLP ratio (4×), same pre-LN, same weight tying. Only the dimensions are smaller. Every lesson transfers directly to the full 124M and beyond.

**Training estimate on M4**: A few thousand steps on Tiny Shakespeare should take **~5–15 minutes**, producing coherent-ish text. Fast enough to re-run experiments dozens of times in an evening.

---

## User Review Required

> [!IMPORTANT]
> **Dataset choice**: We'll start with **Tiny Shakespeare** (~1MB) for instant feedback. Once the pipeline works, we can optionally swap in a FineWeb-Edu shard for a more realistic dataset. Does this sound good, or do you have a preference?

> [!IMPORTANT]
> **Scaling up later**: The code will be written so you can trivially change `GPTConfig` to train a 30M or even 124M model later, just by editing 4 numbers. Nothing in the architecture code changes.

---

## Project Structure

```
~/Projects/Andrej/
├── model.py              # GPT architecture (Attention, MLP, Block, GPT)
├── train.py              # Training loop, optimizer, scheduler, grad accumulation
├── data.py               # Tokenization + DataLoaderLite
├── generate.py           # Text generation / inference
├── benchmark.py          # MPS vs CPU, precision, and speed profiling
├── hellaswag.py          # HellaSwag evaluation
└── README.md             # Your learning notes and experiment log
```

---

## Module 1: Model Architecture & Weight Loading

**What you'll build**: The complete GPT model skeleton. Then we'll prove it works by loading OpenAI's pre-trained GPT-2 weights into it and generating text.

**Key topics**:
- `GPTConfig` dataclass — the 5 numbers that define the entire model
- `GPT(nn.Module)` — the container that wires everything together
- Weight tying: why `wte` and `lm_head` share the same tensor (saves ~38M params in full GPT-2, saves ~19M in ours)
- Loading Hugging Face weights into our custom model via state dict surgery
- Basic greedy/top-k text generation to verify correctness

**What you'll understand after this module**:
- How a 10M-parameter model is just 5 config numbers + a few hundred lines of PyTorch
- Why weight tying matters (parameter efficiency + better gradients)
- How state dicts work and how to map between different model implementations

**PyTorch concepts**: `nn.Module`, `nn.ModuleDict`, `nn.ModuleList`, `nn.Embedding`, `nn.Linear`, `.state_dict()`, `.load_state_dict()`

---

## Module 2: Transformer Internals — Attention, MLP, Residuals

**What you'll build**: The three core components that go inside each transformer block.

**Key topics**:
- **Token + Positional Embeddings**: How the model knows *what* a token is and *where* it is
- **Causal Self-Attention**: The $QK^TV$ mechanism, scaling by $1/\sqrt{d_k}$, causal masking, multi-head splitting
- **MLP (Feed-Forward Network)**: Project up 4×, apply GELU, project back down — and *why* this ratio
- **Pre-Layer Normalization**: Why GPT-2 normalizes *before* attention/MLP (not after), and how this affects training stability
- **Residual connections**: The "gradient highway" that lets you stack 6+ layers without vanishing gradients

**What you'll understand after this module**:
- Why attention is $O(T^2)$ and what that means for context length
- What would break if you removed residual connections (answer: everything)
- Why Pre-LN is preferred over Post-LN for training stability
- How GELU differs from ReLU and why it matters

**PyTorch concepts**: `nn.LayerNorm`, `F.scaled_dot_product_attention`, `F.gelu`, `torch.tril`, tensor reshaping (`view`, `transpose`, `contiguous`)

---

## Module 3: Data Pipeline & Basic Training Loop

**What you'll build**: Tokenization, a lightweight data loader, and a bare-bones training loop that makes the loss go down.

**Key topics**:
- **BPE tokenization** via `tiktoken` — how GPT-2's tokenizer works, why subword tokenization, what the vocab looks like
- **DataLoaderLite**: A simple class that reads pre-tokenized data and yields `(input, target)` batches where targets are shifted by 1 position
- **The training step**: forward → loss → backward → optimizer step
- **Cross-entropy loss**: Why it's the right loss for next-token prediction, what the numbers mean (loss of 4.0 ≈ random over ~55 tokens)
- **Overfitting a single batch**: The first sanity check — can the model memorize 1 batch?

**What you'll understand after this module**:
- Why targets are `input[1:]` (the autoregressive shift)
- What "loss = 10.8" means at initialization (≈ log(50257) = uniform over vocab)
- How to debug a training loop that isn't learning

**PyTorch concepts**: `F.cross_entropy`, `loss.backward()`, `optimizer.step()`, `optimizer.zero_grad()`, `torch.no_grad()`

---

## Module 4: Speed Optimizations (M4 Profiling)

**What you'll build**: A benchmark script that measures the impact of each optimization on your M4.

**Key topics**:
- **Mixed precision**: FP32 vs FP16/BF16 — what MPS supports, when to use `torch.amp.autocast("mps")`
- **`torch.compile`**: What it does (graph capture + kernel fusion), current MPS support status, measuring the speedup
- **`scaled_dot_product_attention`**: PyTorch's fused attention kernel — FlashAttention theory (memory-efficient, IO-aware) and how SDPA dispatches on MPS
- **Profiling methodology**: Warming up, timing with `torch.mps.synchronize()`, measuring tokens/sec

**What you'll understand after this module**:
- Why mixed precision can give ~2× speedup (and when it doesn't on MPS)
- The difference between compute-bound and memory-bound operations
- How to profile PyTorch on Apple Silicon properly

**PyTorch concepts**: `torch.amp.autocast`, `torch.compile`, `F.scaled_dot_product_attention`, `torch.mps.synchronize()`, timing

---

## Module 5: Training Loop Polish & Hyperparameters

**What you'll build**: A production-quality training loop matching GPT-2/GPT-3 paper practices.

**Key topics**:
- **Weight initialization**: Normal(0, 0.02), zeroing biases, scaling residual projections by $1/\sqrt{2N}$
- **AdamW optimizer**: Why weight decay is separate from L2 regularization, parameter group splitting (decay 2D weights, don't decay biases/layernorm)
- **Cosine LR schedule with warmup**: Why warmup prevents early instability, why cosine decay outperforms step decay
- **Gradient accumulation**: How to simulate a 0.5M-token batch with only 4K tokens in memory
- **Gradient clipping**: `clip_grad_norm_(1.0)` — preventing exploding gradients

**What you'll understand after this module**:
- Why initialization matters enormously (bad init → the model never learns)
- Why AdamW ≠ Adam + L2, and why this distinction matters
- How gradient accumulation works mathematically (averaging gradients across micro-batches)
- The trade-offs of batch size, learning rate, and training steps

**PyTorch concepts**: `nn.init.normal_`, parameter groups, `torch.nn.utils.clip_grad_norm_`, manual LR scheduling

---

## Module 6: Evaluation, Generation & Checkpointing

**What you'll build**: Validation monitoring, text generation during training, checkpointing, and HellaSwag evaluation.

**Key topics**:
- **Validation loss**: Monitoring generalization on held-out data
- **Periodic text generation**: Sampling from the model every N steps to watch it improve
- **Checkpointing**: Saving/loading model + optimizer + step so you can resume training
- **HellaSwag evaluation**: A downstream multiple-choice benchmark — how to evaluate completion likelihood across answer choices
- **Temperature, top-k, top-p sampling**: How each affects generation quality

**What you'll understand after this module**:
- How to tell if a model is overfitting vs underfitting
- Why HellaSwag accuracy correlates with model quality even though we never trained on it
- How checkpointing enables fault tolerance and experiment management

**PyTorch concepts**: `torch.save`, `torch.load`, `torch.no_grad()`, `torch.multinomial`, softmax temperature

---

## Verification Plan

### Automated
1. **Logit parity test**: Load HF GPT-2 weights → run same input through both models → assert logits match within $10^{-4}$
2. **Overfit-one-batch test**: Train on 1 batch for 100 steps → loss should drop below 0.01
3. **Benchmark matrix**: CPU vs MPS × FP32 vs FP16 × Eager vs Compiled — logged as a table

### Manual
- Watch generated text improve from gibberish → semi-coherent → recognizable Shakespeare
- Confirm training loss drops from ~10.8 → ~3.0 range over a few thousand steps
- Verify checkpoint save/load produces identical outputs
