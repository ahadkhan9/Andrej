# 00. Project Overview

This project implements a custom, scaled-down reproduction of the OpenAI GPT-2 language model from scratch in PyTorch. The design is structured to mirror the exact mathematical and structural properties of the original GPT-2 (124M) architecture, but it is sized to run and train efficiently on a local **Mac Mini M4 (16GB)**.

---

## 1. Model Configuration

To achieve fast iteration while preserving all structural details (such as LayerNorm ordering, residual scaling, feed-forward ratios, and weight tying), the hyperparameter configuration is adjusted from the original 124M model to a ~10.4M parameter version:

| Hyperparameter | GPT-2 (Original) | GPT-2 (Learning Edition) |
| :--- | :---: | :---: |
| **Number of Layers (`n_layer`)** | 12 | 6 |
| **Attention Heads (`n_head`)** | 12 | 6 |
| **Embedding Dimension (`n_embd`)** | 768 | 384 |
| **Context Length (`block_size`)** | 1024 | 256 |
| **Vocabulary Size (`vocab_size`)** | 50,257 | 50,257 |
| **Parameter Count** | ~124 Million | ~10.4 Million |

---

## 2. Directory Layout

The codebase consists of modular, typed files separating the data pipeline, model definition, benchmarking, evaluation, and training logic:

```text
Andrej/
├── model.py         # GPT-2 model architecture and state dictionary mappings
├── data.py          # BPE tokenization and sliding window DataLoader
├── train.py         # Main training loop with learning rate schedules and optimizer configs
├── generate.py      # Autoregressive text generation and logit parity tests
├── hellaswag.py     # Evaluation script for downstream multiple-choice accuracy
├── benchmark.py     # Profiling matrix comparing CPU vs. MPS, precision, and compilation
├── input.txt        # Tiny Shakespeare corpus used for training
└── docs/            # Deep-dive system documentation (this directory)
```

---

## 3. Core System Goals

1. **Architecture Fidelity**: Ensure that the forward pass outputs identical logits to the official Hugging Face GPT-2 weights. This is verified using a logit parity test (differences must stay under $1 \times 10^{-4}$).
2. **First-Principles Understanding**: Avoid relying on black-box abstractions. Implement the transformer block, causal self-attention, and MLP layers manually.
3. **Hardware Acceleration**: Leverage Apple Silicon GPU cores via the PyTorch **MPS (Metal Performance Shaders)** backend, using mixed-precision (AMP float16) to maximize throughput.
