# Reproducing GPT-2 (Learning Edition)

This repository contains a modular, typed PyTorch implementation of the GPT-2 model architecture from scratch. It is scaled down to a ~10M parameter version, allowing you to train it on your local **Mac Mini M4 (16GB)** in under 15 minutes, while preserving the exact mathematical and structural properties of the original 124M parameter model.

## Model Configuration (~10M parameters)

- **Layers (`n_layer`)**: 6
- **Heads (`n_head`)**: 6
- **Embedding Dim (`n_embd`)**: 384
- **Context Length (`block_size`)**: 256
- **Vocabulary Size**: 50,257
- **Total Parameters**: ~10.4 Million

---

## Speed Benchmarks on Apple Silicon (M4)

We ran speed profiling comparing CPU execution against the **MPS (Metal Performance Shaders)** backend using various optimization configurations on a sequence length of 256 with a batch size of 4 (1,024 tokens per step, accumulating gradients to simulate 8,192 tokens/batch).

### Baseline Capabilities (Peak Unloaded Performance)

| Configuration | Avg Step Time (ms) | Throughput (tokens/sec) | Speedup vs CPU |
|---|---|---|---|
| **CPU \| FP32 \| Eager** | 219.06 ms | 4,674.5 tok/s | 1.00× (Baseline) |
| **MPS \| FP32 \| Eager** | 26.68 ms | 38,380.4 tok/s | 8.21× |
| **MPS \| FP16 (AMP) \| Eager** | 14.50 ms | 70,624.9 tok/s | 15.11× |
| **MPS \| FP32 \| Compiled** | 26.47 ms | 38,692.6 tok/s | 8.28× |
| **MPS \| FP16 (AMP) \| Compiled** | 14.41 ms | 71,038.5 tok/s | 15.20× |

### Key Performance Insights:
1. **GPU Acceleration**: Training on MPS provides an instant **8.2× speedup** over CPU execution in eager FP32 mode.
2. **Mixed Precision (AMP)**: Enabling float16 mixed precision on MPS increases speed by another **1.85×**, yielding an overall **15.2× speedup** compared to CPU.
3. **Graph Compilation (`torch.compile`)**: Currently, PyTorch's MPS compilation backend is in active development. In this version, it runs successfully but adds negligible performance gains over eager execution (it is recommended to use Eager FP16 mode for simplicity).

---

## Getting Started

### 1. Environment & Setup
Activate the virtual environment containing PyTorch and other required dependencies:
```bash
source ~/venvs/ai/bin/activate  # or use your 'aivenv' alias
```

### 2. Run Logit Parity Verification
Verify the architecture correctness by comparing its outputs against OpenAI's official GPT-2 weights:
```bash
python generate.py --run_test
```
*Output: All tests pass with a maximum absolute difference under $9.2 \times 10^{-5}$!*

### 3. Generate Text from Pre-trained Weights
Generate text using the pre-trained 124M parameter model:
```bash
python generate.py --prompt "Alan Turing determined that" --max_tokens 40
```

### 4. Overfit Check (Sanity Test)
Verify that backpropagation works correctly by overfitting the 10M model on a single batch:
```bash
python train.py --overfit
```
*Expected: Loss drops below 0.01 in less than 120 steps.*

### 5. Start Training
Train the ~10M model on Tiny Shakespeare:
```bash
python train.py --steps 100 --amp
```

### 6. Run HellaSwag Evaluation
Evaluate a trained local checkpoint on the multiple-choice HellaSwag benchmark validation subset:
```bash
python hellaswag.py --checkpoint checkpoint_step_20.pt
```

---

## Architecture Reference

```
GPT Model
├── Token Embedding (wte) -- Weight Tied to Output Head (lm_head)
├── Positional Embedding (wpe)
├── Transformer Blocks [0..5]
│   ├── Pre-LayerNorm 1
│   ├── Causal Self-Attention (supports FlashAttention / SDPA)
│   ├── Residual Connection 1
│   ├── Pre-LayerNorm 2
│   ├── MLP (Linear -> GELU (approximate="tanh") -> Linear)
│   └── Residual Connection 2
└── Final LayerNorm (ln_f) -> Linear Output Head (lm_head)
```
