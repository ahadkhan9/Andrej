# 11. Experiments & Benchmarks

This document records the optimization baselines, validation loss curves, and speed profile benchmarks run on the **Mac Mini M4 (16GB)**.

---

## 1. Overfit Single Batch Baseline (Sanity Test)

Before training a model on a large dataset, we run a sanity check to verify that the model can successfully memorize (overfit) a single batch of data:
```bash
python train.py --overfit
```
This script disables dropout, locks the input to a single batch of shape $(4, 256)$, and runs optimization for 120 steps.

### Expected Behavior:
- **Starting Loss**: $\approx 10.9$ (representing random selection across the $50,257$ vocabulary size, where $-\ln(1/50257) \approx 10.82$).
- **Ending Loss**: Must drop below **$0.01$** within 120 steps.
- **Why it matters**: If the loss does not decrease monotonically or fails to reach 0.01, it indicates a bug in backpropagation, incorrect weight initialization, or an incorrect learning rate.

---

## 2. Speed Benchmarks on Apple Silicon (M4)

We profiled the model's throughput on the Mac Mini M4. The benchmark runs training steps using a batch size of $B=4$ and context length of $T=256$ (1,024 tokens per step) and measures the average step duration:

| Backend / Hardware | Precision | Mode | Avg Step Time (ms) | Throughput (tokens/sec) | Speedup vs. CPU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CPU** | FP32 | Eager | 219.06 ms | 4,674.5 tok/s | 1.00x (Baseline) |
| **MPS (Apple Silicon)** | FP32 | Eager | 26.68 ms | 38,380.4 tok/s | 8.21x |
| **MPS (Apple Silicon)** | FP32 | Compiled | 26.47 ms | 38,692.6 tok/s | 8.28x |
| **MPS (Apple Silicon)** | FP16 (AMP) | Eager | 14.50 ms | 70,624.9 tok/s | 15.11x |
| **MPS (Apple Silicon)** | FP16 (AMP) | Compiled | 14.41 ms | 71,038.5 tok/s | **15.20x** |

### Benchmark Analysis:
1. **MPS Acceleration**: Moving from CPU to the Metal Performance Shaders (MPS) GPU backend yields an immediate **8.2x speedup** in standard FP32 mode.
2. **Mixed Precision (AMP FP16)**: Enabling float16 mixed-precision on MPS nearly doubles the performance, resulting in a **15.1x speedup** over CPU.
3. **Graph Compilation**: PyTorch `torch.compile` is supported and compiles successfully. However, on Apple Silicon MPS, it currently yields negligible performance gains ($< 1\%$) over Eager mode. For local development on MPS, using Eager FP16 mode is recommended for simplicity and fast startup times.
