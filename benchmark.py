"""
Speed profiling and benchmarking script for custom GPT-2 training.
Measures execution times and throughput (tokens/second) across devices (CPU, MPS),
numerical precision modes (FP32, FP16 mixed precision), and execution backends (Eager, Compiled).
"""

import time
import torch
import torch.optim as optim

from model import GPT, GPTConfig
from data import DataLoaderLite


def benchmark_run(
    device: str, 
    use_amp: bool, 
    use_compile: bool, 
    warmup_steps: int = 5, 
    active_steps: int = 20
) -> float:
    """
    Runs training steps and measures average step duration.
    Returns the average time per step in milliseconds.
    """
    # Fix seed for reproducibility
    torch.manual_seed(42)

    config = GPTConfig(
        n_layer=6,
        n_head=6,
        n_embd=384,
        block_size=256,
        vocab_size=50257,
        dropout=0.0
    )
    
    # Instantiate model on CPU first for deterministic initialization, then transfer
    model = GPT(config).to(device)
    model.train()

    if use_compile:
        # Compile the model
        model = torch.compile(model)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # We use B=4, T=256 which is 1024 tokens per step
    B, T = 4, 256
    loader = DataLoaderLite(B=B, T=T)

    # Cache target inputs/targets on device
    x_batches = []
    y_batches = []
    for _ in range(warmup_steps + active_steps):
        x, y = loader.next_batch()
        x_batches.append(x.to(device))
        y_batches.append(y.to(device))

    # Helper function to synchronize device execution
    def sync():
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

    # Warmup loop (to trigger compile or caching)
    for step in range(warmup_steps):
        optimizer.zero_grad()
        x, y = x_batches[step], y_batches[step]
        if use_amp:
            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
    
    sync()

    # Measurement loop
    total_time_ms = 0.0
    for step in range(warmup_steps, warmup_steps + active_steps):
        optimizer.zero_grad()
        x, y = x_batches[step], y_batches[step]
        
        sync()
        t0 = time.time()
        
        if use_amp:
            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
            
        loss.backward()
        optimizer.step()
        
        sync()
        t1 = time.time()
        
        total_time_ms += (t1 - t0) * 1000

    avg_step_ms = total_time_ms / active_steps
    return avg_step_ms


def main() -> None:
    print("==================================================")
    print("       GPT-2 Training Speed Benchmarks            ")
    print("==================================================")

    # Check device availability
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    elif torch.cuda.is_available():
        devices.append("cuda")

    print(f"Available devices: {devices}")

    # Benchmark configurations
    # Format: (device, use_amp, use_compile)
    configs = []
    
    # Run CPU configs
    configs.append(("cpu", False, False)) # CPU FP32 Eager
    
    # Run MPS/CUDA configs if available
    for d in devices:
        if d == "cpu":
            continue
        configs.append((d, False, False)) # Device FP32 Eager
        configs.append((d, True, False))  # Device FP16 Eager (AMP)
        configs.append((d, False, True))  # Device FP32 Compiled
        configs.append((d, True, True))   # Device FP16 Compiled (AMP)

    results = []
    B, T = 4, 256
    tokens_per_step = B * T

    for device, use_amp, use_compile in configs:
        config_name = f"{device.upper()} | {'FP16 (AMP)' if use_amp else 'FP32'} | {'Compiled' if use_compile else 'Eager'}"
        print(f"\nBenchmarking: {config_name}...")
        try:
            # Warm up for 5 steps, benchmark for 15 steps (compile takes time on first steps)
            warmup = 15 if use_compile else 5
            steps = 20
            avg_step_ms = benchmark_run(device, use_amp, use_compile, warmup_steps=warmup, active_steps=steps)
            tokens_per_sec = tokens_per_step / (avg_step_ms / 1000.0)
            
            results.append({
                "config": config_name,
                "time_ms": avg_step_ms,
                "throughput": tokens_per_sec,
                "status": "SUCCESS"
            })
            print(f"-> Step time: {avg_step_ms:.2f} ms | Throughput: {tokens_per_sec:.1f} tok/sec")
        except Exception as e:
            results.append({
                "config": config_name,
                "time_ms": 0.0,
                "throughput": 0.0,
                "status": f"FAILED: {e}"
            })
            print(f"-> Benchmark failed: {e}")

    # Print results summary table in markdown
    print("\n\n==================================================")
    print("               BENCHMARK RESULTS                  ")
    print("==================================================")
    print("| Configuration | Avg Step Time (ms) | Throughput (tokens/sec) | Status |")
    print("|---|---|---|---|")
    for r in results:
        if r["status"] == "SUCCESS":
            print(f"| {r['config']} | {r['time_ms']:.2f} ms | {r['throughput']:.1f} tok/s | {r['status']} |")
        else:
            print(f"| {r['config']} | N/A | N/A | {r['status']} |")
    print("==================================================\n")


if __name__ == "__main__":
    main()
