"""
Training script for the custom GPT-2 model (Learning Edition).
Implements a production-grade training loop featuring:
- CPU-deterministic initialization with MPS/CUDA support
- AdamW optimizer with parameter group splitting (weight decay applied only to 2D parameters)
- Cosine learning rate scheduling with linear warmup
- Gradient accumulation simulating larger batch sizes
- Gradient clipping preventing gradient explosion
- Mixed precision training (AMP autocasting) and torch.compile support
"""

import argparse
import math
import time
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import tiktoken

from model import GPT, GPTConfig
from data import DataLoaderLite


def overfit_single_batch(device: str) -> None:
    """
    Sanity check: train the model on a single batch for 120 steps.
    The loss should decrease monotonically and drop close to 0 (< 0.01).
    """
    print("\n--- Running Overfit Single Batch Test ---")
    
    # Set seed for deterministic CPU-first initialization across all backends
    torch.manual_seed(42)
    
    # Configuration for ~10M model
    config = GPTConfig(
        n_layer=6,
        n_head=6,
        n_embd=384,
        block_size=256,
        vocab_size=50257,
        dropout=0.0
    )
    
    # Initialize on CPU, then transfer to target device to guarantee matching weights
    model = GPT(config).to(device)
    model.train()
    
    # Standard AdamW optimizer with learning rate suited for fast overfitting with custom init
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    
    # Loader
    B, T = 4, 256
    loader = DataLoaderLite(B=B, T=T)
    
    # Grab one batch and hold it constant
    x, y = loader.next_batch()
    x, y = x.to(device), y.to(device)
    
    print(f"Input batch shape: {x.shape}")
    print("Beginning overfit optimization loop...")
    
    for step in range(120):
        t0 = time.time()
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        
        # Synchronize for timing precision
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
            
        t1 = time.time()
        
        if step % 20 == 0 or step == 119:
            print(f"Step {step:3d} | Loss: {loss.item():.6f} | Step time: {(t1-t0)*1000:.2f}ms")
            
    if loss.item() < 0.01:
        print("-> OVERFIT TEST PASSED ✅ (Loss < 0.01)\n")
    else:
        print("-> OVERFIT TEST FAILED ❌ (Loss >= 0.01)\n")


def configure_optimizers(
    model: nn.Module, 
    weight_decay: float, 
    learning_rate: float, 
    device_type: str
) -> optim.AdamW:
    """
    Configures the AdamW optimizer with weight decay parameters splitting.
    Any 2D parameters (weights in Linears and Embeddings) will decay.
    Any 1D parameters (biases, LayerNorm gains/biases) will not decay.
    """
    # Candidate parameters that require gradients
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    
    # Split into decay and nodecay groups
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]
    
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"Optimizer configuration details:")
    print(f" -> Decayed parameters: {len(decay_params)} tensors, {num_decay_params:,} parameters")
    print(f" -> Non-decayed parameters: {len(nodecay_params)} tensors, {num_nodecay_params:,} parameters")
    
    # Use fused AdamW kernel if available (CUDA only, not supported on CPU/MPS)
    use_fused = (device_type == "cuda") and ("fused" in optim.AdamW.__init__.__code__.co_varnames)
    print(f" -> Using fused AdamW: {use_fused}")
    
    optimizer = optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer


def get_lr(it: int, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """
    Computes learning rate for step 'it' using Cosine Annealing with linear warmup.
    """
    # 1) Linear warmup
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) Post max_steps
    if it > max_steps:
        return min_lr
    # 3) Cosine decay in between
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train custom GPT-2 model.")
    parser.add_argument("--overfit", action="store_true", help="Run the overfit single batch test")
    parser.add_argument("--steps", type=int, default=60, help="Number of training steps")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Device to run on")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision training (autocast FP16)")
    parser.add_argument("--compile", action="store_true", help="Compile model using torch.compile")
    parser.add_argument("--batch_size", type=int, default=8192, help="Target total batch size in tokens")
    
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    if args.overfit:
        overfit_single_batch(device)
        return

    # Configuration for ~10M model
    config = GPTConfig(
        n_layer=6,
        n_head=6,
        n_embd=384,
        block_size=256,
        vocab_size=50257,
        dropout=0.1
    )
    
    # Initialize model on CPU for deterministic initialization, then transfer
    torch.manual_seed(42)
    model = GPT(config).to(device)
    model.train()
    
    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)
        
    # Data pipeline configuration
    # Micro-batch: B=4, T=256 -> 1024 tokens
    B, T = 4, 256
    tokens_per_microbatch = B * T
    
    # Calculate gradient accumulation steps to simulate target batch size
    grad_accum_steps = max(1, args.batch_size // tokens_per_microbatch)
    total_batch_tokens = grad_accum_steps * tokens_per_microbatch
    
    # Dynamic train/val split of input.txt
    print("Preparing dataset splits...")
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    n = len(text)
    train_text = text[:int(n*0.9)]
    val_text = text[int(n*0.9):]
    with open("input_train.txt", "w", encoding="utf-8") as f:
        f.write(train_text)
    with open("input_val.txt", "w", encoding="utf-8") as f:
        f.write(val_text)
        
    train_loader = DataLoaderLite(B=B, T=T, file_path="input_train.txt")
    val_loader = DataLoaderLite(B=B, T=T, file_path="input_val.txt")
    
    # Configure AdamW optimizer with weight decay splitting
    max_lr = 6e-4
    min_lr = 6e-5
    warmup_steps = 10
    max_steps = args.steps
    
    optimizer = configure_optimizers(model, weight_decay=0.1, learning_rate=max_lr, device_type=device)
    
    print("\nStarting training loop...")
    for step in range(args.steps):
        t0 = time.time()
        
        # Every 20 steps, evaluate validation loss, generate sample text, and save checkpoint
        if step % 20 == 0 or step == args.steps - 1:
            model.eval()
            val_loader.current_position = 0  # Reset validation position
            val_loss_accum = 0.0
            val_loss_steps = 10  # Run 10 validation batches
            
            with torch.no_grad():
                for _ in range(val_loss_steps):
                    x_val, y_val = val_loader.next_batch()
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    if args.amp:
                        with torch.amp.autocast(device_type=device, dtype=torch.float16):
                            _, val_loss = model(x_val, y_val)
                    else:
                        _, val_loss = model(x_val, y_val)
                    val_loss_accum += val_loss.item() / val_loss_steps
            
            print(f" -> Validation Loss at step {step}: {val_loss_accum:.4f}")
            
            # Autoregressive generation demonstration
            enc = tiktoken.get_encoding("gpt2")
            start_tokens = enc.encode("Alan Turing determined that")
            x_gen = torch.tensor([start_tokens], dtype=torch.long, device=device)
            # Run model generation
            y_gen = model.generate(x_gen, max_new_tokens=30, temperature=1.0, top_k=50)
            generated_text = enc.decode(y_gen[0].tolist())
            print(f" -> Generated sample at step {step}:")
            print(f"    \"{generated_text.strip()}\"\n")
            
            # Save checkpoint
            checkpoint_path = f"checkpoint_step_{step}.pt"
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "step": step,
                "val_loss": val_loss_accum
            }
            torch.save(checkpoint, checkpoint_path)
            print(f" -> Saved checkpoint to {checkpoint_path}\n")
            
            # Revert model to training mode
            model.train()
            
        # Calculate current learning rate for this step and set it
        lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
        optimizer.zero_grad()
        
        # Accumulate gradients over micro-batches
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            # Wrap forward pass in autocast if mixed precision is enabled
            if args.amp:
                with torch.amp.autocast(device_type=device, dtype=torch.float16):
                    logits, loss = model(x, y)
            else:
                logits, loss = model(x, y)
                
            # Normalize loss to account for accumulation
            loss = loss / grad_accum_steps
            loss_accum += loss.item()
            loss.backward()
            
        # Clip gradient norm to 1.0 to prevent gradient explosion
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Ensure execution finishes before measuring time
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
            
        t1 = time.time()
        dt = (t1 - t0) * 1000  # milliseconds
        tokens_per_sec = total_batch_tokens / (dt / 1000.0)
        
        # Print logs (every step, since the steps represent large batch runs)
        print(
            f"Step {step:3d} | Train Loss: {loss_accum:.4f} | "
            f"Grad Norm: {grad_norm:.4f} | LR: {lr:.2e} | "
            f"dt: {dt:.1f}ms | Throughput: {tokens_per_sec:.1f} tokens/sec"
        )


if __name__ == "__main__":
    main()
