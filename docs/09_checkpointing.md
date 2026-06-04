# 09. Checkpointing: Serialization & State Restoration

This document details the serialization, parameter mapping, and initialization requirements for saving and restoring model states.

---

## 1. What Goes Into a Checkpoint?

A model checkpoint must contain more than just the model weights. To resume training seamlessly, we must also save the optimizer state (such as momentum vectors) and the configuration details.

In [train.py:L254-L264](file:///Users/ahadk9/Projects/Andrej/train.py#L254-L264), we construct the checkpoint dictionary:
```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "config": config,
    "step": step,
    "val_loss": val_loss_accum
}
torch.save(checkpoint, checkpoint_path)
```

### Components:
- **`model.state_dict()`**: A dictionary mapping each parameter layer name (e.g. `transformer.h.0.attn.c_attn.weight`) to its corresponding tensor.
- **`optimizer.state_dict()`**: Contains the tracking states of the AdamW optimizer (specifically, the first moment `exp_avg` and second moment `exp_avg_sq` tensors for each parameter).
- **`config`**: The `GPTConfig` configuration dataclass. This ensures that when we load the checkpoint later, we construct the model with the exact same hyperparameters (layer count, head count, etc.).
- **`step`**: The training step index, allowing us to resume the learning rate scheduler and dataset position.

---

## 2. Safe and Deterministic Weight Restoration

When loading a checkpoint, we must reconstruct the model architecture before applying the saved weights:

In [hellaswag.py:L168-L176](file:///Users/ahadk9/Projects/Andrej/hellaswag.py#L168-L176):
```python
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
config = checkpoint["config"]
config.dropout = 0.0  # Disable dropout for evaluation
model = GPT(config)
model.load_state_dict(checkpoint["model"])
model.to(device)
```

### Critical Implementation Details:
1. **`map_location=device`**: Forces PyTorch to load tensors directly onto the target hardware device (e.g. MPS or CPU). Without this parameter, loading a checkpoint saved on a CUDA GPU on a machine with only MPS/CPU will cause a crash due to missing drivers.
2. **`weights_only=False`**: Since `GPTConfig` is a custom Python class saved within the pickle file, we must set `weights_only=False` to allow PyTorch to unpickle and reconstruct the configuration object.
3. **Deterministic CPU-First Initialization**: 
   When initializing a new model from scratch, random weights are generated using seed values:
   ```python
   torch.manual_seed(42)
   model = GPT(config).to(device)
   ```
   *Rule*: Always initialize the model on the CPU *before* moving it to a GPU or MPS device using `.to(device)`. Different hardware backends can generate different sequences of random numbers from the same seed. By initializing on the CPU first, we guarantee that the initial model weights are identical across all machines.
