# 06. Training Loop: Mechanics & Schedule

This document breaks down the forward and backward passes, optimizer math, and schedules defined in [train.py](file:///Users/ahadk9/Projects/Andrej/train.py).

---

## 1. Anatomy of a Step

Each training step follows four phases:
1. **Forward Pass**: The model computes output logits and computes cross-entropy loss against target labels:
   ```python
   logits, loss = model(x, y)
   ```
2. **Backward Pass**: PyTorch computes the gradients of the loss with respect to all trainable parameters using backpropagation (reverse-mode autodiff):
   ```python
   loss.backward()
   ```
3. **Gradient Clipping**: Restricts the magnitude of gradients to prevent parameter updates from blowing up:
   ```python
   grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   ```
4. **Optimizer Update**: The optimizer adjusts the model weights based on the accumulated gradients, and then zeroes out gradients for the next step:
   ```python
   optimizer.step()
   optimizer.zero_grad()
   ```

---

## 2. Gradient Accumulation Math

To train a model stably, we need a large batch size (e.g. $8,192$ tokens per batch). However, a batch of that size may not fit into local GPU/MPS memory.
We solve this using **gradient accumulation**, which splits a large batch into multiple smaller "micro-batches" and sums their gradients before taking an optimizer step.

Mathematically, the average loss over a batch $D$ partitioned into $N$ micro-batches $d_i$ is:
$$\mathcal{L}(D) = \frac{1}{N} \sum_{i=1}^N \mathcal{L}(d_i)$$

Therefore, the gradient of the total loss is the average of the gradients of the micro-batches:
$$\nabla \mathcal{L}(D) = \frac{1}{N} \sum_{i=1}^N \nabla \mathcal{L}(d_i)$$

To match this mathematically, we divide each micro-batch loss by $N$ (`grad_accum_steps`) before calling `.backward()`:

In [train.py:L278-L292](file:///Users/ahadk9/Projects/Andrej/train.py#L278-L292):
```python
loss_accum = 0.0
for micro_step in range(grad_accum_steps):
    x, y = train_loader.next_batch()
    # ...
    logits, loss = model(x, y)
    loss = loss / grad_accum_steps
    loss_accum += loss.item()
    loss.backward()
```
Calling `loss.backward()` accumulates (adds) the scaled gradients directly into the `.grad` attributes of the model parameters. After $N$ steps, we call `optimizer.step()`.

---

## 3. AdamW and Weight Decay Splitting

The **AdamW** optimizer combines adaptive learning rates with **weight decay** (L2 regularization):
$$\theta_{t+1} = \theta_t - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$
Where $\theta$ represents parameters, $\hat{m}_t$ and $\hat{v}_t$ are bias-corrected first and second moment estimates, $\eta_t$ is learning rate, and $\lambda$ is the weight decay coefficient (set to $0.1$).

### Weight Decay Parameter Splitting:
Applying weight decay to *every* parameter can hurt performance. Specifically:
- **Weights in Linear/Embedding layers** (2D tensors) represent structural transformations and *should* be decayed to prevent overfitting.
- **Biases and LayerNorm scales/biases** (1D tensors) represent shift and scale baselines and should *not* be decayed.

In [train.py:L87-L122](file:///Users/ahadk9/Projects/Andrej/train.py#L87-L122):
```python
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

optim_groups = [
    {"params": decay_params, "weight_decay": weight_decay},
    {"params": nodecay_params, "weight_decay": 0.0}
]
```

---

## 4. Learning Rate Schedule

Training starts with a low learning rate, warms up to a peak value, and then decays to a minimum value following a cosine curve:

```mermaid
xychart-beta
    title "Cosine Learning Rate Schedule"
    x-axis [0, 10, 20, 30, 40, 50, 60]
    y-axis [0, 1e-4, 6e-4]
    "Learning Rate": [0, 6e-4, 5.2e-4, 3.8e-4, 2.2e-4, 1e-4, 6e-5]
```

### Schedule Mechanics:
1. **Warmup Phase (Steps $0 \dots 10$)**: Linearly ramp up learning rate to prevent early training instability.
2. **Cosine Decay Phase (Steps $10 \dots 60$)**: Smoothly decay the learning rate to find local minima.
3. **Floor**: Hold learning rate constant at `min_lr` ($6 \times 10^{-5}$) once decay is complete.

This is calculated step-by-step using `get_lr` [train.py:L125-L139](file:///Users/ahadk9/Projects/Andrej/train.py#L125-L139).
