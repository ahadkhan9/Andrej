# 03. Transformer Block: Internals & Normalization

This document details the architecture, normalization, and optimization choices within a single Transformer Block in [model.py:L144-L170](file:///Users/ahadk9/Projects/Andrej/model.py#L144-L170).

---

## 1. Pre-LN vs. Post-LN

The location of the Layer Normalization (LN) layer relative to the residual block is critical for training stability.

### Post-LN (Original Transformer / GPT-1):
In the original Transformer, normalization is applied *after* adding the residual connection:
$$x_{l+1} = \text{LayerNorm}(x_l + \text{SubLayer}(x_l))$$
- **Issue**: The gradient path through the residual connections is scaled by LayerNorm at every block. As the depth of the network increases, gradients in early layers become extremely small, requiring a carefully tuned learning rate warmup schedule.

### Pre-LN (GPT-2 & Modern LLMs):
GPT-2 introduced Pre-LN, where normalization is applied to the input *before* passing it to the sub-layer:
$$x_{l+1} = x_l + \text{SubLayer}(\text{LayerNorm}(x_l))$$
This is implemented in our `Block.forward` function:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x
```
- **Why it works**: There is now an clean, unobstructed path from the output layer back to the input layer:
  $$\frac{\partial x_{l+1}}{\partial x_l} = I + \frac{\partial \text{SubLayer}(\text{LayerNorm}(x_l))}{\partial x_l}$$
  Gradients can flow directly through the identity additions ($I$) without being scaled down. This allows training much deeper networks (e.g. 100+ layers) stably from step one.

---

## 2. The MLP Block & GELU Activation

The Causal Self-Attention block mixes information across *tokens* in the sequence. The Multi-Layer Perceptron (MLP) block operates on each token *individually* to extract features.

In [model.py:L41-L68](file:///Users/ahadk9/Projects/Andrej/model.py#L41-L68), the MLP is defined as:
```python
self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
self.gelu = nn.GELU(approximate="tanh")
self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
```

### The 4x Expansion:
The hidden dimension is projected from $C \to 4C$, and then back from $4C \to C$. This expansion (usually called the hidden dimension ratio) provides the model with the capacity to map complex non-linear functions per token.

### GELU Activation:
Instead of ReLU, GPT-2 uses the **Gaussian Error Linear Unit (GELU)** with a `tanh` approximation:
$$\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} (x + 0.044715 x^3)\right)\right)$$

```mermaid
xychart-beta
    title "GELU vs ReLU"
    x-axis [-3, -2, -1, 0, 1, 2, 3]
    y-axis [-1, 0, 1, 2, 3]
    "ReLU": [0, 0, 0, 0, 1, 2, 3]
    "GELU": [-0.01, -0.05, -0.15, 0, 0.85, 1.95, 3]
```

- **Why GELU?**: Unlike ReLU (which has a hard zero gradient for $x < 0$ and can cause "dead neurons"), GELU is smooth and differentiable everywhere. It allows a small, non-zero gradient for negative values, which improves gradient flow during backpropagation.

---

## 3. Residual Scaling (`NANOGPT_SCALE_INIT`)

When using residual connections, the variance of the hidden states increases with the depth of the network. If each block adds noise of variance $\sigma^2$ to the residual stream:
$$\text{Var}(X_L) \approx \text{Var}(X_0) + L \cdot \sigma^2$$
This growth in variance can destabilize training at the final layers.

To counteract this, GPT-2 scales the weights of the output projection layers (`c_proj` in both attention and MLP blocks) during initialization by a factor of $1 / \sqrt{2 \times N_{\text{layers}}}$:

In `model.py`, we flag the projection layers:
```python
self.c_proj.NANOGPT_SCALE_INIT = True
```
And apply the scaling during weight initialization [model.py:L208-L210](file:///Users/ahadk9/Projects/Andrej/model.py#L208-L210):
```python
if hasattr(module, 'NANOGPT_SCALE_INIT'):
    std *= (2 * self.config.n_layer) ** -0.5
```
- **Why $2 \times N_{\text{layers}}$?**: Each block contains **two** residual addition points (one after attention, one after MLP). For a 6-layer model, the scaling factor is $1 / \sqrt{2 \times 6} = 1 / \sqrt{12} \approx 0.288$. This normalization keeps the variance of the activations stable as they propagate deeper through the model.
