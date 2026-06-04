# 02. Attention: Causal Multi-Head Self-Attention

This document derives the mathematics, complexity, and performance details of the Causal Self-Attention mechanism in [model.py](file:///Users/ahadk9/Projects/Andrej/model.py#L70-L142).

---

## 1. Mathematical Derivation of Q, K, V

The self-attention mechanism allows tokens in a sequence to dynamically weigh their relationships with all other tokens.
Given an input matrix $X$ of shape $(B, T, C)$, we project it into three spaces: Queries ($Q$), Keys ($K$), and Values ($V$).

In our code, this projection is done in a single unified linear layer for efficiency:
```python
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
```
Where `self.c_attn` is `nn.Linear(C, 3C)`.

Mathematically, if $W_{attn}$ is the projection weight matrix of shape $(C, 3C)$:
$$[Q, K, V] = X W_{attn} + b_{attn}$$
We then split this output along the final dimension to extract three separate tensors, each of shape $(B, T, C)$.

### Intuition:
- **Query ($Q$)**: "What am I looking for?" (Representing the current token seeking context).
- **Key ($K$)**: "What information do I contain?" (Representing all tokens that could be paid attention to).
- **Value ($V$)**: "What is my actual content?" (Representing the features that will be extracted once attention weights are computed).

---

## 2. Multi-Head Attention (MHA)

Instead of performing attention once on the full embedding dimension $C$, we partition $C$ into $H$ multiple "heads" of dimension $D = C / H$. In our configuration, $C = 384$ and $H = 6$, giving $D = 64$.

We reshape the $Q$, $K$, and $V$ matrices:
$$\text{Shape: }(B, T, C) \xrightarrow{\text{Reshape \& Transpose}} (B, H, T, D)$$
This permits parallel computations across all heads:
```python
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
```

The dot product of $Q$ and $K^T$ computes similarity scores between every query and key token:
$$A = \frac{Q K^T}{\sqrt{D}}$$
The division by $\sqrt{D}$ (which is $\sqrt{64} = 8$) is the **scaling factor**. If $D$ is large, dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients.

---

## 3. Causal Masking

Since our language model is autoregressive (predicting the next token given past tokens), we must prevent tokens from looking into the future.
We enforce this causal structure by setting future attention logits to $-\infty$ before applying softmax:

$$A_{masked} = A + M$$
Where $M$ is a causal mask matrix where:
$$M_{i,j} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

In the manual fallback code:
```python
self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)))
# ...
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
```
When softmax is applied, $e^{-\infty} = 0$, completely zeroing out any information leakage from future tokens.

```mermaid
matrix
  0.8   -inf  -inf  -inf
  0.4   0.9   -inf  -inf
  0.1   0.2   0.7   -inf
  0.3   0.5   0.2   0.8
```

---

## 4. Why Attention is $O(T^2)$

Let's look at the shape of the attention score matrix:
$$\text{Attention Matrix } A = \text{Softmax}\left(\frac{Q K^T}{\sqrt{D}}\right)$$
The matrix multiplication $Q K^T$ multiplies a matrix of shape $(T, D)$ by a matrix of shape $(D, T)$.
This results in a square matrix of shape $(T, T)$ for every batch and head.

- **Computation Complexity**: The matrix multiplication requires $T \times T \times D$ multiplications. Thus, the compute cost scales as $O(T^2 \cdot D)$.
- **Memory Complexity**: Storing the attention matrix $A$ requires saving $T^2$ floating-point numbers. For a context length of $T = 256$, this is $65,536$ values. If $T = 8192$, it requires $67,108,864$ values per head!

This quadratic memory explosion is why long context windows are incredibly expensive.

---

## 5. FlashAttention (Scaled Dot Product Attention)

To bypass the high memory cost of storing the $(T, T)$ matrix in GPU High Bandwidth Memory (HBM), PyTorch provides `torch.nn.functional.scaled_dot_product_attention` (SDPA). This is enabled in our model:

```python
if hasattr(F, 'scaled_dot_product_attention'):
    y = F.scaled_dot_product_attention(
        q, k, v, 
        attn_mask=None, 
        dropout_p=self.dropout if self.training else 0.0, 
        is_causal=True
    )
```

### How it works:
Instead of computing the intermediate $(T, T)$ attention matrix, writing it to HBM, reading it back to compute softmax, and reading it again to multiply by $V$, FlashAttention computes attention incrementally in GPU/MPS SRAM (on-chip cache).
- **Memory footprint**: Reduced from $O(T^2)$ to $O(T)$ in terms of HBM writes.
- **Compute speedup**: Up to 2-4x speedup due to avoiding memory read/write bottlenecks.
- **Under the hood**: Since `is_causal=True` is set, PyTorch automatically constructs and optimizes the causal mask operations.
