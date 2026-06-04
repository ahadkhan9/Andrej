# 12. Interview Preparation: Q&A Matrix

This document provides a graded set of technical questions commonly asked in GenAI, LLM engineering, and ML system interviews, referencing specific patterns in our codebase.

---

## 1. Beginner Level

### Q: Why do we tie the weights of the token embedding matrix (`wte`) and the output projection head (`lm_head`)?
- **Ideal Answer**: Weight tying shares the parameters between the input lookup table and the output classification layer. It reduces the overall parameter footprint of the model by $V \times C$ (approx. $19.3\text{M}$ parameters in our model). Mathematically, it works because word embeddings and language model output projection represent symmetric mappings ($V \to C$ and $C \to V$). Tying them acts as a regularizer, forcing semantic representation alignment.
- **Common Wrong Answer**: "We tie them so that the model runs twice as fast." (Weight tying saves memory, but doesn't change the number of matrix multiplications or ALU compute steps).
- **Follow-up Q**: How does this impact the optimizer update?
  - *Answer*: Gradients from both the forward embedding lookup and the backward cross-entropy projection accumulate into the same shared weight tensor. The optimizer updates this shared tensor once per step.

---

## 2. Mid Level

### Q: Why do we divide the loss by the number of gradient accumulation steps before calling `loss.backward()`?
- **Ideal Answer**: PyTorch loss functions (like `F.cross_entropy`) compute the *mean* loss over the batch. When using gradient accumulation, we split a target batch into $N$ micro-batches. If we simply call `loss.backward()` on each micro-batch, the gradients will accumulate (sum up) over all $N$ steps. This makes the effective gradient step size $N$ times larger than if we processed the entire batch at once. Dividing each micro-batch loss by $N$ normalizes the gradients, ensuring the accumulated updates represent the true average over the entire target batch.
- **Common Wrong Answer**: "We divide it to prevent the gradients from becoming zero due to underflow."
- **Follow-up Q**: If we forgot to divide the loss by the accumulation steps, what would happen?
  - *Answer*: The model would update parameters using a step size scaled by $N$. This is equivalent to scaling the learning rate by $N$, which typically causes training to diverge and loss to blow up.

---

## 3. Senior Level

### Q: What is the purpose of scaling the initialization weights of the residual projection layers by $1 / \sqrt{2 \times N_{\text{layers}}}$?
- **Ideal Answer**: In a transformer block with residual connections ($x_{l+1} = x_l + f(x_l)$), the variance of the hidden states grows linearly with the number of layers: $\text{Var}(x_L) \approx \text{Var}(x_0) + 2L \cdot \text{Var}(f(x))$. If left unchecked, this variance growth destabilizes training as activations flow deeper. By scaling the output projection layers (`c_proj`) by $1 / \sqrt{2 \times N_{\text{layers}}}$, we normalize the variance growth, ensuring the signal variance remains stable throughout the model's depth.
- **Common Wrong Answer**: "It is a standard scaling factor from the Xavier/Glorot initialization framework to match input and output channels." (Xavier matches variance across input/output channels, whereas this scaling scales specifically across *network depth* due to residual accumulation).
- **Follow-up Q**: Why is the scaling factor $\sqrt{2 \times N}$ instead of $\sqrt{N}$?
  - *Answer*: Because each transformer block contains **two** residual addition layers: one after causal attention, and one after the MLP.

---

## 4. Staff Level

### Q: Explain how FlashAttention circumvents the $O(T^2)$ memory bottleneck of self-attention.
- **Ideal Answer**: The classic self-attention bottleneck is not compute, but GPU memory bandwidth. Standard attention computes the intermediate score matrix $S = \text{Softmax}(Q K^T / \sqrt{D})$ of shape $(B, H, T, T)$, writes it to GPU High Bandwidth Memory (HBM), reads it back to apply softmax, and reads it again to multiply by $V$. 
  FlashAttention reorganizes the attention computation by tiling the input matrices $Q, K, V$ into blocks that fit within the GPU's fast, on-chip SRAM cache. It computes attention incrementally using online softmax normalization (tracking scaling factors dynamically), writing only the final output matrix of shape $(B, T, C)$ back to HBM. This reduces HBM memory access from $O(T^2)$ to $O(T)$, converting the attention operation from a memory-bandwidth-bound operation to a compute-bound operation.
- **Common Wrong Answer**: "FlashAttention uses sparse matrices or approximations to drop low-attention connections." (FlashAttention computes the *exact* same mathematical attention values as standard attention; it is a hardware-aware IO optimization, not an approximation).
- **Follow-up Q**: How does FlashAttention affect the backward pass?
  - *Answer*: It recalculates the intermediate attention matrix on-the-fly during the backward pass using stored SRAM blocks, eliminating the need to store the massive forward attention matrix in memory.
