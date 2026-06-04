# Glossary of Core Terms

This glossary defines the foundational terms, metrics, and concepts used throughout the GPT-2 Reproduction project.

---

### **Activation**
The intermediate tensor outputs computed during the forward pass of a neural network (e.g., outputs of Linear layers, Attention weights, GELU activations). These must be saved in memory during training to calculate gradients during backpropagation.

### **AMP (Automatic Mixed Precision)**
An optimization technique that automatically casts certain operations to lower-precision formats (like float16) to speed up execution and reduce memory usage, while keeping critical parameters (like loss and optimizer variables) in float32 to maintain numerical stability.

### **Autoregressive Decoding**
A text generation method where the model generates one token at a time. Each generated token is appended to the input sequence and used as context to predict the next token.

### **BPE (Byte Pair Encoding)**
A subword tokenization algorithm that iteratively merges the most frequent byte pairs in a training text to build a vocabulary of subwords, bytes, and words.

### **Causal Masking**
A mechanism used in decoder-only transformers that blocks attention heads from looking at future tokens in a sequence. This is done by adding a upper-triangular mask of $-\infty$ values to the attention scores before the softmax operation.

### **FlashAttention**
A hardware-aware attention algorithm that computes exact causal self-attention in fast on-chip GPU SRAM cache instead of slow High Bandwidth Memory (HBM). It reduces the memory footprint of attention from $O(T^2)$ to $O(T)$, converting the operation from memory-bandwidth-bound to compute-bound.

### **GQA (Grouped Query Attention)**
An attention mechanism where multiple Query heads share a single Key and Value head. GQA acts as a middle ground between standard Multi-Head Attention (MHA) and Multi-Query Attention (MQA), offering fast generation speed and small memory footprint while maintaining model capacity.

### **Gradient Accumulation**
A training technique that splits a large target batch size into multiple smaller micro-batches. Gradients are computed and summed over these micro-batches before calling `optimizer.step()`, allowing developers to train models with large effective batch sizes on limited hardware.

### **GELU (Gaussian Error Linear Unit)**
A smooth, non-linear activation function ($x \cdot \Phi(x)$) that scales inputs by their cumulative distribution function under a Gaussian distribution. It allows small, non-zero gradients for negative inputs, preventing dead neurons.

### **KV Cache**
A technique used during inference to store the Key ($K$) and Value ($V$) tensors of all past tokens in memory. This prevents the model from recalculating them for previous tokens when generating new tokens.

### **MQA (Multi-Query Attention)**
An extreme variant of grouped attention where all Query heads share a single, unified Key and Value head, reducing the size of the KV Cache to a minimum.

### **Perplexity**
A evaluation metric for language models defined as the exponentiated cross-entropy loss:
$$\text{PPL} = e^{\mathcal{L}}$$
It represents the effective branching factor of the model (i.e., how many tokens the model is choosing between at any given step). A lower perplexity indicates better performance.

### **Pre-LN**
A transformer layout where Layer Normalization is applied to activations *before* they enter attention or MLP sub-layers. Pre-LN provides a clear gradient path through residual connections, stabilizing training for deep architectures.

### **RoPE (Rotary Position Embedding)**
A relative positioning method that applies a rotation matrix to Query and Key projections in the complex plane. RoPE injects relative position information directly into the attention weights, enabling long context window scaling.

### **SwiGLU**
A gated feed-forward network activation that combines the Swish activation with a Gated Linear Unit, enhancing training stability and representational capacity in modern LLMs.

### **Weight Tying**
A memory-saving parameter sharing technique where the input token embedding matrix `wte` and the output language modeling head `lm_head` share the exact same weight tensor in memory.
