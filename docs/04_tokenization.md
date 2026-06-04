# 04. Tokenization: Byte Pair Encoding (BPE)

This document explains the concepts, vocabulary representation, and execution of tokenization within the model using the `tiktoken` library.

---

## 1. Byte Pair Encoding (BPE) Theory

Computers do not understand raw characters or words directly. We use a **tokenizer** to translate raw text strings into lists of integer token IDs.

GPT-2 uses **Byte Pair Encoding (BPE)**, a subword tokenization algorithm. BPE balances character-level representation (small vocabulary, long sequence length) with word-level representation (large vocabulary, short sequence length).

### How BPE works:
1. **Initialize**: Treat all individual bytes ($256$ possibilities) as the base vocabulary.
2. **Frequency Counting**: Read the training corpus and count frequencies of adjacent byte pairs.
3. **Merge**: Merge the most frequent byte pair (e.g. `['t', 'h']` $\to$ `'th'`) and add it to the vocabulary.
4. **Repeat**: Repeat this merging process iteratively until the vocabulary reaches the target size ($50,257$).

- **Why bytes?**: By using bytes as the base vocabulary, BPE can represent *any* unicode character. This avoids the Out-Of-Vocabulary (OOV) problem entirely; if a word has never been seen before, it is broken down into its individual byte components (e.g. `\xe2\x8c\x98`).

---

## 2. Vocabulary Sizing and Special Tokens

The vocabulary size for GPT-2 is **50,257**:
- **50,000** merged subword tokens.
- **256** base byte tokens.
- **1** special token: `<|endoftext|>` (index `50256`), which marks boundaries between documents.

In our dataset loading pipeline [data.py:L26-L29](file:///Users/ahadk9/Projects/Andrej/data.py#L26-L29):
```python
# Tokenize using gpt2 BPE tokenizer
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(text)
self.tokens = torch.tensor(tokens, dtype=torch.long)
```

And in validation text generation [train.py:L245-L251](file:///Users/ahadk9/Projects/Andrej/train.py#L245-L251):
```python
enc = tiktoken.get_encoding("gpt2")
start_tokens = enc.encode("Alan Turing determined that")
x_gen = torch.tensor([start_tokens], dtype=torch.long, device=device)
y_gen = model.generate(x_gen, max_new_tokens=30, temperature=1.0, top_k=50)
generated_text = enc.decode(y_gen[0].tolist())
```

---

## 3. Computational and Parameter Implications

The vocabulary size has a major impact on the model's footprint:
- **Embedding Matrix Weight Size**: The token embedding matrix `wte` requires $V \times C$ weights.
  $$50,257 \times 384 \approx 19.3\text{ Million weights}$$
- **Language Model Head Weight Size**: The classification layer `lm_head` requires $C \times V$ weights.
  $$384 \times 50,257 \approx 19.3\text{ Million weights}$$

Combined, they would require **$38.6$ Million parameters**, which is almost **80%** of our model parameters!

### Weight Tying:
To reduce this footprint, we tie the weights of `wte` and `lm_head`:
```python
self.transformer.wte.weight = self.lm_head.weight
```
This forces them to share the exact same physical memory layout, saving $19.3\text{ Million}$ parameters.
- **Mathematical justification**: Embeddings map tokens to representations ($V \to C$). The output head maps representations to token probabilities ($C \to V$). Tying these layers enforces a mathematical symmetry: tokens that are close in representation space will also receive similar output probabilities.
