# 08. Generation: Autoregressive Decoding

This document details the autoregressive generation loop, sampling parameters, and logic implemented in [model.py:L327-L370](file:///Users/ahadk9/Projects/Andrej/model.py#L327-L370).

---

## 1. The Autoregressive Generation Loop

Language models generate text one token at a time. The newly generated token is appended to the input sequence, and the updated sequence is fed back into the model to predict the next token. This process is called **autoregressive decoding**.

```mermaid
flowchart TD
    Prompt[Prompt Tokens: idx] -->|Forward Pass| Predict[Predict Next Token Logits]
    Predict -->|Sample| NextToken[idx_next]
    NextToken -->|Concat| Append[idx = [idx, idx_next]]
    Append -->|Check Length| LengthLimit{Reached Max Tokens?}
    LengthLimit -->|No| Prompt
    LengthLimit -->|Yes| End[Return Completed Sequence]
```

### Context Length Constraints:
As we append tokens, the sequence length increases. Because the model's position embedding matrix `wpe` has a fixed size of `block_size` (256 in our configuration), the model cannot accept sequences longer than `block_size`.

To prevent crashes, we crop the context window before passing it to the forward pass:
```python
idx_cond = idx[:, -self.config.block_size:]
```

---

## 2. Temperature Scaling

During generation, the raw output scores from the model (`logits`) are converted into a probability distribution using the softmax function:
$$P(x_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$
Where $z_i$ is the logit score for token $i$.

We use **temperature scaling** ($T$) to control the randomness of the generated text by scaling the logits before applying softmax:
$$P(x_i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

In our code:
```python
logits = logits[:, -1, :] / temperature
```

### Temperature settings:
- **$T = 1.0$**: Standard sampling. Probabilities are unchanged.
- **$0 < T < 1.0$**: Compresses the distribution. High-probability tokens become much more likely, and low-probability tokens are suppressed, leading to more deterministic and repetitive text.
- **$T > 1.0$**: flattens the distribution. The differences between logits are minimized, making token selection more uniform and increasing randomness/creativity.

---

## 3. Top-k Filtering

Even with low temperature, a model might sample a highly inappropriate or nonsensical token from the long tail of the distribution. To prevent this, we use **top-k filtering** to restrict sampling to only the $k$ most probable tokens:

In `model.py`:
```python
if top_k is not None:
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float('Inf')
```

### Mechanics:
1. Find the logit value of the $k$-th most probable token.
2. Set all logits smaller than this threshold to $-\infty$.
3. When softmax is applied, these filtered tokens receive a probability of 0, ensuring they are never sampled.

---

## 4. Sampling from the Distribution

Once logits are filtered and normalized to probabilities via softmax:
```python
probs = F.softmax(logits, dim=-1)
```
We use `torch.multinomial` to sample a token ID based on these probabilities:
```python
idx_next = torch.multinomial(probs, num_samples=1)
```
- **Why not Argmax?**: Selecting the most probable token (greedy decoding) often leads to repetitive loops (e.g. "and the and the and the"). Multinomial sampling introduces natural variation, producing more coherent and human-like text.
