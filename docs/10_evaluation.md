# 10. Evaluation: Validation & Downstream Benchmarks

This document details the evaluation loops and downstream benchmarking logic implemented in [train.py](file:///Users/ahadk9/Projects/Andrej/train.py) and [hellaswag.py](file:///Users/ahadk9/Projects/Andrej/hellaswag.py).

---

## 1. Validation Loss Loop

To track overfitting, the training loop periodically pauses and runs a validation loop:
```python
val_loader.current_position = 0  # Reset validation position
val_loss_accum = 0.0
val_loss_steps = 10  # Run 10 validation batches
```
We take the average validation loss over 10 batches. Because we call `model.eval()` before the loop, all dropout layers are deactivated, ensuring the loss metrics are clean and deterministic.

---

## 2. HellaSwag Multiple-Choice Benchmark

The **HellaSwag** benchmark measures a model's common sense reasoning capabilities. Each example consists of a context sentence followed by four possible completions. Only one completion is correct.

```text
Context: A man is playing a basketball game. He dribbles the ball towards the basket and...
Ending 0: shoots the ball, scoring a three-pointer. [Correct]
Ending 1: eats a sandwich on the court.
Ending 2: starts swimming in the pool.
Ending 3: reads a book silently.
```

### Evaluation Strategy:
Instead of asking the model to write out the completion, we use the model's logits to compute the **log-likelihood** of each ending. The ending that yields the highest average log-likelihood (lowest negative cross-entropy loss) is chosen as the model's prediction.

---

## 3. Completion Log-Likelihood Alignment Math

When evaluating a completion option, we concatenate the context tokens and the ending tokens:
$$\text{Full sequence} = [\underbrace{x_0, x_1, \dots, x_{C-1}}_{\text{Context (Length } L_{ctx}\text{)}}, \underbrace{x_C, \dots, x_{N-1}}_{\text{Completion (Length } L_{comp}\text{)}}]$$

We only want to evaluate how well the model predicts the **completion** tokens. We must ignore the prediction errors (losses) on the context tokens, because those tokens were already provided to the model as input.

In [hellaswag.py:L121-L136](file:///Users/ahadk9/Projects/Andrej/hellaswag.py#L121-L136):
```python
num_ctx = len(ctx_tokens)

# Slice logits and targets to align only with the completion tokens
logits_ending = logits[0, num_ctx - 1:]
targets_ending = y[0, num_ctx - 1:]

# Log probabilities (log_softmax)
log_probs = F.log_softmax(logits_ending, dim=-1)

# Gather log probs for the target tokens
target_log_probs = log_probs[torch.arange(len(targets_ending)), targets_ending]

# Average log probability for the option
mean_log_prob = target_log_probs.mean().item()
```

### Why `num_ctx - 1`?
In standard language modeling, the logit at index $i$ predicts the token at index $i+1$. 
- The last token of the context is at index `num_ctx - 1`.
- The prediction logit for this token predicts the *first* token of the completion (index `num_ctx`).
- Therefore, to evaluate the completion, we slice the logits starting at `num_ctx - 1`.

We compute the log-softmax across the vocabulary dimension for these sliced logits, gather the log probabilities of the actual target completion tokens, and average them. The option with the maximum average log probability is selected.
