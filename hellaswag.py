"""
HellaSwag evaluation script.
Downloads a tiny subset of the HellaSwag multiple-choice benchmark validation dataset,
tokenizes it, and evaluates model predictions by computing completion log-likelihoods.
"""

import json
import os
import urllib.request
import torch
import torch.nn.functional as F
import tiktoken

from model import GPT


# URL for raw HellaSwag validation dataset
HELLASWAG_VAL_URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
LOCAL_FILE = "hellaswag_val_subset.jsonl"


def download_hellaswag_subset(num_lines: int = 50) -> None:
    """
    Downloads a small subset (first N lines) of the HellaSwag validation set.
    """
    if os.path.exists(LOCAL_FILE):
        return
        
    print(f"Downloading first {num_lines} examples of HellaSwag val dataset...")
    try:
        req = urllib.request.Request(
            HELLASWAG_VAL_URL, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response:
            with open(LOCAL_FILE, "w", encoding="utf-8") as f:
                for _ in range(num_lines):
                    line = response.readline().decode("utf-8")
                    if not line:
                        break
                    f.write(line)
        print(f"Saved subset to {LOCAL_FILE}")
    except Exception as e:
        print(f"Failed to download HellaSwag: {e}. Creating a fallback local mock dataset.")
        # Fallback mock dataset for local execution when network fails
        mock_data = [
            {
                "ctx": "A man is playing a basketball game. He dribbles the ball towards the basket and",
                "endings": [
                    "shoots the ball, scoring a three-pointer.",
                    "eats a sandwich on the court.",
                    "starts swimming in the pool.",
                    "reads a book silently."
                ],
                "label": 0
            },
            {
                "ctx": "A chef is in the kitchen preparing dinner. He chops some onions, adds them to a hot pan, and",
                "endings": [
                    "runs out of the restaurant screaming.",
                    "stirs them with a wooden spoon as they sizzle.",
                    "throws the pan out of the window.",
                    "turns off the lights and goes to sleep."
                ],
                "label": 1
            }
        ]
        with open(LOCAL_FILE, "w", encoding="utf-8") as f:
            for item in mock_data:
                f.write(json.dumps(item) + "\n")
        print(f"Created fallback mock dataset in {LOCAL_FILE}")


@torch.no_grad()
def evaluate_hellaswag(model: GPT, device: str) -> float:
    """
    Evaluates the model on the local HellaSwag validation subset.
    Returns the accuracy (fraction of correct predictions).
    """
    download_hellaswag_subset()
    
    enc = tiktoken.get_encoding("gpt2")
    model.eval()
    
    examples = []
    with open(LOCAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
            
    correct = 0
    total = len(examples)
    
    print(f"Evaluating model on {total} HellaSwag examples...")
    
    for i, ex in enumerate(examples):
        # Context (prompt prefix)
        ctx = ex["ctx"]
        # Options
        endings = ex["endings"]
        # Ground truth option index
        label = int(ex["label"])
        
        ctx_tokens = enc.encode(ctx)
        
        # We will compute the log-likelihood (sum of log probs) for each option
        option_log_probs = []
        
        for ending in endings:
            # Construct sequence: context + ending
            ending_tokens = enc.encode(ending)
            full_tokens = ctx_tokens + ending_tokens
            
            x = torch.tensor([full_tokens[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([full_tokens[1:]], dtype=torch.long, device=device)
            
            # Forward pass to get logits
            logits, _ = model(x)
            
            # Compute loss specifically for the ending tokens only
            # The context tokens logits are ignored since they are context
            num_ctx = len(ctx_tokens)
            
            # Slice logits and targets to align only with the completion tokens
            # Targets start at index 1, so the completion targets start at index num_ctx - 1
            logits_ending = logits[0, num_ctx - 1:]
            targets_ending = y[0, num_ctx - 1:]
            
            # Log probabilities (log_softmax)
            log_probs = F.log_softmax(logits_ending, dim=-1)
            
            # Gather log probs for the target tokens
            target_log_probs = log_probs[torch.arange(len(targets_ending)), targets_ending]
            
            # Average log probability for the option
            mean_log_prob = target_log_probs.mean().item()
            option_log_probs.append(mean_log_prob)
            
        # Select ending with the highest average log likelihood (minimum negative loss)
        pred = option_log_probs.index(max(option_log_probs))
        
        if pred == label:
            correct += 1
            
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f" -> Example {i+1:2d}/{total} | Pred: {pred} | Label: {label} | Correct: {pred == label} | Logprobs: {[f'{lp:.2f}' for lp in option_log_probs]}")
            
    accuracy = correct / total
    print(f"\nHellaSwag val subset accuracy: {accuracy * 100:.2f}% ({correct}/{total})\n")
    return accuracy


import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on HellaSwag benchmark.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to local checkpoint file (e.g. checkpoint_step_40.pt)")
    parser.add_argument("--model_type", type=str, default="gpt2", help="Pretrained model type if no checkpoint is provided")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    print(f"Using device: {device}")
    
    if args.checkpoint is not None:
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = checkpoint["config"]
        # Disable dropout for evaluation
        config.dropout = 0.0
        model = GPT(config)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
    else:
        model = GPT.from_pretrained(args.model_type).to(device)
        
    evaluate_hellaswag(model, device)


if __name__ == "__main__":
    main()
