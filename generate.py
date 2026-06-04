"""
Text generation and validation script for custom GPT-2 implementation.
This script performs a logit parity test comparing custom model outputs
against Hugging Face's official GPT-2 model, and provides an interface
to generate text from prompts using loaded pre-trained weights.
"""

import argparse
import torch
import torch.nn.functional as F
import tiktoken

from model import GPT


def run_logit_parity_test(device: str = "cpu") -> bool:
    """
    Run a logit parity test comparing our scratch GPT model
    against Hugging Face's official GPT-2 (124M) model.
    """
    print("\n--- Running Logit Parity Test ---")
    from transformers import GPT2LMHeadModel

    # Load both models
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model_custom = GPT.from_pretrained("gpt2").to(device)

    model_hf.eval()
    model_custom.eval()

    # Encode test prompts
    enc = tiktoken.get_encoding("gpt2")
    prompts = [
        "Hello, I'm a language model,",
        "The quick brown fox jumps over the lazy dog.",
        "Alan Turing determined that computers"
    ]

    all_passed = True

    for prompt in prompts:
        print(f"Testing prompt: '{prompt}'")
        tokens = enc.encode(prompt)
        x = torch.tensor([tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            # Get logits from HF model
            outputs_hf = model_hf(x)
            logits_hf = outputs_hf.logits

            # Get logits from our custom model
            logits_custom, _ = model_custom(x)

        # Compare shape
        if logits_hf.shape != logits_custom.shape:
            print(f"Shape mismatch: HF {logits_hf.shape} vs Custom {logits_custom.shape}")
            all_passed = False
            continue

        # Compute differences
        diff = torch.abs(logits_hf - logits_custom)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"-> Max absolute difference: {max_diff:.2e}")
        print(f"-> Mean absolute difference: {mean_diff:.2e}")

        # Assert parity within tolerance (usually 1e-4 is safe)
        tolerance = 1e-4
        if max_diff < tolerance:
            print("-> PASSED ✅")
        else:
            print(f"-> FAILED ❌ (exceeded tolerance {tolerance:.1e})")
            all_passed = False

    if all_passed:
        print("Logit parity test PASSED successfully! 🎉\n")
    else:
        print("Logit parity test FAILED! ⚠️\n")

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text or verify custom GPT-2 implementation.")
    parser.add_argument("--prompt", type=str, default="Alan Turing determined that", help="Prompt to generate text from")
    parser.add_argument("--model_type", type=str, default="gpt2", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"], help="Pretrained model type")
    parser.add_argument("--max_tokens", type=int, default=30, help="Maximum number of tokens to generate")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature for generation (randomness)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Device to run on")
    parser.add_argument("--run_test", action="store_true", help="Run the logit parity test and exit")
    
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    if args.run_test:
        run_logit_parity_test(device)
        return

    # Load custom model with pretrained weights
    model = GPT.from_pretrained(args.model_type).to(device)
    model.eval()

    # Encode prompt
    enc = tiktoken.get_encoding("gpt2")
    start_tokens = enc.encode(args.prompt)
    x = torch.tensor([start_tokens], dtype=torch.long, device=device)

    print(f"\nGenerating text with prompt: '{args.prompt}'...")
    
    # Generate text
    y = model.generate(x, max_new_tokens=args.max_tokens, temperature=args.temp, top_k=args.top_k)
    
    # Decode and print output
    generated_text = enc.decode(y[0].tolist())
    print("\n--- Generated Output ---")
    print(generated_text)
    print("------------------------\n")


if __name__ == "__main__":
    main()
