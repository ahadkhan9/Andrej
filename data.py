"""
Data loading and tokenization pipeline for training the GPT model.
Contains DataLoaderLite which handles subword tokenization via tiktoken
and generates batched inputs/targets shifted by one position.
"""

from typing import Tuple
import torch
import tiktoken


class DataLoaderLite:
    """
    A lightweight, efficient data loader that tokenizes text using tiktoken
    and yields batches of inputs (x) and shifted targets (y).
    """
    def __init__(self, B: int, T: int, file_path: str = "input.txt") -> None:
        self.B = B
        self.T = T
        self.file_path = file_path

        # Load raw text
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize using gpt2 BPE tokenizer
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        
        print(f"Loaded {len(self.tokens)} tokens from {file_path}")
        print(f"Dataset has {len(self.tokens) // (B * T)} batches per epoch")

        # State variable for tracking position
        self.current_position = 0

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the next batch of input-target pairs.
        Returns:
            x: Input tensor of shape (B, T)
            y: Target tensor of shape (B, T) shifted by 1 relative to x
        """
        B, T = self.B, self.T
        
        # Determine the slice boundary (need B * T + 1 tokens)
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        
        # x are the inputs, y are the targets (shifted by 1)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        # Advance current position by B * T
        self.current_position += B * T

        # Wrap around to start if we exceed dataset boundaries
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = 0

        return x, y
