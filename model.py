"""
GPT-2 model architecture implemented in PyTorch from scratch.
This module contains the complete definition of the GPT model, including the
configuration, MLP block, Causal Self-Attention block, Transformer Block, and
the main GPT module. It also includes utility methods to load pre-trained weights
from Hugging Face and to generate text autoregressively.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """
    Configuration class for the GPT model.
    Attributes:
        block_size: The maximum sequence length (context window) the model can handle.
        vocab_size: Number of tokens in the vocabulary.
        n_layer: Number of Transformer blocks.
        n_head: Number of attention heads.
        n_embd: Dimensionality of the embeddings and hidden states.
        dropout: Dropout probability.
        bias: Whether to use bias in Linear and LayerNorm layers.
    """
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (FFN) block inside the Transformer block.
    This block projects the hidden states up by a factor of 4, applies GELU,
    and then projects back down to n_embd. It implements the standard Feed-Forward
    Network with residual-ready projections.
    """
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        # Residual projection initialization scaling flag
        self.c_proj.NANOGPT_SCALE_INIT = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP block.
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_embd)
        Returns:
            Output tensor of shape (batch_size, sequence_length, n_embd)
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention block.
    This block implements the standard causal multi-head self-attention mechanism,
    mapping queries, keys, and values. It supports both manual PyTorch attention computation
    and the optimized FlashAttention-based scaled dot product attention.
    """
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by head count"
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Residual projection initialization scaling flag
        self.c_proj.NANOGPT_SCALE_INIT = True
        
        # regularization
        self.dropout = config.dropout
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # causal mask/bias, not a parameter, registered as buffer
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CausalSelfAttention block.
        Args:
            x: Input tensor of shape (B, T, C) where B is batch size,
               T is sequence length, C is config.n_embd.
        Returns:
            Output tensor of shape (B, T, C) after self-attention projection.
        """
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # reshape to (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if hasattr(F, 'scaled_dot_product_attention'):
            # FlashAttention-based dispatch (highly optimized on both CPU, CUDA, and MPS)
            y = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0.0, 
                is_causal=True
            )
        else:
            # Manual causal attention fallback
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            if self.dropout > 0.0 and self.training:
                att = F.dropout(att, p=self.dropout)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            
        # re-assemble all head outputs side-by-side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    """
    A single Transformer Block.
    Composes Pre-Layer Normalization, Causal Self-Attention, Multi-Layer Perceptron,
    and residual connections.
    """
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Transformer Block.
        Uses Pre-LN structure: LayerNorm is applied to the input *before* the
        attention and feed-forward sub-layers. The outputs of these sub-layers are
        added back to the input (residual connection).
        Args:
            x: Input tensor of shape (B, T, C).
        Returns:
            Output tensor of shape (B, T, C).
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    The main GPT-style Language Model.
    Houses token and position embeddings, a list of Transformer blocks,
    a final LayerNorm, and the output language modeling head.
    Supports loading pre-trained weights from Hugging Face GPT-2 checkpoints.
    """
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between input embeddings and output head
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize module weights following standard GPT-2 initialization:
        - nn.Linear weights: normal(0, 0.02), zero bias.
        - nn.Linear weights on residual output projections: scaled by 1/sqrt(2 * n_layer).
        - nn.Embedding weights: normal(0, 0.02).
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the GPT model.
        Args:
            idx: Input token IDs of shape (B, T), integers.
            targets: Optional ground-truth target token IDs of shape (B, T), integers.
        Returns:
            A tuple of (logits, loss).
            logits: Shape (B, T, vocab_size) if targets is provided, or shape (B, 1, vocab_size) during inference.
            loss: Cross-entropy loss scalar tensor if targets are provided, otherwise None.
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"

        # positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device) # shape (T)
        
        # token embeddings
        tok_emb = self.transformer.wte(idx) # shape (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos) # shape (T, n_embd)
        x = tok_emb + pos_emb

        # forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # final layer norm and lm_head projection
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # shape (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str) -> GPT:
        """
        Class method to load weights from Hugging Face's pre-trained GPT-2 checkpoints.
        Valid model_type strings are: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'.
        This method handles configuration alignment, downloading weights via
        transformers, and transposing Conv1D layers to PyTorch's standard nn.Linear layers.
        Args:
            model_type: The Hugging Face model identifier (e.g., 'gpt2').
        Returns:
            An instance of GPT initialized with the loaded pre-trained weights.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pre-trained GPT-2 model: {model_type}")

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        
        # Instantiate our scratch model
        config = GPTConfig(**config_args)
        model = cls(config)
        
        sd = model.state_dict()
        sd_keys = sd.keys()
        # Filter out attention mask buffers, as they are not model parameters
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # Initialize the Hugging Face model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Filter out attention mask / bias keys from HF model
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]

        # Hugging Face GPT-2 weights use Conv1D instead of standard nn.Linear layers.
        # This requires transposing these weight matrices when loading them into our Linear layers.
        transposed_keys = [
            'attn.c_attn.weight', 
            'attn.c_proj.weight', 
            'mlp.c_fc.weight', 
            'mlp.c_proj.weight'
        ]

        assert len(sd_keys_hf) == len(sd_keys), f"State dict key count mismatch: HF={len(sd_keys_hf)} vs Custom={len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(tk) for tk in transposed_keys):
                # Transpose the weight from Conv1D to standard Linear
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"Shape mismatch for {k}: HF={sd_hf[k].shape[::-1]} vs Custom={sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Copy directly
                assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch for {k}: HF={sd_hf[k].shape} vs Custom={sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        print(f"Successfully loaded and aligned weights for {model_type}.")
        return model

    @torch.no_grad()
    def generate(
        self, 
        idx: torch.Tensor, 
        max_new_tokens: int, 
        temperature: float = 1.0, 
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Autoregressive text generation.
        Args:
            idx: Starting token IDs of shape (B, T), integers.
            max_new_tokens: Number of tokens to generate.
            temperature: Softmax scaling parameter. Higher values increase randomness.
            top_k: If provided, only sample from the top k most probable tokens.
        Returns:
            Tensor of shape (B, T + max_new_tokens) containing the original and
            newly generated token IDs.
        """
        for _ in range(max_new_tokens):
            # Crop sequence context length if it exceeds block_size
            idx_cond = idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Scale logits by temperature and get the last step
            logits = logits[:, -1, :] / temperature
            
            # Crop to top_k if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append next token to context
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
