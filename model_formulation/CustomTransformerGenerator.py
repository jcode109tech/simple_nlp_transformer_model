import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from TransformerBlock import TransformerBlock
from PositionEncoding import PositionalEncoding 


class CustomTransformerGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Final layer
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        # Token embedding + positional encoding
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)

        # Final normalization and projection
        x = self.layer_norm(x)
        logits = self.output_projection(x)

        return logits

    def generate(
        self,
        start_tokens: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ):
        """
        Generate text autoregressively
        """
        self.eval()
        generated = start_tokens.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get predictions for the last token
                logits = self(generated)
                next_token_logits = logits[:, -1, :] / temperature

                # Apply top-k sampling
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))[0][:, -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Apply top-p (nucleus) sampling
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float('Inf'))

                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)

        return generated

