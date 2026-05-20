import torch
from torch import nn
from .modules import Linear, RMSNorm, Embedding, TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.context_length = context_length

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[-1]
        if seq_len > self.context_length:
            raise ValueError(f'Input sequence length {seq_len} exceeds context length {self.context_length}')

        x = self.token_embeddings(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits