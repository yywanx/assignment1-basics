import math
import torch
from torch import nn
from einops import einsum, rearrange
from .utils import get_d_ff


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')
        return out


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        std = math.sqrt(2 / (num_embeddings + embedding_dim))
        nn.init.trunc_normal_(self.weight)

    def forward(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        x = x / rms
        x = x * self.weight

        return x.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        if not d_ff:
            d_ff = get_d_ff(d_model)

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.silu(self.w1(x))
        value = self.w3(x)
        return self.w2(gate * value)

    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ROPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        super().__init__()

        assert d_k % 2 == 0, 'd_k must be even'
        self.d_k = d_k

        i = torch.arange(0, max_seq_len, device=device)
        k = torch.arange(0, d_k, 2, device=device)

        inv_freq = 1 / theta ** (k / d_k)
        freq = einsum(i, inv_freq, 'dim, d -> dim d')

        cos = torch.cos(freq)
        sin = torch.sin(freq)

        self.register_buffer('cos_cached', cos, persistent=False)
        self.register_buffer('sin_cached', sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None
    ) -> torch.Tensor:
        *dims, seq_len, d_k = x.shape
        assert d_k == self.d_k

        even = x[..., 0::2]
        odd = x[..., 1::2]

        if token_positions is None:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        else:
            cos = self.cos_cached[token_positions]
            sin = self.sin_cached[token_positions]

        while cos.ndim < even.ndim:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        new_even = even * cos - odd * sin
        new_odd = even * sin + odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = new_even
        out[..., 1::2] = new_odd

        return out


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_val = torch.max(x, dim=dim, keepdim=True)[0]
    x -= max_val

    x = torch.exp(x)
    x /= torch.sum(x, dim=dim, keepdim=True)

    return x


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    d = query.shape[-1]

    scores = einsum(query, key, '... n d, ... m d -> ... n m')
    scores /= math.sqrt(d)
    
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))

    attention = softmax(scores, dim=-1)

    out = einsum(attention, value, '... n m, ... m d -> ... n d')
    return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model:int,
        num_heads:int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, 'batch seq (head d) -> batch head seq d', head=self.num_heads)
        k = rearrange(k, 'batch seq (head d) -> batch head seq d', head=self.num_heads)
        v = rearrange(v, 'batch seq (head d) -> batch head seq d', head=self.num_heads)

        mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )

        out = scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            mask=mask
        )

        out = rearrange(out, 'batch head seq d -> batch seq (head d)')
        out = self.output_proj(out)

        return out

    
class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(
        self,
        d_model:int,
        num_heads:int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = ROPE(
            theta=theta,
            d_k=d_model // num_heads,
            max_seq_len=max_seq_len,
            device=device
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        seq_len = x.shape[-2]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, 'batch seq (head d) -> batch head seq d', head=self.num_heads)
        k = rearrange(k, 'batch seq (head d) -> batch head seq d', head=self.num_heads)
        v = rearrange(v, 'batch seq (head d) -> batch head seq d', head=self.num_heads)

        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )

        out = scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            mask=mask
        )

        out = rearrange(out, 'batch head seq d -> batch seq (head d)')
        out = self.output_proj(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = MultiHeadAttentionWithRoPE(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.attn(self.ln1(x))
        out = out + self.ffn(self.ln2(out))

        return out