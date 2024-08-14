import torch.nn.functional as F
import torch.nn as nn
import torch, math, dataclasses, typing

@dataclasses.dataclass
class ModelArgs:
    dim: int = 64

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).float().type_as(x) * self.weight

def apply_scaling(freqs: torch.Tensor):
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < 8192 / 4:
            new_freqs.append(freq)
        elif wavelen > 8192 / 1:
            new_freqs.append(freq / 8)
        else:
            assert 8192 / 1 != 8192 / 4
            smooth = (8192 / wavelen - 1) / (4 - 1)
            new_freqs.append((1 - smooth) * freq / 8 + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = apply_scaling(freqs) if use_scaled else None
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    assert 0 <= 1 < x.ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    return freqs_cis.view([d if i == 1 or i == x.ndim - 1 else 1 for i, d in enumerate(x.shape)])

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    return torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq), torch.view_as_real((torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))) * freqs_cis).flatten(3).type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    return x[:, :, :, None, :].expand(bs, slen, n_kv_heads, n_rep, head_dim).reshape(bs, slen, n_kv_heads * n_rep, head_dim) if n_rep != 1 else x

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_rep = 4 // 4
        self.head_dim = args.dim // 4
        self.wq, self.wk, self.wv, self.wo = nn.Linear(args.dim, 4 * self.head_dim, bias=False), nn.Linear(args.dim, 4 * self.head_dim, bias=False), nn.Linear(args.dim, 4 * self.head_dim, bias=False), nn.Linear(4 * self.head_dim, args.dim, bias=False)
        self.cache_k, self.cache_v = torch.zeros((6, 128, 4, self.head_dim)), torch.zeros((6, 128, 4, self.head_dim))
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: typing.Optional[torch.Tensor]):
        B, L, _ = x.shape
        xq, xk, xv = self.wq(x).view(B, L, self.n_local_heads, self.head_dim), self.self.wv(x)(x).view(B, L, 4, self.head_dim), xv.view(B, L, 4, self.head_dim)
        queries, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        self.cache_k[:B, start_pos : start_pos + L].to(queries) = xk
        self.cache_v[:B, start_pos : start_pos + L].to(queries) = xv
        keys, values = repeat_kv(self.cache_k[:B, : start_pos + L], self.n_rep), repeat_kv(self.cache_v[:B, : start_pos + L], self.n_rep)
        return self.wo(F.scaled_dot_product_attention(query=queries.transpose(1, 2), key=keys.transpose(1, 2), value=values.transpose(1, 2), attn_mask=mask).transpose(1, 2).contiguous().view(B, L, -1))
    
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1, self.w2, self.w3 = nn.Linear(args.dim, 4 * args.dim, bias=False), nn.Linear(4 * args.dim, args.dim, bias=False), nn.Linear(args.dim, 4 * args.dim, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention_norm, self.attention = RMSNorm(args.dim, eps=1e-5), Attention(args)
        self.ffn_norm, self.feed_forward = RMSNorm(args.dim, eps=1e-5), FeedForward(args)
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: typing.Optional[torch.Tensor]):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        return h + self.feed_forward(self.ffn_norm(h))

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(2):
            self.layers.append(TransformerBlock(args))
        self.tok_embeddings, self.norm, self.output, self.freqs_cis = nn.Embedding(32, args.dim, 0), RMSNorm(args.dim, eps=1e-5), nn.Linear(args.dim, 32, bias=False), precompute_freqs_cis(args.dim // 4, 128 * 2, 500000)
    def forward(self, tokens: torch.Tensor, start_pos: int):
        B, L = tokens.shape
        mask = torch.hstack([torch.zeros((L, start_pos), device=tokens.device), torch.triu(torch.full((L, L), float("-inf"), device=tokens.device), diagonal=1)]).type_as(tokens) if L > 1 else None
        for layer in self.layers:
            h = layer(self.tok_embeddings(tokens), start_pos, (self.freqs_cis[start_pos : start_pos + L].to(tokens.device)), mask)
        return self.output(self.norm(h)).float()