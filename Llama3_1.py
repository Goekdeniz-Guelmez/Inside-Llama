from typing import Optional, Tuple
from dataclasses import dataclass

import math

import torch
import torch.nn as nn
import torch.functional as F


@dataclass
class ModelArgs:
    # Global
    dim: int = 16
    n_layers: int = 2

    # Token
    vocab_size: int = 32
    pad_idx: int = 0
    max_seq_len: int = 1028
    max_batch_size: int = 16

    # RMS Norm
    norm_eps: float = 1e-5

    # RoPE
    rope_theta: float = 500000
    use_scaled_rope: bool = False

    # Multi-Head-Attention
    n_heads: 4
    n_kv_heads: Optional[int] = None

    # MLP
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[int] = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0




# RMS Norm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # Create trainable Parameters based on the input with shape: torch.Size([dim])
        # It can look like: [0.03409577161073685, 0.045597221702337265, ... -0.002031194744631648]

    def _norm(self, x):
        # y = x.pow(2)
        # y = x.mean(-1, keepdim=True)
        # y = torch.rsqrt(x + self.eps)
        # x = x * y
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    


# RoPE
def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    # This is a function that applies scaling to a given tensor of frequencies
    # It takes a tensor of frequencies as input and returns a new tensor with scaled frequencies
    # Values obtained from grid search
    scale_factor = 8 # The scale factor for low frequency components. A value obtained from grid search.
    low_freq_factor = 1 # The lower bound for the scaling, also obtained from a grid search.
    high_freq_factor = 4 # The upper bound for the scaling, also obtained from a grid search.
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor # Calculate the wavelength corresponding to the lower frequency bound.
    high_freq_wavelen = old_context_len / high_freq_factor # Calculate the wavelength corresponding to the upper frequency bound.
    new_freqs = [] # Initialize an empty list to store the new frequencies after scaling.
    for freq in freqs: # Iterate over each frequency in the input tensor.
        wavelen = 2 * math.pi / freq # Calculate the wavelength corresponding to the current frequency.
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq) # If the wavelength is less than the upper bound, append the original frequency to the result list.
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor) # If the wavelength is greater than the lower bound, divide the frequency by the scale factor and append it to the result list.
        else:
            assert low_freq_wavelen != high_freq_wavelen # This condition should never be reached due to the previous conditions. It's an assertion that checks for any unexpected behavior.
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor) # Calculate a smooth interpolation factor based on the wavelength and frequency bounds.
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq) # Interpolate the frequency using the calculated smooth factor. This is likely some kind of smoothing or interpolation technique.
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device) # Convert the list of new frequencies to a tensor and return it with the same data type and device as the input.


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats the input tensor along a specified dimension.
    torch.repeat_interleave(x, dim=2, repeats=n_rep)

    Args:
        x (torch.Tensor): Input tensor to be repeated.
        n_rep (int): Number of times to repeat each element.

    Returns:
        torch.Tensor: The repeated tensor.
    """

    # Get the shape of the input tensor
    bs, slen, n_kv_heads, head_dim = x.shape # Unpack the dimensions

    # If we're not repeating anything, just return the original tensor
    if n_rep == 1:
        return x
    
    # Repeat each element 'n_rep' times along the specified dimension (dim=2)
    # and then reshape the result to match the desired output shape
    return (
        # Expand the input tensor along the specified dimension
        x[:, :, :, None, :] # Add a new dimension for broadcasting
        .expand(bs, slen, n_kv_heads, n_rep, head_dim) # Repeat elements 'n_rep' times
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim) # Reshape the output
    )


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    """
    Precomputes frequencies for a given dimension and end value.

    Args:
        dim (int): The dimension to compute frequencies for.
        end (int): The end value for the frequency computation.
        theta (float, optional): A parameter used in the frequency computation. Defaults to 10000.0.
        use_scaled (bool, optional): Whether to apply scaling to the computed frequencies. Defaults to False.

    Returns:
        torch.Tensor: The precomputed complex frequencies.
    """

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) # Initialize an array of frequencies based on the given dimension and theta value
    t = torch.arange(end, device=freqs.device, dtype=torch.float32) # Generate a tensor of values from 0 to 'end'
    if use_scaled: # If use_scaled is True, apply the scaling function to the frequencies
        freqs = apply_scaling(freqs) # Apply frequency scaling
    freqs = torch.outer(t, freqs) # Create a tensor with the same shape as 'freqs' and fill it with ones
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # Compute the outer product of 't' and 't_outer', then multiply by 'freqs', complex64
    return freqs_cis



def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshapes the given frequency tensor to be compatible with broadcasting.

    Args:
        freqs_cis (torch.Tensor): The input frequency tensor.
        x (torch.Tensor): The input tensor for reshaping.

    Returns:
        torch.Tensor: The reshaped frequency tensor.
    """

    ndim = x.ndim # Get the number of dimensions in the input tensor 'x'
    assert 0 <= 1 < ndim # Check if the number of dimensions is valid
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]) # Check if the shape of 'freqs_cis' matches the last two dimensions of 'x'
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)] # Creats a list to store the new shape then iterates over the dimensions of 'x'
    # If we're at the second or last dimension, keep the original size otherwise, set the new dimension to 1 (for broadcasting)
    return freqs_cis.view(*shape) # Reshape 'freqs_cis' based on the computed shape



def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the rotary embedding transformation to input tensors.

    Args:
        xq (torch.Tensor): The query tensor.
        xk (torch.Tensor): The key tensor.
        freqs_cis (torch.Tensor): The frequency tensor used for broadcasting.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the transformed query and key tensors.
    """
    
    # Convert 'xq' to complex numbers and reshape it
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # Reshape for complex representation
    # Perform same operations on 'xk'
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)  # Apply broadcasting for complex representation
    # Apply rotary embedding transformation to 'xq'
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # Convert back to real numbers and flatten
    # Perform the same operation on 'xk'
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    # Return the transformed tensors
    return xq_out.type_as(xq), xk_out.type_as(xk)



# Grouped-Query-Multi-Head-Attention with KV-Cache
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads

        self.head_dim = self.dim // self.n_heads # The dimension of every Attention Head, is 16 // 4 = 4
        self.n_rep = self.n_heads // self.n_kv_heads

        # Create the Linear Layers for Query, Key, Value and output
        self.q_proj = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False) # Shape [16, 4 * 4] -> [16, 16] therefore Ouputs 16 Values
        self.k_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False) # Same here
        self.v_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False) # Same here

        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False) # Same here

        # Cache stuff
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, input: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        B, L, _ = input.shape
        xq, xk, xv = self.q_proj(input), self.k_proj(input), self.v_proj(input)

        xq = xq.view(B, L, self.n_heads, self.head_dim)
        xk = xq.view(B, L, self.n_kv_heads, self.head_dim)
        xq = xq.view(B, L, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:B, start_pos : start_pos + L] = xk
        self.cache_v[:B, start_pos : start_pos + L] = xv

        keys = self.cache_k[:B, : start_pos + L]
        values = self.cache_v[:B, : start_pos + L]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # [B, cache_len + L, n_heads, head_dim]
        values = repeat_kv(values, self.n_rep)  # [B, cache_len + L, n_heads, head_dim]

        queries = xq.transpose(1, 2)  # [bs, n_heads, seqlen, head_dim]
        keys = keys.transpose(1, 2)  # [bs, n_heads, cache_len + seqlen, head_dim]
        values = values.transpose(1, 2)  # [bs, n_heads, cache_len + seqlen, head_dim]

        # Scaled-Dot-Product-Attention
        scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask # [bs, n_heads, seqlen, cache_len + seqlen]
        
        scores = F.softmax(scores.float(), dim=-1).type_as(queries)
        output = torch.matmul(scores, values) # [bs, n_heads, seqlen, head_dim]
        output = output.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(output)



# SwiGLU Activated Multi-Layer-Perceptron



# Decoder-Only-Transformer-Block



# Transformer
