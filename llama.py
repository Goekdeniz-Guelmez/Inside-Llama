from typing import Optional, Tuple
from dataclasses import dataclass

import torch.nn.functional as F
import torch.nn as nn
import torch

import math

@dataclass
class ModelArgs:
    # Transformer
    hidden_dim: int = 24
    n_layers: int = 2

    # Token
    vocab_size: int = 32
    pad_idx: int = 0

    max_seq_len: int = 32
    max_batch_size: int = 6

    # RMS Norm
    norm_eps: float = 1e-5

    # RoPE
    rope_theta: float = 500000
    use_scaled_rope: bool = False

    # Multi-Head-Attention
    n_heads: int = 12
    n_kv_heads: Optional[int] = 6

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.hidden_dim % self.n_heads == 0




# ------------------- Root Mean Square Normalisation (RMSNorm)
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        """
        Compute Root Mean Square Normalisation with the input.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float): The epsilon value to prevent division by zero.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as the input.
        """
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # Create trainable parameters based on the input dimension with shape: torch.Size([dim])
        # Initial weights are all set to one but can be learned during training.

    def _norm(self, x):
        # Compute the Root Mean Square Normalization
        # Step 1: Square each element of the tensor
        # Step 2: Compute the mean of these squared values along the last dimension, keeping the dimensions for broadcasting
        # Step 3: Add epsilon to the mean to ensure numerical stability and compute the reciprocal square root
        # Step 4: Multiply the input tensor by the computed normalization factor
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        # Apply normalization to the input and scale by the learnable weights
        output = self._norm(x.float()).type_as(x)
        return output * self.weight




# ------------------- Rotary Positional Encoding (RoPE)
def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply scaling to a given tensor of frequencies.

    Args:
        freqs (torch.Tensor): The input tensor containing frequencies.

    Returns:
        torch.Tensor: A new tensor with scaled frequencies.
    """
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


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    """
    Precomputes frequencies for a given dimension and end value.

    Args:
        dim (int): The dimension to compute frequencies for.
        end (int): The end value for the frequency computation.
        theta (float, optional): A parameter used in the frequency computation. Defaults to 10000.0.
        use_scaled (bool, optional): Whether to apply scaling to the computed frequencies. Defaults to False.

    Returns:
        torch.Tensor: The precomputed complex frequencies. -> Size [end, dim]

        Example: dim = 17, end = 4 -> print(precompute_freqs_cis(17, 4))
        tensor([[ 1.0000+0.0000e+00j,  1.0000+0.0000e+00j,  1.0000+0.0000e+00j,
          1.0000+0.0000e+00j,  1.0000+0.0000e+00j,  1.0000+0.0000e+00j,
          1.0000+0.0000e+00j,  1.0000+0.0000e+00j],
        [ 0.5403+8.4147e-01j,  0.9433+3.3196e-01j,  0.9935+1.1425e-01j,
          0.9992+3.8737e-02j,  0.9999+1.3111e-02j,  1.0000+4.4367e-03j,
          1.0000+1.5013e-03j,  1.0000+5.0802e-04j],
        [-0.4161+9.0930e-01j,  0.7796+6.2628e-01j,  0.9739+2.2701e-01j,
          0.9970+7.7416e-02j,  0.9997+2.6220e-02j,  1.0000+8.8733e-03j,
          1.0000+3.0026e-03j,  1.0000+1.0160e-03j],
        [-0.9900+1.4112e-01j,  0.5275+8.4956e-01j,  0.9416+3.3680e-01j,
          0.9933+1.1598e-01j,  0.9992+3.9324e-02j,  0.9999+1.3310e-02j,
          1.0000+4.5039e-03j,  1.0000+1.5241e-03j]])
    """

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) # Initialize an array of frequencies based on the given dimension and theta value
    t = torch.arange(end, device=freqs.device, dtype=torch.float32) # Generate a tensor of values from 0 to 'end'
    if use_scaled: # If use_scaled is True, apply the scaling function to the frequencies
        freqs = apply_scaling(freqs) # Apply frequency scaling
    freqs = torch.outer(t, freqs) # Create a tensor with the same shape as 'freqs' and fill it with ones
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # Compute the outer product of 't' and 't_outer', then multiply by 'freqs', complex64
    return freqs_cis


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats the input tensor along a specified dimension.

    Args:
        x (torch.Tensor): Input tensor to be repeated. Shape [B, L, n_kv_heads, head_dim].
        n_rep (int): Number of times to repeat each element.

    Returns:
        torch.Tensor: The repeated tensor. Shape [B, L, n_kv_heads * n_rep, head_dim].
    """
    # Get the shape of the input tensor
    bs, slen, n_kv_heads, head_dim = x.shape  # Unpack the dimensions: batch size, sequence length, number of key-value heads, and head dimension.

    # If we're not repeating anything, just return the original tensor
    if n_rep == 1:
        return x  # If n_rep is 1, return the input tensor as is. Shape remains [B, L, n_kv_heads, head_dim].

    # Repeat each element 'n_rep' times along the specified dimension (dim=2)
    # and then reshape the result to match the desired output shape
    return (
        x[:, :, :, None, :]  # Add a new dimension for broadcasting. Shape [B, L, n_kv_heads, 1, head_dim].
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # Expand the input tensor along the new dimension. Shape [B, L, n_kv_heads, n_rep, head_dim].
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # Reshape the expanded tensor to combine the n_kv_heads and n_rep dimensions. Shape [B, L, n_kv_heads * n_rep, head_dim].
    )


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshapes the given frequency tensor to be compatible with broadcasting.

    Args:
        freqs_cis (torch.Tensor): The input frequency tensor. Shape [L, head_dim].
        x (torch.Tensor): The input tensor for reshaping. Shape [B, L, n_heads, head_dim].

    Returns:
        torch.Tensor: The reshaped frequency tensor. Shape [1, L, 1, head_dim].
    """
    ndim = x.ndim  # Get the number of dimensions in the input tensor 'x'.
    assert 0 <= 1 < ndim  # Ensure 'x' has at least two dimensions.
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])  # Ensure 'freqs_cis' matches the second and last dimensions of 'x'.
    
    # Create a new shape for 'freqs_cis' compatible with broadcasting to 'x'.
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    # If we're at the second or last dimension, keep the original size; otherwise, set to 1 for broadcasting.

    return freqs_cis.view(*shape)  # Reshape 'freqs_cis' to the new shape and return it.


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the rotary embedding transformation to input tensors.

    Args:
        xq (torch.Tensor): The query tensor. Shape [B, L, n_heads, head_dim].
        xk (torch.Tensor): The key tensor. Shape [B, L, n_heads, head_dim].
        freqs_cis (torch.Tensor): The frequency tensor used for broadcasting. Shape [L, head_dim].

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the transformed query and key tensors. Shapes [B, L, n_heads, head_dim].
    """
    # Convert 'xq' to complex numbers and reshape it for complex operations.
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # Shape [B, L, n_heads, head_dim//2].
    # Perform the same conversion and reshape for 'xk'.
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  # Shape [B, L, n_heads, head_dim//2].
    
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)  # Reshape 'freqs_cis' for broadcasting. Shape [1, L, 1, head_dim//2].
    
    # Apply rotary embedding transformation to 'xq' by element-wise multiplication with 'freqs_cis'.
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)  # Convert back to real numbers and flatten. Shape [B, L, n_heads, head_dim].
    # Apply the same transformation to 'xk'.
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)  # Convert back to real numbers and flatten. Shape [B, L, n_heads, head_dim].

    # Return the transformed query and key tensors, ensuring they have the same dtype as the original inputs.
    return xq_out.type_as(xq), xk_out.type_as(xk)




# ------------------- Grouped-Query-Multi-Head-Attention with KV-Cache
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): The model configuration parameters.

        Attributes:
            dim (int): The dimensionality of the input.
            n_heads (int): The number of attention heads.
            n_kv_heads (int): The number of key-value attention heads.
            head_dim (int): The dimensionality of each attention head.
            n_rep (int): The number of repetitions of key-value heads.
            q_proj (nn.Linear): Linear layer for queries.
            k_proj (nn.Linear): Linear layer for keys.
            v_proj (nn.Linear): Linear layer for values.
            o_proj (nn.Linear): Linear layer for outputs.
            cache_k (torch.Tensor): Cache for keys.
            cache_v (torch.Tensor): Cache for values.
        """
        self.hidden_dim = args.hidden_dim
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads

        self.head_dim = self.hidden_dim // self.n_heads  # The dimension of each attention head.
        self.n_rep = self.n_heads // self.n_kv_heads  # Number of repetitions of key-value heads.

        # Create the linear layers for query, key, value, and output projections.
        self.q_proj = nn.Linear(self.hidden_dim, self.n_heads * self.head_dim, bias=False)  # Shape (dim, n_heads * head_dim).
        self.k_proj = nn.Linear(self.hidden_dim, self.n_kv_heads * self.head_dim, bias=False)  # Shape (dim, n_kv_heads * head_dim).
        self.v_proj = nn.Linear(self.hidden_dim, self.n_kv_heads * self.head_dim, bias=False)  # Shape (dim, n_kv_heads * head_dim).
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden_dim, bias=False)  # Shape (n_heads * head_dim, dim).

        # Initialize key and value caches.
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Perform a forward pass through the Attention module.

        Args:
            x (torch.Tensor): Input tensor. Shape (B, L, dim).
            start_pos (int): The starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed frequency components for rotary positional encodings. Shape (L, head_dim, 2).
            mask (Optional[torch.Tensor]): Mask tensor for attention. Shape (L, L).

        Returns:
            torch.Tensor: Output tensor after applying attention. Shape (B, L, hidden_dim).
        """
        B, L, _ = x.shape  # Get the batch size and sequence length. Shape [B, L, dim].
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)  # Apply linear projections. Shapes [B, L, n_heads * head_dim] for xq, [B, L, n_kv_heads * head_dim] for xk and xv.

        xq = xq.view(B, L, self.n_heads, self.head_dim)  # Reshape queries. Shape [B, L, n_heads, head_dim].
        xk = xk.view(B, L, self.n_kv_heads, self.head_dim)  # Reshape keys. Shape [B, L, n_kv_heads, head_dim].
        xv = xv.view(B, L, self.n_kv_heads, self.head_dim)  # Reshape values. Shape [B, L, n_kv_heads, head_dim].

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)  # Apply rotary positional embeddings. Shapes [B, L, n_heads, head_dim] for xq and xk.

        self.cache_k = self.cache_k.to(xq.device)  # Move key cache to the same device as xq.
        self.cache_v = self.cache_v.to(xq.device)  # Move value cache to the same device as xq.

        self.cache_k[:B, start_pos : start_pos + L] = xk  # Update key cache. Shape [max_batch_size, max_seq_len, n_kv_heads, head_dim].
        self.cache_v[:B, start_pos : start_pos + L] = xv  # Update value cache. Shape [max_batch_size, max_seq_len, n_kv_heads, head_dim].

        keys = self.cache_k[:B, : start_pos + L]  # Retrieve keys from cache. Shape [B, start_pos + L, n_kv_heads, head_dim].
        values = self.cache_v[:B, : start_pos + L]  # Retrieve values from cache. Shape [B, start_pos + L, n_kv_heads, head_dim].

        # Repeat key-value heads if n_kv_heads < n_heads.
        keys = repeat_kv(keys, self.n_rep)  # Shape [B, start_pos + L, n_heads, head_dim].
        values = repeat_kv(values, self.n_rep)  # Shape [B, start_pos + L, n_heads, head_dim].

        queries = xq.transpose(1, 2)  # Transpose queries for attention. Shape [B, n_heads, L, head_dim].
        keys = keys.transpose(1, 2)  # Transpose keys for attention. Shape [B, n_heads, start_pos + L, head_dim].
        values = values.transpose(1, 2)  # Transpose values for attention. Shape [B, n_heads, start_pos + L, head_dim].

        # Scaled-Dot-Product Attention
        scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)  # Compute attention scores. Shape [B, n_heads, L, start_pos + L].

        if mask is not None:
            scores = scores + mask  # Apply mask to attention scores. Shape [B, n_heads, L, start_pos + L].
        
        scores = F.softmax(scores.float(), dim=-1).type_as(queries)  # Apply softmax to get attention weights. Shape [B, n_heads, L, start_pos + L].
        output = torch.matmul(scores, values)  # Compute attention output. Shape [B, n_heads, L, head_dim].
        output = output.transpose(1, 2).contiguous().view(B, L, -1)  # Reshape output. Shape [B, L, n_heads * head_dim].
        return self.o_proj(output)  # Apply output projection. Shape [B, L, hidden_dim].




# ------------------- SwiGLU Activated Multi-Layer-Perceptron or Feed-Forward Layer
class MLP(nn.Module):
    def __init__(self, hidden_dim: int, intermediete_size: int):
        super().__init__()
        """
        Initialize the MLP model.

        Args:
            hidden_dim (int): The dimensionality of the input and output.
            intermediate_size (int): The dimensionality of the intermediate layer.

        Attributes:
            w1 (nn.Linear): First linear layer, projecting from hidden_dim to intermediate_size.
            w3 (nn.Linear): Second linear layer, projecting from hidden_dim to intermediate_size.
            w2 (nn.Linear): Third linear layer, projecting from intermediate_size back to hidden_dim.
        """
        self.w1 = nn.Linear(in_features=hidden_dim, out_features=intermediete_size, bias=False)
        self.w3 = nn.Linear(in_features=hidden_dim, out_features=intermediete_size, bias=False)
        self.w2 = nn.Linear(in_features=intermediete_size, out_features=hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the MLP model.

        Args:
            x (torch.Tensor): Input tensor. Shape (batch_size, hidden_dim).

        Returns:
            torch.Tensor: Output tensor after passing through the MLP. Shape (batch_size, hidden_dim).
        """
        # z = self.w1(x)
        # y = self.w3(x)
        # multiplied = z * y
        # activated = F.silu(multiplied)
        # output = self.w2(activated)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))




# ------------------- Decoder-Only-Transformer-Block
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        """
        Initialize the Transformer block.

        Args:
            layer_id (int): The layer index in the Transformer.
            args (ModelArgs): The model configuration parameters.

        Attributes:
            layer_id (int): The layer index in the Transformer.
            attention_norm (RMSNorm): Normalization layer before attention.
            attention (Attention): Grouped-Query-Multi-Head self-attention mechanism with KV-Cache.
            ffn_norm (RMSNorm): Normalization layer before feed-forward network.
            feed_forward (nn.Linear): Feed-forward neural network.
        """
        self.layer_id = layer_id

        self.attention_norm = RMSNorm(args.hidden_dim, eps=args.norm_eps)
        self.attention = Attention(args)

        self.mlp_norm = RMSNorm(args.hidden_dim, eps=args.norm_eps)
        self.mlp = MLP(hidden_dim=args.hidden_dim, intermediete_size=4 * args.hidden_dim)

    def forward(self, embeddings: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Perform a forward pass through the Transformer block.

        Args:
            embeddings (torch.Tensor): Input Embeddings tensor. Shape [B, L, hidden_dim].
            start_pos (int): The starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed frequency components for rotary positional encodings. Shape [L, head_dim, 2].
            mask (Optional[torch.Tensor]): Mask tensor for attention. Shape [L, L].

        Returns:
            torch.Tensor: Output tensor after passing through the block. Shape [B, L, hidden_dim].
        """
        # Apply RMS normalization to the input and then pass it through the attention mechanism
        first_residual_conection = embeddings + self.attention(x=self.attention_norm(embeddings), start_pos=start_pos, freqs_cis=freqs_cis, mask=mask) # Shape after attention: [B, L, hidden_dim]
        # Apply RMS normalization to the output of the attention mechanism and pass it through the feed-forward aka mlp network
        final_second_residual_conection = first_residual_conection + self.mlp(self.mlp_norm(x=first_residual_conection)) # Shape after feed-forward network: [B, L, hidden_dim]
        return final_second_residual_conection




# ------------------- Transformer
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        """
        Initialize the Transformer model.

        Args:
            args (ModelArgs): The model configuration parameters.
        
        Attributes:
            args (ModelArgs): The model configuration parameters.
            tok_embeddings (nn.Embedding): Token embeddings layer.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Root Mean Square Normalization layer.
            lm_head (nn.Linear): Output linear layer.
            freqs_cis (torch.Tensor): Precomputed frequency components for rotary positional encodings.
        """
        self.args = args

        self.tok_embeddings = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.hidden_dim, padding_idx=args.pad_idx)

        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(layer_id, args))

        self.norm = RMSNorm(args.hidden_dim)

        self.lm_head = nn.Linear(in_features=args.hidden_dim, out_features=args.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            dim=args.hidden_dim // args.n_heads,
            end=args.max_seq_len,
            theta=args.rope_theta,
            use_scaled=args.use_scaled_rope
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token ids.
            start_pos (int): The starting position in the sequence.

        Returns:
            torch.Tensor: The output logits from the Transformer.
        """
        B, L = tokens.shape

        embeddings = self.tok_embeddings(tokens) # Shape [B, L, hidden_dim]
        self.freqs_cis = self.freqs_cis.to(embeddings.device)  # Move precomputed frequencies to the same device as `embeddings`.
        freqs_cis = self.freqs_cis[start_pos : start_pos + L]  # Select the relevant frequencies for the current sequence. Shape [L, hidden_dim // n_heads, 2].

        mask = None # Initialize mask as None.
        if L > 1:
            mask = torch.full((L, L), float("-inf"), device=tokens.device)  # Create a mask with -inf values. Shape (L, L).
            mask = torch.triu(mask, diagonal=1)  # Apply upper triangular mask to prevent attending to future tokens. Shape (L, L).
            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([torch.zeros((L, start_pos), device=tokens.device), mask]).type_as(embeddings) # Concatenate zeros for the past tokens and the mask. Shape [L, start_pos + L].

        for layer in self.layers:
            hidden_states = layer(embeddings=embeddings, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask) # Pass through each Transformer block. Shape [B, L, hidden_dim].

        hidden_states = self.norm(hidden_states) # Apply RMS normalization. Shape [B, L, hidden_dim].

        output = self.lm_head(hidden_states).float()  # Generate the final output logits. Shape [B, L, vocab_size].

        return output  # Return the output logits.