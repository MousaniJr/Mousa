"""
Model Architecture for DevMentor AI

Transformer-based decoder-only architecture optimized for code generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE)
    Better length generalization than absolute positional embeddings
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute theta
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache rotary embeddings
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        """Pre-compute cos and sin cache"""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor
            seq_len: Sequence length

        Returns:
            Tuple of (cos, sin) embeddings
        """
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...]
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions"""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """
    SwiGLU activation function
    Used in feedforward layers for better gradient flow
    """

    def __init__(self, dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 4, bias=False)
        self.w2 = nn.Linear(dim, dim * 4, bias=False)
        self.w3 = nn.Linear(dim * 4, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with causal masking
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        use_flash_attention: bool = True
    ):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_flash_attention = use_flash_attention

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            use_cache: Whether to return key-value cache
            past_key_value: Cached key-value from previous step

        Returns:
            Tuple of (output, new_kv_cache)
        """
        batch_size, seq_len, _ = hidden_states.size()

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(v, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Use cached key-value if available
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        # Save cache
        new_kv_cache = (k, v) if use_cache else None

        # Compute attention
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch 2.0+ Flash Attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            # Standard attention
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Apply causal mask
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)

            attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, new_kv_cache


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        use_flash_attention: bool = True
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            hidden_size, num_heads, dropout, use_flash_attention
        )
        self.feed_forward = SwiGLU(hidden_size)

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask
            use_cache: Whether to use KV cache
            past_key_value: Past key-value cache

        Returns:
            Tuple of (output, new_cache)
        """
        # Pre-norm architecture
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)

        # Self-attention
        attn_output, new_cache = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_value=past_key_value
        )

        hidden_states = residual + self.dropout(attn_output)

        # Feed-forward
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        ff_output = self.feed_forward(hidden_states)
        hidden_states = residual + self.dropout(ff_output)

        return hidden_states, new_cache


class DevMentorModel(nn.Module):
    """
    DevMentor AI Base Model
    Decoder-only transformer for code generation
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 2048,
        num_layers: int = 24,
        num_heads: int = 16,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
        use_flash_attention: bool = True
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout, use_flash_attention)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(hidden_size)

        # Language modeling head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Tie weights (input embeddings = output embeddings)
        self.lm_head.weight = self.token_embeddings.weight

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Tuple] = None
    ) -> dict:
        """
        Forward pass

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] (for language modeling loss)
            use_cache: Whether to use KV cache
            past_key_values: Past key-value cache

        Returns:
            Dictionary with logits, loss, and cache
        """
        batch_size, seq_len = input_ids.size()

        # Token embeddings
        hidden_states = self.token_embeddings(input_ids)
        hidden_states = self.dropout(hidden_states)

        # Prepare causal mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device)

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=input_ids.device) * float('-inf'),
            diagonal=1
        )

        # Pass through transformer layers
        new_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None

            hidden_states, new_cache = layer(
                hidden_states,
                attention_mask=causal_mask,
                use_cache=use_cache,
                past_key_value=past_kv
            )

            if use_cache:
                new_key_values.append(new_cache)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Language modeling head
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return {
            "logits": logits,
            "loss": loss,
            "past_key_values": new_key_values if use_cache else None
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Generate text autoregressively

        Args:
            input_ids: Initial input tokens [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample or use greedy decoding

        Returns:
            Generated token IDs
        """
        self.eval()

        with torch.no_grad():
            past_key_values = None

            for _ in range(max_length):
                # Forward pass
                outputs = self.forward(
                    input_ids=input_ids[:, -1:] if past_key_values else input_ids,
                    use_cache=True,
                    past_key_values=past_key_values
                )

                logits = outputs["logits"][:, -1, :]
                past_key_values = outputs["past_key_values"]

                # Apply temperature
                logits = logits / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')

                # Sample or greedy
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                # Stop if EOS token generated
                # (assuming EOS token ID is 2, adjust as needed)
                if next_token.item() == 2:
                    break

        return input_ids


def create_model(model_size: str = "medium") -> DevMentorModel:
    """
    Create a DevMentor model with predefined configuration

    Args:
        model_size: "small", "medium", "large", or "xl"

    Returns:
        Initialized model
    """
    configs = {
        "small": {
            "vocab_size": 50000,
            "hidden_size": 1024,
            "num_layers": 12,
            "num_heads": 8,
            "max_seq_len": 4096
        },
        "medium": {
            "vocab_size": 50000,
            "hidden_size": 2048,
            "num_layers": 24,
            "num_heads": 16,
            "max_seq_len": 4096
        },
        "large": {
            "vocab_size": 50000,
            "hidden_size": 4096,
            "num_layers": 32,
            "num_heads": 32,
            "max_seq_len": 4096
        },
        "xl": {
            "vocab_size": 50000,
            "hidden_size": 5120,
            "num_layers": 40,
            "num_heads": 40,
            "max_seq_len": 8192
        }
    }

    if model_size not in configs:
        raise ValueError(f"Invalid model size: {model_size}")

    config = configs[model_size]
    return DevMentorModel(**config)


if __name__ == "__main__":
    # Example usage
    model = create_model("small")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 50000, (batch_size, seq_len))

    outputs = model(input_ids)
    print(f"Logits shape: {outputs['logits'].shape}")
