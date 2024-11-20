from typing import Dict, NamedTuple, Optional
import jax
jax.config.update('jax_disable_jit', True)

import jax.numpy as jnp
import math
from jax import tree_util


DEFAULT_MASK_VALUE = -1e10

class AttnParams(NamedTuple):
    n_heads: int
    n_kv_heads: int
    head_dim: int

class LlamaXfmrParams(NamedTuple):
    n_layers: int
    dim: int
    max_seq_len: int
    attn_params: AttnParams

class ScaledRopeParams(NamedTuple):
    scale_factor: int # 8
    low_freq_factor: int # 1
    high_freq_factor: int # 4
    old_context_len: int # 8192


class LlamaXfmrConfig(NamedTuple):
    """Configuration for the transformer model."""
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    ffn_dim_multiplier: float
    multiple_of: int
    norm_eps: float
    rope_theta: float
    scaled_rope_params: Optional[ScaledRopeParams]
    max_seq_len: int

    @property
    def ffn_dim(self) -> int:
        return int(self.dim * self.ffn_dim_multiplier)

    @property
    def head_dim(self):
        return self.dim // self.n_heads


DEFAULT_SCALE_PARAMS = ScaledRopeParams(
    scale_factor=8, 
    low_freq_factor=1, 
    high_freq_factor=4, 
    old_context_len=8192
)
# Define standard model configurations
MODEL_CONFIGS = {
  "1B": LlamaXfmrConfig(
    dim=2048,
    n_layers=16,
    n_heads=32,
    n_kv_heads=8,
    vocab_size=128256,
    ffn_dim_multiplier=4.0,
    multiple_of=256,
    norm_eps=1e-05,
    rope_theta=500000.0,
    scaled_rope_params=DEFAULT_SCALE_PARAMS,
    max_seq_len=4096,
  ),
  "70B": LlamaXfmrConfig(
    dim=8192,
    n_layers=80,
    n_heads=64,
    n_kv_heads=8,
    vocab_size=128256,
    ffn_dim_multiplier=1.5,
    multiple_of=256,
    norm_eps=1e-05,
    rope_theta=500000.0,
    scaled_rope_params=DEFAULT_SCALE_PARAMS,
    max_seq_len=4096,
  ),
}

def create_llama_params(config: LlamaXfmrConfig) -> LlamaXfmrParams:
    """Creates LlamaLayerParams based on the model name."""
    return LlamaXfmrParams(
        n_layers=config.n_layers,
        max_seq_len=config.max_seq_len,
        dim=config.dim,
        attn_params=AttnParams(
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.dim // config.n_heads,
        ),
    )
