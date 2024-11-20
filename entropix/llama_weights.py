from typing import List, NamedTuple, Optional, Union, Tuple
from dataclasses import dataclass
import jax.random as random
from jax.sharding import NamedSharding
from rich import print
from entropix.llama_config import ScaledRopeParams, LlamaXfmrConfig
import math

import jax
jax.config.update('jax_disable_jit', True)

import jax.numpy as jnp

from pathlib import Path
from rich import print
import numpy as np

class LlamaLayerWeights(NamedTuple):
    wq: jax.Array  # Shape: (dim, n_heads, head_dim)
    wk: jax.Array  # Shape: (dim, n_kv_heads, head_dim)
    wv: jax.Array  # Shape: (dim, n_kv_heads, head_dim)
    wo: jax.Array  # Shape: (dim, n_heads * head_dim)
    w1: jax.Array  # Shape: (dim, ffn_dim)
    w2: jax.Array  # Shape: (ffn_dim, dim)
    w3: jax.Array  # Shape: (dim,)
    attn_norm: jax.Array  # Shape: (dim,)
    ffn_norm: jax.Array  # Shape: (dim,)

class LlamaXfmrWeights(NamedTuple):
    embedding: jax.Array
    final_norm: jax.Array
    unembedding: jax.Array
    layer_weights: List[LlamaLayerWeights]

@dataclass
class ShardingConfig:
    """Configuration for weight loading and sharding."""

    dp_dim: str = "dp"
    mp_dim: str = "mp"
    fsdp_dim: str = "fsdp"

def create_mesh(device_count: int) -> jax.sharding.Mesh:
    """Creates device mesh for distributed execution."""
    devices = jax.devices()
    
    if device_count > len(devices):
        raise ValueError(f"Requested device_count {device_count} exceeds available devices {len(devices)}")
    
    mesh_shape = (device_count, 1)
    selected_devices = devices[:device_count]
    device_mesh = np.reshape(selected_devices, mesh_shape)
    
    mesh = jax.sharding.Mesh(device_mesh, ("mp", "fsdp"))
    return mesh

def create_partition_spec(key):
    dp = "dp"
    mp = "mp"
    fsdp = "fsdp"
    if "norm" in key or key == "w3":
        return jax.sharding.PartitionSpec()
    if "rope.freqs" in key:
        return jax.sharding.PartitionSpec()
    elif "tok_embeddings" in key:
        return jax.sharding.PartitionSpec(fsdp, mp)
    elif "output" in key:
        return jax.sharding.PartitionSpec(fsdp, mp)
    elif "w2" in key or "wo" in key:
        return jax.sharding.PartitionSpec(mp, fsdp)
    else:
        return jax.sharding.PartitionSpec(fsdp, mp)

class LlamaLayerWeights(NamedTuple):
    wq: jax.Array  # Shape: (dim, n_heads, head_dim)
    wk: jax.Array  # Shape: (dim, n_kv_heads, head_dim)
    wv: jax.Array  # Shape: (dim, n_kv_heads, head_dim)
    wo: jax.Array  # Shape: (dim, n_heads * head_dim)
    w1: jax.Array  # Shape: (dim, ffn_dim)
    w2: jax.Array  # Shape: (ffn_dim, dim)
    w3: jax.Array  # Shape: (dim,)
    attn_norm: jax.Array  # Shape: (dim,)
    ffn_norm: jax.Array  # Shape: (dim,)

class LlamaXfmrWeights(NamedTuple):
    embedding: jax.Array
    final_norm: jax.Array
    unembedding: jax.Array
    freqs_cis: jax.Array
    layer_weights: List[LlamaLayerWeights]

def load_llama_weights(
    ckpt_dir: Path,
    config: LlamaXfmrConfig,
    weight_config: Optional[ShardingConfig] = None
) -> Tuple[LlamaXfmrWeights, jax.sharding.Mesh]:
    """Load and shard model weights across devices."""
    if ckpt_dir is not None and isinstance(ckpt_dir, str):
        ckpt_dir = Path(ckpt_dir)
    weight_config = weight_config or ShardingConfig()
    mesh = create_mesh(jax.device_count())

    w = {}
    layer_weights = []

    for file in ckpt_dir.glob("*.npy"):
        name = ".".join(str(file).split("/")[-1].split(".")[:-1])
        weight = jnp.load(file=file, mmap_mode="r", allow_pickle=True)
        partition_spec = create_partition_spec(name)
        sharding = NamedSharding(mesh, partition_spec)
        if any(lyr in name for lyr in ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]):
            weight = weight.T
            if "wq" in name or "wk" in name or "wv" in name:
                weight = weight.reshape(
                -1,
                config.n_heads if "wq" in name else config.n_kv_heads,
                config.head_dim,
                )
        w[name] = jax.device_put(weight, sharding)
    for i in range(config.n_layers):
        layer_weights.append(
            LlamaLayerWeights(
                wq=w[f"layers.{i}.attention.wq.weight"],
                wk=w[f"layers.{i}.attention.wk.weight"],
                wv=w[f"layers.{i}.attention.wv.weight"],
                wo=w[f"layers.{i}.attention.wo.weight"],
                w1=w[f"layers.{i}.feed_forward.w1.weight"],
                w2=w[f"layers.{i}.feed_forward.w2.weight"],
                w3=w[f"layers.{i}.feed_forward.w3.weight"],
                ffn_norm=w[f"layers.{i}.ffn_norm.weight"],
                attn_norm=w[f"layers.{i}.attention_norm.weight"],
            )
        )
    return LlamaXfmrWeights(
        embedding=w["tok_embeddings.weight"],
        final_norm=w["norm.weight"],
        unembedding=w["output.weight"].T,
        freqs_cis=precompute_freqs_cis(config),
        layer_weights=layer_weights,
    ), mesh

def precompute_freqs_cis(
        config: LlamaXfmrConfig, 
        dtype = jnp.float32
    ) -> jax.Array:
    freqs = 1.0 / (config.rope_theta ** (jnp.arange(0, config.dim, 2)[: (config.dim // 2)].astype(dtype) / config.dim))
    if config.scaled_rope_params:
      freqs = apply_scaling(freqs, config.scaled_rope_params)
    t = jnp.arange(config.max_seq_len, dtype=dtype)
    freqs = jnp.outer(t, freqs)
    freqs_cis = jnp.exp(1j * freqs)
    return freqs_cis

def apply_scaling(freqs: jax.Array, scale_params: ScaledRopeParams):
    SCALE_FACTOR = scale_params.scale_factor
    LOW_FREQ_FACTOR = scale_params.low_freq_factor
    HIGH_FREQ_FACTOR = scale_params.high_freq_factor
    OLD_CONTEXT_LEN = scale_params.old_context_len

    low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
    high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

    def scale_freq(freq):
        wavelen = 2 * math.pi / freq

        def scale_mid(_):
            smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (
                HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR
            )
            return (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

        return jax.lax.cond(
            wavelen < high_freq_wavelen,
            lambda _: freq,
            lambda _: jax.lax.cond(
                wavelen > low_freq_wavelen, lambda _: freq / SCALE_FACTOR, scale_mid, None
            ),
            None,
        )
    return jax.vmap(scale_freq)(freqs)
