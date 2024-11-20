import jax 
jax.config.update('jax_disable_jit', True)

import jax.numpy as jnp
import jax.nn as jnn
from jax.sharding import PartitionSpec as PS
from math import pi
from typing import Optional, Dict, Tuple
from functools import partial
from typing import NamedTuple
from entropix.kvcache import KVCache
from entropix.llama_config import AttnParams

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

shard = jax.lax.with_sharding_constraint

@jax.jit
def apply_rotary_emb(x: jax.Array, freqs_cis: jax.Array, dtype: jnp.dtype = jnp.bfloat16) -> jax.Array:
    x = x.astype(jnp.float32)
    x_real, x_imag = jnp.split(x, 2, axis=-1)
    x_complex = jax.lax.complex(x_real, x_imag)
    x_out = x_complex * freqs_cis[:, :x_complex.shape[-1]].reshape(1, -1, 1, x_complex.shape[-1])
    x_out = jnp.concatenate([jnp.real(x_out), jnp.imag(x_out)], axis=-1)
    return x_out.astype(dtype)

@jax.jit
def rms_norm(x: jax.Array, w: jax.Array, eps: float = 1e-6) -> jax.Array:
  x = shard(x, PS())
  return w * (x * jax.lax.rsqrt(jax.lax.pow(x, 2).mean(-1, keepdims=True) + eps))

def silu_ff(w1: jax.Array, w2: jax.Array, w3: jax.Array, x: jax.Array) -> jax.Array:
    x = shard(x, PS())
    h1 = jnn.silu(shard(jnp.dot(x, w1), PS(None, None, 'mp')))
    h = h1 * shard(jnp.dot(x, w3), PS(None, None, 'mp'))
    return shard(jnp.dot(h, w2), PS())

def rope_attn(
    attn_params: AttnParams,
    xq: jax.Array,
    xk: jax.Array,
    xv: jax.Array,
    wq: jax.Array,
    wk: jax.Array,
    wv: jax.Array,
    wo: jax.Array,
    freqs_cis_slice: jax.Array,
    seqlen: int,
    cur_pos: int,
    layer_idx: int,
    kvcache: KVCache,
    attn_mask: Optional[jax.Array] = None,
) -> Tuple[jax.Array, KVCache, jax.Array]:
    bsz = xq.shape[0]
    # Compute query, key, value projections
    xq = jnp.einsum('...e,enh->...nh', xq, wq)
    xk = jnp.einsum('...e,enh->...nh', xk, wk)
    xv = jnp.einsum('...e,enh->...nh', xv, wv)
    xq = apply_rotary_emb(xq, freqs_cis_slice)
    xk = apply_rotary_emb(xk, freqs_cis_slice)
    xq = xq.reshape(bsz, -1, attn_params.n_heads, attn_params.head_dim)
    xk = xk.reshape(bsz, -1, attn_params.n_kv_heads, attn_params.head_dim)
    xv = xv.reshape(bsz, -1, attn_params.n_kv_heads, attn_params.head_dim)
    kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, seqlen)
    keys = kvcache.k[layer_idx, :, :seqlen, :, :]
    values = kvcache.v[layer_idx, :, :seqlen, :, :]
    keys = jnp.repeat(keys, attn_params.n_heads // attn_params.n_kv_heads, axis=2)
    values = jnp.repeat(values, attn_params.n_heads // attn_params.n_kv_heads, axis=2)
    attn_logits = jnp.einsum('...qnh,...knh->...nqk', xq, keys)/ jnp.sqrt(attn_params.head_dim)
    if attn_mask is not None:
        attn_logits = attn_logits.at[...,:seqlen].add(attn_mask)
    mask = jnp.where(attn_logits != 0.0, attn_logits, DEFAULT_MASK_VALUE)
    padded_logits = jnp.where((mask >= DEFAULT_MASK_VALUE * 0.5), attn_logits, DEFAULT_MASK_VALUE)
    attn_probs = jax.nn.softmax(padded_logits, axis=-1)
    attn_output = jnp.einsum('...hqk,...khd->...qhd', attn_probs, values)
    attn_output = attn_output.reshape(bsz, -1, attn_params.n_heads * attn_params.head_dim)
    h_attn = jnp.dot(attn_output, wo)
    return h_attn, kvcache, attn_logits


