



import jax
import jax.numpy as jnp
from typing import Optional, Dict, Union
from functools import partial
from pathlib import Path
from rich import print

from entropix.llama_config import AttnParams, LlamaXfmrParams
from entropix.llama_weights import LlamaXfmrWeights, LlamaLayerWeights
from entropix.kvcache import KVCache
from entropix.xfmr_ops import rms_norm, rope_attn, silu_ff
from entropix.prob_utils import renyi_entropy, normalize_logits

ATTN_NOISE_FLOOR = jnp.log(1e-5)

@partial(jax.jit, static_argnames=('attn_params'))
def llama_layer(
        h: jax.Array,
        attn_params: AttnParams,
        layer_weights: LlamaLayerWeights,
        layer_idx: int,
        freqs_cis_slice: jax.Array,
        kvcache: KVCache,
        attn_mask: Optional[jax.Array] = None,
        cur_pos: int = None,
        seqlen: int = None,
):
    # Normalize input
    h_norm = rms_norm(h, layer_weights.attn_norm)
    # Perform attention
    h_attn, kvcache, attn_logits = rope_attn(
        attn_params=attn_params,
        xq=h_norm,
        xk=h_norm,
        xv=h_norm,
        wq=layer_weights.wq,
        wk=layer_weights.wk,
        wv=layer_weights.wv,
        wo=layer_weights.wo,
        freqs_cis_slice=freqs_cis_slice,
        kvcache=kvcache,
        attn_mask=attn_mask,
        cur_pos=cur_pos,
        seqlen=seqlen,
        layer_idx=layer_idx,
    )
    h = h + h_attn
    # Feed-forward network
    h_norm = rms_norm(h, layer_weights.ffn_norm)
    h_ffn = silu_ff(
        w1=layer_weights.w1,
        w2=layer_weights.w2,
        w3=layer_weights.w3,
        x=h_norm
    )
    h = h + h_ffn
    return h, kvcache, attn_logits

# @partial(jax.jit, static_argnames=('xfmr_params', 'seqlen', 'attn_mask'))
def llama_xfmr(
    h: jax.Array,
    xfmr_params: LlamaXfmrParams,
    xfmr_weights: LlamaXfmrWeights,
    kvcache: KVCache,
    seqlen: int,
    renyi_params: Optional[jax.Array],
    cur_pos: Union[int, jax.Array],
    attn_mask: Optional[jax.Array] = None,
) -> Dict[str, jax.Array]:
    if renyi_params is not None:
        shape = (
            xfmr_params.n_layers,
            h.shape[0],
            xfmr_params.attn_params.n_heads,
            h.shape[1],
            renyi_params.shape[0]
        )
        attn_ent_array = jnp.zeros(shape)
    else:
        attn_ent_array = None  # Or jnp.array([])
    freqs_cis_slice = jax.lax.dynamic_slice_in_dim(
        xfmr_weights.freqs_cis,
        start_index=cur_pos,
        slice_size=seqlen,
        axis=0
    )
    carry = (h, kvcache, attn_ent_array, 0)
    h, kvcache, attn_ent_array, layer_idx = carry
    for layer_idx, layer_weights in enumerate(xfmr_weights.layer_weights):
        h, kvcache, attn_logits = llama_layer(
            h=h,
            attn_params=xfmr_params.attn_params,
            layer_weights=layer_weights,
            layer_idx=layer_idx,
            kvcache=kvcache,
            attn_mask=attn_mask,
            cur_pos=cur_pos,
            seqlen=seqlen,
            freqs_cis_slice=freqs_cis_slice,
        )
        attn_probs = normalize_logits(attn_logits, noise_floor=ATTN_NOISE_FLOOR)
        attn_ent = renyi_entropy(attn_probs, renyi_params)
        attn_ent_array = attn_ent_array.at[layer_idx, :, :, :, :].set(attn_ent)
    h_norm = rms_norm(h, xfmr_weights.final_norm)
    return {
        "h": h_norm,
        "kv_cache": kvcache,
        "attn_ent": attn_ent_array,
    }

