from dataclasses import dataclass
from functools import partial
from typing import Dict, Tuple
from rich import print

import jax
import jax.numpy as jnp
import jax.nn as jnn
from typing import NamedTuple
from functools import partial
from entropix.emwa import EMWA
from entropix.prob_utils import normalize_logits, temp_tune, EFF_NEG_INF, renyi_entropy

class MLPWeights(NamedTuple):
  w1: jax.Array
  w2: jax.Array
  b1: jax.Array
  b2: jax.Array

class RenyiWeights(NamedTuple):
  renyi_params: jax.Array
  ent_target: MLPWeights
  min_p: MLPWeights
  emwa_ent_scaffold_coeff: float
  emwa_ent_naked_coeff: float
  token_cross_ent_naked_coeff: float
  token_cross_ent_scaffold_coeff: float
  noise_floor: float

DEFAULT_RENYI_WEIGHTS = RenyiWeights(
  renyi_params=jnp.array([0.5, 1.0, 2.0, 4.0]),
  ent_target=MLPWeights(
    w1=jnp.zeros((20, 10)),
    w2=jnp.zeros((10, 1)),
    b1=jnp.array([0.0] * 10),
    b2=jnp.array([0.0])
  ),
  min_p=MLPWeights(
    w1=jnp.zeros((20, 10)),
    w2=jnp.zeros((10, 1)),
    b1=jnp.array([0.0] * 10),
    b2=jnp.array([0.0])
  ),
  token_cross_ent_naked_coeff=0.1,
  token_cross_ent_scaffold_coeff=0.2,
  emwa_ent_scaffold_coeff=0.1,
  emwa_ent_naked_coeff=0.1,
  noise_floor=-16.0
)
class DecodeState(NamedTuple):
  # table mapping token positions to disallowed tokens
  disallowed_tokens: jax.Array
  last_argmax_pos: int

class RenyiState(NamedTuple):
  # emwa_attn_ent: EMWA # peerhaps not used in the basic bitch sampler 
  emwa_ent_scaffold: EMWA
  emwa_ent_naked: EMWA
  token_cross_ent_scaffold: EMWA
  token_cross_ent_naked: EMWA

@partial(jax.jit)
def init_renyi_state(tokens: jax.Array, logits: jax.Array, cfg: RenyiWeights) -> RenyiState:
    bsz = logits.shape[0]
    n_renyi = cfg.renyi_params.shape[0]
    logprobs = normalize_logits(logits, cfg.noise_floor)
    naked_ent = renyi_entropy(logprobs, cfg.renyi_params)  # Shape: (bsz, seq_len, n_renyi)
    # Mean and var across sequence length
    avg_ent = naked_ent.mean(axis=1)  # Shape: (bsz, n_renyi)
    avg_ent_var = naked_ent.var(axis=1) / logprobs.shape[-1]  # Shape: (bsz, n_renyi) 
    # Add batch dimension to tokens and expand for broadcasting
    token_logprobs = jnp.take_along_axis(logprobs, tokens[..., None], axis=2).squeeze(-1)
    # Compute token entropy for each Renyi parameter
    token_avg_ent = token_logprobs.mean(axis=-1)  # Shape: (bsz, 1)
    token_avg_ent_var = token_logprobs.var(axis=-1) / logprobs.shape[-1]  # Shape: (bsz, 1)
    return RenyiState(
        emwa_ent_scaffold=EMWA.new(cfg.emwa_ent_scaffold_coeff, (bsz, n_renyi), avg_ent, avg_ent_var),
        emwa_ent_naked=EMWA.new(cfg.emwa_ent_naked_coeff, (bsz, n_renyi), avg_ent, avg_ent_var),
        token_cross_ent_scaffold=EMWA.new(cfg.token_cross_ent_scaffold_coeff, (bsz,), token_avg_ent, token_avg_ent_var),
        token_cross_ent_naked=EMWA.new(cfg.token_cross_ent_naked_coeff, (bsz,), token_avg_ent, token_avg_ent_var),
    )

def create_vector(state: RenyiState) -> jax.Array:
    # First reshape all components to have shape (1, 4) or (1, 1)
    components = [
        state.emwa_ent_naked.mean,                    # (1, 4)
        state.emwa_ent_scaffold.mean,                # (1, 4)
        state.token_cross_ent_naked.mean[..., None],           # (1, 1)
        state.token_cross_ent_scaffold.mean[..., None],         # (1, 1)
        jnp.sqrt(state.emwa_ent_naked.unbiased_var),  # (1, 4)
        jnp.sqrt(state.emwa_ent_scaffold.unbiased_var), # (1, 4)
        jnp.sqrt(state.token_cross_ent_naked.unbiased_var[..., None]),   # (1, 1)
        jnp.sqrt(state.token_cross_ent_scaffold.unbiased_var[..., None])  # (1, 1)
    ]
    # Concatenate along the last axis
    return jnp.concatenate(components, axis=-1)  # Result shape: (1, 32)

def mlp(w1, w2, b1, b2, vector):
    hidden_layer = jnp.matmul(vector, w1) + b1  # Shape: (bsz, hidden_dim)
    hidden_layer = jnn.gelu(hidden_layer)  # GELU activation
    output_layer = jnp.matmul(hidden_layer, w2) + b2  # Shape: (bsz, output_dim)
    return output_layer

def compute_target_entropy(
    state: RenyiState, 
    cfg: RenyiWeights
) -> jax.Array:
  vector = create_vector(state)
  weights = cfg.ent_target
  return jnp.maximum(0, mlp(weights.w1, weights.w2, weights.b1, weights.b2, vector))

def compute_log_base_min_p(
    state: RenyiState, 
    cfg: RenyiWeights
) -> jax.Array:
  vector = create_vector(state)
  weights = cfg.min_p
  return mlp(weights.w1, weights.w2, weights.b1, weights.b2, vector)

def basic_sample(
    state: RenyiState, 
    logits: jax.Array, 
    cfg: RenyiWeights, 
    key: jax.random.PRNGKey
) -> Tuple[jax.Array, RenyiState]:
    target_entropy = compute_target_entropy(state, cfg).squeeze(-1)
    temp, _, _ = temp_tune(logits, target_entropy) 
    assert temp.ndim == 1
    tuned_logprobs = normalize_logits(logits / temp[..., None], cfg.noise_floor).astype(jnp.bfloat16)
    assert tuned_logprobs.ndim == 2
    log_base_min_p = compute_log_base_min_p(state, cfg).astype(jnp.bfloat16)
    assert log_base_min_p.ndim == 2 
    log_min_p = jnp.max(tuned_logprobs, axis=-1, keepdims=True) + log_base_min_p
    min_p_logits = jnp.where(
        tuned_logprobs > log_min_p,
        tuned_logprobs,
        jnp.full_like(tuned_logprobs, EFF_NEG_INF, dtype=tuned_logprobs.dtype)
    )
    sampled_token = jax.random.categorical(key, min_p_logits)    
    naked_logprobs = normalize_logits(logits)
    scaffold_logprobs = normalize_logits(min_p_logits)
    # Compute entropies for each Renyi parameter
    naked_entropy = renyi_entropy(naked_logprobs, cfg.renyi_params)  # Shape: (bsz, n_renyi)
    scaffold_entropy = renyi_entropy(scaffold_logprobs, cfg.renyi_params)  # Shape: (bsz, n_renyi)
    
    # Get token logprobs and compute their entropies
    token_naked_logprob = jnp.take_along_axis(naked_logprobs, sampled_token[..., None], axis=-1).squeeze(-1)
    token_scaffold_logprob = jnp.take_along_axis(scaffold_logprobs, sampled_token[..., None], axis=-1).squeeze(-1)
    return sampled_token, RenyiState(
        emwa_ent_scaffold=state.emwa_ent_scaffold.update(scaffold_entropy),
        emwa_ent_naked=state.emwa_ent_naked.update(naked_entropy),
        token_cross_ent_scaffold=state.token_cross_ent_scaffold.update(token_scaffold_logprob),
        token_cross_ent_naked=state.token_cross_ent_naked.update(token_naked_logprob)
    )


# def renyi_step(
#     state: RenyiState,
#     xfmr_output: Dict[jax.Array],
#     embedding: jnp.ndarray,
#     cfg: RenyiWeights,
#     key: jax.random.PRNGKey
# ) -> Tuple[DecodeState, RenyiState]:
#   # compute 3 scalars based on state and xfmr_output:
#     # I: inject score
#     # B: backtrack score
#     # T: basic score
#   # method is picked by argmax over 3 scalars
#   # (injection tactic): return argmax over injection tokens (some finite curated list)
#   # (backtrack tactic): move cur_pos to max(last_attn_spike, cur_pos -max_backtrack) and update disallowed tokens (i.e. add the token following last attention spike to disallowed and remove all tokens in positions greater than last attention spike)
#   # (basic score tactic): sample with tuned temperature min_p with params functions of RenyiState (this covers the argmax case)
#   pass