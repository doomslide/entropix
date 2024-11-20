from dataclasses import dataclass
from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax.nn as jnn
from typing import NamedTuple
from functools import partial
from entropix.emwa import EMWA
from entropix.prob_utils import normalize_logits, temp_tune, EFF_NEG_INF


class RenyiConfig(NamedTuple):
  ent_target_weights: jax.Array
  ent_target_biases: jax.Array
  inject_weights: jax.Array
  inject_biases: jax.Array
  backtrack_weights: jax.Array
  backtrack_biases: jax.Array
  adaptive_weights: jax.Array
  adaptive_biases: jax.Array

class DecodeState(NamedTuple):
  # table mapping token positions to disallowed tokens
  disallowed_tokens: jax.Array
  last_argmax_pos: int

class RenyiState(NamedTuple):
  emwa_attn_ent: EMWA # not used in the basic bitch sampler 
  emwa_ent_scaffold: EMWA
  emwa_ent_naked: EMWA
  token_cross_ent_scaffold: EMWA
  token_cross_ent_naked: EMWA

def compute_target_entropy(
    state: RenyiState, 
    cfg: RenyiConfig
) -> jax.Array:
  vector = jnp.concatenate([
    state.emwa_ent_naked.mean,
    state.emwa_ent_scaffold.mean,
    state.emwa_attn_ent.mean,
    state.token_cross_ent_naked.mean,
    state.token_cross_ent_scaffold.mean,
    state.emwa_ent_naked.var,
    state.emwa_ent_scaffold.var,
    state.emwa_attn_ent.var,
    state.token_cross_ent_naked.var,
    state.token_cross_ent_scaffold.var
  ])
  weights = cfg.ent_weights
  biases = cfg.ent_biases
  hidden_layer = jnp.dot(vector, weights[0]) + biases[0]
  hidden_layer = jnn.gelu(hidden_layer)  # GELU activation
  output_layer = jnp.dot(hidden_layer, weights[1]) + biases[1]
  return jnp.max(0, output_layer)

def compute_log_base_min_p(
    state: RenyiState, 
    cfg: RenyiConfig
) -> jax.Array:
  vector = jnp.concatenate([
    state.emwa_ent_naked.mean,
    state.emwa_ent_scaffold.mean,
    state.emwa_attn_ent.mean,
    state.token_cross_ent_naked.mean,
    state.token_cross_ent_scaffold.mean,
    state.emwa_ent_naked.var,
    state.emwa_ent_scaffold.var,
    state.emwa_attn_ent.var,
    state.token_cross_ent_naked.var,
    state.token_cross_ent_scaffold.var
  ])
  weights = cfg.min_p_weights
  biases = cfg.min_p_biases
  hidden_layer = jnp.dot(vector, weights[0]) + biases[0]
  hidden_layer = jnn.gelu(hidden_layer)  # GELU activation
  output_layer = jnp.dot(hidden_layer, weights[1]) + biases[1]
  return output_layer

def basic_sample(
    state: RenyiState, 
    logits: jax.Array, 
    cfg: RenyiConfig, 
    key: jax.random.PRNGKey
) -> Tuple[DecodeState, RenyiState]:
  target_entropy = compute_target_entropy(state, cfg)
  tuned_logprobs = normalize_logits(logits/temp_tune(logits, target_entropy))
  log_base_min_p = compute_log_base_min_p(state, cfg)
  log_min_p = jnp.max(tuned_logprobs, axis=-1) + log_base_min_p
  min_p_logits = jnp.where(tuned_logprobs > log_min_p[:, None], tuned_logprobs, EFF_NEG_INF)
  resampled_token = jax.random.categorical(key, min_p_logits)
  scaffold_logprobs = tuned_logprobs[resampled_token == 0]
  naked_logprobs = tuned_logprobs[resampled_token == 1]

  return resampled_token, state
  
def renyi_step(
    state: RenyiState,
    xfmr_output: Dict[jax.Array],
    embedding: jnp.ndarray,
    cfg: RenyiConfig,
    key: jax.random.PRNGKey
) -> Tuple[DecodeState, RenyiState]:
  # compute 4 scalars based on state and xfmr_output:
    # I: inject score
    # B: backtrack score
    # T: basic score
  # method is picked by argmax over 4 scalars
  # (injection tactic): return argmax over injection tokens (some finite curated list)
  # (backtrack tactic): move cur_pos to max(last_attn_spike, cur_pos -max_backtrack) and update disallowed tokens (i.e. add the token following last attention spike to disallowed and remove all tokens in positions greater than last attention spike)
  # (basic score tactic): sample with tuned temperature min_p with params functions of RenyiState (this covers the argmax case)
  pass
