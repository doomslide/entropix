from functools import partial
from typing import NamedTuple, Tuple
from entropix.emwa import EMWA
import jax
import jax.numpy as jnp
import jax.scipy as jsp

from entropix.dslider_config import (
    MIN_TEMP,
    MAX_TEMP,
    EPS,
    DSConfig,
    DEFAULT_DS_CONFIG,
)
from entropix.dslider_utils import *


@jax.jit
def kl_divergence(logp: jnp.ndarray, logq: jnp.ndarray) -> jnp.ndarray:
  """Compute KL divergence between two log probability distributions."""
  p = jnp.exp(logp)
  return jnp.sum(jnp.where(p > 0, p * (logp - logq), 0.0), axis=-1)


@jax.jit
def ent_varent(logp: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Compute entropy and varentropy from log probabilities."""
  p = jnp.exp(logp)
  ent = -jnp.sum(p * logp, axis=-1)
  diff = logp + ent[..., None]
  varent = jnp.sum(p * diff**2, axis=-1)
  return ent, varent


@jax.jit
def normalize_logits(logits: jnp.ndarray, noise_floor: float) -> jnp.ndarray:
  """Normalize logits to log probabilities with noise floor truncation."""
  shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
  normalized = shifted - jax.nn.logsumexp(shifted + EPS, axis=-1, keepdims=True)
  # noise floor calculated for bfloat16
  return jnp.where(normalized < noise_floor, jnp.log(EPS), normalized)


def renyi_entropy(logp: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
  """
  Compute Renyi entropy for given log probabilities and Renyi parameters.

  Args:
    logp: Log probabilities, shape (..., vocab_size)
    alpha: Renyi parameters, shape (num_renyi,)

  Returns:
    Renyi entropies, shape (..., num_renyi)
  """
  p = jnp.exp(logp)
  # For p with shape (s1,s2,...,sk) and alpha with shape (n,)
  # Create shape (s1,...,sk-1,n,sk) by broadcasting alpha
  p_alpha = p[..., None, :] * alpha[None, ..., None]
  sum_p_alpha = jnp.sum(p_alpha, axis=-1)
  return 1.0 / (1.0 - alpha) * jnp.log(sum_p_alpha + EPS)


class DSState(NamedTuple):
  emwa_dir: EMWA # shape of mean: (bsz, n_renyi)
  emwa_logp_on_supp: EMWA # shape of mean: (bsz, vsz)
  emwa_naked_ent: EMWA # EMWA of shape of mean: (bsz, n_cat, n_renyi)
  emwa_scaffold_ent: EMWA # EMWA of shape of mean: (bsz, n_cat, n_renyi)
  token_cross_ent_naked: EMWA # EMWA of shape of mean: (bsz,)
  token_cross_ent_scaffold: EMWA # EMWA of shape of mean: (bsz,) # (scaffold means the empirical average of the negative log likelihood of the actual sampled tokens according to the scaffold probabilitiies)
  dir_cross_ent: EMWA # EMWA of shape of mean: (bsz,)

# first gate determines if dirichlet, second gate determines topk, third gate determines temperature 

@jax.jit
def topk_entropies(logits: jax.Array, cat_size: int, renyi_params: jax.Array, noise_floor: float) -> jax.Array:
  """Compute Renyi entropies for top-k truncated distributions.
  
  Args:
    logits: Logits of shape (bsz, vocab_size)
    cat_size: Size k for top-k truncation
    renyi_params: Array of Renyi parameters
    noise_floor: Minimum probability threshold
    
  Returns:
    Array of shape (bsz, n_renyi) containing Renyi entropies
  """
  # Get top k values and indices
  topk_values, topk_indices = jax.lax.top_k(logits, cat_size)
  
  # Normalize the top k logits
  topk_logprobs = normalize_logits(topk_values, noise_floor)
  
  # Compute Renyi entropy on truncated distribution
  return renyi_entropy(topk_logprobs, renyi_params)



# @partial(jax.jit, static_argnames=("config", "dtype"))
def initialize_state(
  logits: jax.Array, bsz: int, config: DSConfig, dtype=jnp.bfloat16
) -> DSState:
  _, seqlen, _ = logits.shape
  logprobs = normalize_logits(logits, config.noise_floor)
  ent_naked = jax.vmap(lambda i: topk_entropies(logprobs, config.cat_sizes[i], config.renyi_params[i], config.noise_floor))(jnp.arange(config.n_cats))
  ent_scaffold = jax.vmap(lambda i: topk_entropies(logprobs, config.cat_sizes[i], config.renyi_params[i], config.noise_floor))(jnp.arange(config.n_cats))
  init_ent_naked = EMWA.new(bsz, config.naked_ent_weights, ent_naked.mean(axis=-1), ent_naked.var(axis=-1))
  init_ent_scaffold = EMWA.new(bsz, config.scaffold_ent_weights, ent_scaffold.mean(axis=-1), ent_scaffold.var(axis=-1))
  logprobs_on_supp = normalize_logits(
    logits[..., config.dirichlet_support], config.noise_floor
  )
  avg_logprobs_on_supp = jnp.mean(logprobs_on_supp, axis=1)

  initial_dir, _, _ = fit_dirichlet(avg_logprobs_on_supp)
  dir_cross_ent = dirichlet_log_likelihood_from_logprob(
    logprobs_on_supp, initial_dir[:, None, :]
  )
  init_dir_cross_ent = EMWA.new(bsz, config.dir_cross_ent_weights, dir_cross_ent.mean(axis=-1), dir_cross_ent.var(axis=-1))
  emwa_dir = jnp.broadcast_to(initial_dir, (bsz,) + initial_dir.shape)
  emwa_logp_on_supp = jnp.broadcast_to(avg_logprobs_on_supp, (bsz,) + avg_logprobs_on_supp.shape)
  token_cross_ent_naked = EMWA.new(bsz, config.token_cross_ent_weights)
  token_cross_ent_scaffold = EMWA.new(bsz, config.token_cross_ent_weights)
  state = DSState(
      emwa_dir=emwa_dir,
      emwa_logp_on_supp=emwa_logp_on_supp,
      emwa_naked_ent=init_ent_naked,
      emwa_scaffold_ent=init_ent_scaffold,
      token_cross_ent_naked=token_cross_ent_naked,
      token_cross_ent_scaffold=token_cross_ent_scaffold,
      dir_cross_ent=init_dir_cross_ent,
  )
  return state


# @partial(jax.jit, static_argnames=("config", "wild"))
def adaptive_dirichlet_step(
  key: jax.random.PRNGKey,
  state: DSState,
  logits: jnp.ndarray,
  config: DSConfig,
  wild: bool = True,
) -> Tuple[DSState, jnp.ndarray]:
  dtype = logits.dtype
  bsz, vsz = logits.shape
  output_tokens = jnp.zeros(bsz, dtype=jnp.int32)
  naked_log_probs = normalize_logits(logits, config.noise_floor)
  # update emwa naked entropy
  new_ent_naked = jax.vmap(lambda i: topk_entropies(logits, 
                                                    config.cat_sizes[i], 
                                                    config.renyi_params[i], 
                                                    config.noise_floor))(jnp.arange(config.n_cats))
  new_emwa_ent_naked = EMWA.update(state.emwa_naked_ent, new_ent_naked)
  # update emwa logp (on dirichlet support) and dirichlet params
  logprobs_on_supp = normalize_logits(logits[:, config.dirichlet_support], config.noise_floor)
  kl = jnp.sum(jnp.exp(logprobs_on_supp) * (logprobs_on_supp - state.emwa_logp_on_supp), axis=-1)
  emwa_logp_coeff = config.emwa_logp_base ** (-config.emwa_logp_exp_factor / (kl + EPS))
  new_emwa_logp_on_supp = update_emwa(logprobs_on_supp, state.emwa_logp_on_supp, emwa_logp_coeff[..., None])
  # update dirichlet cross entropy
  new_dir_log_likelihood = dirichlet_log_likelihood_from_logprob(logprobs_on_supp, state.emwa_dir)
  new_emwa_dir_cross_ent = EMWA.update(state.dir_cross_ent, -new_dir_log_likelihood)
  # update dirichlet params
  new_emwa_dir, _, _ = fit_dirichlet(new_emwa_logp_on_supp)
  """
  topk gate
  """
  topk = compute_topk(state, config, new_ent_naked, new_dir_log_likelihood)
  """
  dirichlet gate
  """
  dirichlet_mask = compute_dirichlet_mask(state, config, new_dir_log_likelihood, new_ent_naked)
  scaffold_pre_logprobs = jnp.where(dirichlet_mask, 
                                      compute_interpolated_logprobs(state, config, logprobs_on_supp), 
                                      naked_log_probs)
  """
  temperature gate
  """
  target_entropy = compute_target_entropy(state, config, dirichlet_mask, topk)
  temp = temp_tune(scaffold_pre_logprobs, target_entropy)
  scaffold_logprobs = normalize_logits(scaffold_pre_logprobs / jnp.clip(temp[:, None], MIN_TEMP, MAX_TEMP), config.noise_floor)
  """
  sample tokens
  """
  output_tokens = jnp.where(dirichlet_mask, typical_p(scaffold_logprobs), min_p(naked_log_probs))
  """
  update emwa scaffold entropy
  """
  new_ent_scaffold = jax.vmap(lambda i: topk_entropies(scaffold_logprobs, 
                                                       config.cat_sizes[i], 
                                                       config.renyi_params[i], 
                                                       config.noise_floor))(jnp.arange(config.n_cats))
  new_emwa_ent_scaffold = EMWA.update(state.emwa_scaffold_ent, new_ent_scaffold)
  """
  update token cross entropy
  """
  batch_indices = jnp.arange(bsz)
  scaffold_token_logprob = jnp.log(scaffold_logprobs[batch_indices, output_tokens] + EPS)
  naked_token_logprob = jnp.log(naked_log_probs[batch_indices, output_tokens] + EPS)
  new_emwa_token_cross_ent_naked = EMWA.update(state.token_cross_ent_naked, -naked_token_logprob)
  new_emwa_token_cross_ent_scaffold = EMWA.update(state.token_cross_ent_scaffold, -scaffold_token_logprob)
  """
  update token cross entropy variance
  """
  # Assemble new state
  new_state = DSState(
    emwa_dir=new_emwa_dir,
    emwa_logp_on_supp=new_emwa_logp_on_supp,
    emwa_naked_ent=new_emwa_ent_naked,
    emwa_scaffold_ent=new_emwa_ent_scaffold,
    token_cross_ent_naked=new_emwa_token_cross_ent_naked,
    token_cross_ent_scaffold=new_emwa_token_cross_ent_scaffold,
    dir_cross_ent=new_emwa_dir_cross_ent,
  )
  
  return (
    new_state,
    output_tokens,
    new_ent_naked,
    new_ent_scaffold,
    naked_token_logprob,
    scaffold_token_logprob,
  )


def compute_target_entropy(state, config, dirichlet_mask, topk):
  pass

def compute_topk(state, config, new_ent_naked, new_dir_log_likelihood):
  pass

def compute_dirichlet_mask(state, config, new_dir_log_likelihood, new_ent_naked):
  pass

def compute_interpolated_logprobs(state, config, logprobs_on_supp):
  pass

@jax.jit
def update_emwa(new: jax.Array, old: jax.Array, coeff: float | jax.Array) -> jax.Array:
  return coeff * new + (1 - coeff) * old


def compute_outlier_threshold_renyi(
    renyi_ent_naked: jnp.ndarray,
    renyi_ent_scaffold: jnp.ndarray,
    config: DSConfig
) -> jnp.ndarray:
  """
  Compute outlier threshold based on Renyi entropies.

  Args:
    renyi_ent_naked: Renyi entropies for naked, shape (batch_size, num_renyi)
    renyi_ent_scaffold: Renyi entropies for scaffold, shape (batch_size, num_renyi)
    config: DSConfig

  Returns:
    Outlier thresholds, shape (batch_size,)
  """
  # Example: bilinear combination of Renyi entropies
  threshold = (
      jnp.einsum("bi,ij,bj->b", renyi_ent_naked, config.outlier_threshold.bilinear, renyi_ent_scaffold)
      + jnp.einsum("bi,i->b", renyi_ent_naked, config.outlier_threshold.linear_state_ent)
      + jnp.einsum("bi,i->b", renyi_ent_scaffold, config.outlier_threshold.linear_state_std)
      + jnp.sum(renyi_ent_naked * config.outlier_threshold.linear_naked_ent, axis=-1)
      + jnp.sum(renyi_ent_scaffold * config.outlier_threshold.linear_naked_varent, axis=-1)
      + config.outlier_threshold.bias
  )
  return threshold


@partial(jax.jit, static_argnames=("config",))
def update_dirichlet_params(tuned_logprobs_on_supp, state, config):
  kl = kl_divergence(tuned_logprobs_on_supp, state.emwa_logp_on_supp)
  emwa_logp_coeff = (
    config.emwa_logp_base ** (-config.emwa_logp_exp_factor / (kl + EPS))
  )[:, None]
  new_emwa_logp_dir_sup = (
    emwa_logp_coeff * tuned_logprobs_on_supp
    + (1 - emwa_logp_coeff) * state.emwa_logp_on_supp
  )
  new_dir_params, _, _ = fit_dirichlet(new_emwa_logp_dir_sup)
  return new_dir_params, new_emwa_logp_dir_sup


@jax.jit
def update_token_cross_entropies(
  state: DSState,
  scaffold_token_logprob: jnp.ndarray,
  naked_token_logprob: jnp.ndarray,
  config: DSConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Update token cross entropy statistics."""
  token_cross_ent_naked = (
    config.token_cross_ent_naked_coeff * (-naked_token_logprob)
    + (1 - config.token_cross_ent_naked_coeff) * state.token_cross_ent_naked
  )
  token_cross_ent_scaffold = (
    config.token_cross_ent_scaffold_coeff * (-scaffold_token_logprob)
    + (1 - config.token_cross_ent_scaffold_coeff) * state.token_cross_ent_scaffold
  )
  token_cross_var_naked = (
    config.token_cross_var_naked_coeff
    * (token_cross_ent_naked - naked_token_logprob) ** 2
    + (1 - config.token_cross_var_naked_coeff) * state.token_cross_var_naked
  )
  token_cross_var_scaffold = (
    config.token_cross_var_scaffold_coeff
    * (token_cross_ent_scaffold - scaffold_token_logprob) ** 2
    + (1 - config.token_cross_var_scaffold_coeff) * state.token_cross_var_scaffold
  )
  return (
    token_cross_ent_scaffold,
    token_cross_ent_naked,
    token_cross_var_scaffold,
    token_cross_var_naked,
  )



from dataclasses import dataclass
from functools import partial
from typing import Dict, Tuple

from entropix.dslider import DSState, adaptive_dirichlet_step
from entropix.dslider_config import DSConfig
import jax
import jax.numpy as jnp

MAX_K = 256
LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E


@dataclass
class SamplerConfig:
  # Naked (logits) entropy thresholds
  low_naked_entropy_threshold = 0.3  # Captures most observed LELV cases
  medium_naked_entropy_threshold = 1.2  # Separates medium from high entropy cases
  high_naked_entropy_threshold = 2.0  # Above this we see clear high entropy cases

  # Naked (logits) varentropy thresholds
  low_naked_varentropy_threshold = 0.8  # Most LELV cases are below this
  high_naked_varentropy_threshold = 2.0  # Clear separation for high variance cases

  # Scaffold (attention) metrics thresholds
  # These don't appear in logs, keeping unchanged
  low_scaffold_entropy_threshold = 1.0
  high_scaffold_entropy_threshold = 2.0
  low_scaffold_varentropy_threshold = 0.3
  high_scaffold_varentropy_threshold = 0.8


@jax.jit
def multinomial_sample_one(probs: jax.Array, key) -> jax.Array:
  """Samples one token from a multinomial distribution."""
  q = jax.random.exponential(key=key, shape=probs.shape)
  return jnp.argmax(probs / q, axis=-1, keepdims=True).astype(jnp.int32)


@jax.jit
def min_p(
  logits: jax.Array,
  temperature: float = 1.666,
  min_p: float = 0.03,
  key=jax.random.PRNGKey(1337),
) -> jax.Array:
  """
  Applies min_p sampling to the logits.

  Tokens with probabilities less than min_p * p_max are filtered out.
  """
  bsz = logits.shape[0]
  # logit = logits[:, -1]  # Get the last token's logits
  probs = jax.nn.softmax(logits / temperature, axis=-1)

  # Min-p sampling logic
  p_max = jnp.max(probs, axis=-1, keepdims=True)
  threshold = min_p * p_max
  indices_to_remove = probs < threshold
  probs = jnp.where(indices_to_remove, 0.0, probs)
  probs = probs / jnp.sum(probs, axis=-1, keepdims=True)  # Re-normalize probabilities

  next_token = multinomial_sample_one(probs, key)
  return next_token.astype(jnp.int32)


@jax.jit
def typical_p(
  logits: jax.Array,
  temperature: float = 0.666,
  typical_p: float = 0.90,
  key=jax.random.PRNGKey(1337),
) -> jax.Array:
  """
  Applies typical_p sampling to the logits.

  Tokens are filtered based on their divergence from the typical entropy.
  """
  bsz = logits.shape[0]
  # logit = logits[-1]
  probs = jax.nn.softmax(logits / temperature, axis=-1)

  # Compute entropy of the distribution
  log_probs = jnp.log(probs + 1e-20)
  entropy = -jnp.sum(probs * log_probs, axis=-1, keepdims=True)

  # Compute KL divergence from the typical entropy
  kl_divergence = jnp.abs(-log_probs - entropy)
  sorted_indices = jnp.argsort(kl_divergence, axis=-1)
  sorted_kl = jnp.take_along_axis(kl_divergence, sorted_indices, axis=-1)
  sorted_probs = jnp.take_along_axis(probs, sorted_indices, axis=-1)

  # Compute cumulative probabilities
  cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)

  # Apply typical_p threshold
  mask = cumulative_probs > typical_p
  sorted_probs = jnp.where(mask, 0.0, sorted_probs)
  sorted_probs = sorted_probs / jnp.sum(
    sorted_probs, axis=-1, keepdims=True
  )  # Re-normalize

  # Sample from the filtered distribution
  next_token = multinomial_sample_one(sorted_probs, key)
  original_indices = jnp.take_along_axis(sorted_indices, next_token, axis=-1)
  return original_indices.astype(jnp.int32)

