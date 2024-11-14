from functools import partial
from typing import NamedTuple, Optional, Tuple

from entropix.dslider_tuning import OnlineTuner
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax
from jax import vmap

from entropix.dslider_config import EPS, MAX_TEMP, MIN_TEMP, DSConfig
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


class DSState(NamedTuple):
  emwa_dir: jnp.ndarray
  emwa_logp_on_supp: jnp.ndarray
  emwa_temp: jnp.ndarray
  emwa_ent_scaffold: jnp.ndarray
  emwa_ent_naked: jnp.ndarray
  emwa_varent_scaffold: jnp.ndarray
  emwa_varent_naked: jnp.ndarray
  token_cross_ent_scaffold: jnp.ndarray
  token_cross_ent_naked: jnp.ndarray
  token_cross_var_scaffold: jnp.ndarray
  token_cross_var_naked: jnp.ndarray
  emwa_dir_ent: jnp.ndarray
  emwa_topk_ent_naked: jnp.ndarray
  tokens: jnp.ndarray


@partial(jax.jit, static_argnames=("bsz", "config", "dtype"))
def initialize_state(
  logits: jax.Array, bsz: int, config: DSConfig, dtype=jnp.bfloat16
) -> DSState:
  _, seqlen, _ = logits.shape
  logprobs = normalize_logits(logits, config.noise_floor)
  ent, varent = ent_varent(logprobs)
  avg_ent, avg_varent = ent.mean(axis=-1), varent.mean(axis=-1)

  topk_logits, topk_indices = jax.lax.top_k(logprobs, config.outlier_topk)
  topk_logprobs = normalize_logits(topk_logits, config.noise_floor)
  topk_ent, _ = ent_varent(topk_logprobs)
  avg_topk_ent = topk_ent.mean(axis=-1)

  logprobs_on_supp = normalize_logits(
    logits[..., config.dirichlet_support], config.noise_floor
  )
  avg_logprobs_on_supp = jnp.mean(logprobs_on_supp, axis=1)

  initial_dir, _, _ = fit_dirichlet(avg_logprobs_on_supp)
  avg_dir_ent = dirichlet_log_likelihood_from_logprob(
    logprobs_on_supp, initial_dir[:, None, :]
  ).mean(axis=-1)

  topk_token_logprobs = jnp.take_along_axis(logprobs, topk_indices, axis=-1)
  initial_cross_ent_naked = -topk_token_logprobs.mean(axis=(1, 2))
  initial_cross_var_naked = topk_token_logprobs.var(axis=(1, 2))

  state = DSState(
    emwa_dir=initial_dir.repeat(bsz, axis=0),
    emwa_logp_on_supp=avg_logprobs_on_supp.repeat(bsz, axis=0),
    emwa_temp=jnp.ones((bsz,), dtype=dtype),
    emwa_ent_scaffold=avg_ent.repeat(bsz, axis=0),
    emwa_ent_naked=avg_ent.repeat(bsz, axis=0),
    emwa_varent_scaffold=jnp.zeros((bsz,), dtype=dtype),
    emwa_varent_naked=avg_varent.repeat(bsz, axis=0),
    token_cross_ent_scaffold=avg_ent.repeat(bsz, axis=0),
    token_cross_ent_naked=initial_cross_ent_naked.repeat(bsz, axis=0),
    token_cross_var_scaffold=jnp.zeros((bsz,), dtype=dtype),
    token_cross_var_naked=initial_cross_var_naked.repeat(bsz, axis=0),
    emwa_dir_ent=avg_dir_ent.repeat(bsz, axis=0),
    emwa_topk_ent_naked=avg_topk_ent.repeat(bsz, axis=0),
    tokens=jnp.zeros((bsz,), dtype=jnp.int32),
  )
  return state


@partial(jax.jit, static_argnames=('wild',))
def adaptive_dirichlet_step(
  key: jax.random.PRNGKey,
  state: DSState,
  logits: jnp.ndarray,
  config: DSConfig,
  tuner: Optional[OnlineTuner] = None,
  wild: bool = True,
) -> Tuple[DSState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  print("\nDirichlet Step - Start")
  dtype = logits.dtype
  bsz, vsz = logits.shape

  output_tokens = jnp.zeros(bsz, dtype=jnp.int32)
  EPS = jnp.array(1e-8, dtype=dtype)
  naked_log_probs = normalize_logits(logits, config.noise_floor)

  # update naked entropy rate
  naked_ent, naked_varent = ent_varent(naked_log_probs)


  # fix shape issue!
  new_emwa_ent_naked = update_emwa(
    naked_ent, state.emwa_ent_naked, config.emwa_ent_naked_coeff
  )
  new_emwa_varent_naked = update_emwa(
    naked_varent, state.emwa_varent_naked, config.emwa_varent_naked_coeff
  )
  print("Updated EMWA values")

  # entropy and varentropy vectors - shape (bsz, 4)
  state_ent = jnp.array(
    [
      state.token_cross_ent_scaffold,
      state.token_cross_ent_naked,
      state.emwa_ent_scaffold,
      state.emwa_ent_naked,
    ]
  ).T
  state_std = jnp.sqrt(
    jnp.array(
      [
        state.token_cross_var_scaffold,
        state.token_cross_var_naked,
        state.emwa_varent_scaffold,
        state.emwa_varent_naked,
      ]
    )
  ).T
  print("Computed state vectors")

  outlier_threshold = compute_outlier_threshold(
    state_ent, state_std, naked_ent, naked_varent, config
  )
  outlier_mask = outlier_threshold > 0
  # update emwa topk entropy
  topk_logits, topk_indices = jax.lax.top_k(naked_log_probs, config.outlier_topk)
  topk_logprobs = normalize_logits(topk_logits, config.noise_floor)
  naked_topk_ent, _ = ent_varent(topk_logprobs)
  new_emwa_topk_ent_naked = update_emwa(
    naked_topk_ent, state.emwa_topk_ent_naked, config.emwa_topk_ent_naked_coeff
  )
  """
  argmax policy for concentrated inliers
  """
  argmax_threshold = (
    config.argmax_threshold.weight * state.emwa_topk_ent_naked
    + config.argmax_threshold.bias
  )
  argmax_mask = ~outlier_mask & (naked_topk_ent < argmax_threshold)
  argmax_indices = jnp.argmax(topk_logprobs, axis=-1)
  argmax_tokens = jnp.take_along_axis(
    topk_indices, argmax_indices[:, None], axis=-1
  ).squeeze(1)
  output_tokens = jnp.where(argmax_mask, argmax_tokens, output_tokens)
  """
  topk temperature tuning policy for dispersed inliers
  """
  inlier_sampling_indices = ~outlier_mask & ~argmax_mask
  inlier_sampling_temp, _, _ = temp_tune(topk_logprobs, state.emwa_topk_ent_naked)
  sampling_inlier_choices = jax.random.categorical(
    key, topk_logprobs / inlier_sampling_temp[:, None]
  )
  sampling_inlier_tokens = jnp.take_along_axis(
    topk_indices, sampling_inlier_choices[:, None], axis=-1
  ).squeeze(1)
  output_tokens = jnp.where(
    inlier_sampling_indices, sampling_inlier_tokens, output_tokens
  )
  """
  tune temperature of outliers to match target entropy
  """
  target_entropy = (
    jnp.dot(state_ent, config.target_entropy.linear)
    + jnp.sum(config.target_entropy.linear_inv_temp / state.emwa_temp, axis=-1)
    + config.target_entropy.bias
  )
  temp, _, _ = temp_tune(naked_log_probs.astype(jnp.float32), target_entropy)
  new_emwa_temp = update_emwa(temp, state.emwa_temp, config.emwa_temp_coeff)
  tuned_logprobs = normalize_logits(
    naked_log_probs / jnp.clip(temp[:, None], MIN_TEMP, MAX_TEMP), config.noise_floor
  )
  """
  update emwa logp (on dirichlet support)
  """
  logprobs_on_supp = normalize_logits(
    tuned_logprobs[:, config.dirichlet_support], config.noise_floor
  )
  kl = jnp.sum(
    jnp.exp(logprobs_on_supp) * (logprobs_on_supp - state.emwa_logp_on_supp), axis=-1
  )
  emwa_logp_coeff = config.emwa_logp_base ** (-config.emwa_logp_exp_factor / (kl + EPS))
  new_emwa_logp_on_supp = update_emwa(
    logprobs_on_supp, state.emwa_logp_on_supp, emwa_logp_coeff[..., None]
  )
  new_emwa_dir, _, _ = fit_dirichlet(new_emwa_logp_on_supp)
  """
  update dirichlet and compute threshold
  """
  dir_log_likelihood = dirichlet_log_likelihood_from_logprob(
    logprobs_on_supp, state.emwa_dir
  )
  new_emwa_dir_ent = update_emwa(
    -dir_log_likelihood, state.emwa_dir_ent, config.emwa_dir_ent_coeff
  )
  dirichlet_threshold = (
    config.dirichlet_threshold.weight * state.emwa_dir_ent
    + config.dirichlet_threshold.bias
  )
  use_dirichlet = outlier_mask & (-dir_log_likelihood < dirichlet_threshold)
  if wild:  # if wild, sample from dirichlet, else use expectation
    dir_probs = sample_dirichlet(key, new_emwa_dir)
  else:
    dir_probs = dirichlet_expectation(new_emwa_dir)
  """
  below dirichlet threshold, interpolate and sample (can improve this in the future)
  """
  kl = jnp.sum(dir_probs * (jnp.log(dir_probs + EPS) - logprobs_on_supp), axis=-1)
  perturb_coeff = config.perturb_base_coeff / (1 + jnp.exp(config.perturb_exp_coeff * kl))
  interpolated_probs = perturb_coeff[:, None] * dir_probs + (
    1 - perturb_coeff[:, None]
  ) * jnp.exp(logprobs_on_supp)
  # Apply EXTREME DRY penalty
  interpolated_probs = apply_dry_penalty(
    interpolated_probs,
    state.tokens,
    multiplier=10.0,    # 2x stronger (was 5.0)
    base=5.0,           # Much faster growth (was 3.0)
    allowed_length=0,   # No repeats allowed! (was 1)
    window_size=1,   # 2x longer context (was 1024)
    max_check_length=64 # 2x longer lookback (was 32)
  )
    
  # Even wilder mode penalties
  if wild:
    # Apply stronger historical token penalties
    token_mask = jnp.zeros(interpolated_probs.shape[-1])
    token_mask = token_mask.at[state.tokens].add(1.0)
    
    # Stronger penalty for historical tokens
    interpolated_probs = jnp.where(
        token_mask > 0,
        interpolated_probs - 5.0,  # 2.5x stronger (was 2.0)
        interpolated_probs
    )
    
    # Add more extreme random noise to coefficients
    noise = jax.random.uniform(key, perturb_coeff.shape, minval=0.5, maxval=2.0)  # Much wider range
    perturb_coeff = perturb_coeff * noise
    
    # More random noise to thresholds
    noise = jax.random.uniform(key, dirichlet_threshold.shape, minval=0.5, maxval=1.1)
    dirichlet_threshold = dirichlet_threshold * noise

  # Ensure we don't have -inf or nan
  interpolated_probs = jnp.nan_to_num(interpolated_probs, nan=-1e9, posinf=1e9, neginf=-1e9)

  # in use_dirichlet case take argmax of the slided probs
  dicihlet_choices = jnp.argmax(interpolated_probs, axis=-1)
  dirichlet_tokens = jnp.take(config.dirichlet_support, dicihlet_choices)
  output_tokens = jnp.where(use_dirichlet, dirichlet_tokens, output_tokens)
  """
  above dirichlet threshold youre ngmi
  """
  ood_choices = jax.random.categorical(key, jnp.log(dir_probs + EPS))
  ood_tokens = jnp.take(config.dirichlet_support, ood_choices)
  output_tokens = jnp.where(outlier_mask & ~use_dirichlet, ood_tokens, output_tokens)
  # update scaffold entropy rate
  scaffold_ent, scaffold_varent = ent_varent(jnp.log(interpolated_probs + EPS))
  new_emwa_ent_scaffold = update_emwa(
    scaffold_ent, state.emwa_ent_scaffold, config.emwa_ent_scaffold_coeff
  )
  new_emwa_varent_scaffold = update_emwa(
    scaffold_varent, state.emwa_varent_scaffold, config.emwa_varent_scaffold_coeff
  )
  # update token cross entropies
  batch_indices = jnp.arange(bsz)
  scaffold_token_logprob = jnp.nan_to_num(
    jnp.log(interpolated_probs[batch_indices, output_tokens] + EPS),
    nan=0.0, posinf=1e6, neginf=-1e6
  )
  naked_token_logprob = jnp.nan_to_num(
    naked_log_probs[batch_indices, output_tokens],
    nan=0.0, posinf=1e6, neginf=-1e6
  )

  (
    new_token_cross_ent_scaffold,
    new_token_cross_ent_naked,
    new_token_cross_var_scaffold,
    new_token_cross_var_naked,
  ) = update_token_cross_entropies(
    state, scaffold_token_logprob, naked_token_logprob, config
  )
  if tuner is not None:
    # Instead of modifying tuner state directly, get new config only
    tuner_outputs = tuner.pure_update(
        jnp.log(interpolated_probs + EPS),
        naked_log_probs,
        new_token_cross_ent_naked,
        new_token_cross_ent_scaffold
    )
    # Unpack the returned values - don't modify tuner state
    new_config = tuner_outputs[0]  # Get just the config
    config = new_config  # Use new config for rest of function

  # First, let's add debug prints to understand the shapes
  jax.debug.print("state.tokens shape: {}", state.tokens.shape)
  jax.debug.print("output_tokens shape: {}", output_tokens.shape)
  jax.debug.print("state.tokens: {}", state.tokens)
  jax.debug.print("output_tokens: {}", output_tokens)

  # Then fix the concatenation and slicing
  if state.tokens.ndim == 1:
    # If 1D, concatenate and take last 32 tokens
    new_tokens = jnp.concatenate([state.tokens, output_tokens], axis=0)[-32:]
  else:
    # If 2D (batch, seq_len), concatenate along sequence dimension
    new_tokens = jnp.concatenate([state.tokens, output_tokens[:, None]], axis=1)[:, -32:]

  # Update state with proper tokens
  new_state = DSState(
    emwa_dir=new_emwa_dir,
    emwa_logp_on_supp=new_emwa_logp_on_supp,
    emwa_temp=new_emwa_temp,
    emwa_ent_scaffold=new_emwa_ent_scaffold,
    emwa_ent_naked=new_emwa_ent_naked,
    emwa_varent_scaffold=new_emwa_varent_scaffold,
    emwa_varent_naked=new_emwa_varent_naked,
    token_cross_ent_scaffold=new_token_cross_ent_scaffold,
    token_cross_ent_naked=new_token_cross_ent_naked,
    token_cross_var_scaffold=new_token_cross_var_scaffold,
    token_cross_var_naked=new_token_cross_var_naked,
    emwa_dir_ent=new_emwa_dir_ent,
    emwa_topk_ent_naked=new_emwa_topk_ent_naked,
    tokens=new_tokens
  )

  # Add debug prints
  jax.debug.print("Outlier mask: {}", outlier_mask)
  jax.debug.print("Dirichlet threshold: {}", dirichlet_threshold)
  jax.debug.print("Use dirichlet: {}", use_dirichlet)
  jax.debug.print("Perturb coeff: {}", perturb_coeff)
  jax.debug.print("KL: {}", kl)
  jax.debug.print("Dir probs entropy: {}", -jnp.sum(dir_probs * jnp.log(dir_probs + EPS), axis=-1))
  jax.debug.print("Logprobs entropy: {}", -jnp.sum(jnp.exp(logprobs_on_supp) * logprobs_on_supp, axis=-1))
  jax.debug.print("Output tokens: {}", output_tokens)

  print("\nDirichlet Step Debug - End")

  # Return all the values needed for tuning
  return (
    new_state, 
    output_tokens,
    naked_token_logprob,      # For cross entropy
    scaffold_token_logprob,   # For cross entropy
    jnp.log(interpolated_probs + EPS),  # scaffold logprobs
    naked_log_probs,          # naked logprobs
    new_token_cross_ent_naked,
    new_token_cross_ent_scaffold
  )


@jax.jit
def update_emwa(new: jax.Array, old: jax.Array, coeff: float | jax.Array) -> jax.Array:
  return coeff * new + (1 - coeff) * old


@partial(jax.jit, static_argnames=("config",))
def compute_outlier_threshold(state_ent, state_std, naked_ent, naked_varent, config):
  return (
    jnp.einsum("bi,ij,bj->b", state_ent, config.outlier_threshold.bilinear, state_std)
    + jnp.einsum("bi,i->b", state_ent, config.outlier_threshold.linear_state_ent)
    + jnp.einsum("bi,i->b", state_std, config.outlier_threshold.linear_state_std)
    + naked_ent * config.outlier_threshold.linear_naked_ent
    + naked_varent * config.outlier_threshold.linear_naked_varent
    + config.outlier_threshold.bias
  )


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

@partial(jax.jit, static_argnums=(6,))
def apply_dry_penalty(
    logits: jnp.ndarray,
    tokens: jnp.ndarray,
    multiplier: float = 5.0,
    base: float = 3.0,
    allowed_length: int = 1,
    window_size: int = 1024,
    max_check_length: int = 32
) -> jnp.ndarray:
    """Apply DRY penalty to logits to prevent repetitive sequences."""
    
    # Get batch size from logits
    bsz = logits.shape[0]
    
    # Ensure tokens have shape (bsz, seq_len)
    tokens = jnp.expand_dims(tokens, axis=0) if tokens.ndim == 1 else tokens
    tokens = jnp.tile(tokens, (bsz, 1)) if tokens.shape[0] == 1 else tokens
    
    # Define static slice sizes
    SLICE_SIZE = (max_check_length,)  # Static tuple for all slicing operations
    
    # Ensure we have at least 2 * max_check_length tokens by padding
    padded_tokens = jnp.pad(
        tokens,
        ((0, 0), (2 * max_check_length, 0)),
        mode='constant',
        constant_values=0
    )

    def compute_penalty(tokens_padded_seq: jnp.ndarray, logits_seq: jnp.ndarray) -> jnp.ndarray:
        penalties = jnp.zeros_like(logits_seq)
        
        def body_fn(i, penalties_acc):
            # Calculate effective length (dynamic)
            length = jnp.minimum(i + 1, max_check_length)
            
            # Calculate start positions (dynamic)
            current_start = 2 * max_check_length - length
            previous_start = 2 * max_check_length - 2 * length
            
            # Create slices with static size
            current = lax.dynamic_slice(
                tokens_padded_seq,
                (current_start,),  # Dynamic start index is fine
                SLICE_SIZE        # Static slice size
            )
            
            previous = lax.dynamic_slice(
                tokens_padded_seq,
                (previous_start,),  # Dynamic start index is fine
                SLICE_SIZE        # Static slice size
            )
            
            # Create length mask
            position_indices = jnp.arange(max_check_length)
            length_mask = position_indices < length
            
            # Compare sequences using JAX operations with masking
            comparison = jnp.equal(current, previous)
            masked_comparison = comparison & length_mask[:, None]  # Broadcasting for potential token dimension
            
            # Check matches only within effective length
            matches = jnp.all(jnp.where(length_mask, comparison, True))
            
            # Calculate penalty using pure JAX operations
            penalty = jnp.where(
                matches & (length > allowed_length),
                multiplier * (base ** (length - allowed_length)),
                0.0
            )
            
            # Get last valid token using dynamic_index_in_dim
            last_token = lax.dynamic_index_in_dim(current, length - 1, axis=0)
            
            # Update penalties using scatter_add
            return penalties_acc.at[last_token].add(penalty)
        
        # Use fori_loop for the iteration
        penalties = lax.fori_loop(
            0,
            max_check_length,
            body_fn,
            penalties
        )
        
        # Handle recent token penalties with static slice size
        seq_length = tokens_padded_seq.shape[0]
        recent_start = seq_length - max_check_length
        
        # Extract recent tokens with static slice size
        recent_tokens = lax.dynamic_slice(
            tokens_padded_seq,
            (recent_start,),
            SLICE_SIZE
        )
        
        # Create one-hot encoding
        one_hot = jax.nn.one_hot(
            recent_tokens,
            num_classes=logits_seq.shape[-1],
            dtype=logits_seq.dtype
        )
        
        # Sum and scale recent penalties
        recent_penalties = jnp.sum(one_hot, axis=0) * 0.5
        
        # Add recent penalties to the accumulated penalties
        return penalties + recent_penalties

    # Use vmap to process each batch element independently
    penalties = jax.vmap(compute_penalty, in_axes=(0, 0))(padded_tokens, logits)
    
    return logits - penalties
