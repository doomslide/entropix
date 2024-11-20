import jax.numpy as jnp
import jax
from typing import Optional
from functools import partial
SAFE_LOG_EPS = 1e-20
DEFAULT_NOISE_FLOOR = jnp.log(1e-5)
ALPHA_NEAR_ONE_THRESHOLD = 1e-3  # Region around α=1
ALPHA_NEAR_ZERO_THRESHOLD = 1e-6  # Region around α=0
ALPHA_LARGE_THRESHOLD = 10.0      # Region for large α
EFF_NEG_INF = -1e30
LARGE_ALPHA = 1e5
SMALL_ALPHA = 1e-5
SMALL_ONE_MINUS_ALPHA= 1e-5
# Create a new function that wraps jnp.nonzero with a static size argument
nonzero_static = jax.jit(jnp.nonzero, static_argnames='size')

@jax.jit
def normalize_logits(logits: jnp.ndarray, noise_floor: float = DEFAULT_NOISE_FLOOR) -> jnp.ndarray:
    """
    Normalize logits to log probabilities with noise floor truncation.

    Args:
        logits (jnp.ndarray): Input logits.
        noise_floor (float): Minimum probability to retain.

    Returns:
        jnp.ndarray: Normalized log probabilities.
    """
    # Shift logits for numerical stability
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
    normalized = shifted - jax.nn.logsumexp(shifted, axis=-1, keepdims=True)
    clipped = jnp.where(normalized<noise_floor, EFF_NEG_INF, normalized)
    renormalized = clipped - jax.nn.logsumexp(clipped, axis=-1, keepdims=True)
    return renormalized

@jax.jit
def entropy(logp: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Shannon entropy (α=1 case) from log probabilities.
    Uses the formula: H = -∑(p_i * log(p_i))
    
    Args:
        logp: Log probabilities, shape (..., vocab_size)
    Returns:
        Shannon entropy, shape (...)
    """
    p = jnp.exp(logp)
    return -jnp.sum(p * logp, axis=-1)

@jax.jit
def renyi_entropy(
    logp: jax.Array, 
    alphas: jax.Array, 
    small_alpha: float = SMALL_ALPHA, 
    large_alpha: float=LARGE_ALPHA, 
    small_one_minus_alpha: float = SMALL_ONE_MINUS_ALPHA
) -> jax.Array:
    """
    Compute the Rényi entropy for given log probabilities and alpha values.
    
    Args:
        logp (jnp.ndarray): Log probabilities, shape (..., vocab_size) where ... represents
                           any number of leading dimensions
        alphas (jnp.ndarray): Rényi parameter(s), shape (num_alphas,)
        
    Returns:
        jnp.ndarray: Rényi entropies, shape (..., num_alphas) where ... matches the
                    leading dimensions of input logp
    """

    # Expand alphas to align with logp dimensions for broadcasting
    alphas_expanded = alphas.reshape((1,) * (logp.ndim - 1) + alphas.shape)
    # Compute probabilities from log probabilities
    p = jnp.exp(logp)
    # Create masks for different alpha cases
    mask_shannon = jnp.abs(alphas_expanded - 1.0) < small_one_minus_alpha
    mask_max = alphas_expanded > large_alpha
    mask_min = alphas_expanded < small_alpha
    # Shannon entropy approximation
    shannon_entropy = -jnp.sum(p * logp, axis=-1)
    # Argmax approximation for very large alpha
    max_p = jnp.max(p, axis=-1)
    max_entropy = -jnp.log(max_p)
    # Argmin approximation for very small alpha
    min_p = jnp.min(p, axis=-1)
    min_entropy = -jnp.log(min_p)
    # General Renyi entropy computation
    sum_p_alpha = jnp.sum(p ** alphas_expanded, axis=-1)
    renyi = jnp.log(sum_p_alpha) / (1 - alphas_expanded)
    # Combine results based on the masks
    entropy = jnp.where(mask_shannon,shannon_entropy,jnp.where(mask_max,max_entropy,jnp.where(mask_min, min_entropy, renyi)))
    return entropy




@jax.jit
def ent_grad_hess(
  logits: jnp.ndarray, T: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  p = jax.nn.softmax(logits / T[..., None], axis=-1)
  log_p = jax.nn.log_softmax(logits / T[..., None], axis=-1)
  mu1 = jnp.sum(p * log_p, axis=-1)
  diff = log_p - mu1[..., None]
  mu2 = jnp.sum(p * diff**2, axis=-1)
  mu3 = jnp.sum(p * diff**3, axis=-1)
  return -mu1, mu2 / T, -(2 * mu3 + 3 * mu2) / (T * T)


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def temp_tune(
  logits: jnp.ndarray,
  target_ent: jnp.ndarray,
  T_init: float = 1.0,
  lr: float = 0.1,
  max_iters: int = 10,
  tol: float = 1e-6,
  dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  batch_size = logits.shape[0]
  logits = logits.astype(jnp.float32)

  def scan_body(carry, _):
    T, iters, converged = carry
    ent, grad, hess = ent_grad_hess(logits, T)
    error = ent - target_ent
    new_converged = converged | (jnp.abs(error) < tol)
    denominator = 2 * grad * grad - error * hess
    halley_step = jnp.where(
      jnp.abs(denominator) > 1e-8,
      2 * error * grad / denominator,
      jnp.full_like(T, jnp.inf),
    )
    newton_step = jnp.where(
      jnp.abs(grad) > 1e-8, error / grad, jnp.full_like(T, jnp.inf)
    )
    grad_step = jnp.where(error > 0, lr * T, -lr * T)

    delta_T = jnp.where(
      jnp.abs(grad) < 1e-8,
      grad_step,
      jnp.where(jnp.abs(denominator) < 1e-8, newton_step, halley_step),
    )
    delta_T = jnp.clip(delta_T, -0.5 * T, 0.5 * T)
    new_T = jnp.where(new_converged, T, jnp.maximum(T - delta_T, T / 2))
    return (new_T, iters + 1, new_converged), None

  init_state = (
    jnp.full((batch_size,), T_init, dtype=jnp.float32),
    jnp.zeros(batch_size, dtype=jnp.int32),
    jnp.zeros(batch_size, dtype=jnp.bool_),
  )
  (final_T, final_iters, final_converged), _ = jax.lax.scan(
    scan_body, init_state, None, length=max_iters
  )
  return final_T.astype(dtype), final_iters, final_converged
