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
def renyi_entropy(logp: jax.Array, alphas: jax.Array) -> jax.Array:
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
    # Convert inputs to arrays
    logp = jnp.asarray(logp)  # Shape: (..., vocab_size)
    alphas = jnp.asarray(alphas)  # Shape: (num_alphas,)
    assert alphas.ndim == 1, "alphas must be a 1D array"
    # Handle special cases
    is_alpha_0 = jnp.isclose(alphas, 0.0)
    is_alpha_1 = jnp.isclose(alphas, 1.0)
    
    # Replace -inf with very negative number to avoid NaN in exp
    logp = jnp.where(jnp.isneginf(logp), -1e30, logp)
    
    # Compute probabilities
    p = jnp.exp(logp)  # Shape: (..., vocab_size)
    
    # For alpha != 1 case: H_alpha = 1/(1-alpha) * log(sum(p^alpha))
    # Reshape alphas for broadcasting: (..., 1) * (num_alphas,) -> (..., num_alphas)
    log_p_alpha = jnp.where(
        p[..., None, :] > 0,  # Only compute for non-zero probabilities
        logp[..., None, :] * alphas[:, None],  # Shape: (..., num_alphas, vocab_size)
        -1e30  # Large negative number for zero probabilities
    )
    
    # Compute log sum exp carefully to avoid overflow/underflow
    max_log_p_alpha = jnp.max(log_p_alpha, axis=-1, keepdims=True)
    log_p_alpha_shifted = log_p_alpha - max_log_p_alpha
    log_sum_p_alpha = jnp.log(jnp.sum(jnp.exp(log_p_alpha_shifted), axis=-1)) + max_log_p_alpha[..., 0]
    
    # Handle the division by (1-alpha) carefully
    renyi_alpha = jnp.where(
        jnp.abs(1.0 - alphas) > 1e-6,  # Avoid division by very small numbers
        log_sum_p_alpha / (1.0 - alphas),
        jnp.zeros_like(log_sum_p_alpha)  # Will be replaced by Shannon entropy
    )
    
    # For alpha = 1 case (Shannon entropy)
    # Only include non-zero probabilities in the sum
    nonzero_mask = p > 0
    shannon = -jnp.sum(
        jnp.where(nonzero_mask, p * logp, 0.0),
        axis=-1
    )[..., None]  # Shape: (..., 1)
    
    # For alpha = 0 case (log of support size)
    log_support_size = jnp.log(jnp.sum(p > 0, axis=-1))[..., None]  # Shape: (..., 1)
    
    # Combine all cases
    result = jnp.where(
        is_alpha_0[None, :],
        jnp.broadcast_to(log_support_size, log_sum_p_alpha.shape),
        jnp.where(
            is_alpha_1[None, :],
            jnp.broadcast_to(shannon, log_sum_p_alpha.shape),
            renyi_alpha
        )
    )
    # Handle degenerate cases (all probability mass in one state)
    is_deterministic = jnp.sum(p > 0, axis=-1) <= 1
    result = jnp.where(is_deterministic[..., None], 0.0, result)
    
    return result  # Shape: (..., num_alphas)

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
