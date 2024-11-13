from dataclasses import dataclass
from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from entropix.dslider import DSState, adaptive_dirichlet_step, initialize_state
from entropix.dslider_config import DSConfig, DEFAULT_DS_CONFIG


MAX_K = 256
LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E


@dataclass
class SamplerConfig:
  # Naked (logits) entropy thresholds
  low_naked_entropy_threshold = 0.3  # Captures most observed LELV cases
  medium_naked_entropy_threshold = 1.2  # Separates medium from high entropy cases
  high_naked_entropy_threshold = 2.5  # Above this we see clear high entropy cases

  # Naked (logits) varentropy thresholds
  low_naked_varentropy_threshold = 1.2  # Most LELV cases are below this
  high_naked_varentropy_threshold = 2.5  # Clear separation for high variance cases

  # Scaffold (attention) metrics thresholds
  # These don't appear in logs, keeping unchanged
  low_scaffold_entropy_threshold = 1.0
  high_scaffold_entropy_threshold = 2.0
  low_scaffold_varentropy_threshold = 0.3
  high_scaffold_varentropy_threshold = 0.8


@partial(jax.jit, static_argnames=("config",))
def sample(
  state: DSState,
  logits: jnp.ndarray,
  config: DSConfig,
  clarifying_question_token: int = 2564,
  key=jax.random.PRNGKey(1337),
) -> Tuple[DSState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Sample from logits using adaptive dirichlet sampling."""
  jax.debug.print("Sampler Debug - Logits shape: {}", logits.shape)
  jax.debug.print("Sampler Debug - Config perturb base: {}", config.perturb_base_coeff)
  
  result = adaptive_dirichlet_step(
    key,
    state,
    logits,
    config,
    tuner=None,  # Don't pass tuner through sampler
    wild=True,
  )
  
  jax.debug.print("Sampler Debug - Output tokens: {}", result[1])
  
  return result
