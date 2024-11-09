import jax.numpy as jnp
import jax
from typing import NamedTuple, Optional, Tuple



class EMWA(NamedTuple):
    # all arrays have the same shape and dtype
    weight: jax.Array
    mean: jax.Array 
    var: jax.Array
    inv_eff_size: jax.Array

    @staticmethod
    def new(bsz: int, weight: jax.Array, initial_mean: Optional[jax.Array] = None, initial_var: Optional[jax.Array] = None) -> 'EMWA':
        shape, dtype = weight.shape, weight.dtype
        if initial_mean is None:
            initial_mean = jnp.zeros((bsz,) + shape, dtype=dtype)
        if initial_var is None:
            initial_var = jnp.zeros((bsz,) + shape, dtype=dtype)
        return EMWA(
            weight=weight,
            mean=initial_mean,
            var=initial_var,
            inv_eff_size=jnp.zeros((bsz,) + shape, dtype=dtype)
        )
    
    def update(state: 'EMWA', sample: jax.Array) -> 'EMWA':
        new_mean = state.weight * sample + (1 - state.weight) * state.mean
        new_var = state.weight * (sample - new_mean) ** 2 + (1 - state.weight) * state.var
        new_inv_eff_size = state.weight ** 2 + (1 - state.weight) ** 2 * state.inv_eff_size
        new_unbiased_var = jnp.where(new_inv_eff_size > 0, new_var / (1 / new_inv_eff_size - 1), jnp.nan)
        return EMWA(
            weight=state.weight,
            mean=new_mean,
            var=new_var, 
            inv_eff_size=new_inv_eff_size
        ), new_mean, new_unbiased_var
    
    @property
    def mean_var(self) -> Tuple[jax.Array, jax.Array]:
        return self.mean, jnp.where(self.inv_eff_size > 0, self.var * self.inv_eff_size  / (1 -self.inv_eff_size), jnp.nan)



