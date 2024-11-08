import jax.numpy as jnp
import jax
from typing import NamedTuple, Optional, Tuple

# state variables: 
# EMWA for renyi entropies
# EMWA for dirichlet parameters
# EMWA for average empirical entropy of topk distribution for a collection of k's determined by the config. example: k=1, 2, 4, 8, 16, 32, 64, 128
# EMWA of empirical entropy rate of the actual sampled tokens. 

# first gate determines if dirichlet or not dirichlet
# second gate determines the top_k
# third gate determines the temperature

# ds config: 
# noise floor 
# naked emwa coeff: shape (n_topk, n_renyi) where n_topk is the number of topk sizes tracked determined by the config. example: n_topk=8 with k=2, 4, 8, 16, 32, 64, 128, vocab_size) and n_renyi is the number of renyi parameters tracked determined by the config. example: n_renyi=8 with alpha=0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8
# scaffold emwa coeff: shape (n_topk, n_renyi) where n_topk is the number of topk sizes tracked determined by the config. example: n_topk=8 with k=2, 4, 8, 16, 32, 64, 128, vocab_size) and n_renyi is the number of renyi parameters tracked determined by the config. example: n_renyi=8 with alpha=0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8
# token cross entropy naked emwa coeff: shape (n_topk) where n_topk is the number of topk sizes tracked determined by the config. example: n_topk=8 with k=2, 4, 8, 16, 32, 64, 128
# token cross entropy scaffold emwa coeff: shape (n_topk) where n_topk is the number of topk sizes tracked determined by the config. example: n_topk=8 with k=2, 4, 8, 16, 32, 64, 128
# dirichlet emwa coeff: shape (vocab_size)
# sampling modes:
# argmax (deterministic) - special case of topk for k=1
# top_k + adaptive temperature (nondeterministic)
    # temperature
    # min_p
# dirichlet 




# temperature
# min_p
# top_p
# top_k



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

# gates are fed EMWA.update(state)[1] and EMWA.update(state)[2] and EMWA.update(state)[0] is how the state is carried forward


