from functools import partial
from typing import NamedTuple, Dict, Tuple
import jax
import jax.numpy as jnp
from entropix.dslider_config import DSConfig, OutlierThreshold, DirichletThreshold, ArgmaxThreshold, TargetEntropy
from jax.tree_util import register_pytree_node_class
class TuningStats(NamedTuple):
    cross_ent_diff: float
    renyi_div: float
    combined_score: float
    kl_div: float
    param_gradients: Dict[str, float]

@jax.jit
def renyi_divergence(p: jnp.ndarray, q: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """
    Compute Rényi divergence of order alpha:
    D_α(P||Q) = 1/(α-1) * log(∑ p^α * q^(1-α))

    For α = 1/R where R > 1:
    D_{1/R}(P||Q) = R/(1-R) * log(∑ p^(1/R) * q^(1-1/R))
    """
    # Ensure numerical stability
    p = p + 1e-10
    q = q + 1e-10

    # Normalize if needed
    p = p / jnp.sum(p, axis=-1, keepdims=True)
    q = q / jnp.sum(q, axis=-1, keepdims=True)

    # Compute powers
    p_power = jnp.power(p, alpha)
    q_power = jnp.power(q, 1 - alpha)

    # Compute sum term
    sum_term = jnp.sum(p_power * q_power, axis=-1)

    # Final computation
    return 1.0 / (alpha - 1.0) * jnp.log(sum_term)

class OnlineTuner:
    def __init__(
        self,
        config: DSConfig,
        R: float = 2.0,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        window_size: int = 100,
    ):
        assert R > 1, "R must be greater than 1"
        self.config = config
        self.R = R
        self.alpha = 1.0 / R  # For Rényi divergence
        self.lr = learning_rate
        self.momentum = momentum
        self.idx = 0
        self.max_idx = 50

        # Initialize parameter momentum buffers
        self.param_momentum = {
            'outlier_bilinear': jnp.zeros_like(config.outlier_threshold.bilinear),
            'outlier_linear_state_ent': jnp.zeros_like(config.outlier_threshold.linear_state_ent),
            'outlier_linear_state_std': jnp.zeros_like(config.outlier_threshold.linear_state_std),
            'dirichlet_weight': jnp.array(0.0),
            'dirichlet_bias': jnp.array(0.0),
            'perturb_base': jnp.array(0.0),
            'perturb_exp': jnp.array(0.0)
        }

        # Statistics tracking
        self.window_size = window_size
        self.stats_buffer = []
        self.total_steps = 0

    def tree_flatten(self):
        """For JAX pytree handling"""
        arrays = [
            self.param_momentum['outlier_bilinear'],
            self.param_momentum['outlier_linear_state_ent'],
            self.param_momentum['outlier_linear_state_std'],
            self.param_momentum['dirichlet_weight'],
            self.param_momentum['dirichlet_bias'],
            self.param_momentum['perturb_base'],
            self.param_momentum['perturb_exp']
        ]
        aux_data = {
            "config": self.config,
            "R": self.R,
            "learning_rate": self.lr,
            "momentum": self.momentum,
            "window_size": self.window_size,
            "stats_buffer": self.stats_buffer,
            "total_steps": self.total_steps
        }
        return arrays, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, arrays):
        """For JAX pytree handling"""
        instance = cls(
            config=aux_data["config"],
            R=aux_data["R"],
            learning_rate=aux_data["learning_rate"],
            momentum=aux_data["momentum"],
            window_size=aux_data["window_size"]
        )
        instance.param_momentum = {
            'outlier_bilinear': arrays[0],
            'outlier_linear_state_ent': arrays[1],
            'outlier_linear_state_std': arrays[2],
            'dirichlet_weight': arrays[3],
            'dirichlet_bias': arrays[4],
            'perturb_base': arrays[5],
            'perturb_exp': arrays[6]
        }
        instance.stats_buffer = aux_data["stats_buffer"]
        instance.total_steps = aux_data["total_steps"]
        return instance

    def __hash__(self):
        return hash((
            self.config,
            self.R,
            self.lr,
            self.momentum,
            self.window_size,
            self.total_steps
        ))

    @partial(jax.jit, static_argnames=('self',))
    def compute_metrics(
        self,
        scaffold_logprobs: jnp.ndarray,
        naked_logprobs: jnp.ndarray,
        token_cross_ent_naked: jnp.ndarray,
        token_cross_ent_scaffold: jnp.ndarray
    ) -> TuningStats:
        """Compute current performance metrics"""
        # Cross entropy difference - add numerical stability
        cross_ent_diff = jnp.mean(jnp.nan_to_num(
            token_cross_ent_naked - token_cross_ent_scaffold,
            nan=0.0, posinf=1e6, neginf=-1e6
        ))

        # Convert logprobs to probs with numerical stability
        scaffold_probs = jnp.exp(jnp.clip(scaffold_logprobs, -1e6, 1e6))
        naked_probs = jnp.exp(jnp.clip(naked_logprobs, -1e6, 1e6))

        # Normalize probabilities
        scaffold_probs = scaffold_probs / jnp.sum(scaffold_probs, axis=-1, keepdims=True)
        naked_probs = naked_probs / jnp.sum(naked_probs, axis=-1, keepdims=True)

        # Compute KL divergence with numerical stability
        kl_div = jnp.mean(jnp.sum(
            scaffold_probs * jnp.clip(scaffold_logprobs - naked_logprobs, -1e6, 1e6),
            axis=-1
        ))

        # Compute Rényi divergence with α = 1/R
        renyi_div = jnp.mean(renyi_divergence(scaffold_probs, naked_probs, self.alpha))

        # Combined score using the same weighting
        combined_score = jnp.nan_to_num(
            (1.0/self.R) * cross_ent_diff + ((self.R-1.0)/self.R) * renyi_div,
            nan=0.0, posinf=1e6, neginf=-1e6
        )

        # Compute gradients for parameters
        param_gradients = self.compute_parameter_gradients(
            scaffold_logprobs, naked_logprobs,
            token_cross_ent_naked, token_cross_ent_scaffold
        )

        return TuningStats(cross_ent_diff, renyi_div, combined_score, kl_div, param_gradients)

    def get_summary(self) -> str:
        """Generate a summary of tuning statistics"""
        if not self.stats_buffer:
            return "No tuning statistics available"

        recent_stats = self.stats_buffer[-self.window_size:]
        # Convert JAX arrays to Python floats
        avg_cross_ent = float(sum(s.cross_ent_diff for s in recent_stats) / len(recent_stats))
        avg_renyi = float(sum(s.renyi_div for s in recent_stats) / len(recent_stats))
        avg_score = float(sum(s.combined_score for s in recent_stats) / len(recent_stats))
        avg_kl = float(sum(s.kl_div for s in recent_stats) / len(recent_stats))

        return f"""
Online Tuning Summary (R={self.R}, α=1/R={self.alpha:.4f}):
---------------------
Total Steps: {self.total_steps}
Recent Window Statistics (last {len(recent_stats)} steps):
- Average Cross Entropy Difference: {avg_cross_ent:.4f}
- Average Rényi Divergence (α=1/R): {avg_renyi:.4f}
- Average KL Divergence: {avg_kl:.4f}
- Average Combined Score: {avg_score:.4f}

Current Parameter Values:
- Outlier Threshold:
  - Bilinear: {float(self.config.outlier_threshold.bilinear.mean()):.4f}
  - Linear State Ent: {float(self.config.outlier_threshold.linear_state_ent.mean()):.4f}
  - Linear State Std: {float(self.config.outlier_threshold.linear_state_std.mean()):.4f}
- Dirichlet Threshold:
  - Weight: {float(self.config.dirichlet_threshold.weight):.4f}
  - Bias: {float(self.config.dirichlet_threshold.bias):.4f}
- Perturbation:
  - Base Coefficient: {float(self.config.perturb_base_coeff):.4f}
  - Exp Coefficient: {float(self.config.perturb_exp_coeff):.4f}
"""

    def update(
        self,
        scaffold_logprobs: jnp.ndarray,
        naked_logprobs: jnp.ndarray,
        token_cross_ent_naked: jnp.ndarray,
        token_cross_ent_scaffold: jnp.ndarray
    ) -> DSConfig:
        """Update tuner state and return optimized config"""
        # Compute current metrics and gradients
        stats = self.compute_metrics(
            scaffold_logprobs,
            naked_logprobs,
            token_cross_ent_naked,
            token_cross_ent_scaffold
        )

        print("\nGradients:")
        for param_name, gradient in stats.param_gradients.items():
            # Handle NaN gradients
            gradient = jnp.nan_to_num(gradient, nan=0.0, posinf=1.0, neginf=-1.0)
            print(f"{param_name}: {gradient}")

        # Update momentum buffers and apply gradients
        print("\nMomentum Updates:")
        for param_name, gradient in stats.param_gradients.items():
            # Handle NaN gradients
            gradient = jnp.nan_to_num(gradient, nan=0.0, posinf=1.0, neginf=-1.0)
            
            old_momentum = self.param_momentum[param_name]
            # Handle NaN momentum
            old_momentum = jnp.nan_to_num(old_momentum, nan=0.0, posinf=1.0, neginf=-1.0)
            
            self.param_momentum[param_name] = (
                self.momentum * old_momentum +
                (1 - self.momentum) * gradient
            )
            print(f"{param_name} momentum: {old_momentum} -> {self.param_momentum[param_name]}")
            
            # Actually apply the gradient update with clipping
            print(f"\nParameter Updates for {param_name}:")
            update = jnp.clip(self.lr * self.param_momentum[param_name], -0.1, 0.1)
            # Handle NaN updates
            update = jnp.nan_to_num(update, nan=0.0, posinf=0.1, neginf=-0.1)
            
            if param_name == 'outlier_bilinear':
                old_val = self.config.outlier_threshold.bilinear
                self.config.outlier_threshold.bilinear += update
                # Keep positive
                self.config.outlier_threshold.bilinear = jnp.maximum(self.config.outlier_threshold.bilinear, 0.1)
                print(f"Old: {old_val}, Update: {update}, New: {self.config.outlier_threshold.bilinear}")
                
            elif param_name == 'outlier_linear_state_ent':
                old_val = self.config.outlier_threshold.linear_state_ent
                self.config.outlier_threshold.linear_state_ent += update
                self.config.outlier_threshold.linear_state_ent = jnp.maximum(self.config.outlier_threshold.linear_state_ent, 0.1)
                print(f"Old: {old_val}, Update: {update}, New: {self.config.outlier_threshold.linear_state_ent}")
                
            elif param_name == 'outlier_linear_state_std':
                old_val = self.config.outlier_threshold.linear_state_std
                self.config.outlier_threshold.linear_state_std += update
                self.config.outlier_threshold.linear_state_std = jnp.maximum(self.config.outlier_threshold.linear_state_std, 0.1)
                print(f"Old: {old_val}, Update: {update}, New: {self.config.outlier_threshold.linear_state_std}")
                
            elif param_name == 'dirichlet_weight':
                old_val = self.config.dirichlet_threshold.weight
                self.config.dirichlet_threshold.weight += update
                self.config.dirichlet_threshold.weight = jnp.maximum(self.config.dirichlet_threshold.weight, 0.01)
                print(f"Old: {old_val}, Update: {update}, New: {self.config.dirichlet_threshold.weight}")
                
            elif param_name == 'dirichlet_bias':
                old_val = self.config.dirichlet_threshold.bias
                self.config.dirichlet_threshold.bias += update
                print(f"Old: {old_val}, Update: {update}, New: {self.config.dirichlet_threshold.bias}")
                
            elif param_name == 'perturb_base':
                old_val = self.config.perturb_base_coeff
                self.config.perturb_base_coeff += update
                self.config.perturb_base_coeff = jnp.maximum(self.config.perturb_base_coeff, 1.0)
                print(f"Old: {old_val}, Update: {update}, New: {self.config.perturb_base_coeff}")
                
            elif param_name == 'perturb_exp':
                old_val = self.config.perturb_exp_coeff
                self.config.perturb_exp_coeff += update
                self.config.perturb_exp_coeff = jnp.maximum(self.config.perturb_exp_coeff, 0.1)
                print(f"Old: {old_val}, Update: {update}, New: {self.config.perturb_exp_coeff}")

        # Update statistics buffer
        self.stats_buffer.append(stats)
        if len(self.stats_buffer) > self.window_size:
            self.stats_buffer.pop(0)
        self.total_steps += 1

        print("\nMetrics:")
        print(f"Cross Entropy Diff: {stats.cross_ent_diff}")
        print(f"Renyi Divergence: {stats.renyi_div}")
        print(f"KL Divergence: {stats.kl_div}")
        print(f"Combined Score: {stats.combined_score}\n")

        return self.config

    @partial(jax.jit, static_argnames=('self',))
    def compute_parameter_gradients(
        self,
        scaffold_logprobs_input: jnp.ndarray,
        naked_logprobs_input: jnp.ndarray,
        token_cross_ent_naked: jnp.ndarray,
        token_cross_ent_scaffold: jnp.ndarray
    ) -> Dict[str, float]:
        """
        Compute gradients for parameters with respect to the objective:
        score = (1/R) * (cross_ent_naked - cross_ent_scaffold) +
                ((R-1)/R) * D_{1/R}(scaffold_logprobs||naked_logprobs)
        """
        def objective_fn(params):
            # Compute entropies - make sure they're all the same shape
            scaffold_entropy = -jnp.sum(jnp.exp(scaffold_logprobs_input) * scaffold_logprobs_input, axis=-1)
            naked_entropy = -jnp.sum(jnp.exp(naked_logprobs_input) * naked_logprobs_input, axis=-1)

            # Reshape cross entropies to match entropy shapes
            token_cross_ent_scaffold_reshaped = token_cross_ent_scaffold.squeeze()
            token_cross_ent_naked_reshaped = token_cross_ent_naked.squeeze()

            # Get state entropy vector - shape (bsz, 4) with NaN handling
            state_ent = jnp.nan_to_num(jnp.stack([
                token_cross_ent_scaffold_reshaped,  # token cross entropy scaffold
                token_cross_ent_naked_reshaped,     # token cross entropy naked
                scaffold_entropy,                    # scaffold entropy
                naked_entropy                        # naked entropy
            ], axis=1), nan=0.0)  # Replace NaN with 0.0

            # Compute variances with NaN handling
            scaffold_varent = jnp.nan_to_num(jnp.var(scaffold_entropy), nan=0.0)
            naked_varent = jnp.nan_to_num(jnp.var(naked_entropy), nan=0.0)
            cross_var_scaffold = jnp.nan_to_num(jnp.var(token_cross_ent_scaffold_reshaped), nan=0.0)
            cross_var_naked = jnp.nan_to_num(jnp.var(token_cross_ent_naked_reshaped), nan=0.0)

            # Get state std vector - shape (bsz, 4) with NaN handling
            state_std = jnp.nan_to_num(jnp.stack([
                jnp.full_like(scaffold_entropy, jnp.sqrt(cross_var_scaffold)),
                jnp.full_like(scaffold_entropy, jnp.sqrt(cross_var_naked)),
                jnp.full_like(scaffold_entropy, jnp.sqrt(scaffold_varent)),
                jnp.full_like(scaffold_entropy, jnp.sqrt(naked_varent))
            ], axis=1), nan=0.0)  # Replace NaN with 0.0

            # Get dirichlet entropy
            dir_ent = scaffold_entropy  # Use scaffold entropy for dirichlet

            # Compute cross entropy difference
            cross_ent_diff = jnp.mean(token_cross_ent_naked - token_cross_ent_scaffold)

            # Convert logprobs to probs with numerical stability
            scaffold_probs = jnp.exp(jnp.clip(scaffold_logprobs_input, -1e6, 1e6))
            naked_probs = jnp.exp(jnp.clip(naked_logprobs_input, -1e6, 1e6))

            # Normalize probabilities
            scaffold_probs = scaffold_probs / jnp.sum(scaffold_probs, axis=-1, keepdims=True)
            naked_probs = naked_probs / jnp.sum(naked_probs, axis=-1, keepdims=True)

            # Compute Renyi divergence
            renyi_div = jnp.mean(renyi_divergence(scaffold_probs, naked_probs, self.alpha))

            # Compute thresholds with NaN handling
            outlier_threshold = jnp.nan_to_num(
                jnp.einsum('bi,ij,bj->b', state_ent, params['outlier_bilinear'], state_std) +
                jnp.einsum('bi,i->b', state_ent, params['outlier_linear_state_ent']) +
                jnp.einsum('bi,i->b', state_std, params['outlier_linear_state_std']),
                nan=0.0
            )

            dirichlet_threshold = params['dirichlet_weight'] * dir_ent + params['dirichlet_bias']

            # Compute perturbation coefficient
            kl = jnp.sum(jnp.exp(scaffold_logprobs_input) * (scaffold_logprobs_input - naked_logprobs_input), axis=-1)
            perturb_coeff = params['perturb_base'] / (1 + jnp.exp(params['perturb_exp'] * kl))

            # Compute objective terms
            outlier_term = jnp.mean(jnp.abs(outlier_threshold))  # Encourage non-zero thresholds
            dirichlet_term = jnp.mean(jnp.abs(dirichlet_threshold))  # Encourage non-zero thresholds
            perturb_term = jnp.mean(perturb_coeff)  # Encourage larger perturbation
            
            # Add entropy bonus to encourage exploration
            entropy_bonus = -jnp.mean(
                jnp.sum(scaffold_probs * jnp.log(scaffold_probs + 1e-10), axis=-1)
            )

            # Combined objective with exploration bonus
            return (1.0/self.R) * cross_ent_diff + ((self.R-1.0)/self.R) * renyi_div + \
                   0.1 * outlier_term + 0.1 * dirichlet_term + 0.1 * perturb_term + \
                   0.05 * entropy_bonus  # Add entropy bonus with smaller weight

        # Get current parameter values - ensure all are float32
        params = {
            'outlier_bilinear': jnp.asarray(self.config.outlier_threshold.bilinear, dtype=jnp.float32),
            'outlier_linear_state_ent': jnp.asarray(self.config.outlier_threshold.linear_state_ent, dtype=jnp.float32),
            'outlier_linear_state_std': jnp.asarray(self.config.outlier_threshold.linear_state_std, dtype=jnp.float32),
            'dirichlet_weight': jnp.asarray(self.config.dirichlet_threshold.weight, dtype=jnp.float32),
            'dirichlet_bias': jnp.asarray(self.config.dirichlet_threshold.bias, dtype=jnp.float32),
            'perturb_base': jnp.asarray(self.config.perturb_base_coeff, dtype=jnp.float32),
            'perturb_exp': jnp.asarray(self.config.perturb_exp_coeff, dtype=jnp.float32)
        }

        # Use the fixed dirichlet_support from config
        dirichlet_support = jnp.asarray(self.config.dirichlet_support, dtype=jnp.int32)

        def wrapped_objective_fn(float_params):
            # Reconstruct full params dict with both float and int params
            full_params = {
                **float_params,
                'dirichlet_support': dirichlet_support  # Use fixed integer array
            }
            return objective_fn(full_params)

        # Compute gradients through the full computation
        gradients = jax.grad(wrapped_objective_fn)(params)
        return gradients

    def pure_update(
        self,
        scaffold_logprobs: jnp.ndarray,
        naked_logprobs: jnp.ndarray,
        token_cross_ent_naked: jnp.ndarray,
        token_cross_ent_scaffold: jnp.ndarray
    ) -> Tuple[DSConfig, "OnlineTuner"]:
        """Pure version of update that returns new state instead of modifying in place"""
        # Compute current metrics and gradients
        stats = self.compute_metrics(
            scaffold_logprobs,
            naked_logprobs,
            token_cross_ent_naked,
            token_cross_ent_scaffold
        )

        # Create new momentum values with updates
        new_momentum = {}
        for param_name, gradient in stats.param_gradients.items():
            old_momentum = self.param_momentum[param_name]
            new_momentum[param_name] = (
                self.momentum * old_momentum +
                (1 - self.momentum) * gradient
            )

        # Create new config with updated parameters
        new_config = DSConfig(
            # Keep all non-tuned parameters the same
            emwa_logp_base=self.config.emwa_logp_base,
            emwa_logp_exp_factor=self.config.emwa_logp_exp_factor,
            emwa_dir_coeff=self.config.emwa_dir_coeff,
            emwa_temp_coeff=self.config.emwa_temp_coeff,
            emwa_dir_ent_coeff=self.config.emwa_dir_ent_coeff,
            emwa_ent_scaffold_coeff=self.config.emwa_ent_scaffold_coeff,
            emwa_varent_scaffold_coeff=self.config.emwa_varent_scaffold_coeff,
            emwa_ent_naked_coeff=self.config.emwa_ent_naked_coeff,
            emwa_varent_naked_coeff=self.config.emwa_varent_naked_coeff,
            emwa_topk_ent_naked_coeff=self.config.emwa_topk_ent_naked_coeff,
            token_cross_ent_scaffold_coeff=self.config.token_cross_ent_scaffold_coeff,
            token_cross_ent_naked_coeff=self.config.token_cross_ent_naked_coeff,
            token_cross_var_scaffold_coeff=self.config.token_cross_var_scaffold_coeff,
            token_cross_var_naked_coeff=self.config.token_cross_var_naked_coeff,
            dirichlet_support=self.config.dirichlet_support,
            noise_floor=self.config.noise_floor,
            outlier_topk=self.config.outlier_topk,

            # Update tuned parameters with clipped values
            outlier_threshold=OutlierThreshold(
                bilinear=jnp.maximum(
                    self.config.outlier_threshold.bilinear + 
                    self.lr * new_momentum['outlier_bilinear'],
                    0.1
                ),
                linear_state_ent=jnp.maximum(
                    self.config.outlier_threshold.linear_state_ent + 
                    self.lr * new_momentum['outlier_linear_state_ent'],
                    0.1
                ),
                linear_state_std=jnp.maximum(
                    self.config.outlier_threshold.linear_state_std + 
                    self.lr * new_momentum['outlier_linear_state_std'],
                    0.1
                ),
                linear_naked_ent=self.config.outlier_threshold.linear_naked_ent,
                linear_naked_std=self.config.outlier_threshold.linear_naked_std,
                linear_naked_varent=self.config.outlier_threshold.linear_naked_varent,
                bias=self.config.outlier_threshold.bias
            ),
            argmax_threshold=self.config.argmax_threshold,
            dirichlet_threshold=DirichletThreshold(
                weight=jnp.maximum(
                    self.config.dirichlet_threshold.weight + 
                    self.lr * new_momentum['dirichlet_weight'],
                    0.01
                ),
                bias=self.config.dirichlet_threshold.bias + 
                     self.lr * new_momentum['dirichlet_bias']
            ),
            target_entropy=self.config.target_entropy,
            perturb_base_coeff=jnp.maximum(
                self.config.perturb_base_coeff + 
                self.lr * new_momentum['perturb_base'],
                1.0
            ),
            perturb_exp_coeff=jnp.maximum(
                self.config.perturb_exp_coeff + 
                self.lr * new_momentum['perturb_exp'],
                0.1
            )
        )

        # Create new tuner with updated state
        new_tuner = OnlineTuner(
            config=new_config,
            R=self.R,
            learning_rate=self.lr,
            momentum=self.momentum,
            window_size=self.window_size
        )
        new_tuner.param_momentum = new_momentum
        new_tuner.stats_buffer = self.stats_buffer + [stats]
        new_tuner.total_steps = self.total_steps + 1

        return new_config, new_tuner

register_pytree_node_class(OnlineTuner)