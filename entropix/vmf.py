import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp, digamma
from functools import partial
from typing import NamedTuple
import warnings
import json
from datetime import datetime

# Suppress warnings and configure JAX
warnings.filterwarnings("ignore", category=FutureWarning)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)


class ModelState(NamedTuple):
    """State of the von Mises-Fisher mixture model."""
    n_clusters: int
    d_model: int
    dir_prior: float
    mean_prior_conc: float
    mix_dir: jnp.ndarray       # Shape: (n_clusters,)
    means: jnp.ndarray         # Shape: (n_clusters, d_model)
    conc: jnp.ndarray          # Shape: (n_clusters,)
    cluster_counts: jnp.ndarray  # Shape: (n_clusters,)
    dir_sums: jnp.ndarray      # Shape: (n_clusters, d_model)
    sample_counts: jnp.ndarray  # Track samples per cluster

@jax.jit
def log_bessel_iv_saddle(nu, kappa):
    """Compute log of modified Bessel function using saddle point approximation."""
    t = kappa / nu
    sqrt_term = jnp.sqrt(1 + t**2)
    eta = sqrt_term + jnp.log(t / (1 + sqrt_term) + 1e-10)
    return nu * eta - 0.5 * jnp.log(2 * jnp.pi * nu) - 0.5 * jnp.log(1 + t**2)

@jax.jit
def estimate_kappa(r_bar, d, n_samples):
    """
    Hybrid kappa estimation combining high-dimensional asymptotics with small-sample corrections.
    
    Args:
        r_bar: Mean resultant length
        d: Dimension of the space
        n_samples: Number of samples used
    """
    r_bar = jnp.clip(r_bar, 1e-8, 1 - 1e-8)
    
    # Base estimation using the critical threshold d/2
    kappa_base = r_bar * d / (1 - r_bar ** 2)
    
    def small_sample_estimate():
        # Best-Fisher estimator with jackknife bias correction
        jackknife_correction = (d - 1) / (2 * kappa_base)
        return kappa_base * (1 - jackknife_correction)
    
    def large_sample_estimate():
        # Original high-dimensional estimator with asymptotic corrections
        c1 = (d - 1) / (2 * kappa_base)
        c2 = (d - 1) * (d - 3) / (8 * kappa_base ** 2)
        correction = 1 + c1 + c2
        
        # Scale kappa to maintain non-degenerate behavior
        kappa = kappa_base * correction
        
        # Ensure minimum value scales with dimension
        kappa = jnp.maximum(kappa, d/2)
        
        # Upper bound based on numerical stability
        max_kappa = 100 * d  # Scale with dimension
        return jnp.clip(kappa, d/2, max_kappa)
    
    # Smooth transition between estimators based on sample size
    def transition_estimate():
        # Linear interpolation between small and large sample estimates
        weight = (n_samples - 8) / 8  # Goes from 0 at n=8 to 1 at n=16
        small_est = small_sample_estimate()
        large_est = large_sample_estimate()
        return (1 - weight) * small_est + weight * large_est
    
    # Choose estimation method based on sample size and dimension
    # For very high dimensions (d > 1000), always use large sample estimate
    use_large = jnp.logical_or(n_samples >= 16, d > 1000)
    use_transition = jnp.logical_and(n_samples >= 8, n_samples < 16)
    
    return jnp.where(use_large,
                    large_sample_estimate(),
                    jnp.where(use_transition,
                             transition_estimate(),
                             small_sample_estimate()))

def generate_well_separated_centers(key, n_clusters, d_model):
    """Generate well-separated mean directions using Gram-Schmidt orthogonalization."""
    vectors = jax.random.normal(key, (n_clusters, d_model))
    Q = jnp.zeros((n_clusters, d_model))
    Q = Q.at[0].set(vectors[0] / jnp.linalg.norm(vectors[0]))
    
    for i in range(1, n_clusters):
        v = vectors[i]
        for j in range(i):
            v = v - jnp.dot(v, Q[j]) * Q[j]
        Q = Q.at[i].set(v / jnp.linalg.norm(v))
    
    return Q

def initialize_state(n_clusters, d_model, seed=0):
    """Initialize model state with appropriate scaling for high dimensions."""
    key = jax.random.PRNGKey(seed)
    
    # Initialize means using well-separated centers
    means_init = generate_well_separated_centers(key, n_clusters, d_model)
    
    # Scale concentration and Dirichlet parameters with dimension
    base_conc = d_model
    conc = jnp.full((n_clusters,), base_conc)
    dir_conc = 1.0 / jnp.sqrt(d_model)
    mix_dir = jnp.full((n_clusters,), dir_conc)
    
    return ModelState(
        n_clusters=n_clusters,
        d_model=d_model,
        dir_prior=1.0,
        mean_prior_conc=0.0,
        mix_dir=mix_dir,
        means=means_init,
        conc=conc,
        cluster_counts=jnp.zeros(n_clusters),
        dir_sums=jnp.zeros((n_clusters, d_model)),
        sample_counts=jnp.zeros(n_clusters)
    )

@jax.jit
def e_step(state: ModelState, embeddings: jnp.ndarray, log_prior: jnp.ndarray):
    """E-step with improved numerical stability for high dimensions."""
    # Scale concentrations with dimension
    scaled_conc = state.conc / jnp.sqrt(state.d_model)
    weighted_means = state.means * scaled_conc[:, None]
    
    # Compute log probabilities
    log_likelihoods = embeddings @ weighted_means.T
    mix_dir_sum = jnp.sum(state.mix_dir)
    log_mix = digamma(state.mix_dir) - digamma(mix_dir_sum)
    log_mix = log_mix - logsumexp(log_mix)
    
    log_resp = log_likelihoods + log_mix + log_prior[:, None]
    return jax.nn.softmax(log_resp, axis=1)

@jax.jit
def m_step(state: ModelState, embeddings: jnp.ndarray, resp: jnp.ndarray):
    """M-step with hybrid parameter estimation."""
    Nk = jnp.sum(resp, axis=0)
    alpha_post = state.dir_prior + Nk
    
    S_k = resp.T @ embeddings
    mu_post = S_k / (jnp.linalg.norm(S_k, axis=1, keepdims=True) + 1e-10)
    
    r_bar = jnp.linalg.norm(S_k, axis=1) / (Nk + 1e-10)
    r_bar = jnp.clip(r_bar, 1e-8, 1 - 1e-8)
    
    # Use vmap to apply estimate_kappa to each cluster
    kappa_post = jax.vmap(estimate_kappa, in_axes=(0, None, 0))(
        r_bar, state.d_model, Nk)
    
    return state._replace(
        mix_dir=alpha_post,
        means=mu_post,
        conc=kappa_post,
        cluster_counts=state.cluster_counts + Nk,
        dir_sums=state.dir_sums + S_k,
        sample_counts=state.sample_counts + Nk
    )

@jax.jit
def compute_log_likelihood(state: ModelState, embeddings: jnp.ndarray):
    """Compute log likelihood with improved numerical stability."""
    embeddings_norm = jnp.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / (embeddings_norm + 1e-10)
    
    mix_dir_sum = jnp.sum(state.mix_dir)
    log_mix = digamma(state.mix_dir) - digamma(mix_dir_sum)
    log_mix = jnp.clip(log_mix, -100, 100)
    
    dot_products = embeddings_normalized @ state.means.T
    dot_products = jnp.clip(dot_products, -1 + 1e-7, 1 - 1e-7)
    
    log_prob = dot_products * state.conc[None, :]
    log_prob += log_mix[None, :]
    
    max_log_prob = jnp.max(log_prob, axis=1, keepdims=True)
    log_prob_shifted = log_prob - max_log_prob
    log_likelihoods = max_log_prob.squeeze() + jnp.log(jnp.sum(jnp.exp(log_prob_shifted), axis=1) + 1e-10)
    
    return jnp.mean(log_likelihoods)

@partial(jax.jit, static_argnums=(3,))
def vmf_sample(key, mu, kappa, num_samples):
    """
    Modified sampling based on exact asymptotics for high dimensions.
    Uses acceptance-rejection sampling with optimized proposal distribution.
    """
    d = mu.shape[0]
    
    # Adjust kappa based on dimension to prevent degeneracy
    effective_kappa = jnp.minimum(kappa, 100 * d)
    
    # Special case for S² (3D sphere)
    if d == 3:
        key1, key2 = jax.random.split(key)
        u = jax.random.uniform(key1, (num_samples,))
        omega = 1/effective_kappa * jnp.log(jnp.exp(effective_kappa) * u + jnp.exp(-effective_kappa) * (1 - u))
    else:
        # General case parameters with improved stability
        b = (-2 * effective_kappa + jnp.sqrt(4 * effective_kappa**2 + (d - 1)**2)) / (d - 1)
        a = ((d - 1) + 2 * effective_kappa + jnp.sqrt(4 * effective_kappa**2 + (d - 1)**2)) / 4
        d_param = 4 * a * b / (1 + b) - (d - 1) * jnp.log(d - 1)
        
        # Vectorized acceptance-rejection sampling
        def sample_omega(carry):
            key, omega, idx = carry
            key1, key2, key3 = jax.random.split(key, 3)
            
            # Sample from Beta((m-1)/2, (m-1)/2)
            eps = jax.random.beta(key1, (d - 1)/2, (d - 1)/2)
            omega_prop = (1 - (1 + b) * eps) / (1 - (1 - b) * eps)
            
            # Accept-reject step
            t = 2 * a * b / (1 - (1 - b) * eps)
            log_accept_ratio = (d - 1) * jnp.log(t) - t + d_param
            u = jnp.log(jax.random.uniform(key2))
            
            # Update omega if accepted
            accept = log_accept_ratio >= u
            omega = jnp.where(accept, omega_prop, omega)
            idx = jnp.where(accept, idx + 1, idx)
            
            return (key3, omega, idx)
        
        # Initialize and run sampling
        init_carry = (key, jnp.zeros(num_samples), 0)
        final_carry = jax.lax.while_loop(
            lambda x: x[2] < num_samples,
            sample_omega,
            init_carry
        )
        omega = final_carry[1]
    
    # Generate directions
    v = jax.random.normal(key, shape=(num_samples, d-1))
    v = v / jnp.linalg.norm(v, axis=1, keepdims=True)
    
    # Construct samples
    samples = jnp.zeros((num_samples, d))
    samples = samples.at[:, 0].set(omega)
    samples = samples.at[:, 1:].set(jnp.sqrt(1 - omega[:, None]**2) * v)
    
    # Rotate samples to desired mean direction
    e1 = jnp.zeros(d)
    e1 = e1.at[0].set(1.0)
    u = e1 - mu
    u = u / (jnp.linalg.norm(u) + 1e-10)
    samples = samples - 2 * jnp.outer(jnp.dot(samples, u), u)
    
    return samples

def train_vmf_model(samples, initial_state, n_epochs=100, batch_size=1000, patience=5, min_improvement=1e-4):
    """
    Train the vMF mixture model.
    
    Args:
        samples: Input data samples
        initial_state: Initial ModelState
        n_epochs: Maximum number of epochs to train
        batch_size: Number of samples per batch
        patience: Number of epochs to wait for improvement before early stopping
        min_improvement: Minimum improvement in log-likelihood to consider progress
    """
    n_samples = samples.shape[0]
    n_batches = n_samples // batch_size
    state = initial_state
    
    # Initialize tracking variables
    best_state = state
    best_ll = float('-inf')
    epochs_without_improvement = 0
    
    for epoch in range(n_epochs):
        # Initialize epoch-specific tracking
        epoch_ll = 0.0
        n_batches_processed = 0
        
        # Shuffle data
        key = jax.random.PRNGKey(epoch)
        perm = jax.random.permutation(key, n_samples)
        shuffled_samples = samples[perm]

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            batch = shuffled_samples[start:end]
            
            # Perform E-M steps
            log_prior = jnp.zeros(batch.shape[0])
            resp = e_step(state, batch, log_prior)
            state = m_step(state, batch, resp)

            # Calculate log-likelihood every 10 batches
            if (i + 1) % 10 == 0:
                ll = compute_log_likelihood(state, batch)
                epoch_ll += ll
                n_batches_processed += 1
                print(f"Epoch {epoch+1}, Batch {i+1}/{n_batches}, Log-Likelihood: {ll:.4f}")

        # Calculate average log-likelihood for epoch
        avg_epoch_ll = epoch_ll / n_batches_processed if n_batches_processed > 0 else float('-inf')
        print(f"Epoch {epoch+1} average log-likelihood: {avg_epoch_ll:.4f}")
        
        # Check for improvement
        if avg_epoch_ll > best_ll + min_improvement:
            best_ll = avg_epoch_ll
            best_state = state
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            return best_state

    return best_state

def run_vmf_test(n_clusters, d_model, kappas, alphas, samples_per_cluster=1000):
    """Run a complete test of the vMF mixture model."""
    # Generate synthetic data
    key = jax.random.PRNGKey(0)
    centers = generate_well_separated_centers(key, n_clusters, d_model)
    
    samples = []
    true_labels = []
    
    for i in range(n_clusters):
        key, subkey = jax.random.split(key)
        n_samples = int(samples_per_cluster * alphas[i])
        cluster_samples = vmf_sample(subkey, centers[i], kappas[i], n_samples)
        samples.append(cluster_samples)
        true_labels.extend([i] * n_samples)
    
    samples = jnp.concatenate(samples, axis=0)
    true_labels = jnp.array(true_labels)
    
    # Train model
    initial_state = initialize_state(n_clusters, d_model)
    trained_state = train_vmf_model(samples, initial_state)
    
    # Compute metrics
    log_prior = jnp.zeros(samples.shape[0])
    resp = e_step(trained_state, samples, log_prior)
    assignments = jnp.argmax(resp, axis=1)
    accuracy = float(jnp.mean(assignments == true_labels))
    
    # Parameter recovery metrics
    mix_props = trained_state.mix_dir / jnp.sum(trained_state.mix_dir)
    mix_error = float(jnp.mean(jnp.abs(mix_props - alphas)))
    kappa_error = float(jnp.mean(jnp.abs(trained_state.conc - kappas) / kappas))
    
    cos_sims = [float(jnp.abs(jnp.dot(trained_state.means[i], centers[i]))) 
                for i in range(n_clusters)]
    mean_error = float(jnp.mean(1 - jnp.array(cos_sims)))
    
    return {
        'accuracy': accuracy,
        'mix_prop_error': mix_error,
        'kappa_rel_error': kappa_error,
        'mu_mean_error': mean_error,
        'final_log_likelihood': float(compute_log_likelihood(trained_state, samples))
    }

if __name__ == "__main__":
    # Example usage
    n_clusters = 5
    d_model = 100
    key = jax.random.PRNGKey(0)
    kappas = jax.random.uniform(key, (n_clusters,), minval=50.0, maxval=100.0)
    alphas = jax.random.uniform(key, (n_clusters,), minval=0.01, maxval=100.0) 
    key, subkey = jax.random.split(key)
    
    results = run_vmf_test(n_clusters, d_model, kappas, alphas)
    print("\nTest Results:")
    for key, value in results.items():
        print(f"{key}: {value}")