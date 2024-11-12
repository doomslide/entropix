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

@jax.jit
def log_bessel_iv_saddle(nu, kappa):
    """Compute log of modified Bessel function using saddle point approximation."""
    t = kappa / nu
    sqrt_term = jnp.sqrt(1 + t**2)
    eta = sqrt_term + jnp.log(t / (1 + sqrt_term) + 1e-10)
    return nu * eta - 0.5 * jnp.log(2 * jnp.pi * nu) - 0.5 * jnp.log(1 + t**2)

@jax.jit
def estimate_kappa(r_bar, d):
    """
    Estimate concentration parameter kappa using asymptotic approximation.
    Includes higher-order corrections for improved accuracy in high dimensions.
    """
    r_bar = jnp.clip(r_bar, 1e-8, 1 - 1e-8)
    
    # First-order approximation
    kappa_0 = r_bar * d / (1 - r_bar ** 2)
    
    # Higher-order corrections
    c1 = (d - 1) / 2
    c2 = (d - 1) * (d - 3) / 8
    correction = 1 + c1 / kappa_0 + c2 / (kappa_0 ** 2)
    
    kappa = kappa_0 * correction
    
    # Add dimension-dependent scaling
    dim_factor = jnp.sqrt(d / 1000)
    kappa = kappa * dim_factor
    
    return jnp.clip(kappa, 100.0, 1e6)

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
        dir_sums=jnp.zeros((n_clusters, d_model))
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
    """M-step with improved parameter estimation for high dimensions."""
    Nk = jnp.sum(resp, axis=0)
    alpha_post = state.dir_prior + Nk
    
    S_k = resp.T @ embeddings
    mu_post = S_k / (jnp.linalg.norm(S_k, axis=1, keepdims=True) + 1e-10)
    
    r_bar = jnp.linalg.norm(S_k, axis=1) / (Nk + 1e-10)
    r_bar = jnp.clip(r_bar, 1e-8, 1 - 1e-8)
    
    kappa_post = estimate_kappa(r_bar, state.d_model)
    
    return state._replace(
        mix_dir=alpha_post,
        means=mu_post,
        conc=kappa_post,
        cluster_counts=state.cluster_counts + Nk,
        dir_sums=state.dir_sums + S_k
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

def vmf_sample(key, mu, kappa, num_samples):
    """Generate samples from von Mises-Fisher distribution using Wood's algorithm."""
    d = mu.shape[0]
    
    # Compute parameters
    b = (-2 * kappa + jnp.sqrt(4 * kappa**2 + (d - 1)**2)) / (d - 1)
    b = jnp.clip(b, -1 + 1e-10, 1 - 1e-10)
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * jnp.log(1 - x0**2)
    
    # Generate samples
    key1, key2, key3 = jax.random.split(key, 3)
    z = jax.random.beta(key1, (d - 1) / 2, (d - 1) / 2, shape=(num_samples,))
    w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
    
    # Generate directions
    v = jax.random.normal(key3, shape=(num_samples, d-1))
    v = v / jnp.linalg.norm(v, axis=1, keepdims=True)
    
    # Construct samples
    samples = jnp.zeros((num_samples, d))
    samples = samples.at[:, 0].set(w)
    samples = samples.at[:, 1:].set(jnp.sqrt(1 - w[:, None]**2) * v)
    
    # Rotate samples to desired mean direction
    e1 = jnp.zeros(d)
    e1 = e1.at[0].set(1.0)
    u = e1 - mu
    u = u / (jnp.linalg.norm(u) + 1e-10)
    samples = samples - 2 * jnp.outer(jnp.dot(samples, u), u)
    
    return samples

def train_vmf_model(samples, initial_state, n_epochs=20, batch_size=1000):
    """Train the vMF mixture model."""
    n_samples = samples.shape[0]
    n_batches = n_samples // batch_size
    state = initial_state

    for epoch in range(n_epochs):
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

            if (i + 1) % 10 == 0:
                ll = compute_log_likelihood(state, batch)
                print(f"Epoch {epoch+1}, Batch {i+1}/{n_batches}, Log-Likelihood: {ll:.4f}")

    return state

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
    
    return {
        'accuracy': accuracy,
        'mix_prop_error': mix_error,
        'kappa_rel_error': kappa_error,
        'mean_cos_sims': cos_sims,
        'final_log_likelihood': float(compute_log_likelihood(trained_state, samples))
    }

if __name__ == "__main__":
    # Example usage
    n_clusters = 10
    d_model = 1000
    key = jax.random.PRNGKey(0)
    thetas = jax.random.uniform(key, (n_clusters,), minval=0.7, maxval=1.3)
    kappas = thetas * d_model
    key, subkey = jax.random.split(key)
    alphas = jax.random.uniform(subkey, (n_clusters,), minval=0.2/n_clusters, maxval=10.0/n_clusters)
    
    results = run_vmf_test(n_clusters, d_model, kappas, alphas)
    print("\nTest Results:")
    for key, value in results.items():
        print(f"{key}: {value}")