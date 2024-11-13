import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional
import warnings
import os

# Suppress warnings and configure JAX
warnings.filterwarnings("ignore", category=FutureWarning)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

# Set JAX_TRACEBACK_FILTERING to 'off' to see the full traceback
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

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
    Estimate concentration parameter kappa using second order asymptotic expansion
    """    
    disc = (d-1)*(2*d*r_bar - d - 6*r_bar + 5)
    term1 = jnp.sqrt(jnp.maximum(disc, 1e-10))/(4*(1-r_bar))
    term2 = (d-1)/(4*(1-r_bar))
    kappa = term1 + term2
    return kappa


@jax.jit
def exact_inverse_r_bar(kappa, d):
    """
    Compute theoretical r_bar for given kappa using exact second order expansion.
    
    Args:
        kappa: Concentration parameter of shape (batch_size, ...)
        d: Dimension of the space
        
    Returns:
        Mean resultant length with same shape as kappa
    """
    # First two terms of exact expansion
    r_bar = 1 - (d - 1)/(2 * kappa) + (d**2 - 4*d + 3)/(8 * kappa**2)
    return r_bar

def generate_centers(key, n_clusters, d_model):
    """Generate well-separated unit vectors in high dimensions."""
    centers = jax.random.normal(key, shape=(n_clusters, d_model))
    centers = centers / jnp.linalg.norm(centers, axis=1, keepdims=True)    
    return centers

def initialize_state(n_clusters, d_model, means_init=None, seed=0):
    """Initialize model state."""
    if means_init is None:
        key = jax.random.PRNGKey(seed)
        means_init = generate_centers(key, n_clusters, d_model)
    
    # Initialize with reasonable concentration parameters
    base_conc = jnp.sqrt(d_model)  # Scale with dimension
    conc = jnp.full((n_clusters,), base_conc)
    
    # Use uniform prior for mixture weights
    dir_conc = 1.0
    mix_dir = jnp.full((n_clusters,), dir_conc)
    
    # Initialize dir_sums with correct shape
    dir_sums = jnp.zeros((n_clusters, d_model))
    
    return ModelState(
        n_clusters=n_clusters,
        d_model=d_model,
        dir_prior=5.0,
        mean_prior_conc=jnp.sqrt(d_model),  # Stronger regularization
        mix_dir=mix_dir,
        means=means_init,
        conc=conc,
        cluster_counts=jnp.zeros(n_clusters),
        dir_sums=dir_sums
    )

@jax.jit
def compute_exact_log_mix_weights(mix_dir):
    """
    Compute log mixture weights using exact series expansion.
    
    Args:
        mix_dir: Dirichlet parameters (n_clusters,)
    
    Returns:
        Log mixture weights (n_clusters,)
    """
    mix_dir_sum = jnp.sum(mix_dir)
    x = mix_dir / mix_dir_sum  # Normalized alphas
    
    # Use exact expansion around x = 1
    # -7/12 + 19x/12 + 25(x-1)²/288 - 1061(x-1)³/10368
    x_centered = x - 1
    log_mix = (19 * x / 12 - 7/12 + 
               25 * x_centered**2 / 288 - 
               1061 * x_centered**3 / 10368)
    
    # Normalize properly in log space
    max_log_mix = jnp.max(log_mix)
    log_mix_shifted = log_mix - max_log_mix
    log_norm = jnp.log(jnp.sum(jnp.exp(log_mix_shifted)))
    
    return log_mix_shifted - log_norm

@jax.jit
def exact_e_step(state: ModelState, embeddings: jnp.ndarray, log_prior: jnp.ndarray):
    """E-step with improved numerical stability."""
    # Normalize embeddings
    embeddings_norm = jnp.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / (embeddings_norm + 1e-10)
    
    # Get exact log mixture weights
    log_mix = compute_exact_log_mix_weights(state.mix_dir)
    
    # Compute dot products with numerical stability
    dot_products = embeddings_normalized @ state.means.T
    # dot_products = jnp.clip(dot_products, -1 + 1e-7, 1 - 1e-7)
    
    # Combine components
    log_prob = dot_products * state.conc[None, :] + log_mix[None, :] + log_prior[:, None]
    
    # Convert to probabilities with improved stability
    max_log_prob = jnp.max(log_prob, axis=1, keepdims=True)
    log_prob_shifted = log_prob - max_log_prob
    resp = jnp.exp(log_prob_shifted)
    
    # Normalize
    normalizer = jnp.sum(resp, axis=1, keepdims=True)
    return resp / (normalizer + 1e-10)

@jax.jit
def exact_m_step(state: ModelState, embeddings: jnp.ndarray, resp: jnp.ndarray):
    """M-step without smoothing or regularization."""
    # Compute counts
    Nk = jnp.sum(resp, axis=0)  # Shape: (n_clusters,)
    alpha_post = Nk
    
    # Update mean directions
    S_k = resp.T @ embeddings  # Shape: (n_clusters, d_model)
    mu_post = S_k
    
    # Normalize means
    mu_post = mu_post / (jnp.linalg.norm(mu_post, axis=1, keepdims=True) + 1e-10)
    
    # Update concentrations
    r_bar = jnp.linalg.norm(S_k, axis=1) / (Nk + 1e-10)
    
    kappa_post = estimate_kappa(r_bar, state.d_model)
    
    return state._replace(
        mix_dir=alpha_post,
        means=mu_post,
        conc=kappa_post,
        cluster_counts=state.cluster_counts + Nk,
        dir_sums=state.dir_sums + S_k
    )

@jax.jit
def compute_exact_log_likelihood(state: ModelState, embeddings: jnp.ndarray):
    """Compute exact log likelihood using series expansions."""
    # Normalize embeddings
    embeddings_norm = jnp.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / (embeddings_norm + 1e-10)
    
    # Get exact log mixture weights
    log_mix = compute_exact_log_mix_weights(state.mix_dir)
    
    # Compute dot products with numerical stability
    dot_products = embeddings_normalized @ state.means.T
    # dot_products = jnp.clip(dot_products, -1 + 1e-7, 1 - 1e-7)
    
    # Combine components
    log_prob = dot_products * state.conc[None, :] + log_mix[None, :]
    
    # Sum in log space with improved stability
    max_log_prob = jnp.max(log_prob, axis=1, keepdims=True)
    log_prob_shifted = log_prob - max_log_prob
    log_sum = jnp.log(jnp.sum(jnp.exp(log_prob_shifted), axis=1) + 1e-10)
    
    return jnp.mean(max_log_prob.squeeze() + log_sum)

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

def train_vmf_model(samples: jnp.ndarray, initial_state: ModelState, 
                   n_epochs: int = 15, batch_size: Optional[int] = None, key: Optional[jax.random.PRNGKey] = None):
    """Train the vMF mixture model with more gradual updates."""
    n_samples = samples.shape[0]
    batch_size = batch_size or n_samples
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    print("\nTraining Progress:")
    print(f"Number of samples: {n_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {n_batches}")
    
    state = initial_state
    
    # More gradual learning rate schedule
    def get_lr(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup * 0.3  # Max lr of 0.3 after warmup
        return 0.3 * (0.95 ** (epoch - warmup))  # Exponential decay after warmup
    
    for epoch in range(n_epochs):
        if key is not None:
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, n_samples)
            shuffled_samples = samples[perm]
        else:
            shuffled_samples = samples
        
        epoch_ll = 0.0
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            batch = shuffled_samples[start:end]
            
            # Perform E-M steps with gradual updates
            log_prior = jnp.zeros(batch.shape[0])
            resp = exact_e_step(state, batch, log_prior)
            
            # Get new state
            new_state = exact_m_step(state, batch, resp)
            
            # More conservative interpolation between old and new state
            lr = get_lr(epoch)
            state = ModelState(
                n_clusters=state.n_clusters,
                d_model=state.d_model,
                dir_prior=state.dir_prior,
                mean_prior_conc=state.mean_prior_conc,
                mix_dir=state.mix_dir * (1-lr) + new_state.mix_dir * lr,
                means=state.means * (1-lr) + new_state.means * lr,
                conc=state.conc * (1-lr) + new_state.conc * lr,
                cluster_counts=new_state.cluster_counts,
                dir_sums=new_state.dir_sums
            )
            
            # Track progress
            batch_ll = compute_exact_log_likelihood(state, batch)
            epoch_ll += batch_ll * (end - start)
        
        avg_ll = epoch_ll / n_samples
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}, Avg Log-Likelihood: {avg_ll:.4f}, Learning Rate: {lr:.4f}")
            print(f"Mix proportions: {state.mix_dir / jnp.sum(state.mix_dir)}")
            print(f"Concentrations: {state.conc}")
    
    return state
