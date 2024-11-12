import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp, digamma
from functools import partial
from typing import NamedTuple
import warnings
import json

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set JAX to use CPU
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

# Print available devices for validation
print("Available devices:", jax.devices())

# Define a namedtuple to hold the model state
class ModelState(NamedTuple):
    n_clusters: int
    d_model: int
    dir_prior: float
    mean_prior_conc: float
    mix_dir: jnp.ndarray       # Shape: (n_clusters,)
    means: jnp.ndarray         # Shape: (n_clusters, d_model)
    conc: jnp.ndarray          # Shape: (n_clusters,)
    cluster_counts: jnp.ndarray  # Shape: (n_clusters,)
    dir_sums: jnp.ndarray        # Shape: (n_clusters, d_model)

@jax.jit
def log_bessel_iv_saddle(nu, kappa):
    t = kappa / nu
    sqrt_term = jnp.sqrt(1 + t**2)
    eta = sqrt_term + jnp.log(t / (1 + sqrt_term) + 1e-10)
    log_iv = nu * eta - 0.5 * jnp.log(2 * jnp.pi * nu) - 0.5 * jnp.log(1 + t**2)
    return log_iv

@jax.jit
def estimate_kappa_from_r_bar(r_bar, d):
    nu = d / 2 - 1
    r_bar = jnp.clip(r_bar, 1e-8, 1 - 1e-8)
    kappa = nu * r_bar / jnp.sqrt(1 - r_bar ** 2)
    return kappa

# Remove @jax.jit from initialize_state since it needs dynamic shapes
def initialize_state(n_clusters, d_model, dir_prior=1.0, mean_prior_conc=0.0, seed=0):
    key = jax.random.PRNGKey(seed)
    means_init = jnp.zeros((n_clusters, d_model))
    angles = jnp.linspace(0, 2 * jnp.pi, n_clusters + 1)[:-1]
    means_init = means_init.at[:, 0].set(jnp.cos(angles))
    means_init = means_init.at[:, 1].set(jnp.sin(angles))
    means_init = means_init / jnp.linalg.norm(means_init, axis=1, keepdims=True)
    conc = jnp.full((n_clusters,), 500.0)
    mix_dir = jnp.full((n_clusters,), 10.0)
    cluster_counts = jnp.zeros(n_clusters)
    dir_sums = jnp.zeros((n_clusters, d_model))

    return ModelState(
        n_clusters=n_clusters,
        d_model=d_model,
        dir_prior=dir_prior,
        mean_prior_conc=mean_prior_conc,
        mix_dir=mix_dir,
        means=means_init,
        conc=conc,
        cluster_counts=cluster_counts,
        dir_sums=dir_sums
    )

@jax.jit
def e_step(state: ModelState, embeddings: jnp.ndarray, log_prior: jnp.ndarray):
    n_samples = embeddings.shape[0]
    weighted_means = state.means * state.conc[:, None]
    log_likelihoods = embeddings @ weighted_means.T
    mix_props = state.mix_dir / jnp.sum(state.mix_dir)
    log_mix = jnp.log(mix_props + 1e-10)
    log_resp = log_likelihoods + log_mix + log_prior[:, None]
    log_resp = log_resp - logsumexp(log_resp, axis=1, keepdims=True)
    return jnp.exp(log_resp)

@jax.jit
def estimate_kappa_high_dim(r_bar, d):
    """
    Refined estimation of kappa for high-dimensional data with higher-order corrections.
    
    Asymptotic expansion:
    kappa ≈ (r_bar * d) / (1 - r_bar^2) * (1 + (d - 1) / (2 * kappa) + (d - 1) * (d - 3) / (8 * kappa**2) + ...)
    
    Args:
        r_bar (jnp.ndarray): Empirical mean resultant length.
        d (int): Dimension of the space.
        
    Returns:
        jnp.ndarray: Estimated kappa values.
    """
    # Initial estimate
    kappa = r_bar * d / (1 - r_bar ** 2)
    
    # Apply higher-order corrections
    correction = 1 + (d - 1) / (2 * kappa) + (d - 1) * (d - 3) / (8 * kappa**2)
    kappa = kappa * correction
    
    # Scaling factor based on dimension
    dim_factor = jnp.sqrt(d / 1000)
    kappa = kappa * dim_factor
    
    # Ensure kappa is within reasonable bounds
    kappa = jnp.clip(kappa, 100.0, 1e6)
    
    return kappa

def compute_A_d_saddle(nu, kappa):
    """Computes A_d(kappa) using the saddlepoint approximation."""
    t = kappa / nu
    sqrt_term = jnp.sqrt(1 + t**2)
    A_d = t / sqrt_term
    return A_d

def log_Cp_saddle(kappa, d):
    """Computes log C_p(kappa) using the saddlepoint approximation."""
    nu = d / 2 - 1
    log_iv = log_bessel_iv_saddle(nu, kappa)
    log_Cp = nu * jnp.log(kappa) - (d / 2) * jnp.log(2 * jnp.pi) - log_iv
    return log_Cp

# Update compute_log_likelihood with improved stability
@jax.jit
def compute_log_likelihood_new(state: ModelState, embeddings: jnp.ndarray):
    """Compute the log likelihoods with improved numerical stability."""
    embeddings_norm = jnp.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / (embeddings_norm + 1e-10)
    
    mix_dir_sum = jnp.sum(state.mix_dir)
    log_mix = digamma(state.mix_dir) - digamma(mix_dir_sum)
    log_mix = jnp.clip(log_mix, -100, 100)

    log_norm = log_Cp_saddle(state.conc, state.d_model)
    log_norm = jnp.clip(log_norm, -100, 100)

    dot_products = embeddings_normalized @ state.means.T
    dot_products = jnp.clip(dot_products, -1 + 1e-7, 1 - 1e-7)

    log_prob = dot_products * state.conc[None, :]
    log_prob += log_mix[None, :]
    log_prob -= log_norm[None, :]

    max_log_prob = jnp.max(log_prob, axis=1, keepdims=True)
    log_prob_shifted = log_prob - max_log_prob
    log_likelihoods = max_log_prob.squeeze() + jnp.log(jnp.sum(jnp.exp(log_prob_shifted), axis=1) + 1e-10)
    
    return jnp.mean(log_likelihoods)

@jax.jit
def m_step(state: ModelState, embeddings: jnp.ndarray, resp: jnp.ndarray):
    Nk = jnp.sum(resp, axis=0)
    alpha_post = state.dir_prior + Nk
    S_k = resp.T @ embeddings
    mu_post = S_k / (jnp.linalg.norm(S_k, axis=1, keepdims=True) + 1e-10)
    r_bar = jnp.linalg.norm(S_k, axis=1) / (Nk + 1e-10)
    r_bar = jnp.clip(r_bar, 1e-8, 1 - 1e-8)
    kappa_post = jax.vmap(lambda r: estimate_kappa_high_dim(r, state.d_model))(r_bar)
    kappa_post = jnp.clip(kappa_post, 100.0, 1e6)
    return ModelState(
        n_clusters=state.n_clusters,
        d_model=state.d_model,
        dir_prior=state.dir_prior,
        mean_prior_conc=state.mean_prior_conc,
        mix_dir=alpha_post,
        means=mu_post,
        conc=kappa_post,
        cluster_counts=state.cluster_counts + Nk,
        dir_sums=state.dir_sums + S_k
    )

@jax.jit
def compute_log_likelihood(state: ModelState, embeddings: jnp.ndarray):
    embeddings_norm = jnp.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / (embeddings_norm + 1e-10)
    mix_dir_sum = jnp.sum(state.mix_dir)
    log_mix = digamma(state.mix_dir) - digamma(mix_dir_sum)
    log_mix = jnp.clip(log_mix, -100, 100)
    log_norm = log_bessel_iv_saddle(state.conc, state.d_model)
    log_norm = jnp.clip(log_norm, -100, 100)
    dot_products = embeddings_normalized @ state.means.T
    dot_products = jnp.clip(dot_products, -1 + 1e-7, 1 - 1e-7)
    log_prob = dot_products * state.conc[None, :]
    log_prob += log_mix[None, :]
    log_prob -= log_norm[None, :]
    max_log_prob = jnp.max(log_prob, axis=1, keepdims=True)
    log_prob_shifted = log_prob - max_log_prob
    log_likelihoods = max_log_prob.squeeze() + jnp.log(jnp.sum(jnp.exp(log_prob_shifted), axis=1) + 1e-10)
    return jnp.mean(log_likelihoods)

@jax.jit
def verify_kappa_estimation():
    print("\nVerifying kappa estimation:")
    test_kappas = [5000.0, 10000.0, 15000.0]
    d_model = 1000
    for true_kappa in test_kappas:
        r_bar = 1 - (d_model - 1) / (2 * true_kappa)
        est_kappa = estimate_kappa_high_dim(r_bar, d_model)
        print(f"\nTrue kappa: {true_kappa:.1f}")
        print(f"r_bar: {r_bar:.4f}")
        print(f"Estimated kappa: {est_kappa:.1f}")
        print(f"Relative error: {abs(est_kappa - true_kappa) / true_kappa:.4f}")

def vmf_sample_wood(key, mu, kappa, num_samples):
    """
    Generate samples from vMF distribution using Wood's algorithm.
    Removed @jax.jit decorator due to dynamic shapes.
    """
    d = mu.shape[0]
    
    # Compute b
    b = (-2 * kappa + jnp.sqrt(4 * kappa**2 + (d - 1)**2)) / (d - 1)
    b = jnp.clip(b, -1 + 1e-10, 1 - 1e-10)
    
    # Compute x0 and c
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * jnp.log(1 - x0**2)
    
    # Generate beta samples
    key1, key2, key3 = jax.random.split(key, 3)
    z = jax.random.beta(key1, (d - 1) / 2, (d - 1) / 2, shape=(num_samples,))
    
    # Compute w
    w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
    w = jnp.clip(w, -1 + 1e-10, 1 - 1e-10)
    
    # Generate uniform samples for rejection
    u = jax.random.uniform(key2, shape=(num_samples,))
    
    # Compute acceptance
    t = kappa * w + (d - 1) * jnp.log(1 - x0 * w)
    accept = (t - c) >= jnp.log(u)
    
    # Generate v
    v = jax.random.normal(key3, shape=(num_samples, d-1))
    v = v / jnp.linalg.norm(v, axis=1, keepdims=True)
    
    # Construct samples
    samples = jnp.zeros((num_samples, d))
    samples = samples.at[:, 0].set(w)
    samples = samples.at[:, 1:].set(jnp.sqrt(1 - w[:, None]**2) * v)
    
    # Rotate samples
    e1 = jnp.zeros(d)
    e1 = e1.at[0].set(1.0)
    u = e1 - mu
    u = u / (jnp.linalg.norm(u) + 1e-10)
    samples = samples - 2 * jnp.outer(jnp.dot(samples, u), u)
    
    return samples

@jax.jit
def verify_sampling(key, mu, kappa, d_model, num_samples=10000):
    samples = vmf_sample_wood(key, mu, kappa, num_samples)
    mean_vector = jnp.mean(samples, axis=0)
    r_bar_empirical = jnp.linalg.norm(mean_vector)
    r_bar_theoretical = 1 - (d_model - 1) / (2 * kappa)
    cos_sim = jnp.abs(jnp.dot(mean_vector, mu))
    return {
        'r_bar_empirical': r_bar_empirical,
        'r_bar_theoretical': r_bar_theoretical,
        'cos_sim': cos_sim
    }

def train_vmf_mixture(samples, true_labels, state, centers, n_epochs=20, batch_size=1000):
    """
    Training function without JIT due to dynamic shapes and Python control flow.
    Added centers parameter to compute correct cosine similarities.
    """
    n_samples = samples.shape[0]
    n_batches = n_samples // batch_size
    current_state = state
    best_state = state
    best_likelihood = float('-inf')
    print("\nStarting training...")
    
    for epoch in range(n_epochs):
        key = jax.random.PRNGKey(epoch)
        perm = jax.random.permutation(key, n_samples)
        shuffled_samples = samples[perm]
        shuffled_labels = true_labels[perm]
        epoch_likelihood = 0.0
        epoch_accuracy = 0.0
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_samples = shuffled_samples[start_idx:end_idx]
            batch_labels = shuffled_labels[start_idx:end_idx]
            
            # Multiple EM steps per batch
            for _ in range(3):
                log_prior = jnp.zeros(batch_samples.shape[0])
                resp = e_step(current_state, batch_samples, log_prior)
                current_state = m_step(current_state, batch_samples, resp)
            
            # Convert JAX arrays to NumPy for accumulation
            batch_likelihood = jnp.asarray(compute_log_likelihood(current_state, batch_samples))
            batch_assignments = jnp.argmax(resp, axis=1)
            batch_accuracy = jnp.mean(jnp.asarray(batch_assignments == batch_labels))
            
            # Accumulate using NumPy values
            epoch_likelihood += float(batch_likelihood) * batch_size
            epoch_accuracy += float(batch_accuracy) * batch_size
            
            if (i + 1) % 5 == 0:
                mix_props = current_state.mix_dir / jnp.sum(current_state.mix_dir)
                print(f"\nEpoch {epoch + 1}, Batch {i + 1}/{n_batches}")
                print(f"  Batch Accuracy: {float(batch_accuracy):.4f}")
                print(f"  Batch Log-likelihood: {float(batch_likelihood):.4f}")
                print(f"  Mixing Proportions: {mix_props}")
                print(f"  Concentrations: {current_state.conc}")
                
                # Compare with true centers instead of initial state means
                for k in range(state.n_clusters):
                    cos_sim = jnp.abs(jnp.dot(current_state.means[k], centers[k]))
                    print(f"  Cluster {k} mean cosine similarity: {float(cos_sim):.4f}")
        
        # Compute epoch
        avg_epoch_likelihood = epoch_likelihood / n_samples
        avg_epoch_accuracy = epoch_accuracy / n_samples
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Log-likelihood: {float(avg_epoch_likelihood):.4f}")
        print(f"  Average Accuracy: {float(avg_epoch_accuracy):.4f}")
        
        if avg_epoch_likelihood > best_likelihood:
            best_likelihood = avg_epoch_likelihood
            best_state = current_state
        
        if epoch > 0 and abs(avg_epoch_likelihood - best_likelihood) < 1e-6:
            print("\nConverged! Stopping early.")
            break
            
    return best_state

def generate_well_separated_centers(key, n_clusters, d_model):
    """
    Generate well-separated mean directions using Gram-Schmidt orthogonalization.
    """
    # Generate random vectors
    vectors = jax.random.normal(key, (n_clusters, d_model))
    
    # Gram-Schmidt orthogonalization
    Q = jnp.zeros((n_clusters, d_model))
    Q = Q.at[0].set(vectors[0] / jnp.linalg.norm(vectors[0]))
    
    for i in range(1, n_clusters):
        v = vectors[i]
        # Subtract projections onto previous vectors
        for j in range(i):
            v = v - jnp.dot(v, Q[j]) * Q[j]
        # Normalize
        Q = Q.at[i].set(v / jnp.linalg.norm(v))
    
    return Q

@partial(jax.jit, static_argnums=(0, 1))
def bayesian_update_state(n_clusters, d_model, state, batch_samples):
    """Perform Bayesian update to the concentration, Dirichlet parameters, and means."""
    log_prior = jnp.zeros(batch_samples.shape[0])
    resp = e_step(state, batch_samples, log_prior)

    Nk = jnp.sum(resp, axis=0)
    alpha_post = state.mix_dir + Nk
    mask = Nk > 1e-8

    S_k = resp.T @ batch_samples
    kappa_prior = state.mean_prior_conc
    kappa_post = kappa_prior + Nk
    kappa_post_safe = jnp.where(kappa_post > 1e-8, kappa_post, 1.0)

    mu_prior = jnp.zeros((n_clusters, d_model))
    mu_numerator = (kappa_prior * mu_prior) + S_k
    mu_denominator = kappa_post_safe[:, None]
    mu_post = mu_numerator / mu_denominator
    mu_post = jnp.where(mask[:, None], mu_post, state.means)

    mu_post_norm = jnp.linalg.norm(mu_post, axis=1, keepdims=True)
    mu_post_norm = jnp.where(mu_post_norm > 1e-8, mu_post_norm, 1.0)
    mu_post = mu_post / mu_post_norm

    r_bar = jnp.where(mask, jnp.linalg.norm(S_k, axis=1) / (Nk + 1e-8), 0.0)
    r_bar = jnp.clip(r_bar, 1e-8, 1 - 1e-8)
    
    kappa_post = estimate_kappa_from_r_bar(r_bar, d_model)
    kappa_post = jnp.clip(kappa_post, 1e-3, 1e5)
    kappa_post = jnp.where(mask, kappa_post, state.conc)

    return state._replace(
        mix_dir=alpha_post,
        means=mu_post,
        conc=kappa_post,
        cluster_counts=state.cluster_counts + Nk,
        dir_sums=state.dir_sums + S_k
    )

def train_vmf_model(samples, initial_state, n_epochs=20, batch_size=1000):
    """
    Train the vMF mixture model using Bayesian updates.

    Args:
        samples (jnp.ndarray): The complete dataset of embeddings.
        initial_state (ModelState): The initialized state of the model.
        n_epochs (int): Number of training epochs.
        batch_size (int): Number of samples per batch.

    Returns:
        ModelState: The trained model state.
    """
    n_samples = samples.shape[0]
    n_batches = n_samples // batch_size
    state = initial_state

    for epoch in range(n_epochs):
        # Shuffle the data at the beginning of each epoch
        key = jax.random.PRNGKey(epoch)
        perm = jax.random.permutation(key, n_samples)
        shuffled_samples = samples[perm]

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            batch = shuffled_samples[start:end]

            # Perform Bayesian update
            state = bayesian_update_state(state.n_clusters, state.d_model, state, batch)

            if (i + 1) % 10 == 0:
                log_likelihood = compute_log_likelihood(state, batch)
                print(f"Epoch {epoch+1}, Batch {i+1}/{n_batches}, Log-Likelihood: {log_likelihood:.4f}")

    return state

def run_vmf_test(n_clusters, d_model, kappas, alphas, batch_size):
    """
    Run a single VMF mixture test with given parameters.
    
    Args:
        n_clusters: Number of clusters
        d_model: Dimension of the space
        kappas: Array of concentration parameters
        alphas: Array of mixing proportions
        batch_size: Size of each batch for processing
    
    Returns:
        dict: Dictionary containing test results and metrics
    """
    # Generate well-separated centers
    key = jax.random.PRNGKey(0)
    centers = generate_well_separated_centers(key, n_clusters, d_model)
    
    # Generate samples
    samples = []
    true_labels = []
    samples_per_cluster = 1000
    
    for i in range(n_clusters):
        key, subkey = jax.random.split(key)
        n_samples = int(samples_per_cluster * alphas[i])
        cluster_samples = vmf_sample_wood(subkey, centers[i], kappas[i], n_samples)
        samples.append(cluster_samples)
        true_labels.extend([i] * n_samples)
    
    samples = jnp.concatenate(samples, axis=0)
    true_labels = jnp.array(true_labels)
    
    # Initialize and train model
    initial_state = initialize_state(n_clusters, d_model, seed=0)
    trained_state = train_vmf_model(samples, initial_state, n_epochs=5, batch_size=batch_size)
    
    # Compute final metrics
    log_prior = jnp.zeros(samples.shape[0])
    resp = e_step(trained_state, samples, log_prior)
    cluster_assignments = jnp.argmax(resp, axis=1)
    accuracy = float(jnp.mean(cluster_assignments == true_labels))
    
    # Parameter recovery metrics
    mix_props = trained_state.mix_dir / jnp.sum(trained_state.mix_dir)
    mix_prop_error = float(jnp.mean(jnp.abs(mix_props - alphas)))
    kappa_rel_error = float(jnp.mean(jnp.abs(trained_state.conc - kappas) / kappas))
    
    mean_cos_sims = [float(jnp.abs(jnp.dot(trained_state.means[i], centers[i]))) 
                    for i in range(n_clusters)]
    
    return {
        'n_clusters': n_clusters,
        'd_model': d_model,
        'accuracy': accuracy,
        'mix_prop_error': mix_prop_error,
        'kappa_rel_error': kappa_rel_error,
        'mean_cos_sims': mean_cos_sims,
        'true_kappas': kappas,
        'recovered_kappas': trained_state.conc,
        'true_alphas': alphas,
        'recovered_alphas': mix_props,
        'final_log_likelihood': compute_log_likelihood(trained_state, samples)
    }


def generate_random_params(rng):
    """Generate random test parameters."""
    n_clusters = int(rng.integers(10, 20))  # Random number of clusters between 2 and 5
    d_model = int(rng.integers(1000, 10001))  # Random dimension between 1e3 and 1e4
    
    # Generate random kappas between 5000 and 20000
    kappas = rng.uniform(5000, 20000, size=n_clusters)
    kappas = jnp.array(kappas)
    
    # Generate random mixing proportions that sum to 1
    alphas = rng.dirichlet(np.ones(n_clusters))
    alphas = jnp.array(alphas)
    
    return n_clusters, d_model, kappas, alphas



def estimate_kappa_improved(r_bar, d):
    """
    Improved estimation of kappa for high-dimensional data using higher-order asymptotic corrections.
    
    From asymptotic analysis:
    kappa ≈ (r_bar * d) / (1 - r_bar^2) * (1 + (d - 1) / (2 * kappa_0) + (d - 1) * (d - 3) / (8 * kappa_0^2))
    
    Args:
        r_bar (jnp.ndarray): Empirical mean resultant length.
        d (int): Dimension of the space.
        
    Returns:
        jnp.ndarray: Estimated kappa values.
    """
    r_bar = jnp.clip(r_bar, 1e-8, 1 - 1e-8)
    
    # First-order approximation
    kappa_0 = r_bar * d / (1 - r_bar ** 2)
    
    # Second-order correction
    c1 = (d - 1) / 2
    c2 = (d - 1) * (d - 3) / 8
    correction = 1 + c1 / kappa_0 + c2 / (kappa_0 ** 2)
    
    kappa = kappa_0 * correction
    
    return kappa

@jax.jit
def compute_log_likelihood_improved(state: ModelState, embeddings: jnp.ndarray):
    """
    Compute log likelihood using asymptotic expansions for numerical stability in high dimensions.
    
    Args:
        state (ModelState): Current state of the model.
        embeddings (jnp.ndarray): Input embeddings.
        
    Returns:
        jnp.ndarray: Mean log likelihood over the batch.
    """
    # Normalize embeddings
    embeddings_norm = jnp.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / (embeddings_norm + 1e-10)
    
    # Dimension
    d = state.d_model
    
    # Compute log C_p(kappa) using improved asymptotic expansions
    def log_cp_asymptotic(kappa):
        nu = d / 2 - 1
        t = kappa / nu
        sqrt_term = jnp.sqrt(1 + t ** 2)
        eta = sqrt_term + jnp.log(t / (1 + sqrt_term))
        return nu * jnp.log(kappa) - (d / 2) * jnp.log(2 * jnp.pi) - nu * eta - 0.5 * jnp.log(1 + t ** 2)
    
    log_norm = jax.vmap(log_cp_asymptotic)(state.conc)
    
    # Compute dot products
    dot_products = embeddings_normalized @ state.means.T
    clip_threshold = 1 - 1e-6 / d
    dot_products = jnp.clip(dot_products, -clip_threshold, clip_threshold)
    
    # Compute log probabilities
    log_prob = dot_products * state.conc[None, :]
    log_prob += digamma(state.mix_dir) - digamma(jnp.sum(state.mix_dir))
    log_prob -= log_norm[None, :]
    
    # Stabilize log probabilities
    max_log_prob = jnp.max(log_prob, axis=1, keepdims=True)
    log_prob_shifted = log_prob - max_log_prob
    log_likelihoods = max_log_prob.squeeze() + jnp.log(jnp.sum(jnp.exp(log_prob_shifted), axis=1) + 1e-10)
    
    return jnp.mean(log_likelihoods)

@jax.jit
def e_step_improved(state: ModelState, embeddings: jnp.ndarray, log_prior: jnp.ndarray):
    """
    E-step with improved numerical stability for high-dimensional data.
    
    Adjustments:
    - Scales concentrations with dimension.
    - Improves Dirichlet expectation normalization.
    
    Args:
        state (ModelState): Current state of the model.
        embeddings (jnp.ndarray): Input embeddings.
        log_prior (jnp.ndarray): Log prior probabilities.
        
    Returns:
        jnp.ndarray: Responsibilities after E-step.
    """
    d = state.d_model
    
    # Scale concentrations with dimension
    scaled_conc = state.conc / jnp.sqrt(d)
    
    # Compute weighted means
    weighted_means = state.means * scaled_conc[:, None]
    
    # Compute log likelihoods
    log_likelihoods = embeddings @ weighted_means.T
    
    # Improved Dirichlet expectation
    mix_dir_sum = jnp.sum(state.mix_dir)
    log_mix = digamma(state.mix_dir) - digamma(mix_dir_sum)
    log_mix = log_mix - logsumexp(log_mix)  # Extra normalization
    
    # Compute log responsibilities
    log_resp = log_likelihoods + log_mix + log_prior[:, None]
    
    # Convert to probabilities using softmax for numerical stability
    return jax.nn.softmax(log_resp, axis=1)

@jax.jit
def m_step_improved(state: ModelState, embeddings: jnp.ndarray, resp: jnp.ndarray):
    """
    M-step with improved high-dimensional stability.
    
    Adjustments:
    - Utilizes improved kappa estimation.
    - Applies dimension-dependent regularization to kappa.
    
    Args:
        state (ModelState): Current state of the model.
        embeddings (jnp.ndarray): Input embeddings.
        resp (jnp.ndarray): Responsibilities from E-step.
        
    Returns:
        ModelState: Updated model state after M-step.
    """
    d = state.d_model
    
    # Compute sufficient statistics
    Nk = jnp.sum(resp, axis=0)
    alpha_post = state.dir_prior + Nk
    
    # Improved mean estimation using asymptotic properties
    S_k = resp.T @ embeddings
    mu_post = S_k / (jnp.linalg.norm(S_k, axis=1, keepdims=True) + 1e-10)
    
    # Improved concentration estimation
    r_bar = jnp.linalg.norm(S_k, axis=1) / (Nk + 1e-10)
    r_bar = jnp.clip(r_bar, 1e-8, 1 - 1e-8)
    
    # Use asymptotic kappa estimation
    kappa_post = estimate_kappa_improved(r_bar, d)
    
    # Add regularization based on dimension
    reg_factor = 1.0 / jnp.sqrt(d)
    kappa_post = kappa_post * reg_factor
    
    return ModelState(
        n_clusters=state.n_clusters,
        d_model=state.d_model,
        dir_prior=alpha_post,
        mean_prior_conc=state.mean_prior_conc,  # Assuming mean_prior_conc remains unchanged
        mix_dir=alpha_post,  # Updated Dirichlet parameters
        means=mu_post,
        conc=kappa_post,
        cluster_counts=state.cluster_counts + Nk,
        dir_sums=state.dir_sums + S_k
    )

def initialize_state_improved(n_clusters, d_model, seed=0):
    """
    Initialize model state using asymptotic properties for improved performance in high dimensions.
    
    Adjustments:
    - Initializes means with well-separated centers.
    - Sets concentration parameters scaling linearly with dimension.
    - Adjusts Dirichlet parameters based on dimension.
    
    Args:
        n_clusters (int): Number of clusters.
        d_model (int): Dimension of the space.
        seed (int): Random seed for initialization.
        
    Returns:
        ModelState: Initialized model state.
    """
    key = jax.random.PRNGKey(seed)
    
    # Initialize means using well-separated centers
    means_init = generate_well_separated_centers(key, n_clusters, d_model)
    
    # Initialize kappa using dimension-dependent scaling
    base_conc = d_model  # Linear scaling with dimension
    conc = jnp.full((n_clusters,), base_conc)
    
    # Initialize Dirichlet with dimension-appropriate concentration
    dir_conc = 1.0 / jnp.sqrt(d_model)  # Scales with dimension
    mix_dir = jnp.full((n_clusters,), dir_conc)
    
    # Initialize other parameters
    cluster_counts = jnp.zeros(n_clusters)
    dir_sums = jnp.zeros((n_clusters, d_model))
    
    return ModelState(
        n_clusters=n_clusters,
        d_model=d_model,
        dir_prior=0.0,  # Assuming prior is handled differently
        mean_prior_conc=0.0,  # Assuming no prior concentration
        mix_dir=mix_dir,
        means=means_init,
        conc=conc,
        cluster_counts=cluster_counts,
        dir_sums=dir_sums
    )



@jax.jit
def bayesian_update_improved(state: ModelState, batch_samples: jnp.ndarray) -> ModelState:
    """
    Perform an improved Bayesian update of the model state using a batch of samples.
    
    This function integrates enhanced E-step and M-step procedures tailored for high-dimensional
    data, leveraging asymptotic insights to improve numerical stability and parameter estimation.
    
    Args:
        state (ModelState): The current state of the vMF mixture model.
        batch_samples (jnp.ndarray): A batch of input embeddings with shape (batch_size, d_model).
        
    Returns:
        ModelState: The updated model state after performing the Bayesian update.
    """
    # Compute responsibilities using the improved E-step
    resp = e_step_improved(state, batch_samples, log_prior=jnp.zeros(batch_samples.shape[0]))
    
    # Update the state using the improved M-step
    updated_state = m_step_improved(state, batch_samples, resp)
    
    return updated_state

def train_vmf_improved_model(samples, initial_state, n_epochs=20, batch_size=1000):
    """
    Train the vMF mixture model using the improved Bayesian update.

    Args:
        samples (jnp.ndarray): The complete dataset of embeddings.
        initial_state (ModelState): The initialized state of the model.
        n_epochs (int): Number of training epochs.
        batch_size (int): Number of samples per batch.

    Returns:
        ModelState: The trained model state.
    """
    n_samples = samples.shape[0]
    n_batches = n_samples // batch_size
    state = initial_state

    for epoch in range(n_epochs):
        # Shuffle the data at the beginning of each epoch
        key = jax.random.PRNGKey(epoch)
        perm = jax.random.permutation(key, n_samples)
        shuffled_samples = samples[perm]

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            batch = shuffled_samples[start:end]

            # Perform Bayesian update
            state = bayesian_update_improved(state, batch)

            if (i + 1) % 10 == 0:
                log_likelihood = compute_log_likelihood_improved(state, batch)
                print(f"Epoch {epoch+1}, Batch {i+1}/{n_batches}, Log-Likelihood: {log_likelihood:.4f}")

    return state


def run_vmf_improved_test(n_clusters, d_model, kappas, alphas, batch_size):
    """
    Run a single VMF mixture test with given parameters.
    
    Args:
        n_clusters: Number of clusters
        d_model: Dimension of the space
        kappas: Array of concentration parameters
        alphas: Array of mixing proportions
        batch_size: Size of each batch for processing
    
    Returns:
        dict: Dictionary containing test results and metrics
    """
    # Generate well-separated centers
    key = jax.random.PRNGKey(0)
    centers = generate_well_separated_centers(key, n_clusters, d_model)
    
    # Generate samples
    samples = []
    true_labels = []
    samples_per_cluster = 1000
    
    for i in range(n_clusters):
        key, subkey = jax.random.split(key)
        n_samples = int(samples_per_cluster * alphas[i])
        cluster_samples = vmf_sample_wood(subkey, centers[i], kappas[i], n_samples)
        samples.append(cluster_samples)
        true_labels.extend([i] * n_samples)
    
    samples = jnp.concatenate(samples, axis=0)
    true_labels = jnp.array(true_labels)
    
    # Initialize and train model
    initial_state = initialize_state_improved(n_clusters, d_model, seed=0)
    trained_state = train_vmf_improved_model(samples, initial_state, n_epochs=5, batch_size=batch_size)
    
    # Compute final metrics
    log_prior = jnp.zeros(samples.shape[0])
    resp = e_step_improved(trained_state, samples, log_prior)
    cluster_assignments = jnp.argmax(resp, axis=1)
    accuracy = float(jnp.mean(cluster_assignments == true_labels))
    
    # Parameter recovery metrics
    mix_props = trained_state.mix_dir / jnp.sum(trained_state.mix_dir)
    mix_prop_error = float(jnp.mean(jnp.abs(mix_props - alphas)))
    kappa_rel_error = float(jnp.mean(jnp.abs(trained_state.conc - kappas) / kappas))
    
    mean_cos_sims = [float(jnp.abs(jnp.dot(trained_state.means[i], centers[i]))) 
                    for i in range(n_clusters)]
    
    return {
        'n_clusters': n_clusters,
        'd_model': d_model,
        'accuracy': accuracy,
        'mix_prop_error': mix_prop_error,
        'kappa_rel_error': kappa_rel_error,
        'mean_cos_sims': mean_cos_sims,
        'true_kappas': kappas,
        'recovered_kappas': trained_state.conc,
        'true_alphas': alphas,
        'recovered_alphas': mix_props,
        'final_log_likelihood': compute_log_likelihood_improved(trained_state, samples)
    }

def serialize_results(results):
    """
    Recursively convert JAX arrays in the results to Python lists or floats.
    
    Args:
        results (list or dict): The results data structure containing JAX arrays.
        
    Returns:
        list or dict: A serialized version of the results with JAX arrays converted.
    """
    if isinstance(results, dict):
        return {key: serialize_results(value) for key, value in results.items()}
    elif isinstance(results, list):
        return [serialize_results(element) for element in results]
    elif isinstance(results, jnp.ndarray):
        return results.tolist()
    else:
        return results
# Main execution
if __name__ == "__main__":
    import numpy as np
    from datetime import datetime
    
    # Set random seed for reproducibility
    rng = np.random.default_rng(42)
    n_tests = 10
    batch_size = 1000
    
    results = {'improved': [], 'baseline': []}
    print(f"Starting {n_tests} random tests comparing improved vs baseline implementations...")
    
    for test_idx in range(n_tests):
        print(f"\nTest {test_idx + 1}/{n_tests}")
        
        # Generate random parameters
        n_clusters, d_model, kappas, alphas = generate_random_params(rng)
        
        print(f"Parameters:")
        print(f"  n_clusters: {n_clusters}")
        print(f"  d_model: {d_model}")
        print(f"  kappas: {kappas}")
        print(f"  alphas: {alphas}")
        
        # Run test with improved implementation
        print("\nRunning improved implementation...")
        improved_result = run_vmf_improved_test(n_clusters, d_model, kappas, alphas, batch_size)
        results['improved'].append(improved_result)
        
        # Run test with baseline implementation
        print("\nRunning baseline implementation...")
        # Use original functions instead of improved versions
        baseline_result = run_vmf_test(n_clusters, d_model, kappas, alphas, batch_size)
        results['baseline'].append(baseline_result)
        
        print(f"\nResults Comparison:")
        print("Improved vs Baseline")
        print(f"  Accuracy: {improved_result['accuracy']:.4f} vs {baseline_result['accuracy']:.4f}")
        print(f"  Mixing proportion error: {improved_result['mix_prop_error']:.4f} vs {baseline_result['mix_prop_error']:.4f}")
        print(f"  Kappa relative error: {improved_result['kappa_rel_error']:.4f} vs {baseline_result['kappa_rel_error']:.4f}")
        print(f"  Mean cosine similarities:")
        print(f"    Improved: {[f'{x:.4f}' for x in improved_result['mean_cos_sims']]}")
        print(f"    Baseline: {[f'{x:.4f}' for x in baseline_result['mean_cos_sims']]}")
    
    # Serialize results
    serializable_results = serialize_results(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"vmf_test_comparison_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Print summary statistics and comparisons
    improved_accuracies = [r['accuracy'] for r in results['improved']]
    baseline_accuracies = [r['accuracy'] for r in results['baseline']]
    improved_mix_errors = [r['mix_prop_error'] for r in results['improved']]
    baseline_mix_errors = [r['mix_prop_error'] for r in results['baseline']]
    improved_kappa_errors = [r['kappa_rel_error'] for r in results['improved']]
    baseline_kappa_errors = [r['kappa_rel_error'] for r in results['baseline']]
    
    print("\nSummary Statistics (Improved vs Baseline):")
    print(f"Accuracy: {np.mean(improved_accuracies):.4f} ± {np.std(improved_accuracies):.4f} vs "
          f"{np.mean(baseline_accuracies):.4f} ± {np.std(baseline_accuracies):.4f}")
    print(f"Mixing Proportion Error: {np.mean(improved_mix_errors):.4f} ± {np.std(improved_mix_errors):.4f} vs "
          f"{np.mean(baseline_mix_errors):.4f} ± {np.std(baseline_mix_errors):.4f}")
    print(f"Kappa Relative Error: {np.mean(improved_kappa_errors):.4f} ± {np.std(improved_kappa_errors):.4f} vs "
          f"{np.mean(baseline_kappa_errors):.4f} ± {np.std(baseline_kappa_errors):.4f}")
