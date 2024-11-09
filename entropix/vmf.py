import os
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp, digamma
from functools import partial
from typing import NamedTuple, Optional
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set JAX to use CPU
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

# Force JAX to initialize CUDA before anything else
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

def log_bessel_iv_saddle(nu, kappa):
    """
    Computes log(I_nu(kappa)) using the saddlepoint approximation.
    """
    t = kappa / nu
    sqrt_term = jnp.sqrt(1 + t**2)
    eta = sqrt_term + jnp.log(t / (1 + sqrt_term) + 1e-10)
    log_iv = nu * eta - 0.5 * jnp.log(2 * jnp.pi * nu) - 0.5 * jnp.log(1 + t**2)
    return log_iv

def compute_A_d_saddle(nu, kappa):
    """
    Computes A_d(kappa) using the saddlepoint approximation.
    """
    t = kappa / nu
    sqrt_term = jnp.sqrt(1 + t**2)
    A_d = t / sqrt_term
    return A_d

def estimate_kappa_from_r_bar(r_bar, d):
    """
    Estimate kappa from r_bar using the inverted saddlepoint approximation.
    """
    nu = d / 2 - 1
    r_bar = jnp.clip(r_bar, 1e-8, 1 - 1e-8)
    kappa = nu * r_bar / jnp.sqrt(1 - r_bar ** 2)
    return kappa

def log_Cp_saddle(kappa, d):
    """
    Computes log C_p(kappa) using the saddlepoint approximation.
    """
    nu = d / 2 - 1
    log_iv = log_bessel_iv_saddle(nu, kappa)
    log_Cp = nu * jnp.log(kappa) - (d / 2) * jnp.log(2 * jnp.pi) - log_iv
    return log_Cp

def initialize_state(n_clusters, d_model, dir_prior=1.0, mean_prior_conc=0.0, seed=0):
    """
    Initialize the model state with improved initialization for high dimensions.
    """
    key = jax.random.PRNGKey(seed)
    
    # Generate orthonormal vectors as initial means using QR decomposition
    key, subkey = jax.random.split(key)
    random_matrix = jax.random.normal(subkey, (d_model, n_clusters))
    Q, _ = jnp.linalg.qr(random_matrix)
    means_init = Q[:, :n_clusters].T  # Shape: (n_clusters, d_model)
    
    # Initialize concentration parameters proportional to dimension
    conc = jnp.full((n_clusters,), d_model)  # Higher initial concentration
    
    # Rest remains the same
    mix_dir = jnp.full((n_clusters,), dir_prior)
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

def e_step(state: ModelState, embeddings: jnp.ndarray, log_prior: jnp.ndarray):
    """
    Perform the E-step: compute responsibilities.
    Args:
        state: ModelState containing current parameters
        embeddings: shape (batch_size, d_model)
        log_prior: shape (batch_size,)
    Returns:
        resp: shape (batch_size, n_clusters)
    """
    # Compute log mixing proportions
    log_mix = digamma(state.mix_dir) - digamma(jnp.sum(state.mix_dir))
    
    # Compute log normalizing constants using the saddlepoint approximation
    log_norm = log_Cp_saddle(state.conc, state.d_model)
    
    # Compute weighted means
    weighted_means = state.means * state.conc[:, None]  # Shape: (n_clusters, d_model)
    
    # Compute log probabilities (unnormalized)
    log_resp = embeddings @ weighted_means.T  # Shape: (batch_size, n_clusters)
    log_resp += log_mix  # Add log mixing proportions
    log_resp -= log_norm  # Subtract log normalizing constants
    log_resp += log_prior[:, None]  # Add log prior if provided
    
    # Normalize log responsibilities
    log_resp = log_resp - logsumexp(log_resp, axis=1, keepdims=True)
    resp = jnp.exp(log_resp)
    return resp

def m_step(state: ModelState, embeddings: jnp.ndarray, resp: jnp.ndarray):
    """
    Perform the M-step: update parameters using responsibilities.
    Args:
        state: ModelState containing current parameters
        embeddings: shape (batch_size, d_model)
        resp: shape (batch_size, n_clusters)
    Returns:
        Updated ModelState
    """
    # Update sufficient statistics
    batch_counts = jnp.sum(resp, axis=0)         # Shape: (n_clusters,)
    batch_dir_sums = resp.T @ embeddings         # Shape: (n_clusters, d_model)
    
    # Update mixing proportions
    new_mix_dir = state.dir_prior + batch_counts
    
    # Compute new means
    norms = jnp.linalg.norm(batch_dir_sums, axis=1)
    new_means = batch_dir_sums / norms[:, None]
    
    # Normalize means
    new_means = new_means / jnp.linalg.norm(new_means, axis=1, keepdims=True)
    
    # Compute mean resultant length r_bar
    r_bar = norms / (batch_counts + 1e-8)
    r_bar = jnp.clip(r_bar, 1e-8, 1 - 1e-8)
    
    # Estimate kappa using the accurate formula
    new_conc = estimate_kappa_from_r_bar(r_bar, state.d_model)
    new_conc = jnp.clip(new_conc, 1e-3, 1e5)
    new_conc = jnp.where(batch_counts > 0, new_conc, state.conc)
    
    # Update cluster counts and directional sums
    new_cluster_counts = state.cluster_counts + batch_counts
    new_dir_sums = state.dir_sums + batch_dir_sums
    
    # Create updated state
    new_state = ModelState(
        n_clusters=state.n_clusters,
        d_model=state.d_model,
        dir_prior=state.dir_prior,
        mean_prior_conc=state.mean_prior_conc,
        mix_dir=new_mix_dir,
        means=new_means,
        conc=new_conc,
        cluster_counts=new_cluster_counts,
        dir_sums=new_dir_sums
    )
    return new_state

def compute_log_likelihood(state: ModelState, embeddings: jnp.ndarray):
    """
    Compute the log likelihoods of embeddings under each component.
    Args:
        state: ModelState containing current parameters
        embeddings: shape (batch_size, d_model)
    Returns:
        log_likelihoods: shape (batch_size,)
    """
    # Compute log mixing proportions
    log_mix = digamma(state.mix_dir) - digamma(jnp.sum(state.mix_dir))

    # Compute log normalizing constants using saddlepoint approximation
    log_norm = log_Cp_saddle(state.conc, state.d_model)

    # Compute weighted means
    weighted_means = state.means * state.conc[:, None]  # Shape: (n_clusters, d_model)

    # Compute log probabilities (unnormalized)
    log_prob = embeddings @ weighted_means.T  # Shape: (batch_size, n_clusters)
    log_prob += log_mix  # Add log mixing proportions
    log_prob -= log_norm  # Subtract log normalizing constants

    # Compute total log likelihood
    log_likelihoods = logsumexp(log_prob, axis=1)
    return log_likelihoods

def vmf_sample(key, mu, kappa, num_samples, rsf=10):
    """
    Sample from a vMF distribution.
    Args:
        key: PRNG key
        mu: Mean direction, shape (d,)
        kappa: Concentration parameter, scalar
        num_samples: Number of samples to generate
        rsf: Rejection sampling factor (oversampling factor)
    Returns:
        samples: Generated samples, shape (num_samples, d)
    """
    d = mu.shape[0]
    mu = mu / jnp.linalg.norm(mu)

    kmr = jnp.sqrt(4 * kappa ** 2 + (d - 1) ** 2)
    bb = (kmr - 2 * kappa) / (d - 1)
    aa = (kmr + 2 * kappa + d - 1) / 4
    dd = (4 * aa * bb) / (1 + bb) - (d - 1) * jnp.log(d - 1)

    total_samples = num_samples * rsf
    key, subkey_beta, subkey_uniform, subkey_v = jax.random.split(key, 4)
    beta_rv = jax.random.beta(subkey_beta, a=0.5 * (d - 1), b=0.5 * (d - 1), shape=(total_samples,))
    uniform_rv = jax.random.uniform(subkey_uniform, shape=(total_samples,))

    w0 = (1 - (1 + bb) * beta_rv) / (1 - (1 - bb) * beta_rv)
    t0 = (2 * aa * bb) / (1 - (1 - bb) * beta_rv)
    det = (d - 1) * jnp.log(t0) - t0 + dd - jnp.log(uniform_rv + 1e-8)
    valid_indices = jnp.nonzero(det >= 0)[0]
    w0_valid = w0[valid_indices]

    # Ensure we have enough samples
    num_valid = w0_valid.shape[0]
    if num_valid < num_samples:
        # Recursively sample more if needed
        key, subkey = jax.random.split(key)
        additional_samples = vmf_sample(subkey, mu, kappa, num_samples - num_valid, rsf)
        w0_valid = jnp.concatenate([w0_valid, additional_samples[:, 0]])

        # Generate corresponding v's for the new w0's
        v = jax.random.normal(subkey_v, shape=(w0_valid.shape[0], d - 1))
        v = v / jnp.linalg.norm(v, axis=1, keepdims=True)
    else:
        w0_valid = w0_valid[:num_samples]
        v = jax.random.normal(subkey_v, shape=(total_samples, d - 1))
        v = v[valid_indices][:num_samples]
        v = v / jnp.linalg.norm(v, axis=1, keepdims=True)

    samples = jnp.concatenate([w0_valid[:, None], jnp.sqrt(1 - w0_valid ** 2 + 1e-8)[:, None] * v], axis=1)

    # Householder transformation
    e1 = jnp.zeros_like(mu)
    e1 = e1.at[0].set(1.0)
    u = e1 - mu
    u = u / jnp.linalg.norm(u + 1e-8)
    # Apply Householder transformation
    samples = samples - 2 * jnp.outer(jnp.dot(samples, u), u)

    return samples

def vmf_mixture_sample(key, mus, kappas, alphas, num_samples):
    """
    Sample from a mixture of vMF distributions.
    Args:
        key: PRNG key
        mus: Mean directions, shape (n_clusters, d)
        kappas: Concentration parameters, shape (n_clusters,)
        alphas: Mixing proportions, shape (n_clusters,)
        num_samples: Number of samples to generate
    Returns:
        samples: Generated samples, shape (num_samples, d)
        cids: Component indices for each sample, shape (num_samples,)
    """
    n_clusters, d = mus.shape
    key, subkey_choice = jax.random.split(key)
    cids = jax.random.choice(subkey_choice, n_clusters, shape=(num_samples,), p=alphas)

    def sample_component(c):
        key_c = jax.random.fold_in(key, c)
        num_c = jnp.sum(cids == c)
        samples_c = vmf_sample(key_c, mus[c], kappas[c], num_c)
        return samples_c

    samples = []
    for c in range(n_clusters):
        samples_c = sample_component(c)
        samples.append(samples_c)

    samples = jnp.concatenate(samples, axis=0)
    # Shuffle the samples to mix components
    key, subkey_shuffle = jax.random.split(key)
    permutation = jax.random.permutation(subkey_shuffle, num_samples)
    samples = samples[permutation]
    cids = cids[permutation]

    return samples, cids

def bayesian_update_state(state: ModelState, embeddings: jnp.ndarray):
    """
    Perform Bayesian update to the concentration, Dirichlet parameters, and means given new data.
    Args:
        state: Current ModelState
        embeddings: New data samples, shape (batch_size, d_model)
    Returns:
        Updated ModelState
    """
    # Compute responsibilities using the current state
    log_prior = jnp.zeros(embeddings.shape[0])  # Assuming uniform prior over data
    resp = e_step(state, embeddings, log_prior)

    # Update the Dirichlet parameters (mixing proportions)
    Nk = jnp.sum(resp, axis=0)  # Effective number of samples per cluster
    alpha_post = state.mix_dir + Nk  # Updated Dirichlet parameters

    # Create a mask for clusters with Nk > 0
    mask = Nk > 1e-8

    # Update the mean directions with prior
    S_k = resp.T @ embeddings  # Sum of data weighted by responsibilities per cluster
    mean_prior_conc = state.mean_prior_conc
    kappa_prior = mean_prior_conc
    kappa_post = kappa_prior + Nk  # Updated concentration parameters

    # Avoid division by zero in kappa_post
    kappa_post_safe = jnp.where(kappa_post > 1e-8, kappa_post, 1.0)

    mu_prior = jnp.zeros((state.n_clusters, state.d_model))  # Assuming zero prior mean
    mu_numerator = (kappa_prior * mu_prior) + S_k
    mu_denominator = kappa_post_safe[:, None]
    mu_post = mu_numerator / mu_denominator

    # For clusters with Nk == 0, keep the previous means
    mu_post = jnp.where(mask[:, None], mu_post, state.means)

    # Normalize means
    mu_post_norm = jnp.linalg.norm(mu_post, axis=1, keepdims=True)
    mu_post_norm = jnp.where(mu_post_norm > 1e-8, mu_post_norm, 1.0)
    mu_post = mu_post / mu_post_norm

    # Compute mean resultant length r_bar
    r_bar = jnp.where(mask, jnp.linalg.norm(S_k, axis=1) / (Nk + 1e-8), 0.0)
    r_bar = jnp.clip(r_bar, 1e-8, 1 - 1e-8)
    
    # Estimate kappa using the accurate formula
    kappa_post = estimate_kappa_from_r_bar(r_bar, state.d_model)
    kappa_post = jnp.clip(kappa_post, 1e-3, 1e5)
    kappa_post = jnp.where(mask, kappa_post, state.conc)

    # Update cluster counts and directional sums
    cluster_counts = state.cluster_counts + Nk
    dir_sums = state.dir_sums + S_k

    # Create updated state
    updated_state = ModelState(
        n_clusters=state.n_clusters,
        d_model=state.d_model,
        dir_prior=state.dir_prior,
        mean_prior_conc=state.mean_prior_conc,
        mix_dir=alpha_post,
        means=mu_post,
        conc=kappa_post,
        cluster_counts=cluster_counts,
        dir_sums=dir_sums
    )

    return updated_state
if __name__ == "__main__":
    # Test the implementation with improved parameters
    d_model = 1000
    n_clusters = 3
    samples_per_cluster = 1000  # Increased sample size
    
    # Initialize with higher concentration
    state = initialize_state(n_clusters, d_model, dir_prior=5.0)
    
    # Generate test data with better separation
    key = jax.random.PRNGKey(0)
    
    # Generate well-separated centers using spherical coding
    angles = jnp.linspace(0, 2 * jnp.pi, n_clusters + 1)[:-1]
    centers = jnp.zeros((n_clusters, d_model))
    centers = centers.at[:, 0].set(jnp.cos(angles))
    centers = centers.at[:, 1].set(jnp.sin(angles))
    # Normalize centers
    centers = centers / jnp.linalg.norm(centers, axis=1, keepdims=True)
    
    # Use much larger kappas for high dimensions
    kappas = jnp.array([2000.0, 2500.0, 3000.0])  # Significantly increased
    alphas = jnp.array([0.3, 0.4, 0.3])
    
    # Generate more samples
    n_samples = samples_per_cluster * n_clusters
    samples, true_labels = vmf_mixture_sample(key, centers, kappas, alphas, n_samples)
    
    # Perform multiple Bayesian updates with smaller batches
    batch_size = 500
    n_batches = n_samples // batch_size
    updated_state = state
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_samples = samples[start_idx:end_idx]
        updated_state = bayesian_update_state(updated_state, batch_samples)
    
    # Rest of the evaluation code remains the same
    log_prior = jnp.zeros(samples.shape[0])
    resp = e_step(updated_state, samples, log_prior)
    cluster_assignments = jnp.argmax(resp, axis=1)
    
    # Print diagnostics
    print("\nTest Results:")
    print("-" * 50)
    
    # Compare cluster assignments with true labels
    from itertools import permutations
    perms = list(permutations(range(n_clusters)))
    best_perm = None
    best_accuracy = 0.0
    
    for perm in perms:
        perm = jnp.array(perm)
        remapped_assignments = jnp.zeros_like(cluster_assignments)
        for i, p in enumerate(perm):
            remapped_assignments = jnp.where(cluster_assignments == i, p, remapped_assignments)
        accuracy = jnp.mean(remapped_assignments == true_labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_perm = perm
    
    print(f"\nClustering Accuracy: {best_accuracy * 100:.2f}%")
    # Compare recovered parameters with true values
    print("\nTrue vs Recovered Parameters:")
    print("-" * 50)
    
    # Get mixing proportions from updated state
    log_mix = digamma(updated_state.mix_dir) - digamma(jnp.sum(updated_state.mix_dir))
    recovered_alphas = jnp.exp(log_mix)
    recovered_alphas = recovered_alphas / jnp.sum(recovered_alphas)
    
    print("\nMixing Proportions (alphas):")
    for i, p in enumerate(best_perm):
        print(f"Cluster {i}:")
        print(f"  True:      {alphas[i]:.3f}")
        print(f"  Recovered: {recovered_alphas[p]:.3f}")
    
    print("\nConcentration Parameters (kappas):")
    for i, p in enumerate(best_perm):
        print(f"Cluster {i}:")
        print(f"  True:      {kappas[i]:.1f}")
        print(f"  Recovered: {updated_state.conc[p]:.1f}")
    
    print("\nMean Directions (cosine similarity):")
    for i, p in enumerate(best_perm):
        cos_sim = jnp.abs(jnp.dot(centers[i], updated_state.means[p]))
        print(f"Cluster {i}: {cos_sim:.3f}")
