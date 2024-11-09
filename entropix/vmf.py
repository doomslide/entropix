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
    Simplified initialization with strong numerical safeguards.
    """
    key = jax.random.PRNGKey(seed)
    
    # Initialize means using fixed angles in first two dimensions
    means_init = jnp.zeros((n_clusters, d_model))
    angles = jnp.linspace(0, 2 * jnp.pi, n_clusters + 1)[:-1]
    means_init = means_init.at[:, 0].set(jnp.cos(angles))
    means_init = means_init.at[:, 1].set(jnp.sin(angles))
    means_init = means_init / jnp.linalg.norm(means_init, axis=1, keepdims=True)
    
    # Initialize with moderate concentration
    conc = jnp.full((n_clusters,), 500.0)
    
    # Initialize mixing proportions uniformly
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
    Modified M-step with minimum thresholds to prevent collapse.
    """
    # Update mixing proportions with minimum threshold
    Nk = jnp.sum(resp, axis=0)  # Effective number of points in each cluster
    alpha_post = state.mix_dir + Nk
    alpha_post = jnp.maximum(alpha_post, 1.0)  # Prevent collapse to zero
    
    # Update means
    S_k = resp.T @ embeddings  # Sum of weighted points
    mu_post = S_k / (jnp.linalg.norm(S_k, axis=1, keepdims=True) + 1e-10)
    
    # Update concentration parameters with minimum threshold
    r_bar = jnp.linalg.norm(S_k, axis=1) / (Nk + 1e-10)
    r_bar = jnp.clip(r_bar, 1e-8, 1 - 1e-8)
    
    # Estimate kappa with minimum threshold
    kappa_post = estimate_kappa_from_r_bar(r_bar, state.d_model)
    kappa_post = jnp.maximum(kappa_post, 100.0)  # Prevent collapse to zero
    
    # Update counts and sums
    cluster_counts = state.cluster_counts + Nk
    dir_sums = state.dir_sums + S_k
    
    return ModelState(
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

def compute_log_likelihood(state: ModelState, embeddings: jnp.ndarray):
    """
    Compute the log likelihoods with improved numerical stability.
    """
    # Normalize embeddings
    embeddings_norm = jnp.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / (embeddings_norm + 1e-10)
    
    # Compute log mixing proportions with stability
    mix_dir_sum = jnp.sum(state.mix_dir)
    log_mix = digamma(state.mix_dir) - digamma(mix_dir_sum)
    log_mix = jnp.clip(log_mix, -100, 100)  # Prevent extreme values

    # Compute log normalizing constants
    log_norm = log_Cp_saddle(state.conc, state.d_model)
    log_norm = jnp.clip(log_norm, -100, 100)  # Prevent extreme values

    # Compute dot products with stability
    dot_products = embeddings_normalized @ state.means.T  # Shape: (batch_size, n_clusters)
    dot_products = jnp.clip(dot_products, -1 + 1e-7, 1 - 1e-7)

    # Compute log probabilities
    log_prob = dot_products * state.conc[None, :]
    log_prob += log_mix[None, :]
    log_prob -= log_norm[None, :]

    # Compute total log likelihood with stability
    max_log_prob = jnp.max(log_prob, axis=1, keepdims=True)
    log_prob_shifted = log_prob - max_log_prob
    log_likelihoods = max_log_prob.squeeze() + jnp.log(jnp.sum(jnp.exp(log_prob_shifted), axis=1) + 1e-10)
    
    return jnp.mean(log_likelihoods)

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

def vmf_sample_wood(key, mu, kappa, num_samples):
    """
    Improved Wood's algorithm implementation with better numerical stability.
    """
    d = mu.shape[0]
    
    # More stable computation of b
    b = (-2 * kappa + jnp.sqrt(4 * kappa**2 + (d - 1)**2)) / (d - 1)
    b = jnp.clip(b, -1 + 1e-10, 1 - 1e-10)
    
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * jnp.log(1 - x0**2)
    
    # Pre-allocate arrays for vectorized operations
    key, subkey = jax.random.split(key)
    z = jax.random.beta(subkey, (d - 1) / 2, (d - 1) / 2, shape=(num_samples,))
    
    # Compute w with numerical stability
    w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
    w = jnp.clip(w, -1 + 1e-10, 1 - 1e-10)
    
    # Generate uniform random numbers for acceptance test
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=(num_samples,))
    
    # Compute acceptance criterion
    t = kappa * w + (d - 1) * jnp.log(1 - x0 * w)
    accept = (t - c) >= jnp.log(u)
    
    # Generate random directions on d-1 sphere
    key, subkey = jax.random.split(key)
    v = jax.random.normal(subkey, shape=(num_samples, d-1))
    v = v / jnp.linalg.norm(v, axis=1, keepdims=True)
    
    # Combine accepted w with random directions
    samples = jnp.zeros((num_samples, d))
    samples = samples.at[:, 0].set(w)
    samples = samples.at[:, 1:].set(jnp.sqrt(1 - w[:, None]**2) * v)
    
    # Rotate samples to align with mu
    e1 = jnp.zeros(d)
    e1 = e1.at[0].set(1.0)
    u = e1 - mu
    u = u / (jnp.linalg.norm(u) + 1e-10)
    samples = samples - 2 * jnp.outer(jnp.dot(samples, u), u)
    
    return samples

def verify_sampling(key, mu, kappa, d_model, num_samples=10000):
    """
    Verify the sampling function by comparing empirical and theoretical statistics.
    """
    samples = vmf_sample_wood(key, mu, kappa, num_samples)
    
    # Compute empirical mean resultant length
    mean_vector = jnp.mean(samples, axis=0)
    r_bar_empirical = jnp.linalg.norm(mean_vector)
    
    # Compute theoretical mean resultant length
    r_bar_theoretical = 1 - (d_model - 1)/(2 * kappa)
    
    # Compute cosine similarity with true mean
    cos_sim = jnp.abs(jnp.dot(mean_vector, mu))
    
    return {
        'r_bar_empirical': r_bar_empirical,
        'r_bar_theoretical': r_bar_theoretical,
        'cos_sim': cos_sim
    }

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
        samples_c = vmf_sample_wood(key_c, mus[c], kappas[c], num_c)
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

def train_vmf_mixture(samples, true_labels, state, n_epochs=20, batch_size=1000):
    """
    Training loop with comprehensive monitoring.
    """
    n_samples = samples.shape[0]
    n_batches = n_samples // batch_size
    current_state = state
    best_state = state
    best_likelihood = float('-inf')
    
    print("\nStarting training...")
    
    for epoch in range(n_epochs):
        # Shuffle data
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
            
            # Multiple EM iterations per batch
            for _ in range(3):
                # E-step
                log_prior = jnp.zeros(batch_samples.shape[0])
                resp = e_step(current_state, batch_samples, log_prior)
                
                # M-step
                current_state = m_step(current_state, batch_samples, resp)
            
            # Compute metrics
            batch_likelihood = compute_log_likelihood(current_state, batch_samples)
            batch_assignments = jnp.argmax(resp, axis=1)
            batch_accuracy = jnp.mean(batch_assignments == batch_labels)
            
            epoch_likelihood += batch_likelihood * batch_size
            epoch_accuracy += batch_accuracy * batch_size
            
            if (i + 1) % 5 == 0:
                mix_props = current_state.mix_dir / jnp.sum(current_state.mix_dir)
                print(f"\nEpoch {epoch + 1}, Batch {i + 1}/{n_batches}")
                print(f"  Batch Accuracy: {batch_accuracy:.4f}")
                print(f"  Batch Log-likelihood: {batch_likelihood:.4f}")
                print(f"  Mixing Proportions: {mix_props}")
                print(f"  Concentrations: {current_state.conc}")
                
                # Print mean directions cosine similarities
                for j in range(state.n_clusters):
                    cos_sim = jnp.abs(jnp.dot(current_state.means[j], state.means[j]))
                    print(f"  Cluster {j} mean cosine similarity: {cos_sim:.4f}")
        
        # Compute epoch metrics
        avg_epoch_likelihood = epoch_likelihood / n_samples
        avg_epoch_accuracy = epoch_accuracy / n_samples
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Log-likelihood: {avg_epoch_likelihood:.4f}")
        print(f"  Average Accuracy: {avg_epoch_accuracy:.4f}")
        
        # Update best state if needed
        if avg_epoch_likelihood > best_likelihood:
            best_likelihood = avg_epoch_likelihood
            best_state = current_state
        
        # Early stopping check
        if epoch > 0 and abs(avg_epoch_likelihood - prev_likelihood) < 1e-4:
            print("\nConverged! Stopping early.")
            break
            
        prev_likelihood = avg_epoch_likelihood
    
    return best_state

if __name__ == "__main__":
    # Test settings
    d_model = 1000
    n_clusters = 3
    samples_per_cluster = 10000
    
    # Generate well-separated centers
    key = jax.random.PRNGKey(0)
    centers = generate_well_separated_centers(key, n_clusters, d_model)
    
    # Print center separations
    cos_sims = centers @ centers.T
    print("\nCenter cosine similarities:")
    print(cos_sims)
    
    # Use moderate kappas for stability
    kappas = jnp.array([5000.0, 6000.0, 7000.0])
    alphas = jnp.array([0.3, 0.4, 0.3])
    
    # Verify sampling function
    print("\nVerifying sampling function...")
    for i in range(n_clusters):
        stats = verify_sampling(key, centers[i], kappas[i], d_model)
        print(f"\nCluster {i}:")
        print(f"Empirical r_bar: {stats['r_bar_empirical']:.4f}")
        print(f"Theoretical r_bar: {stats['r_bar_theoretical']:.4f}")
        print(f"Cosine similarity with true mean: {stats['cos_sim']:.4f}")
    
    # Generate mixture samples
    n_samples = samples_per_cluster * n_clusters
    samples = []
    true_labels = []
    
    for i in range(n_clusters):
        key, subkey = jax.random.split(key)
        cluster_samples = vmf_sample_wood(
            subkey, 
            centers[i], 
            kappas[i], 
            int(samples_per_cluster * alphas[i])
        )
        samples.append(cluster_samples)
        true_labels.extend([i] * int(samples_per_cluster * alphas[i]))
    
    samples = jnp.concatenate(samples, axis=0)
    true_labels = jnp.array(true_labels)
    
    # Initialize state with true parameters
    state = ModelState(
        n_clusters=n_clusters,
        d_model=d_model,
        dir_prior=10.0,
        mean_prior_conc=100.0,
        mix_dir=alphas * 100,  # Initialize close to true values
        means=centers,         # Initialize with true centers
        conc=kappas,          # Initialize with true kappas
        cluster_counts=jnp.zeros(n_clusters),
        dir_sums=jnp.zeros((n_clusters, d_model))
    )

    # Train the model
    final_state = train_vmf_mixture(samples, true_labels, state)
    
    # Final evaluation
    print("\nFinal Evaluation:")
    log_prior = jnp.zeros(samples.shape[0])
    resp = e_step(final_state, samples, log_prior)
    cluster_assignments = jnp.argmax(resp, axis=1)
    
    # Compute final accuracy
    accuracy = jnp.mean(cluster_assignments == true_labels)
    print(f"\nFinal Clustering Accuracy: {accuracy:.4f}")
    
    # Print parameter comparisons
    print("\nParameter Comparison:")
    mix_props = final_state.mix_dir / jnp.sum(final_state.mix_dir)
    print("\nMixing Proportions:")
    for i in range(n_clusters):
        print(f"Cluster {i}:")
        print(f"  True: {alphas[i]:.4f}")
        print(f"  Recovered: {mix_props[i]:.4f}")
    
    print("\nConcentration Parameters:")
    for i in range(n_clusters):
        print(f"Cluster {i}:")
        print(f"  True: {kappas[i]:.1f}")
        print(f"  Recovered: {final_state.conc[i]:.1f}")
    
    print("\nMean Directions (cosine similarity):")
    for i in range(n_clusters):
        cos_sim = jnp.abs(jnp.dot(final_state.means[i], centers[i]))
        print(f"Cluster {i}: {cos_sim:.4f}")
