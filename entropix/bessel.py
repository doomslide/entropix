import mpmath

# Set high precision
mpmath.mp.dps = 50  # Set decimal places to 50 for high precision

# Define the numerical evaluation for the exact value
def Cp_exact(kappa, R):
    p = 1 / R
    numerator = mpmath.power(kappa, (p / 2) - 1)
    denominator = mpmath.power(2 * mpmath.pi, p / 2) * mpmath.besseli((p / 2) - 1, kappa)
    return numerator / denominator

# Define a function for the log of Cp(kappa) for exact calculation
def log_Cp_exact(kappa, R):
    return mpmath.log(Cp_exact(kappa, R))

# Calculate the exact value for comparison
kappa_value = 2.0  # Arbitrary choice for kappa
R_value = 1/10000    # Small R to simulate the limit as R -> 0
exact_log_Cp_value = log_Cp_exact(kappa_value, R_value)

# # Implement the asymptotic approximation using the uniform asymptotic expansion
# def log_Cp_asymptotic(kappa, R):
#     nu = (1 / (2 * R)) - 1
#     t = kappa / nu

#     sqrt_one_plus_t2 = mpmath.sqrt(1 + t**2)
#     eta = sqrt_one_plus_t2 + mpmath.log(t / (1 + sqrt_one_plus_t2))

#     # Compute log of the leading term
#     log_leading_term = nu * eta - 0.5 * mpmath.log(2 * mpmath.pi * nu) - 0.25 * mpmath.log(1 + t**2)

#     # Compute u1(t) and its contribution in logarithmic form
#     u1_t = (3 * t**2 - 1) / (24 * (1 + t**2)**1.5)
#     correction_term = 1 + u1_t / nu
#     log_correction_term = mpmath.log(correction_term)

#     # Compute log of I_nu approximation
#     log_I_nu_approx = log_leading_term + log_correction_term

#     # Compute log of the prefactor
#     log_prefactor = ((1 / (2 * R)) - 1) * mpmath.log(kappa) - (1 / (2 * R)) * mpmath.log(2 * mpmath.pi)

#     # Compute log of Cp(kappa) directly
#     log_Cp_value = log_prefactor - log_I_nu_approx
#     return log_Cp_value
import jax
import jax.numpy as jnp
from jax import jit

@jit
def log_Cp_asymptotic(kappa, R):
    # Define nu and t
    nu = (1 / (2 * R)) - 1
    t = kappa / nu
    
    # Compute sqrt(1 + t^2) and eta
    sqrt_one_plus_t2 = jnp.sqrt(1 + t**2)
    eta = sqrt_one_plus_t2 + jnp.log(t / (1 + sqrt_one_plus_t2))
    
    # Compute log of the leading term
    log_leading_term = nu * eta - 0.5 * jnp.log(2 * jnp.pi * nu) - 0.25 * jnp.log(1 + t**2)
    
    # Compute u1(t) and its logarithmic contribution
    u1_t = (3 * t**2 - 1) / (24 * (1 + t**2)**1.5)
    correction_term = 1 + u1_t / nu
    log_correction_term = jnp.log(correction_term)
    
    # Compute log of I_nu approximation
    log_I_nu_approx = log_leading_term + log_correction_term
    
    # Compute log of the prefactor
    log_prefactor = ((1 / (2 * R)) - 1) * jnp.log(kappa) - (1 / (2 * R)) * jnp.log(2 * jnp.pi)
    
    # Compute log of Cp(kappa)
    log_Cp_value = log_prefactor - log_I_nu_approx
    return log_Cp_value

# Example usage
kappa_value = 2.0  # Arbitrary choice for kappa
R_value = 1/10000  # Small R to simulate the limit as R -> 0
asymptotic_log_Cp_value = log_Cp_asymptotic(kappa_value, R_value)

# Compare the exact and asymptotic values
print("Exact log Cp(kappa):", exact_log_Cp_value)
print("Asymptotic log Cp(kappa):", asymptotic_log_Cp_value)
print("Difference:", exact_log_Cp_value - float(asymptotic_log_Cp_value))
