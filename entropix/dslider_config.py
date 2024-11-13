from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import math

# Constants
MIN_TEMP = 1e-4
MAX_TEMP = 1e4
EPS = 1e-8
VOCAB_SIZE = 128256


@register_pytree_node_class
class OutlierThreshold:
  def __init__(self, bilinear, linear_state_ent, linear_state_std, 
               linear_naked_ent, linear_naked_std, linear_naked_varent, bias):
    self.bilinear = bilinear
    self.linear_state_ent = linear_state_ent
    self.linear_state_std = linear_state_std
    self.linear_naked_ent = linear_naked_ent
    self.linear_naked_std = linear_naked_std
    self.linear_naked_varent = linear_naked_varent
    self.bias = bias

  def tree_flatten(self):
    """For JAX pytree handling"""
    arrays = (self.bilinear, self.linear_state_ent, self.linear_state_std)
    aux_data = {
      'linear_naked_ent': self.linear_naked_ent,
      'linear_naked_std': self.linear_naked_std,
      'linear_naked_varent': self.linear_naked_varent,
      'bias': self.bias
    }
    return arrays, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, arrays):
    """For JAX pytree handling"""
    return cls(
      bilinear=arrays[0],
      linear_state_ent=arrays[1],
      linear_state_std=arrays[2],
      linear_naked_ent=aux_data['linear_naked_ent'],
      linear_naked_std=aux_data['linear_naked_std'],
      linear_naked_varent=aux_data['linear_naked_varent'],
      bias=aux_data['bias']
    )


@register_pytree_node_class
class ArgmaxThreshold:
  def __init__(self, weight, bias):
    self.weight = weight
    self.bias = bias

  def tree_flatten(self):
    """For JAX pytree handling"""
    return (self.weight, self.bias), {}

  @classmethod
  def tree_unflatten(cls, aux_data, arrays):
    """For JAX pytree handling"""
    return cls(weight=arrays[0], bias=arrays[1])


@register_pytree_node_class
class DirichletThreshold:
  def __init__(self, weight, bias):
    self.weight = weight
    self.bias = bias

  def tree_flatten(self):
    """For JAX pytree handling"""
    return (self.weight, self.bias), {}

  @classmethod
  def tree_unflatten(cls, aux_data, arrays):
    """For JAX pytree handling"""
    return cls(weight=arrays[0], bias=arrays[1])


@register_pytree_node_class
class TargetEntropy:
  def __init__(self, linear, linear_inv_temp, bias):
    self.linear = linear
    self.linear_inv_temp = linear_inv_temp
    self.bias = bias

  def tree_flatten(self):
    """For JAX pytree handling"""
    return (self.linear, self.linear_inv_temp), {'bias': self.bias}

  @classmethod
  def tree_unflatten(cls, aux_data, arrays):
    """For JAX pytree handling"""
    return cls(linear=arrays[0], linear_inv_temp=arrays[1], bias=aux_data['bias'])


@register_pytree_node_class
class DSConfig:
  def __init__(self, emwa_logp_base, emwa_logp_exp_factor, emwa_dir_coeff, 
               emwa_temp_coeff, emwa_dir_ent_coeff, emwa_ent_scaffold_coeff,
               emwa_varent_scaffold_coeff, emwa_ent_naked_coeff, emwa_varent_naked_coeff,
               emwa_topk_ent_naked_coeff, token_cross_ent_scaffold_coeff,
               token_cross_ent_naked_coeff, token_cross_var_scaffold_coeff,
               token_cross_var_naked_coeff, perturb_base_coeff, perturb_exp_coeff,
               dirichlet_support, noise_floor, outlier_threshold, argmax_threshold,
               dirichlet_threshold, target_entropy, outlier_topk):
    self.emwa_logp_base = emwa_logp_base
    self.emwa_logp_exp_factor = emwa_logp_exp_factor
    self.emwa_dir_coeff = emwa_dir_coeff
    self.emwa_temp_coeff = emwa_temp_coeff
    self.emwa_dir_ent_coeff = emwa_dir_ent_coeff
    self.emwa_ent_scaffold_coeff = emwa_ent_scaffold_coeff
    self.emwa_varent_scaffold_coeff = emwa_varent_scaffold_coeff
    self.emwa_ent_naked_coeff = emwa_ent_naked_coeff
    self.emwa_varent_naked_coeff = emwa_varent_naked_coeff
    self.emwa_topk_ent_naked_coeff = emwa_topk_ent_naked_coeff
    self.token_cross_ent_scaffold_coeff = token_cross_ent_scaffold_coeff
    self.token_cross_ent_naked_coeff = token_cross_ent_naked_coeff
    self.token_cross_var_scaffold_coeff = token_cross_var_scaffold_coeff
    self.token_cross_var_naked_coeff = token_cross_var_naked_coeff
    self.perturb_base_coeff = perturb_base_coeff
    self.perturb_exp_coeff = perturb_exp_coeff
    self.dirichlet_support = dirichlet_support
    self.noise_floor = noise_floor
    self.outlier_threshold = outlier_threshold
    self.argmax_threshold = argmax_threshold
    self.dirichlet_threshold = dirichlet_threshold
    self.target_entropy = target_entropy
    self.outlier_topk = outlier_topk

  def tree_flatten(self):
    """Improved flattening for JAX pytree"""
    arrays = (
      self.outlier_threshold,
      self.dirichlet_threshold,
      self.target_entropy
    )
    aux_data = {
      'emwa_logp_base': self.emwa_logp_base,
      'emwa_logp_exp_factor': self.emwa_logp_exp_factor,
      'emwa_dir_coeff': self.emwa_dir_coeff,
      'emwa_temp_coeff': self.emwa_temp_coeff,
      'emwa_dir_ent_coeff': self.emwa_dir_ent_coeff,
      'emwa_ent_scaffold_coeff': self.emwa_ent_scaffold_coeff,
      'emwa_varent_scaffold_coeff': self.emwa_varent_scaffold_coeff,
      'emwa_ent_naked_coeff': self.emwa_ent_naked_coeff,
      'emwa_varent_naked_coeff': self.emwa_varent_naked_coeff,
      'emwa_topk_ent_naked_coeff': self.emwa_topk_ent_naked_coeff,
      'token_cross_ent_scaffold_coeff': self.token_cross_ent_scaffold_coeff,
      'token_cross_ent_naked_coeff': self.token_cross_ent_naked_coeff,
      'token_cross_var_scaffold_coeff': self.token_cross_var_scaffold_coeff,
      'token_cross_var_naked_coeff': self.token_cross_var_naked_coeff,
      'perturb_base_coeff': self.perturb_base_coeff,
      'perturb_exp_coeff': self.perturb_exp_coeff,
      'dirichlet_support': self.dirichlet_support,
      'noise_floor': self.noise_floor,
      'argmax_threshold': self.argmax_threshold,
      'outlier_topk': self.outlier_topk
    }
    return arrays, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, arrays):
    """Improved unflattening for JAX pytree"""
    return cls(
      outlier_threshold=arrays[0],
      dirichlet_threshold=arrays[1],
      target_entropy=arrays[2],
      **aux_data
    )


DEFAULT_DS_CONFIG = DSConfig(
  emwa_logp_base=4.0,
  emwa_logp_exp_factor=3.0,
  emwa_dir_coeff=0.70,
  emwa_temp_coeff=0.70,
  emwa_dir_ent_coeff=0.70,
  emwa_ent_scaffold_coeff=0.70,
  emwa_varent_scaffold_coeff=0.70,
  emwa_ent_naked_coeff=0.70,
  emwa_varent_naked_coeff=0.70,
  emwa_topk_ent_naked_coeff=0.70,
  token_cross_ent_scaffold_coeff=0.65,
  token_cross_ent_naked_coeff=0.65,
  token_cross_var_scaffold_coeff=0.75,
  token_cross_var_naked_coeff=0.65,
  perturb_base_coeff=1,
  perturb_exp_coeff=1,
  dirichlet_support=jnp.arange(VOCAB_SIZE, dtype=jnp.int32),
  noise_floor=-12.0,
  outlier_threshold=OutlierThreshold(
    bilinear=jnp.ones((4, 4)) * 1.3,
    linear_state_ent=jnp.ones(4) * 0.80,
    linear_state_std=jnp.ones(4) * 0.80,
    linear_naked_ent=1.2,
    linear_naked_std=1.2,
    linear_naked_varent=1.2,
    bias=0.0,
  ),
  argmax_threshold=ArgmaxThreshold(weight=0.1, bias=1.2),
  dirichlet_threshold=DirichletThreshold(weight=0.1, bias=1.2),
  target_entropy=TargetEntropy(
    linear=jnp.array([1.0, 1.0, 1.0, 1.0]), linear_inv_temp=jnp.ones(1) * 8.0, bias=0.0
  ),
  outlier_topk=6,
)
