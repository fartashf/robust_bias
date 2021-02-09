# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for generating adversarial attacks."""
import copy

import jax
from jax import numpy as jnp
import scipy.fft
import numpy as np

from norm import norm_projection
from norm import norm_f


def sd_dir(grad, eps_iter):
  dft = np.matrix(scipy.linalg.dft(grad.shape[0], scale='sqrtn'))
  dftxgrad = dft @ grad
  dftz = dftxgrad.reshape(1, -1)
  dftz = jnp.concatenate((jnp.real(dftz), jnp.imag(dftz)), axis=0)
  # projection does not scale bigger, here we want to scale it
  def l2_normalize(delta, eps):
    avoid_zero_div = 1e-15
    norm2 = jnp.sum(delta**2, axis=0, keepdims=True)
    norm = jnp.sqrt(jnp.maximum(avoid_zero_div, norm2))
    # only decrease the norm, never increase
    delta = delta * eps / norm
    return delta
  dftz = l2_normalize(dftz, eps_iter)
  dftz = (dftz[0, :] + 1j * dftz[1, :]).reshape(grad.shape)
  adv_step = dft.getH() @ dftz
  return adv_step


def find_adversarial_samples(
    data, loss_f, dloss_dx, linearize_f, model_param0, normalize_f,
    config, rng_key):
  """Generates an adversarial example in the epsilon-ball centered at data.

  Args:
    data: An array of size dim x num, with input vectors as the initialization
      for the adversarial attack.
    loss_f: Loss function for attack.
    dloss_dx: The gradient function of the adversarial loss w.r.t. the input
    linearize_f: Linearize function used for the DFT attack.
    model_param0: Current model parameters.
    normalize_f: A function to normalize the weights of the model.
    config: Dictionary of hyperparameters.
    rng_key: JAX random number generator key.

  Returns:
    An array of size dim x num, one adversarial example per input data point.

  - Can gradient wrt parameters be zero but not the gradient wrt inputs? No.
  f = w*x
  df/dw = x (for a linear model)

  dL/dw = dL/df1 df1/dw = G x
  if dL/dw ~= 0, and x != 0 => G~=0

  dL/dx = dL/df1 df1/dx = G w
  G ~= 0 => dL/dx ~= 0
  """

  x0, y = data
  eps_iter, eps_tot = config.eps_iter, config.eps_tot
  norm_type = config.norm_type
  # - Normalize model params, prevents small gradients
  # For both linear and non-linear models the norm of the gradient can be
  # artificially very small. This might be less of an issue for linear models
  # and when we use sign(grad) to adv.
  # For separable data, the norm of the weights of the separating classifier
  # can increase to increase the confidence and decrease the gradient.
  # But adversarial examples within the epsilon ball still exist.
  # For linear models, divide one layer by the norm of the product of weights
  model_param = model_param0
  if config.pre_normalize:
    model_param = normalize_f(model_param0, norm_type)

  # - Reason for starting from a random delta instead of zero:
  # A non-linear model can have zero dL/dx at x0 but huge d^2L/dx^2 which means
  # a gradient-based attack fails if it always starts the optimization from x0
  # but succeed if starts from a point nearby with non-zero gradient.
  # It is not trivial what the distribution for the initial perturbation should
  # be. Uniform within the epsilon ball has its merits but then we have to use
  # different distributions for different norm-balls. We instead config for
  # sampling from a uniform distribution and clipping delta to lie within the
  # norm ball.
  delta = jax.random.normal(rng_key, x0.shape)
  if not config.rand_init:
    # Makes it harder to find the optimal adversarial direction for linear
    # models
    delta *= 0
  assert eps_iter <= eps_tot, 'eps_iter > eps_tot'
  delta = norm_projection(delta, norm_type, eps_iter)
  options = {'bound_step': True, 'step_size': 1000.}
  m_buffer = None
  for _ in range(config.niters):
    x_adv = x0 + delta
    # Untargeted attack: increases the loss for the correct label
    if config.step_dir == 'sign_grad':
      # Linf attack, FGSM and PGD attacks use only sign
      grad = dloss_dx(model_param, x_adv, y)
      adv_step = config.lr * jnp.sign(grad)
    elif config.step_dir == 'grad':
      # For L2 attack
      grad = dloss_dx(model_param, x_adv, y)
      adv_step = config.lr * grad
    elif config.step_dir == 'grad_max':
      grad = dloss_dx(model_param, x_adv, y)
      grad_max = grad * (jnp.abs(grad) == jnp.abs(grad).max())
      adv_step = config.lr * grad_max
    elif config.step_dir == 'dftinf_sd':
      # Linf attack, FGSM and PGD attacks use only sign
      grad = dloss_dx(model_param, x_adv, y)
      adv_step = sd_dir(grad, eps_iter)
      adv_step = config.lr * jnp.real(adv_step)
    # - Reason for having both a per-step epsilon and a total epsilon:
    # Needed for non-linear models. Increases attack success if dL/dx at x0 is
    # huge and f(x) is correct on the entire shell of the norm-ball but wrong
    # inside the norm ball.
    delta_i = norm_projection(adv_step, norm_type, eps_iter)
    delta = norm_projection(delta + delta_i, norm_type, eps_tot)
  delta = norm_projection(delta, norm_type, eps_tot)

  if config.post_normalize:
    delta_norm = jax.vmap(lambda x: norm_f(x, norm_type), (1,), 0)(delta)
    delta = delta/jnp.maximum(1e-12, delta_norm)*eps_tot
    delta = norm_projection(delta, norm_type, eps_tot)
  x_adv = x0 + delta
  return x_adv


def find_adversarial_samples_multi_attack(
    data, loss_f, dloss_dx, linearize_f, model_param0, normalize_f, config,
    rng_key):
  """Generates adversarial samples with multiple attacks and returns all samples."""
  config = copy.deepcopy(config)

  # Setting from the config
  rng_key, rng_subkey = jax.random.split(rng_key, 2)
  x_adv_multi = []
  x_adv_multi += [
      find_adversarial_samples(
        data, loss_f, dloss_dx, linearize_f, model_param0, normalize_f,
        config, rng_subkey)
  ]

  return x_adv_multi
