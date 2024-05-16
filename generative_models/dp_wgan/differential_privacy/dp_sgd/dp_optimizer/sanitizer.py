# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Defines Sanitizer class for sanitizing tensors.

A sanitizer first limits the sensitivity of a tensor and then adds noise
to the tensor. The parameters are determined by the privacy_spending and the
other parameters. It also uses an accountant to keep track of the privacy
spending.
"""
from __future__ import division

import collections

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from generative_models.dp_wgan.differential_privacy.dp_sgd.dp_optimizer import utils


ClipOption = collections.namedtuple("ClipOption",
                                    ["l2norm_bound", "clip"])


class AmortizedGaussianSanitizer(object):
  """Sanitizer with Gaussian noise and amoritzed privacy spending accounting.

  This sanitizes a tensor by first clipping the tensor, summing the tensor
  and then adding appropriate amount of noise. It also uses an amortized
  accountant to keep track of privacy spending.
  """

  def __init__(self, accountant, default_option, audit_world=None, insert_canary=None, data_dim=None):
    """Construct an AmortizedGaussianSanitizer.

    Args:
      accountant: the privacy accountant. Expect an amortized one.
      default_option: the default ClipOptoin.
    """

    self._accountant = accountant
    self._default_option = default_option
    self._options = {}
    self.audit_world = audit_world
    self.insert_canary = insert_canary
    self.data_dim = data_dim

  def set_option(self, tensor_name, option):
    """Set options for an individual tensor.

    Args:
      tensor_name: the name of the tensor.
      option: clip option.
    """

    self._options[tensor_name] = option

  def sanitize(self, x, eps_delta, sigma=None,
               option=ClipOption(None, None), tensor_name=None,
               num_examples=None, add_noise=True):
    """Sanitize the given tensor.

    This santize a given tensor by first applying l2 norm clipping and then
    adding Gaussian noise. It calls the privacy accountant for updating the
    privacy spending.

    Args:
      x: the tensor to sanitize.
      eps_delta: a pair of eps, delta for (eps,delta)-DP. Use it to
        compute sigma if sigma is None.
      sigma: if sigma is not None, use sigma.
      option: a ClipOption which, if supplied, used for
        clipping and adding noise.
      tensor_name: the name of the tensor.
      num_examples: if None, use the number of "rows" of x.
      add_noise: if True, then add noise, else just clip.
    Returns:
      a pair of sanitized tensor and the operation to accumulate privacy
      spending.
    """
    l2norm_bound, clip = option
    with tf.name_scope('original'):
        tf.summary.histogram(x.name, tf.norm(x))
    if l2norm_bound is None:
      l2norm_bound, clip = self._default_option
      if ((tensor_name is not None) and
          (tensor_name in self._options)):
        l2norm_bound, clip = self._options[tensor_name]

    # insert canary gradient when auditing `in` world (target datapoint is in)
    # and gradient corresponds to discriminator (add_noise = True)
    x_shape = tf.shape(x)
    if self.audit_world == 'in' and add_noise:
      if tensor_name == 'critic/fc1/w':
        canary_grad = np.zeros((self.data_dim, 1024))
        canary_grad[0, 0] = l2norm_bound
      elif tensor_name == 'critic/fc1/b':
        canary_grad = np.zeros(1024)
        canary_grad[0] = l2norm_bound
      elif tensor_name == 'critic/fc3/w':
        canary_grad = np.zeros((1024, 1))
        canary_grad[0, 0] = l2norm_bound
      elif tensor_name == 'critic/fc3/b':
        canary_grad = np.zeros(1)
        canary_grad[0] = l2norm_bound
      canary_grad = tf.constant(canary_grad, dtype=tf.float32)

      # create canary gradient
      batch_size = tf.slice(x_shape, [0], [1])
      x_flat = tf.reshape(x, tf.concat(axis=0, values=[batch_size, [-1]]))
      canary_grad = tf.add(tf.multiply(x_flat[0], tf.constant(0, dtype=tf.float32)), tf.reshape(canary_grad, tf.shape(x_flat[0])))

      # if insert_canary, add canary gradient, else add original gradient
      # i.e. actual gradient = (1 - insert_canary) * original_gradient + insert_canary * canary_gradient
      neg_insert_canary = tf.multiply(tf.constant(-1, dtype=tf.float32), self.insert_canary)
      canary_grad = tf.add(tf.multiply(x_flat[-1], tf.add(tf.constant(1, dtype=tf.float32), neg_insert_canary)), tf.multiply(canary_grad, self.insert_canary))

      x_flat = tf.concat(axis=0, values=[x_flat[:-1], [canary_grad]])
      x = tf.reshape(x_flat, x_shape)

    if sigma is None:
      # pylint: disable=unpacking-non-sequence
      eps, delta = eps_delta
      with tf.control_dependencies(
          [tf.Assert(tf.greater(eps, 0),
                     ["eps needs to be greater than 0"]),
           tf.Assert(tf.greater(delta, 0),
                     ["delta needs to be greater than 0"])]):
        # The following formula is taken from
        #   Dwork and Roth, The Algorithmic Foundations of Differential
        #   Privacy, Appendix A.
        #   http://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
        sigma = tf.sqrt(2.0 * tf.log(1.25 / delta)) / eps

    if clip:
      x = utils.BatchClipByL2norm(x, l2norm_bound)
    with tf.name_scope('clipped'):
      tf.summary.histogram(x.name, tf.norm(x))

    if add_noise:
      if num_examples is None:
        num_examples = tf.slice(tf.shape(x), [0], [1])
      privacy_accum_op = self._accountant.accumulate_privacy_spending(
          eps_delta, sigma, num_examples)
      with tf.control_dependencies([privacy_accum_op]):
        saned_x = utils.AddGaussianNoise(tf.reduce_sum(x, 0),
                                         sigma * l2norm_bound)
    else:
      saned_x = tf.reduce_sum(x, 0)

    with tf.name_scope('sanitized'):
        tf.summary.histogram(x.name, tf.norm(saned_x))
    return saned_x
