"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.

    Adapted from: https://github.com/nesl/nist_differential_privacy_synthetic_data_challenge/blob/master/tf_gan.py
    Adapted by: Meenatchi Sundaram Muthu Selva Annamalai
"""
import math
import numpy as np
from .data_utils import preprocess_data, postprocess_data
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import dill

from tensorflow.compat.v1.distributions import Bernoulli, Categorical

from .differential_privacy.dp_sgd.dp_optimizer import dp_optimizer
from .differential_privacy.dp_sgd.dp_optimizer import sanitizer
from .differential_privacy.privacy_accountant.tf import accountant

import sys; sys.path.insert(0, '../..')
from generative_models.base import GenerativeModel

#########################################################################
# Utility functions for building the WGAN model
#########################################################################
def lrelu(x, alpha=0.01):
    """ leaky relu activation function """
    return tf.nn.leaky_relu(x, alpha)


def fully_connected(input_node, output_dim, activation=tf.nn.relu, scope='None'):
    """ returns both the projection and output activation """
    with tf.variable_scope(scope or 'FC'):
        w = tf.get_variable('w', shape=[input_node.get_shape()[1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('b', shape=[output_dim],
                            initializer=tf.constant_initializer())
        tf.summary.histogram('w', w)
        tf.summary.histogram('b', b)
        z = tf.matmul(input_node, w) + b
        h = activation(z)
    return z, h


def critic_f(input_node, hidden_dim):
    """ Defines the critic model architecture """
    z1, h1 = fully_connected(input_node, hidden_dim, lrelu, scope='fc1')
    # z2, h2 = fully_connected(h1, hidden_dim, lrelu, scope='fc2')
    z3, _ = fully_connected(h1, 1, tf.identity, scope='fc3')
    return z3


def generator(input_node, hidden_dim, output_dim):
    """ Defines the generator model architecture """
    z1, h1 = fully_connected(input_node, hidden_dim, lrelu, scope='fc1')
    # z2, h2 = fully_connected(h1, hidden_dim, lrelu, scope='fc2')
    z3, _ = fully_connected(h1, output_dim, tf.identity, scope='fc3')
    return z3


def nist_data_format(output, metadata, columns_list, col_maps):
    """ Output layer format for generator data """
    with tf.name_scope('nist_format'):
        output_list = []
        cur_idx = 0
        for k in columns_list:
            v = col_maps[k]
            if isinstance(v, dict):
                if len(v) == 2:
                    output_list.append(tf.nn.sigmoid(
                        output[:, cur_idx:cur_idx+1]))
                    cur_idx += 1
                else:
                    output_list.append(
                        tf.nn.softmax(output[:, cur_idx: cur_idx+len(v)]))
                    cur_idx += len(v)
            elif v == 'int':
                output_list.append(output[:, cur_idx:cur_idx+1])
                cur_idx += 1
            elif v == 'int_v':
                output_list.append(tf.nn.sigmoid(output[:, cur_idx:cur_idx+1]))
                output_list.append(output[:, cur_idx+1:cur_idx+2])
                cur_idx += 2
            elif v == 'void':
                pass
            else:
                raise Exception('ivnalid mapping for col {}'.format(k))
        return tf.concat(output_list, axis=1)


def nist_sampling_format(output, metadata, columns_list, col_maps):
    """
    Output layer format for generator data plus performing random sampling
     from the output softmax and bernoulli distributions.
    """
    with tf.name_scope('nist_sampling_format'):
        output_list = []
        cur_idx = 0
        for k in columns_list:
            v = col_maps[k]
            if isinstance(v, dict):
                if len(v) == 2:
                    output_list.append(
                        tf.cast(
                            tf.expand_dims(
                                Bernoulli(logits=output[:, cur_idx]).sample(), axis=1), tf.float32)
                    )
                    cur_idx += 1
                else:
                    output_list.append(
                        tf.cast(tf.expand_dims(
                            Categorical(logits=output[:, cur_idx: cur_idx+len(v)]).sample(), axis=1), tf.float32))

                    cur_idx += len(v)
            elif v == 'int':
                output_list.append(
                    tf.nn.relu(output[:, cur_idx:cur_idx+1]))
                cur_idx += 1
            elif v == 'int_v':
                output_list.append(tf.nn.sigmoid(output[:, cur_idx:cur_idx+1]))
                output_list.append(tf.nn.relu(output[:, cur_idx+1:cur_idx+2]))
                cur_idx += 2
            elif v == 'void':
                pass
        return tf.concat(output_list, axis=1)


def sample_dataset(sess, sampling_output, columns_list, sampling_size,
                   metadata, col_maps, original_df_columns, feed_dict=None):
    """ Performs sampling to output synthetic data from the generative model. """
    sampling_result = []
    num_samples = 0
    while num_samples < sampling_size:
        batch_samples = sess.run(sampling_output, feed_dict=feed_dict)
        num_samples += batch_samples.shape[0]
        sampling_result.append(batch_samples)
    sampling_result = np.concatenate(sampling_result, axis=0)
    final_df = postprocess_data(
        sampling_result, metadata, col_maps, columns_list, greedy=False)
    final_df = pd.DataFrame(
        data=final_df, columns=original_df_columns, index=None)
    return final_df

class DPWGAN(GenerativeModel):
    def __init__(self, metadata, epsilon=1, delta=1e-5,
                 # Training parameters
                 batch_size=64, lr=1e-3, num_epochs=10, weight_clip=0.01,
                 # Model parameters
                 z_size=64, hidden_dim=1024,
                 # Privacy parameters
                 gradient_l2norm_bound=1.0,
                 # Custom auditing parameters
                 audit_world=None,
                 early_stop=True,
                 critic_iters=10,
                 sigma=None,
                ):
        tf.reset_default_graph()
        self.metadata = metadata
        self.epsilon = epsilon
        self.delta = delta
        if self.epsilon is None:
            self.epsilon = 0
            self.delta = None
            self.with_privacy = False
        else:
            self.with_privacy = True

        # Training parameters
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.weight_clip = weight_clip

        # Model parameters
        self.z_size = z_size
        self.hidden_dim = hidden_dim

        # Privacy parameters
        self.gradient_l2norm_bound = gradient_l2norm_bound

        # Custom
        self.saved_model = None
        self.audit_world = audit_world
        self.early_stop = early_stop
        self.critic_iters = critic_iters
        self.sigma = sigma
        self.disc_obs = 0
        self.gen_obs = 0
        self.disc_steps = 0
        self.gen_steps = 0
    
    def fit(self, df):
        # Reading input data
        original_df, input_data, metadata, self.col_maps, self.columns_list = preprocess_data(
            df, self.metadata, subsample=False)
        self.original_df_columns = original_df.columns

        input_data = input_data.values  # .astype(np.float32)
        data_dim = input_data.shape[1]
        format_fun = nist_data_format
        num_examples = input_data.shape[0]

        batch_size = self.batch_size
        num_batches = math.ceil(num_examples / batch_size)
        T = self.num_epochs * num_batches
        q = float(self.batch_size) / num_examples

        max_eps = self.epsilon

        if self.delta is None:
            max_delta = 1.0 / (num_examples**2)
        else:
            max_delta = self.delta

        eps_per_step = None  # unused for moments accountant
        delta_per_step = None  # unused for moments accountant
        if self.with_privacy:
            # Decide which accountanint_v to use
            use_moments_accountant = max_eps > 0.7
            if use_moments_accountant:
                if self.sigma is not None:
                    sigma = self.sigma
                elif max_eps > 5.0:
                    sigma = 1.0
                else:
                    sigma = 3.0
            else:
                sigma = None  # unused for amortized accountant
                # bound of eps_per_step from lemma 2.3 in https://arxiv.org/pdf/1405.7085v2.pdf
                eps_per_step = max_eps / (q * math.sqrt(2 * T * math.log(1/max_delta)))
                delta_per_step = max_delta / (T * q)

        with tf.name_scope('inputs'):
            x_holder = tf.placeholder(tf.float32, [None, data_dim], 'x')
            z_holder = tf.random_normal(shape=[self.batch_size, self.z_size],
                                        dtype=tf.float32, name='z')
            sampling_noise = tf.random_normal([self.batch_size, self.z_size],
                                            dtype=tf.float32, name='sample_z')
            eps_holder = tf.placeholder(tf.float32, [], 'eps')
            delta_holder = tf.placeholder(tf.float32, [], 'delta')

        with tf.variable_scope('generator') as scope:
            gen_output = generator(z_holder, self.hidden_dim, data_dim)
            gen_output = format_fun(gen_output, metadata, self.columns_list, self.col_maps)
            scope.reuse_variables()
            self.sampling_output = generator(sampling_noise, self.hidden_dim, data_dim)
            self.sampling_output = nist_sampling_format(
                self.sampling_output, metadata, self.columns_list, self.col_maps)

        with tf.variable_scope('critic') as scope:
            critic_real = critic_f(x_holder, self.hidden_dim)
            scope.reuse_variables()
            critic_fake = critic_f(gen_output, self.hidden_dim)

        with tf.name_scope('train'):
            global_step = tf.Variable(
                0, dtype=tf.int32, trainable=False, name='global_step')
            insert_canary = tf.Variable(
                0, dtype=tf.float32, trainable=False, name='insert_canary')
            loss_critic_real = - tf.reduce_mean(critic_real)
            loss_critic_fake = tf.reduce_mean(critic_fake)
            loss_critic = loss_critic_real + loss_critic_fake
            critic_vars = [x for x in tf.trainable_variables()
                        if x.name.startswith('critic')]
            if self.with_privacy:
                # assert FLAGS.sigma > 0, 'Sigma has to be positive when with_privacy=True'
                with tf.name_scope('privacy_accountant'):
                    if use_moments_accountant:
                        # Moments accountant introduced in (https://arxiv.org/abs/1607.00133)
                        # we use same implementation of
                        # https://github.com/tensorflow/models/blob/master/research/differential_privacy/privacy_accountant/tf/accountant.py
                        priv_accountant = accountant.GaussianMomentsAccountant(
                            num_examples)
                    else:
                        # AmortizedAccountant which tracks the privacy spending in the amortized way.
                        # It uses privacy amplication via sampling to compute the privacyspending for each
                        # batch and strong composition (specialized for Gaussian noise) for
                        # accumulate the privacy spending (http://arxiv.org/pdf/1405.7085v2.pdf)
                        # we use the implementation of
                        # https://github.com/tensorflow/models/blob/master/research/differential_privacy/privacy_accountant/tf/accountant.py
                        priv_accountant = accountant.AmortizedAccountant(
                            num_examples)

                    # per-example Gradient l_2 norm bound.
                    example_gradient_l2norm_bound = self.gradient_l2norm_bound / self.batch_size

                    # Gaussian sanitizer, will enforce differential privacy by clipping the gradient-per-example.
                    # Add gaussian noise, and sum the noisy gradients at each weight update step.
                    # It will also notify the privacy accountant to update the privacy spending.
                    gaussian_sanitizer = sanitizer.AmortizedGaussianSanitizer(
                        priv_accountant,
                        [example_gradient_l2norm_bound, True],
                        audit_world=self.audit_world,
                        insert_canary=insert_canary,
                        data_dim=data_dim)

                    critic_step = dp_optimizer.DPGradientDescentOptimizer(
                        self.lr,
                        # (eps, delta) unused parameters for the moments accountant which we are using
                        [eps_holder, delta_holder],
                        gaussian_sanitizer,
                        sigma=sigma,
                        batches_per_lot=1,
                        var_list=critic_vars).minimize((loss_critic_real, loss_critic_fake),
                                                    global_step=global_step, var_list=critic_vars)

            else:
                # This is used when we train without privacy.
                critic_step = tf.train.RMSPropOptimizer(self.lr).minimize(
                    loss_critic, var_list=critic_vars)

            # Weight clipping to ensure the critic function is K-Lipschitz as required
            # for WGAN training.
            clip_c = [tf.assign(var, tf.clip_by_value(
                var, -self.weight_clip, self.weight_clip)) for var in critic_vars]
            with tf.control_dependencies([critic_step]):
                critic_step = tf.tuple(clip_c)

            # Traing step of generator
            generator_vars = [x for x in tf.trainable_variables()
                            if x.name.startswith('generator')]
            loss_generator = -tf.reduce_mean(critic_fake)
            generator_step = tf.train.RMSPropOptimizer(self.lr).minimize(
                loss_generator, var_list=generator_vars)

            tb_c_op = tf.summary.scalar('critic_loss', loss_critic)
            tb_g_op = tf.summary.scalar('generator_loss', loss_generator)

        self.final_eps = 0.0
        self.final_delta = 0.0

        self.disc_obs = 0
        self.gen_obs = 0
        self.disc_steps = 0
        self.gen_steps = 0

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        abort_early = False  # Flag that will be changed to True if we exceed the privacy budget
        for e in range(self.num_epochs):
            if abort_early:
                break
            # One epoch is one full pass over the whole training data
            # Randomly shuffle the data at the beginning of each epoch
            rand_idxs = np.arange(num_examples)
            target_idx = num_examples - 1
            np.random.shuffle(rand_idxs)
            idx = 0
            abort_early = False
            while idx < num_batches and not abort_early:
                critic_i = 0
                while critic_i < self.critic_iters and idx < num_batches and not abort_early:
                    # Train the critic.
                    if self.with_privacy:
                        disc_param_flat_before = self.get_disc_param_flat()

                    batch_idxs = rand_idxs[idx*batch_size: (idx+1)*batch_size]
                    do_insert_canary = 1 if target_idx in batch_idxs else 0
                    batch_xs = input_data[batch_idxs, :]
                    feed_dict = {x_holder: batch_xs,
                                eps_holder: eps_per_step,
                                delta_holder: delta_per_step,
                                insert_canary: do_insert_canary
                                }
                    _, tb_c = self.sess.run(
                        [critic_step, tb_c_op], feed_dict=feed_dict)
                    critic_i += 1
                    idx += 1
                    self.disc_steps += 1

                    if self.with_privacy:
                        disc_param_flat_after = self.get_disc_param_flat()
                        grad_flat = disc_param_flat_after - disc_param_flat_before
                        # print(grad_flat[:5])
                        canary_grad_flat = np.zeros_like(grad_flat)
                        canary_grad_flat[0] = -example_gradient_l2norm_bound
                        canary_grad_flat[data_dim * 1024] = -example_gradient_l2norm_bound
                        canary_grad_flat[data_dim * 1024 + 1024] = -example_gradient_l2norm_bound
                        canary_grad_flat[data_dim * 1024 + 1024 + 1024] = -example_gradient_l2norm_bound
                        self.disc_obs += np.dot(grad_flat, canary_grad_flat)

                    if self.with_privacy:
                        if use_moments_accountant:
                            spent_eps_deltas = priv_accountant.get_privacy_spent(
                                self.sess, target_deltas=[max_delta])[0]
                        else:
                            spent_eps_deltas = priv_accountant.get_privacy_spent(
                                self.sess, target_eps=None)[0]

                        # Check whether we exceed the privacy budget
                        if (spent_eps_deltas.spent_delta > max_delta or
                                spent_eps_deltas.spent_eps > max_eps) and self.early_stop:
                            abort_early = True
                        else:
                            self.final_eps = spent_eps_deltas.spent_eps
                            self.final_delta = spent_eps_deltas.spent_delta
                    else:
                        # Training without privacy
                        spent_eps_deltas = accountant.EpsDelta(np.inf, 1)

                # Train the generator
                if not abort_early:
                    if self.with_privacy:
                        gen_param_flat_before = self.get_gen_param_flat()

                    # Check for abort_early because we stop updating the generator
                    #  once we exceeded privacy budget.
                    _, tb_g = self.sess.run([generator_step, tb_g_op])
                    self.gen_steps += 1

                    if self.with_privacy:
                        gen_param_flat_after = self.get_gen_param_flat()
                        grad_flat = gen_param_flat_after - gen_param_flat_before
                        canary_grad_flat = np.zeros_like(grad_flat)
                        canary_grad_flat[64 * 1024:65 * 1024] = 1
                        self.gen_obs += np.dot(grad_flat, canary_grad_flat)
    
    def sample(self, n_synth):
        """ Performs sampling to output synthetic data from the generative model. """
        # Sample synthetic data from the model after training is done.
        synth_df = sample_dataset(self.sess, self.sampling_output, self.columns_list, n_synth,
                    self.metadata, self.col_maps, self.original_df_columns, feed_dict=self.saved_model)[:n_synth]
        return synth_df

    def get_model_params(self):
        # save weights and biases of all layers
        suffixes = ['/w', '/b']
        saved_model = {f'{n.name}:0': self.sess.run(f'{n.name}:0')
                       for n in tf.get_default_graph().as_graph_def().node
                       if (any([n.name.endswith(suffix) for suffix in suffixes]) and '_' not in n.name)
                      }
        return saved_model

    def save_model(self, save_path):
        saved_model = self.get_model_params()
        dill.dump(saved_model, open(save_path, 'wb'))
    
    def _restore_model(self, dummy_df, save_path):
        self.num_epochs = 0
        self.saved_model = dill.load(open(save_path, 'rb'))
        self.fit(dummy_df)
    
    def get_logan_score(self, record):
        tf.get_variable_scope().reuse_variables()
        _, record, _, _, _ = preprocess_data(record.astype(str), self.metadata)
        record = record.to_numpy()[[0]]
        feed_dict = {
            'inputs/x:0': record
        }
        score = self.sess.run('critic/fc3/Identity:0', feed_dict=feed_dict)[0,0]
        return score

    ### Auditing Utility Functions ###
    def get_disc_param_flat(self):
        """Get all weights and biases from discriminator in a 1D flat array """
        return np.concatenate([param.flatten() for name, param in self.get_model_params().items() if 'critic' in name])

    def get_gen_param_flat(self):
        """Get all weights and biases from generator in a 1D flat array """
        return np.concatenate([param.flatten() for name, param in self.get_model_params().items() if 'generator' in name])
    ##################################