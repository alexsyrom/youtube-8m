# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")

class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}


class OneHiddenModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    net = slim.fully_connected(
        model_input, 2048, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    output = slim.fully_connected(
        net, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}


class OneHiddenSeparateModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    audio = slim.fully_connected(
            model_input[:, -128:], 64, activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty))
    video = slim.fully_connected(
            model_input[:, :-128], 512, activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty))
    paired = slim.fully_connected(
            model_input, 512, activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty))
    net = tf.concat([audio, video, paired], -1)
    output = slim.fully_connected(
        net, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}


class OneHiddenShortcutModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    net = slim.fully_connected(
        model_input, 512, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    net_concated = tf.concat([model_input, net], -1)
    output = slim.fully_connected(
        net_concated, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}


class TwoHiddenShortcutModel(models.BaseModel):

  def create_model(self, 
          model_input, 
          vocab_size, 
          l2_penalty=1e-8, 
          is_training=False,
          **unused_params):
    audio = model_input[:, -128:]
    audio = tf.nn.l2_normalize(audio, -1)
    video = model_input[:, :-128]
    video = tf.nn.l2_normalize(video, -1)

    model_input = tf.concat([video, audio], -1)
    net = slim.fully_connected(
        model_input, 512, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    net_list = [model_input, net]
    net_concated = tf.concat(net_list, -1)

    net = slim.fully_connected(
        net_concated, 256, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    net_list.append(net)
    net_concated = tf.concat(net_list, -1)
 
    output = slim.fully_connected(
        net_concated, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}


def gaussian_noise_layer(input_layer, std, training=False):
  if not training:
      return input_layer
  noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
  return input_layer + noise


def threshold_layer(input_layer, shape):
    shape = 1024 + 128
    layer = tf.layers.dense(
            input_layer,
            shape,
            activation=tf.nn.relu,
            trainable=False,
            kernel_initializer=tf.initializers.identity)
    return layer

_vertical_label = None

def get_vertical_label():
  if _vertical_label is not None:
    return _vertical_label


class THSModel(models.BaseModel):

  def create_model(self, 
          model_input, 
          num_frames,
          vocab_size, 
          labels,
          l2_penalty=1e-8, 
          is_training=False,
          trainable=False,
          compute_loss=False,
          augment=True,
          **unused_params):
    if augment:
      model_input = gaussian_noise_layer(
             model_input, 
             0.05, 
             training=is_training)

    # bn_input = tf.layers.batch_normalization(
    #        model_input,
    #        center=False,
    #        scale=False,
    #        training=is_training)
    print(is_training)
    print(trainable)
    if model_input.shape.ndims == 2:
      audio = model_input[:, -128:]
      video = model_input[:, :-128]
    else:
      audio = model_input[:, :, -128:]
      video = model_input[:, :, :-128]

    audio = tf.nn.l2_normalize(audio, -1)
    video = tf.nn.l2_normalize(video, -1)

    model_input_norm = tf.concat(
            [video, audio], 
            -1)
    model_input = self.cor_layer(
            model_input_norm, l2_penalty, is_training, trainable)

    wide = self.wide_layer(
            model_input, 
            model_input_norm, 
            l2_penalty,
            is_training,
            trainable)
    shortcut = self.shortcut_layer(model_input, l2_penalty, trainable)
    res = self.res_layer(model_input, l2_penalty, is_training, trainable)

    net_list = [wide, shortcut, res]
    net_concated = tf.concat(net_list, -1)

    logits = slim.fully_connected(
        net_concated, vocab_size, activation_fn=None,
        trainable=trainable,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    output = tf.nn.sigmoid(logits)

    if compute_loss:
        with tf.variable_scope("loss_xent"):
          loss = tf.losses.sigmoid_cross_entropy(
                  labels,
                  logits,
                  reduction=tf.losses.Reduction.NONE
          )
          loss = tf.reduce_mean(tf.reduce_sum(loss, 1))
        reg_loss = tf.constant(0.)
    else:
        loss = None
        reg_loss = None
    return {
        "logits": logits,
        "predictions": output,
        "loss": loss,
        "regularization_loss": reg_loss
    }

  def correct_input(self, model_input, l2_penalty, trainable):
    with tf.variable_scope("correct_input"):
      shape = 1024 + 128
      weight = 2 * slim.fully_connected(
          model_input, shape, activation_fn=tf.nn.sigmoid,
          trainable=trainable,
          weights_regularizer=slim.l2_regularizer(l2_penalty))
      result = weight * model_input
      return result 

  def wide_layer(self, in_layer, in_layer2, l2_penalty, is_training, trainable):
    with tf.variable_scope("wide_layer"):
      net_list = [in_layer2] 
      for weight in [-1, 1]:
        net = tf.nn.relu(weight * in_layer)
        net_list.append(net)
      for bias in [0.5,]:
        net = tf.nn.relu(in_layer - bias)
        net_list.append(net)

      net_list.append(net)
      net_concated = tf.concat(net_list, -1)

      # net = slim.fully_connected(
      #     net_concated, 1024, activation_fn=tf.nn.relu,
      #     weights_regularizer=slim.l2_regularizer(l2_penalty))
      return net_concated

  def shortcut_layer(self, in_layer, l2_penalty, trainable):
    with tf.variable_scope("shortcut_layer"):
      net = in_layer
      net_list = list()
      for i in range(3):
        net = slim.fully_connected(
            net, 1024, activation_fn=tf.nn.relu,
            trainable=trainable,
            weights_regularizer=slim.l2_regularizer(l2_penalty))
        #net = tf.layers.dropout(net, rate=0.1, training=is_training) 
        net_list.append(net)
        to_concat = net_list[-2:]
        to_concat.append(in_layer)
        net = tf.concat(to_concat, -1)

      net = slim.fully_connected(
          net, 1024, activation_fn=tf.nn.relu,
          trainable=trainable,
          weights_regularizer=slim.l2_regularizer(l2_penalty))
      return net
 
  def res_bn(self, net, training, l2_penalty, trainable):
    bn_layer = tf.layers.BatchNormalization(
           beta_regularizer=slim.l2_regularizer(l2_penalty),
           gamma_regularizer=slim.l2_regularizer(l2_penalty),
           trainable=trainable,
           center=True,
           scale=True)
    return bn_layer.apply(net, training=training)

  def res_block(
          self, 
          in_layer, 
          l2_penalty, 
          is_training, 
          shape,
          suf,
          trainable):
    net = in_layer
    with tf.variable_scope("res_block_" + suf):
      net = slim.fully_connected(
          net, shape, activation_fn=None,
          trainable=trainable,
          biases_initializer=None,
          weights_regularizer=slim.l2_regularizer(l2_penalty))
      net = self.res_bn(
              net, training=is_training,
              trainable=trainable,
              l2_penalty=l2_penalty)
      net = tf.nn.relu(net)

      net = slim.fully_connected(
          net, shape, activation_fn=None,
          trainable=trainable,
          biases_initializer=None,
          weights_regularizer=slim.l2_regularizer(l2_penalty))
      net = self.res_bn(
              net, training=is_training, 
              trainable=trainable,
              l2_penalty=l2_penalty)

      net = net + in_layer
      net = self.res_bn(
              net, training=is_training, 
              trainable=trainable,
              l2_penalty=l2_penalty)
      net = tf.nn.relu(net)
      return net

  def res_layer(self, in_layer, l2_penalty, is_training, trainable):
    with tf.variable_scope("res_layer"):
      shape = 1024 + 128
      net = in_layer
      for i in range(2):
        net = self.res_block(
                net, 
                l2_penalty, 
                is_training, 
                shape, 
                str(i),
                trainable)
      return net

  def cor_block(self, in_layer, l2_penalty, is_training, shape, trainable):
    inner = in_layer
    weight = 2 * slim.fully_connected(
        inner, shape, activation_fn=tf.nn.sigmoid,
        trainable=trainable,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    bias = 0.1 * slim.fully_connected(
        inner, shape, activation_fn=tf.nn.tanh,
        trainable=trainable,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    result = weight * in_layer + bias
    return result 

  def cor_layer(self, in_layer, l2_penalty, is_training, trainable):
    with tf.variable_scope("cor_layer"):
      shape = 1024 + 128
      net = in_layer
      for i in range(1):
        net = self.cor_block(net, l2_penalty, is_training, shape, trainable)
      net = tf.layers.dropout(net, rate=0.05, training=is_training) 
      return net


class SSModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    net = slim.fully_connected(
        model_input, 256, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    audio = slim.fully_connected(
            model_input[:, -128:], 32, activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty))
    video = slim.fully_connected(
            model_input[:, :-128], 256, activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty))
    net_list = [model_input, net, video, audio]
    net_concated = tf.concat(net_list, -1)

    net = slim.fully_connected(
        net_concated, 256, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    net_list.append(net)
    net_concated = tf.concat(net_list, -1)
 
    output = slim.fully_connected(
        net_concated, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}


class ManyHiddenShortcutModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    net = slim.fully_connected(
        model_input, 512, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    net_list = [model_input, net]
    net_concated = tf.concat(net_list, -1)

    net = slim.fully_connected(
        net_concated, 256, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    net_list.append(net)
    net_concated = tf.concat(net_list, -1)

    net = slim.fully_connected(
        net_concated, 128, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    net_list.append(net)
    net_concated = tf.concat(net_list, -1)
 
    net = slim.fully_connected(
        net_concated, 64, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    net_list.append(net)
    net_concated = tf.concat(net_list, -1)
 
    net = slim.fully_connected(
        net_concated, 32, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    net_list.append(net)
    net_concated = tf.concat(net_list, -1)
 
    net = slim.fully_connected(
        net_concated, 16, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    net_list.append(net)
    net_concated = tf.concat(net_list, -1)
 
    output = slim.fully_connected(
        net_concated, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}


class LabelProfileModel(models.BaseModel):

  def create_model(self, 
                   model_input,
                   vocab_size, 
                   l2_penalty=1e-8,
                   profile_size=100,
                   **unused_params):
    one = tf.constant([[1.]])
    label_profiles_flatten = tf.layers.dense(
            one, profile_size * vocab_size, 
            use_bias=False, activation=tf.nn.sigmoid, 
            kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_penalty)
        )
    label_profiles = tf.reshape(label_profiles_flatten, [vocab_size, profile_size])

    video_profiles = tf.layers.dense(
            model_input, profile_size,
            activation=tf.nn.sigmoid, 
            kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_penalty)
    )
    recovered_input = tf.layers.dense(
            video_profiles, model_input.shape[1].value,
            activation=tf.nn.sigmoid, 
            kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_penalty)
    )
    ae_loss = tf.losses.mean_squared_error(
        model_input,
        recovered_input,
        weights=1.0
    )

    dot_product = tf.tensordot(video_profiles, label_profiles, [[1], [1]])
    output = tf.nn.sigmoid(dot_product)
    return {
        "predictions": output,
        "regularization_loss": ae_loss
    }


class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}
