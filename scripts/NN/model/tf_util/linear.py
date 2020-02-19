#! -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from .variable_util import get_const_variable, get_rand_variable, get_dim
from .sn import spectral_norm, spectral_norm_substitute

def linear(name, inputs, out_dim):
    in_dim = get_dim(inputs)
    w = get_rand_variable(name, [in_dim, out_dim], 1/np.sqrt(in_dim))
    b = get_const_variable(name, [out_dim], 0.0)
    return tf.matmul(inputs, w) + b

def linear_with_weight_l1(name, inputs, out_dim):
    in_dim = get_dim(inputs)
    w = get_rand_variable(name, [in_dim, out_dim], 1/np.sqrt(in_dim))
    b = get_const_variable(name, [out_dim], 0.0)
    return tf.matmul(inputs, w) + b, tf.reduce_sum(tf.abs(w))

def linear_with_weight_l2(name, inputs, out_dim):
    in_dim = get_dim(inputs)
    w = get_rand_variable(name, [in_dim, out_dim], 1/np.sqrt(in_dim))
    b = get_const_variable(name, [out_dim], 0.0)
    return tf.matmul(inputs, w) + b, tf.reduce_sum(tf.square(w))

def sn_linear(name, inputs, out_dim, lipschitz, num_layers, update_collection):
    in_dim = get_dim(inputs)
    w = get_rand_variable(name, [in_dim, out_dim], 1/np.sqrt(in_dim))
    b = get_const_variable(name, [out_dim], 0.0)
    W_shape = w.shape.as_list()
    u = tf.get_variable("u_{}".format(name), [1, W_shape[-1]], trainable=False)
    output = spectral_norm(w, lipschitz, num_layers, u=u, update_collection=update_collection)
    return tf.matmul(inputs, output) + b

def sn_sub_linear(name, inputs, out_dim, a, num_layers, update_collection):
    in_dim = get_dim(inputs)
    w = get_rand_variable(name, [in_dim, out_dim], 1/np.sqrt(in_dim))
    b = get_const_variable(name, [out_dim], 0.0)
    W_shape = w.shape.as_list()
    u = tf.get_variable("u_{}".format(name), [1, W_shape[-1]], trainable=False)
    output, singular = spectral_norm_substitute(w, a, num_layers, u=u, update_collection=update_collection)
    return tf.matmul(inputs, output) + b, singular