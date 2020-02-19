# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from .tf_util import Layers, lrelu, linear_with_weight_l1
from .regression import Regression

class _network(Layers):
    def __init__(self, name_scopes, layer_channels):
        assert(len(name_scopes) == 1)
        super().__init__(name_scopes)
        self.layer_channels = layer_channels

    def set_model(self, inputs, is_training = True, reuse = False):

        h  = inputs
        all_weight_sum = 0.0
        # fully connect
        with tf.variable_scope(self.name_scopes[0], reuse = reuse):
            for i, s in enumerate(self.layer_channels):
                lin, weight_sum = linear_with_weight_l1(i, h, s)
                h = lrelu(lin)
                all_weight_sum += weight_sum
        return lin, all_weight_sum

class RegressionL1(Regression):
    
    def __init__(self, input_dim):
        super().__init__(input_dim)
        self.network = _network(['NN'], self.network_layer)
        
    def set_model(self, lr, param):
        
        self.lr = tf.Variable(
            name = "learning_rate",
            initial_value = lr,
            trainable = False)

        self.param = tf.Variable(
            name = "hyper_parameter",
            initial_value = param,
            trainable = False)

        self.lr_op = tf.assign(self.lr, 0.95 * self.lr)
        
        # -- place holder ---
        self.input = tf.placeholder(tf.float32, [None, self.input_dim])
        self.target_val = tf.placeholder(tf.float32, [None, 1])

        # -- set network ---
        self.v_s, self.weight = self.network.set_model(self.input, is_training = True, reuse = False)
        self.td = self.target_val - self.v_s
        loss_data = tf.reduce_sum(tf.square(self.td), axis = 1)
        loss_l1 = self.param*self.weight
        self.obj = tf.reduce_mean(loss_data) + loss_l1

        self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.obj,
                                                            var_list = self.network.get_variables())
        
        # -- for test --
        self.v_s_wo_train, self.weight_ = self.network.set_model(self.input, is_training = False, reuse = True)