# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from .tf_util import Layers, lrelu, sn_sub_linear
from .regression import Regression

class _network(Layers):
    def __init__(self, name_scopes, layer_channels):
        assert(len(name_scopes) == 1)
        super().__init__(name_scopes)
        self.layer_channels = layer_channels

    def set_model(self, inputs, a, update_collection, is_training = True, reuse = False):

        h  = inputs
        singular_multiplication = 1.0
        # fully connect
        with tf.variable_scope(self.name_scopes[0], reuse = reuse):
            for i, s in enumerate(self.layer_channels):
                lin, singular = sn_sub_linear(i, h, s, a, len(self.layer_channels), update_collection)
                h = lrelu(lin)
                singular_multiplication *= singular
        return lin, singular_multiplication

class RegressionLN(Regression):
    
    update_collection = 'SN_UPDATE_OP'
    
    def __init__(self, input_dim):
        super().__init__(input_dim)
        self.network = _network(['NN'], self.network_layer)
        
    def set_model(self, lr, param, lipschitz):
        
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

        # -- variable ---
        self.a = tf.get_variable('a', [1], trainable = True)
        
        # -- set network ---
        self.v_s, self.singular = self.network.set_model(self.input, self.a, self.update_collection, is_training = True, reuse = False)
        self.td = self.target_val - self.v_s
        loss_data = tf.reduce_sum(tf.square(self.td), axis = 1)
        loss_singular = self.param*(tf.square((1/lipschitz)*self.singular-tf.pow(tf.exp(self.a), len(self.network_layer))))
        #loss_singular = self.param*(tf.square((1/lipschitz)*self.singular-tf.exp(tf.pow(self.a, len(self.network_layer)))))
        self.obj = tf.reduce_mean(loss_data) + loss_singular

        self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.obj,
                                                            var_list = self.network.get_variables().append(self.a))
        # -- update collections --
        self.sn_update_ops = tf.get_collection(self.update_collection)
        
        # -- for test --
        self.v_s_wo_train, self.singular_ = self.network.set_model(self.input, self.a, self.update_collection, is_training = False, reuse = True)


    def train(self, sess, input_data, target_val):
        feed_dict = {self.input: input_data,
                     self.target_val: target_val}
        obj, _ = sess.run([self.obj, self.train_op], feed_dict = feed_dict)
        
        for update_op in self.sn_update_ops:
            sess.run(update_op)
            
        return obj

    def get_value(self, sess, input_data):
        v_s = sess.run([self.v_s_wo_train], feed_dict = {self.input: input_data})
        return v_s