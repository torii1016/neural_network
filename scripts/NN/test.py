# -*- coding:utf-8 -*-

import os
import sys
import math

import numpy as np
import tensorflow as tf

from .model.regression import Regression

class Train(object):

    def __init__(self, train_data, episode_num, save_name):
        self.train_data = train_data
        #self.train_label = train_label

        self.nn = Regression(1)
        self.nn.set_model(0.0001)

        self.episode_num = episode_num
        self.save_name = save_name
        
    
    def __call__(self):
        # -- begin training --
        with tf.Session() as sess:
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            sess.run(init)
            try:
                for i in range(self.episode_num):
                    print("### begin episode {}".format(i))

                    for j in range(len(self.train_data)):
                        obj = self.nn.train(sess, [[j]], [[self.train_data[j]]])

                # save model
                saver.save(sess, self.save_name)
        
                return sess
        
            except KeyboardInterrupt:
                pass