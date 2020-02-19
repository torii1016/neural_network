# -*- coding:utf-8 -*-

import os
import sys
import math

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from .model.regression import Regression
from .model.regression_l1 import RegressionL1
from .model.regression_l2 import RegressionL2
from .model.regression_sn import RegressionSN
from .model.regression_ln import RegressionLN
from .model.regression_ln_autotuning import RegressionLNAutoTuning

"""
def show_train_data(input_data, input_label_ideal, input_label, save_name):
    ax = plt.subplot2grid((1,1), (0,0))
    ax.plot(input_data, input_label_ideal, color="g")
    for i in range(len(input_data)):
        ax.scatter(np.array(input_data[i]), np.array([input_label[i]]) , color="r")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid()
    plt.savefig(save_name)
"""

class Train(object):

    def __init__(self, train_data, train_label, mode, lr_rate, hyper_param, lipschitz, episode_num, save_name):
        self.train_data = train_data
        self.train_label = train_label

        if mode=="normal":
            self.nn = Regression(1)
            self.nn.set_model(lr_rate)
        elif mode=="l1":
            self.nn = RegressionL1(1)
            self.nn.set_model(lr_rate, hyper_param)
        elif mode=="l2":
            self.nn = RegressionL2(1)
            self.nn.set_model(lr_rate, hyper_param)
        elif mode=="sn":
            self.nn = RegressionSN(1)
            self.nn.set_model(lr_rate, lipschitz)
        elif mode=="ln":
            self.nn = RegressionLN(1)
            self.nn.set_model(lr_rate, hyper_param, lipschitz)
        elif mode=="ln-auto":
            self.nn = RegressionLNAutoTuning(1)
            self.nn.set_model(lr_rate, hyper_param, lipschitz)

        self.episode_num = episode_num
        self.save_name = save_name
        self.step = self.episode_num/10

    
    def __call__(self):
        # -- begin training --
        with tf.Session() as sess:
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            sess.run(init)
            try:
                for i in range(1, self.episode_num+1):
                    if i%self.step==0: 
                        bar = ('=' * int(i/self.step) ) + (' ' * (int(self.episode_num/self.step-int(i/self.step)))) 
                        print('\r### begin episode[{0}] {1}% ({2}/{3})'.format(bar, int((i/self.step)/(self.episode_num/self.step)*100), i, self.episode_num), end='') 
                    elif i==1:
                        bar = ('=' * 0 ) + (' ' * (int(self.episode_num/self.step))) 
                        print('\r### begin episode[{0}] {1}% ({2}/{3})'.format(bar, int((i/self.step)/(self.episode_num/self.step)*100), 0, self.episode_num), end='') 

                    l, a, obj_awb, obj_l, loss_data, loss_singular, loss_singular_, loss_data_l, loss_singular_l, loss_singular_l_ = self.nn.train(sess, np.array(self.train_data).reshape(-1,1), np.array(self.train_label).reshape(-1,1))
                    #print(l, obj_awb, loss_data, loss_singular)
                    print("awb   l: {},  loss: {}, loss_awb: {}, loss_l: {}, loss_l_: {}".format(format(float(l[0]), ".3f"), format(float(obj_awb[0]), ".3f"), format(float(loss_data), ".3f"), format(float(loss_singular), ".3f"), format(float(loss_singular_), ".3f")))
                    print("l     l: {},  loss: {}, loss_awb: {}, loss_l: {}, loss_l_: {}".format(format(float(l[0]), ".3f"), format(float(obj_l[0]), ".3f"), format(float(loss_data_l), ".3f"), format(float(loss_singular_l), ".3f"), format(float(loss_singular_l_), ".3f")))

                    """
                    if i%self.step==0: 
                        tmp_output = []
                        for j in range(len(self.train_data)):
                            obj, _ = self.nn.get_value(sess, [[self.train_data[j]]])
                            tmp_output.append(obj[0][0])

                        show_train_data(self.train_data, tmp_output, self.train_label, "result_" + str(i/self.step) + ".png")
                    """

                #print("\nobj: {}".format(obj))
                # save model
                #saver.save(sess, self.save_name)

                output = []
                for j in range(len(self.train_data)):
                    obj = self.nn.get_value(sess, [[self.train_data[j]]])
                    output.append(obj[0][0])
                    print(obj[0][0])
        
                return output
        
            except KeyboardInterrupt:
                pass