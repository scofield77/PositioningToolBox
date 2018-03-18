# -*- coding:utf-8 -*-
# Created by steve @ 18-3-18 下午6:53
'''
                   _ooOoo_ 
                  o8888888o 
                  88" . "88 
                  (| -_- |) 
                  O\  =  /O 
               ____/`---'\____ 
             .'  \\|     |//  `. 
            /  \\|||  :  |||//  \ 
           /  _||||| -:- |||||-  \ 
           |   | \\\  -  /// |   | 
           | \_|  ''\---/''  |   | 
           \  .-\__  `-`  ___/-. / 
         ___`. .'  /--.--\  `. . __ 
      ."" '<  `.___\_<|>_/___.'  >'"". 
     | | :  `- \`.;`\ _ /`;.`/ - ` : | | 
     \  \ `-.   \_ __\ /__ _/   .-` /  / 
======`-.____`-.___\_____/___.-`____.-'====== 
                   `=---=' 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
         佛祖保佑       永无BUG 
'''

import numpy as np
import scipy as sp

import numdifftools as nd


class KalmanFilterBase:
    def __init__(self, state_dim):
        self.state_x = np.zeros(state_dim)
        self.state_prob = np.ones([state_dim, state_dim])

    def initial_filter(self, initial_state, initial_prob):
        self.state_x = initial_state
        self.state_prob = initial_prob



    def state_transaction_function(self, input, input_cov, transaction_function=None):
        if transaction_function is None:
            print("Error: transaction function is None.")
        else:
            # self.A, self.B = transaction_function(self.state_x,input)
            # compute jacobian matrix
            # state_function = lambda s: transaction_function(s, input)
            def state_function(s):
                return transaction_function(s,input)
            self.A = nd.Jacobian(state_function)(self.state_x)
            # input_function = lambda i: transaction_function(self.state_x, i)
            def input_function(i):
                return transaction_function(self.state_x,i)
            self.B = nd.Jacobian(input_function)(input)

            self.state_x = transaction_function(self.state_x, input)

            self.state_prob = self.A.dot(self.state_prob).dot(np.transpose(self.A)) + \
                              (self.B.dot(input_cov).dot(np.transpose(self.B)))

            self.state_prob = (self.state_prob.copy() + np.transpose(self.state_prob.copy())) * 0.5


    def measurement_function(self, measurement, m_cov, measurement_function=None, state_update_function=None):
        if measurement_function is None:
            print("Error: measurement function is None.")

        else:
            self.H = nd.Jacobian(measurement_function)(self.state_x)

            self.y = measurement - measurement_function(self.state_x)

            self.K = (self.state_prob.dot(np.transpose(self.H))).dot(
                np.linalg.inv(self.H.dot(self.state_prob).dot(np.transpose(self.H)) + m_cov)
            )

            if state_update_function is None:
                self.state_x = self.state_x + self.K.dot(self.y)
            else:
                self.state_x = state_update_function(self.state_x,self.K.dot(self.y))

