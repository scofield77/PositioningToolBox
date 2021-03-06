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
from numpy import linalg
import scipy as sp

# import numdifftools as nd

from numba import jit


class KalmanFilterBase:
    def __init__(self, state_dim):
        self.state_x = np.zeros(state_dim)
        self.state_prob = np.ones([state_dim, state_dim])

    def initial_filter(self, initial_state, initial_prob):
        self.state_x = initial_state
        self.state_prob = initial_prob

    @jit
    def state_transaction_function(self, input, input_cov, transaction_function=None):
        '''
        normal state transaction function.
        :param input:
        :param input_cov:
        :param transaction_function:
        :return:
        '''
        if transaction_function is None:
            print("Error: transaction function is None.")
        else:
            # self.A, self.B = transaction_function(self.state_x,input)
            # compute jacobian matrix
            # state_function = lambda s: transaction_function(s, input)
            def state_function(s):
                return transaction_function(s, input)

            self.A = nd.Jacobian(state_function)(self.state_x)

            # input_function = lambda i: transaction_function(self.state_x, i)
            def input_function(i):
                return transaction_function(self.state_x, i)

            self.B = nd.Jacobian(input_function)(input)

            self.state_x = transaction_function(self.state_x, input)

            self.state_prob = self.A.dot(self.state_prob).dot(np.transpose(self.A)) + \
                              (self.B.dot(input_cov).dot(np.transpose(self.B)))

            self.state_prob = (self.state_prob.copy() + np.transpose(self.state_prob.copy())) * 0.5

    @jit
    def state_transaction_function_imu_ukf(self,
                                       input: np.ndarray,
                                       input_cov: np.ndarray):
        Sigma_matrix = np.zeros([self.state_x.shape[0] + input.shape[0],
                                 self.state_x.shape[0] + input.shape[0]])

        Sigma_matrix[:self.state_x.shape[0],:self.state_x.shape[0]] = self.state_prob*1.0
        Sigma_matrix[self.state_x.shape[0],self.state_x.shape[0]:] = input_cov
        L = linalg.cholesky(Sigma_matrix)

        print('L',L)





    @jit
    def measurement_function(self, measurement, m_cov, H=None, measurement_function=None, state_update_function=None):
        '''
        normal measurement function
        :param measurement:
        :param m_cov:
        :param H:
        :param measurement_function:
        :param state_update_function:
        :return:
        '''

        if measurement_function is None:
            print("Error: measurement function is None.")

        else:

            if H is None:
                self.H = nd.Jacobian(measurement_function)(self.state_x)
            else:
                self.H = H

            self.y = measurement - measurement_function(self.state_x)

            self.K = (self.state_prob.dot(np.transpose(self.H))).dot(
                np.linalg.inv(self.H.dot(self.state_prob).dot(np.transpose(self.H)) + m_cov)
            )

            if state_update_function is None:
                self.state_x = self.state_x + self.K.dot(self.y)
            else:
                self.state_x = state_update_function(self.state_x, self.K.dot(self.y))
