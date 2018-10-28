# -*- coding:utf-8 -*-
# carete by steve at  2018 / 09 / 25　9:35 PM
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
import matplotlib.pyplot as plt

from numba import jit

from AlgorithmTool.ImuTools import *
from PositioningAlgorithm.BayesStateEstimation.ImuEKF import *
from PositioningAlgorithm.BayesStateEstimation.DualFeetImu import *

class DualImuEKFComplexCombine:
    def __init__(self, initial_prob, local_g=-9.8, time_interval=0.01):
        '''

        :param initial_prob:
        :param local_g:
        :param time_interval:
        '''
        self.l_ekf = ImuEKFComplex(initial_prob, local_g, time_interval)
        self.r_ekf = ImuEKFComplex(initial_prob, local_g, time_interval)

        self.whole_P = np.zeros([self.l_ekf.prob_state.shape[0] + self.r_ekf.prob_state.shape[0],
                                 self.l_ekf.prob_state.shape[1] + self.r_ekf.prob_state.shape[1]])

        first_index = self.l_ekf.state.shape[0]

        self.whole_P[0:first_index, 0:first_index] = 1.0 * self.l_ekf.prob_state
        self.whole_P[first_index:, first_index:] = 1.0 * self.r_ekf.prob_state

    def initial_state(self, left_imu_data: np.ndarray,
                      right_imu_data: np.ndarray,
                      left_pos=np.asarray((0.0, 0.0, 0.0)),
                      right_pos=np.asarray((0.0, 0.0, 0.0)),
                      ori: float = 0.0):
        '''

        :param left_imu_data:
        :param right_imu_data:
        :param left_pos:
        :param right_pos:
        :param ori:
        :param mag:
        :return:
        '''
        # assert left_imu_data.shape[0] > 10 and left_imu_data.shape[1] > 6
        # assert right_imu_data.shape[0] > 10 and right_imu_data.shape[1] > 6

        # if np.linalg.norm(self.state)

        self.l_ekf.initial_state(left_imu_data, left_pos, ori)
        self.r_ekf.initial_state(right_imu_data, right_pos, ori)  # +25.0/180.0*np.pi)

        print(dcm2euler(q2dcm(self.l_ekf.rotation_q)))
        print(dcm2euler(q2dcm(self.r_ekf.rotation_q)))
        self.distance_counter = 0

    def state_transaction_function(self,
                                   left_imu_data,
                                   right_imu_data,
                                   left_noise_matrix,
                                   right_noise_matrix):
        '''

        :param left_imu_data:
        :param right_imu_data:
        :param left_noise_matrix:
        :param right_noise_matrix:
        :return:
        '''
        self.l_ekf.state_transaction_function(left_imu_data, left_noise_matrix)
        self.r_ekf.state_transaction_function(right_imu_data, right_noise_matrix)

    def zv_update(self, left_flag, right_flag, max_distance=2.5):

        if left_flag > 0.5:
            self.l_ekf.measurement_function_zv(np.asarray((0.0, 0.0, 0.0)),
                                               np.asarray((0.0001, 0.0001, 0.0001)))

        if right_flag > 0.5:
            self.r_ekf.measurement_function_zv(np.asarray((0.0, 0.0, 0.0)),
                                               np.asarray((0.0001, 0.0001, 0.0001)))

        # print('----', np.linalg.norm(self.l_ekf.state[0:3] - self.r_ekf.state[0:3]), 'of', max_distance)
        # if (left_flag > 0.5 and right_flag > 0.5) and \
        self.distance_counter = self.distance_counter - 1
        if np.linalg.norm(
                self.l_ekf.state[0:3] - self.r_ekf.state[0:3]) > max_distance and self.distance_counter < 1:

            print(np.linalg.norm(self.l_ekf.state[0:3] - self.r_ekf.state[0:3]), 'of', max_distance)
            # print('before left state:', self.l_ekf.state)
            # print('before left state:', self.r_ekf.state)
            try:
                self.distance_constrain(max_distance)
                self.distance_counter = 5
            except np.linalg.LinAlgError:
                return
            # else:
            #     return

    def distance_constrain(self, eta):
        '''
        distance constrained function.

        :param eta: range constraint
        :return:
        '''

        total_x = np.zeros(self.l_ekf.state.shape[0] + self.r_ekf.state.shape[0])
        total_P = np.zeros([total_x.shape[0], total_x.shape[0]])

        total_x[:self.l_ekf.state.shape[0]] = self.l_ekf.state
        total_x[-self.r_ekf.state.shape[0]:] = self.r_ekf.state

        total_P[:self.l_ekf.state.shape[0], :self.l_ekf.state.shape[0]] = self.l_ekf.prob_state * 1.0
        total_P[-self.r_ekf.state.shape[0]:, -self.r_ekf.state.shape[0]:] = self.r_ekf.prob_state * 1.0

        W = np.linalg.inv(total_P)
        W = (np.transpose(W) + (W)) * 0.5

        L = np.zeros([3, total_x.shape[0]])
        for i in range(3):
            L[i, i] = 1.0
            L[i, i + self.l_ekf.state.shape[0]] = -1.0

        ### Projection
        try:

            G = np.linalg.cholesky(W)

            U, S, V = np.linalg.svd(L.dot(np.linalg.inv(G)))


        except np.linalg.LinAlgError:
            print('lina')
            return
        # else:
        #     print('unknown error')
        #     return

        e = np.transpose(V).dot(G).dot(total_x)

        # Newton search to find lambda
        lam = 0.0
        delta = 1e100
        ctr = 0

        while abs(delta) > 1e-4 and ctr < 25:
            g = e[0] * e[0] * S[0] * S[0] / ((1 + lam * S[0] * S[0]) ** 2.0) + \
                e[1] * e[1] * S[1] * S[1] / ((1 + lam * S[1] * S[1]) ** 2.0) + \
                e[2] * e[2] * S[2] * S[2] / ((1 + lam * S[2] * S[2]) ** 2.0) - \
                eta * eta

            # dg = -2.0 * (e[0] ** 2.0 * S[0, 0] ** 4.0 / ((1 + lam * S[0, 0] ** 2.0) ** 3.0) +
            #              e[1] ** 2 * S[1, 1] ** 4.0 / ((1 + lam * S[1, 1] ** 2.0) ** 3.0) +
            #              e[2] ** 2.0 * S[2, 2] ** 4.0 / ((1 + lam * S[2, 2] ** 2.0) ** 3.0)
            #              )

            dg = -2.0 * (e[0] * e[0] * S[0] ** 4.0 / ((1 + lam * S[0] * S[0]) ** 3.0) +
                         e[1] * e[1] * S[1] ** 4.0 / ((1 + lam * S[1] * S[1]) ** 3.0) +
                         e[2] * e[2] * S[2] ** 4.0 / ((1 + lam * S[2] * S[2]) ** 3.0)
                         )
            delta = g / dg
            lam = lam - delta
            ctr = ctr + 1
        print('ctr', ctr)

        if (lam < 0):
            print("ERROR : lam must bigger than zero.")
            z = total_x
        else:
            z = np.linalg.inv(W + (lam * np.transpose(L).dot(L))).dot(W.dot(total_x))

        # Correct data
        print('Angle corrected')
        self.l_ekf.rotation_q = quaternion_left_update(self.l_ekf.rotation_q, z[6:9], -1.0)
        self.l_ekf.state = z[:self.l_ekf.state.shape[0]]
        # self.l_ekf.state[0:3] = z[0:3]
        self.l_ekf.state[6:9] = dcm2euler(q2dcm(self.l_ekf.rotation_q))

        self.r_ekf.rotation_q = quaternion_left_update(self.r_ekf.rotation_q,
                                                       z[self.l_ekf.state.shape[0] + 6:self.l_ekf.state.shape[0] + 9],
                                                       -1.0)
        self.r_ekf.state = z[-self.r_ekf.state.shape[0]:]
        # self.r_ekf.state[0:3] = z[self.l_ekf.state.shape[0]:self.l_ekf.state.shape[0] + 3]
        self.r_ekf.state[6:9] = dcm2euler(q2dcm(self.r_ekf.rotation_q))
        print('optimized distance:', np.linalg.norm(self.l_ekf.state[0:3] - self.r_ekf.state[0:3]))

        z = (np.transpose(L).dot(L)).dot(z)

        A = np.linalg.inv(np.linalg.inv(total_P) + lam * (np.transpose(L).dot(L)))

        alpha = (np.transpose(z).dot(A)).dot(z)
        # Jp = (np.identity(total_P.shape[0]) - np.linalg.inv(alpha * (A).dot(z.dot(np.transpose(z))))).dot(A).dot(W)
        Jp = ((np.identity(total_P.shape[0]) - (1.0 / alpha * (A).dot(z.dot(np.transpose(z))))).dot(A)).dot(W)
        total_P = (Jp.dot(total_P)).dot(np.transpose(Jp))

        self.l_ekf.prob_state = total_P[:self.l_ekf.state.shape[0], :self.l_ekf.state.shape[0]]
        self.r_ekf.prob_state = total_P[-self.r_ekf.state.shape[0]:, -self.r_ekf.state.shape[0]:]
        # print('total x:', total_x)
        # print('after z:', z)




