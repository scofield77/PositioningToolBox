# -*- coding:utf-8 -*-
# carete by steve at  2018 / 04 / 23　下午9:18
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
# from numba import float64,jitclass

from AlgorithmTool.ImuTools import *


# spec=[
#     ('rotation_q',float64[:]),
#     ('state',float64[:]),
#     ('prob_state',float64[:,:]),
#     ('local_g',float64),
#     ('time_interval',float64),
#     ('F',float64[:,:]),
#     ('G',float64[:,:]),
#     ('acc',float64[:])
# ]

# @jitclass(spec)
class ImuEKFComplex:
    def __init__(self, initial_prob, local_g=-9.8, time_interval=0.01):
        self.rotation_q = np.zeros([4])
        self.state = np.zeros([15])
        self.prob_state = initial_prob
        self.local_g = local_g
        self.time_interval = time_interval

        self.F = np.zeros([15, 15])
        self.G = np.zeros([15, 6])

    def initial_state(self, imu_data: np.ndarray,
                      pos=np.asarray((0.0, 0.0, 0.0)),
                      ori: float = 0.0):
        '''
        Initial state based on given position and orientation(angle of z-axis)
        :param imu_data:
        :param pos:
        :param ori: float orientation
        :return:
        '''
        assert imu_data.shape[1] >= 6
        if np.linalg.norm(self.state) > 1e-19:
            self.state *= 0.0
        self.state[0:3] = pos

        self.state[6:9], self.rotation_q = get_initial_rotation(imu_data[:, 0:3], ori)
        print('q:', self.rotation_q)

        self.I = np.identity(3)

    # @jit(nopython=True)
    def state_transaction_function(self,
                                   imu_data,
                                   noise_matrix):
        '''
        State transaction function. 15 state, with bias of acc and gyr
        :param imu_data: acc(m/s^2), gyr(rad /s)
        :param noise_matrix: (noise matrix)
        :return:
        '''
        self.rotation_q = quaternion_right_update(self.rotation_q,
                                                  imu_data[3:6] + self.state[12:15],
                                                  self.time_interval)

        Rb2t = q2dcm(self.rotation_q)
        acc = Rb2t.dot(imu_data[0:3] + self.state[9:12]) + np.asarray((0.0, 0.0, self.local_g))  # + self.state[9:12])
        # print('acc:',acc)
        self.acc = acc

        self.state[0:3] = self.state[0:3] + self.state[3:6] * self.time_interval
        self.state[3:6] = self.state[3:6] + acc * self.time_interval

        self.state[6:9] = dcm2euler(q2dcm(self.rotation_q))

        f_t = Rb2t.dot(imu_data[0:3])

        St = np.asarray((
            0.0, -f_t[2], f_t[1],
            f_t[2], 0.0, -f_t[0],
            -f_t[1], f_t[0], 0.0
        )).reshape([3, 3])

        aux_build_F_G(self.F, self.G, St, q2dcm(self.rotation_q), self.time_interval)

        self.prob_state = (self.F.dot(self.prob_state)).dot(np.transpose(self.F)) + (self.G.dot(noise_matrix)).dot(
            np.transpose(self.G))

        self.prob_state = 0.5 * self.prob_state + 0.5 * self.prob_state.transpose()
        # print('F:\n', tF - self.F)
        # print('G:\n', tG - self.G)

    # @jit(cache=True)
    def measurement_function_zv(self, m, cov_matrix):
        '''
        Zero-velocity measurement.
        Suitable for ekf with more than 15 state model.
        :param m: actually is Vector3d(0,0,0)
        :param cov_matrix:
        :return:
        '''
        H = np.zeros([3, self.state.shape[0]])
        H[0:3, 3:6] = np.identity(3)

        K = (self.prob_state.dot(np.transpose(H))).dot(
            np.linalg.inv((H.dot(self.prob_state)).dot(np.transpose(H)) + cov_matrix)
        )

        before_p_norm = np.linalg.norm(self.prob_state)
        self.prob_state = (np.identity(self.prob_state.shape[0]) - K.dot(H)).dot(self.prob_state)

        self.prob_state = 0.5 * self.prob_state + 0.5 * np.transpose(self.prob_state)

        dx = K.dot(m - H.dot(self.state))

        self.state[0:6] = self.state[0:6] + dx[0:6]
        #
        self.rotation_q = quaternion_left_update(self.rotation_q, dx[6:9], -1.0)
        # self.rotation_q = quaternion_right_update(self.rotation_q, dx[6:9], 1.0)
        #
        self.state[6:9] = dcm2euler(q2dcm(self.rotation_q))

        self.state[9:] = self.state[9:] + dx[9:]

    def measurement_uwb(self, measurement, cov_m, beacon_pos):
        '''
        correct system state based on UWB measurements.
        :param measurement:
        :param cov_m:
        :param beacon_pos:
        :return:
        '''
        z = np.zeros(1)
        y = np.zeros(1)

        z = measurement
        y[0] = np.linalg.norm(self.state[0:3] - beacon_pos)

        self.H = np.zeros(shape=(1, self.state.shape[0]))
        self.H[0, 0:3] = (self.state[0:3] - beacon_pos).transpose() / y[0]

        self.K = (self.prob_state.dot(np.transpose(self.H))).dot(
            np.linalg.inv((self.H.dot(self.prob_state)).dot(np.transpose(self.H)) + cov_m)
        )

        dx = self.K.dot(z - y)

        self.state = self.state + dx

        self.rotation_q = quaternion_left_update(self.rotation_q, dx[6:9], -1.0)

        self.state[6:9] = dcm2euler(q2dcm(self.rotation_q))


@jit((float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64), nopython=True, parallel=True)
def aux_build_F_G(F, G, St, Rb2t, time_interval):
    '''
    Build up Jacbian matrix for compute probability update
    # O = np.diag((0.0, 0.0, 0.0))
    # I = np.diag((1.0, 1.0, 1.0))
    #
    # Fc = np.zeros_like(self.F)
    # Fc[0:3, 3:6] = self.I
    #
    # Fc[3:6, 6:9] = St
    # Fc[3:6, 9:12] = Rb2t
    #
    # Fc[6:9, 12:15] = -1.0 * Rb2t
    #
    # Gc = np.zeros_like(self.G)
    # Gc[3:6, 0:3] = Rb2t
    # Gc[6:9, 3:6] = -1.0 * Rb2t
    #
    # # self.F = np.identity(self.F.shape[0]) + Fc * self.time_interval
    #
    # # self.G = Gc * self.time_interval
    # tF = np.identity(self.F.shape[0]) + Fc * self.time_interval
    # tG = Gc * self.time_interval

    # self.F, self.G = aux_build_F_G(St, Rb2t, self.time_interval)
    # self.G = np.zeros_like(self.G)
    # self.F = np.identity(self.F.shape[0])
    :param F:
    :param G:
    :param St:
    :param Rb2t: rotation matrix represent
    :param time_interval:
    :return:
    '''
    # F = np.identity(15)
    # F = np.identity(15)
    # G = np.zeros(shape=(15, 6))

    # G = np.zeros(shape=(15, 6))
    for k in range(F.shape[0]):
        F[k, k] = 1.0

    for i in range(3):

        F[i, 3 + i] = time_interval
        for j in range(3):
            # F[i, 3 + j] = 1.0 * time_interval

            F[3 + i, 6 + j] = St[i, j] * time_interval
            F[3 + i, 9 + j] = Rb2t[i, j] * time_interval

            F[6 + i, 12 + j] = -1.0 * Rb2t[i, j] * time_interval

            G[3 + i, 0 + j] = Rb2t[i, j] * time_interval
            G[6 + i, 3 + j] = -1.0 * time_interval * Rb2t[i, j]
    # return F, G
