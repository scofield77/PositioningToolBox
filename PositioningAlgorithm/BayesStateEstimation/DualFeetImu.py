# -*- coding:utf-8 -*-
# carete by steve at  2018 / 10 / 25　9:29 PM
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


class DualFeetImu:
    def __init__(self, initial_prob, local_g=-9.8, time_interval=0.01):
        self.state = np.zeros([18])
        self.l_q = np.zeros([4])
        self.r_q = np.zeros([4])
        self.r_offset = 9
        self.prob_state = np.zeros([self.state.shape[0], self.state.shape[0]])
        self.prob_state[:self.r_offset, :self.r_offset] = 1.0 * initial_prob
        self.prob_state[-self.r_offset:, -self.r_offset:] = 1.0 * initial_prob

        self.local_g = local_g
        self.time_interval = time_interval

        self.F = np.zeros([18, 18])
        self.G = np.zeros([18, 12])

        self.uwb_eta_dict = dict()
        self.dx_dict = dict()

        self.error_counter = 0
        self.right_counter = 0

    def initial_state(self, l_imu_data,
                      r_imu_data,
                      l_pos=np.asarray((0.0, 0.0, 0.0)),
                      r_pos=np.asarray((0.0, 0.0, 0.0)),
                      l_ori=0.0,
                      r_ori=0.0):
        self.state[0:3] = l_pos * 1.0
        self.state[self.r_offset + 0:self.r_offset + 3] = r_pos * 1.0

        self.state[6:9], self.l_q = get_initial_rotation(l_imu_data[:, 0:3], l_ori)
        self.state[self.r_offset + 6:self.r_offset + 9], self.r_q = get_initial_rotation(r_imu_data[:, 0:3], r_ori)

        self.I = np.identity(3)

    def state_transaction_function(self, l_imu_data,
                                   r_imu_data,
                                   l_noise_matrix,
                                   r_noise_matrix):

        self.l_q = quaternion_right_update(self.l_q,
                                           l_imu_data[3:6],
                                           self.time_interval)

        self.r_q = quaternion_right_update(self.r_q,
                                           r_imu_data[3:6],
                                           self.time_interval)

        l_Rb2t = q2dcm(self.l_q)
        r_Rb2t = q2dcm(self.r_q)

        l_acc = l_Rb2t.dot(l_imu_data[0:3]) + np.asarray((0.0, 0.0, self.local_g))
        r_acc = r_Rb2t.dot(r_imu_data[0:3]) + np.asarray((0.0, 0.0, self.local_g))

        self.state[0:3] = self.state[0:3] + self.state[3:6] * self.time_interval
        self.state[3:6] = self.state[3:6] + l_acc * self.time_interval

        self.state[self.r_offset + 0:self.r_offset + 3] = self.state[self.r_offset + 0:self.r_offset + 3] + \
                                                          self.state[
                                                          self.r_offset + 3:self.r_offset + 6] * self.time_interval
        self.state[self.r_offset + 3:self.r_offset + 6] = self.state[self.r_offset + 3:self.r_offset + 6] + \
                                                          r_acc * self.time_interval

        self.state[6:9] = dcm2euler(q2dcm(self.l_q))
        self.state[self.r_offset + 6:self.r_offset + 9] = dcm2euler(q2dcm(self.r_q))

        lf_t = l_Rb2t.dot(l_imu_data[0:3])
        rf_t = r_Rb2t.dot(r_imu_data[0:3])

        lSt = np.asarray((
            0.0, -lf_t[2], lf_t[1],
            lf_t[2], 0.0, -lf_t[0],
            -lf_t[1], lf_t[0], 0.0
        )).reshape([3, 3])

        rSt = np.asarray((
            0.0, -rf_t[2], rf_t[1],
            rf_t[2], 0.0, -rf_t[0],
            -rf_t[1], rf_t[0], 0.0
        )).reshape([3, 3])

        aux_build_F_G_dual(self.F, self.G, lSt, rSt,
                           q2dcm(self.l_q), q2dcm(self.r_q),
                           self.time_interval)

        noise_matrix = np.ones([12, 12])
        noise_matrix[0:6, 0:6] = l_noise_matrix * 1.0
        noise_matrix[6:12, 6:12] = r_noise_matrix * 1.0

        self.prob_state = (self.F.dot(self.prob_state)).dot(np.transpose(self.F)) + \
                          (self.G.dot(noise_matrix)).dot(np.transpose(self.G))

        self.prob_state = 0.5 * self.prob_state + 0.5 * (self.prob_state.transpose())

    def measurement_zv(self,
                       left_zv_flag,
                       right_zv_flag,
                       max_distance=2.0):

        # correct velocity of dual feet speratelly.
        if left_zv_flag > 0.5:
            before_p = self.prob_state * 1.0
            # return
            H = np.zeros([3, 18])
            H[0:3, 3:6] = np.identity(3)

            m = np.zeros(3)
            cov_matrix = np.identity(3) * 0.0001

            K = (self.prob_state.dot(np.transpose(H))).dot(
                np.linalg.inv((H.dot(self.prob_state)).dot(np.transpose(H)) + cov_matrix)
            )

            before_p_norm = np.linalg.norm(self.prob_state)
            self.prob_state = (np.identity(self.prob_state.shape[0]) - K.dot(H)).dot(self.prob_state)

            self.prob_state[9:18, 9:18] = before_p[9:18, 9:18] * 1.0

            dx = K.dot(m - H.dot(self.state))

            # print('before:', self.state[3:6], self.state[12:15])
            self.state[0:9] = self.state[0:9] + dx[0:9]

            self.l_q = quaternion_left_update(self.l_q, dx[6:9], -1.0)
            # self.r_q = quaternion_left_update(self.r_q, dx[self.r_offset + 6:self.r_offset + 9], -1.0)

            self.state[6:9] = dcm2euler(q2dcm(self.l_q))
            self.state[6 + self.r_offset:9 + self.r_offset] = dcm2euler(q2dcm(self.r_q))
            # print('after:', self.state[3:6], self.state[12:15])

        if right_zv_flag > 0.5:
            before_p = self.prob_state * 1.0
            H = np.zeros([3, 18])
            H[0:3, 3 + self.r_offset:6 + self.r_offset] = np.identity(3)

            m = np.zeros(3)
            cov_matrix = np.identity(3) * 0.0001

            K = (self.prob_state.dot(np.transpose(H))).dot(
                np.linalg.inv((H.dot(self.prob_state)).dot(np.transpose(H)) + cov_matrix)
            )

            self.prob_state = (np.identity(self.prob_state.shape[0]) - K.dot(H)).dot(self.prob_state)

            # self.prob_state = 0.5 * self.prob_state + 0.5 * np.transpose(self.prob_state)

            self.prob_state[0:9, 0:9] = before_p[0:9, 0:9] * 1.0

            dx = K.dot(m - H.dot(self.state))

            # print('before:', self.state[3:6], self.state[12:15])
            self.state[9:18] = self.state[9:18] + dx[9:18]

            # self.l_q = quaternion_left_update(self.l_q, dx[6:9], -1.0)
            self.r_q = quaternion_left_update(self.r_q, dx[self.r_offset + 6:self.r_offset + 9], -1.0)

            self.state[6:9] = dcm2euler(q2dcm(self.l_q))
            self.state[6 + self.r_offset:9 + self.r_offset] = dcm2euler(q2dcm(self.r_q))
            # print('after:', self.state[3:6], self.state[12:15])

        if np.linalg.norm(self.state[0:3] - self.state[9:12]) > max_distance and \
                (left_zv_flag > 0.5 or \
                 right_zv_flag > 0.5):
            self.distance_constrain(max_distance)

    def distance_constrain(self, eta):
        '''

        :param eta:
        :return:
        '''

        W = np.linalg.inv(self.prob_state)
        W = (W + np.transpose(W)) * 0.5

        L = np.zeros([3, 18])
        L[0:3, 0:3] = np.identity(3)
        L[0:3, 9:12] = np.identity(3) * -1.0
        print(self.right_counter, self.error_counter)
        try:
            # G = np.linalg.cholesky(W)
            G, r = np.linalg.qr(W)
            # q, G = np.linalg.qr(W)

            U, S, V = np.linalg.svd(L.dot(np.linalg.inv(G)))


        except np.linalg.LinAlgError:
            print('LinAlg Error')
            self.error_counter = self.error_counter + 1
            return

        self.right_counter = self.right_counter + 1

        e = np.transpose(V).dot(G).dot(self.state)

        lam = 0.0
        delta = 1e100
        ctr = 0

        while abs(delta) > 1e-4 and ctr < 125:
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
            z = self.state * 1.0
        else:
            z = np.linalg.inv(W + (lam * np.transpose(L).dot(L))).dot(W.dot(self.state))

        self.state = z * 1.0

        self.l_q = quaternion_left_update(self.l_q, z[6:9], -1.0)
        self.r_q = quaternion_left_update(self.r_q, z[12:15], -1.0)
        self.state[6:9] = dcm2euler(q2dcm(self.l_q))
        self.state[12:15] = dcm2euler(q2dcm(self.r_q))

        z = (np.transpose(L).dot(L)).dot(z)

        A = np.linalg.inv(W + lam * (np.transpose(L).dot(L)))

        alpha = (np.transpose(z).dot(A)).dot(z)
        # Jp = (np.identity(total_P.shape[0]) - np.linalg.inv(alpha * (A).dot(z.dot(np.transpose(z))))).dot(A).dot(W)
        Jp = ((np.identity(self.prob_state.shape[0]) - (1.0 / alpha * (A).dot(z.dot(np.transpose(z))))).dot(A)).dot(W)
        self.prob_state = (Jp.dot(self.prob_state)).dot(np.transpose(Jp))


@jit((float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64), nopython=True,
     parallel=True)
def aux_build_F_G_dual(F, G, lSt, rSt, lRb2t, rRb2t, time_interval):
    for k in range(F.shape[0]):
        F[k, k] = 1.0

    for i in range(3):
        F[i, 3 + i] = time_interval
        F[i + 9, 3 + i + 9] = time_interval

        for j in range(3):
            F[3 + i, 6 + j] = lSt[i, j] * time_interval
            F[3 + i + 9, 6 + j + 9] = rSt[i, j] * time_interval

            G[3 + i, 0 + j] = lRb2t[i, j] * time_interval
            G[6 + i, 3 + j] = -1.0 * lRb2t[i, j] * time_interval

            G[3 + i + 9, 0 + j + 6] = rRb2t[i, j] * time_interval
            G[6 + i + 9, 3 + j + 6] = -1.0 * rRb2t[i, j] * time_interval
