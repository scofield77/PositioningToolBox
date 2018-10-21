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


class DualImuEKFComplex:
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
        self.r_ekf.initial_state(right_imu_data, right_pos, ori)

        print(dcm2euler(q2dcm(self.l_ekf.rotation_q)))
        print(dcm2euler(q2dcm(self.r_ekf.rotation_q)))

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
                                               np.asarray((0.001, 0.001, 0.001)))

        if right_flag > 0.5:
            self.r_ekf.measurement_function_zv(np.asarray((0.0, 0.0, 0.0)),
                                               np.asarray((0.001, 0.001, 0.001)))

        # if (left_flag > 0.5 and right_flag > 0.5) and\
        # if        np.linalg.norm(
        #         self.l_ekf.state[0:3] - self.r_ekf.state[0:3]) > max_distance:
        #     print(np.linalg.norm(self.l_ekf.state[0:3] - self.r_ekf.state[0:3]), 'of', max_distance)
        #     print(self.l_ekf.state)
        #     print(self.r_ekf.state)
        #     self.distance_constrain(max_distance)

    def distance_constrain(self, eta):
        '''

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
        W = (np.transpose(W )+(W)) * 0.5

        L = np.zeros([3, total_x.shape[0]])
        for i in range(3):
            L[i, i] = 1.0
            L[i, i + self.l_ekf.state.shape[0]] = -1.0

        ### Projection

        # G = np.linalg.cholesky(W)
        G = sp.linalg.cholesky(W, lower=True)

        U, S, V = np.linalg.svd(L.dot(np.linalg.inv(G)))

        e = np.transpose(V).dot(G).dot(total_x)

        # Newton search to find lambda
        lam = 0.0
        delta = 1e100
        ctr = 0.0

        while abs(delta) > 1e-4 and ctr < 25:
            g = e[0] * e[0] * S[0] * S[0] / ((1 + lam * S[0] * S[0]) ** 2) + \
                e[1] * e[1] * S[1] * S[1] / ((1 + lam * S[1] * S[1]) ** 2) + \
                e[2] * e[2] * S[2] * S[2] / ((1 + lam * S[2] * S[2]) ** 2) - \
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

        if (lam < 0):
            print("ERROR : lam must bigger than zero.")
            z = total_x
        else:
            z = np.linalg.inv(W + (np.transpose(lam * L).dot(L))).dot(total_P.dot(total_x))

        # Correct data
        print('Angle corrected')
        # self.l_ekf.rotation_q = quaternion_left_update(self.l_ekf.rotation_q, z[6:9], -1.0)
        self.l_ekf.state = z[:self.l_ekf.state.shape[0]]
        self.l_ekf.state[6:9] = dcm2euler(q2dcm(self.l_ekf.rotation_q))

        # self.r_ekf.rotation_q = quaternion_left_update(self.r_ekf.rotation_q,
        #                                                 z[self.l_ekf.state.shape[0] + 6:self.l_ekf.state.shape[0] + 9],
        #                                                 -1.0)
        self.r_ekf.state = z[-self.r_ekf.state.shape[0]:]
        self.r_ekf.state[6:9] = dcm2euler(q2dcm(self.l_ekf.rotation_q))
        print('optimized distance:',np.linalg.norm(self.l_ekf.state[0:3]-self.r_ekf.state[0:3]))

        z = (np.transpose(L).dot(L)).dot(z)

        A = np.linalg.inv(W + lam * (np.transpose(L).dot(L)))

        alpha = (np.transpose(z).dot(A)).dot(z)
        Jp = (np.identity(total_P.shape[0]) - np.linalg.inv(alpha * (A).dot(z.dot(np.transpose(z))))).dot(A).dot(W)
        total_P = (Jp.dot(total_P)).dot(np.transpose(Jp))

        self.l_ekf.prob_state = total_P[:self.l_ekf.state.shape[0], :self.l_ekf.state.shape[0]]
        self.r_ekf.prob_state = total_P[-self.r_ekf.state.shape[0]:, -self.r_ekf.state.shape[0]:]


if __name__ == '__main__':

    print('dual feet test')

    import math
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib

    # from transforms3d import euler, quaternions

    from PositioningAlgorithm.BayesStateEstimation.KalmanFIlterBase import *

    from scipy.optimize import minimize

    from AlgorithmTool.ImuTools import *

    from PositioningAlgorithm.BayesStateEstimation.ImuEKF import *
    # from gr import pygr

    # from AlgorithmTool
    import time

    import mkl

    mkl.set_num_threads(6)
    # print(np.show_config())
    # print(mk)
    matplotlib.use('Qt5Agg')
    # matplotlib.rcParams['toolbar'] = 'toolmanager'
    start_time = time.time()
    # dir_name = '/home/steve/Data/FusingLocationData/0017/'
    # dir_name = '/home/steve/Data/NewFusingLocationData/0039/'
    # dir_name = 'C:/Data/NewFusingLocationData/0039/'
    dir_name = '/home/steve/Data/PDR/0012/'
    # dir_name = 'D:\\NewFusingLocationData\\0035\\'

    left_imu_data = np.loadtxt(dir_name + 'LEFT_FOOT.data', delimiter=',')
    # imu_data = np.loadtxt(dir_name + 'RIGHT_FOOT.data', delimiter=',')
    left_imu_data = left_imu_data[:, 1:]
    left_imu_data[:, 1:4] = left_imu_data[:, 1:4] * 9.81
    left_imu_data[:, 4:7] = left_imu_data[:, 4:7] * (np.pi / 180.0)

    right_imu_data = np.loadtxt(dir_name + 'RIGHT_FOOT.data', delimiter=',')
    right_imu_data = right_imu_data[:, 1:]
    right_imu_data[:, 1:4] = right_imu_data[:, 1:4] * 9.81
    right_imu_data[:, 4:7] = right_imu_data[:, 4:7] * (np.pi / 180.0)

    # initial_state = get_initial_state(imu_data[:40, 1:4], np.asarray((0, 0, 0)), 0.0, 9)

    left_trace = np.zeros([left_imu_data.shape[0], 3])
    vel = np.zeros([left_imu_data.shape[0], 3])
    ang = np.zeros([left_imu_data.shape[0], 3])
    ba = np.zeros([left_imu_data.shape[0], 3])
    bg = np.zeros([left_imu_data.shape[0], 3])

    right_trace = np.zeros([right_imu_data.shape[0], 3])

    dual_left_trace = np.zeros([right_imu_data.shape[0], 3])
    dual_right_trace = np.zeros([right_imu_data.shape[0], 3])

    iner_acc = np.zeros([left_imu_data.shape[0], 3])

    left_zv_state = np.zeros([left_imu_data.shape[0]])

    # kf = KalmanFilterBase(9)
    # kf.state_x = initial_state
    # kf.state_prob = np.diag((0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001))

    average_time_interval = (left_imu_data[-1, 0] - left_imu_data[0, 0]) / float(left_imu_data.shape[0])
    print('average time interval ', average_time_interval)

    lkf = ImuEKFComplex(np.diag((
        0.001, 0.001, 0.001,
        0.001, 0.001, 0.001,
        0.001 * np.pi / 180.0, 0.001 * np.pi / 180.0, 0.0001 * np.pi / 180.0,
        0.0001,
        0.0001,
        0.0001,
        0.0001 * np.pi / 180.0,
        0.0001 * np.pi / 180.0,
        0.0001 * np.pi / 180.0
    )),
        local_g=-9.81, time_interval=average_time_interval)

    lkf.initial_state(left_imu_data[:50, 1:7], mag=left_imu_data[0, 7:10])

    left_zv_state = GLRT_Detector(left_imu_data[:, 1:7], sigma_a=1.0,
                                  sigma_g=1.0 * np.pi / 180.0,
                                  gamma=250,
                                  gravity=9.8,
                                  time_Window_size=5)

    rkf = ImuEKFComplex(np.diag((
        0.001, 0.001, 0.001,
        0.001, 0.001, 0.001,
        0.001 * np.pi / 180.0, 0.001 * np.pi / 180.0, 0.0001 * np.pi / 180.0,
        0.0001,
        0.0001,
        0.0001,
        0.0001 * np.pi / 180.0,
        0.0001 * np.pi / 180.0,
        0.0001 * np.pi / 180.0
    )),
        local_g=-9.81, time_interval=average_time_interval)

    rkf.initial_state(left_imu_data[:50, 1:7], mag=left_imu_data[0, 7:10])

    right_zv_state = GLRT_Detector(right_imu_data[:, 1:7], sigma_a=1.0,
                                   sigma_g=1.0 * np.pi / 180.0,
                                   gamma=250,
                                   gravity=9.8,
                                   time_Window_size=5)

    dkf = DualImuEKFComplex(np.diag((
        0.001, 0.001, 0.001,
        0.001, 0.001, 0.001,
        0.001 * np.pi / 180.0, 0.001 * np.pi / 180.0, 0.0001 * np.pi / 180.0,
        0.0001,
        0.0001,
        0.0001,
        0.0001 * np.pi / 180.0,
        0.0001 * np.pi / 180.0,
        0.0001 * np.pi / 180.0
    )),
        local_g=-9.81, time_interval=average_time_interval)

    dkf.initial_state(left_imu_data[:50, 1:7], right_imu_data[:50, 1:7])

    for i in range(left_imu_data.shape[0]):
        # print('i:',i)

        lkf.state_transaction_function(left_imu_data[i, 1:7],
                                       np.diag((0.01, 0.01, 0.01,
                                                0.01 * np.pi / 180.0,
                                                0.01 * np.pi / 180.0,
                                                0.01 * np.pi / 180.0))
                                       )
        if (i > 5) and (i < left_imu_data.shape[0] - 5):
            # print('i:',i)
            # zv_state[i] = z_tester.GLRT_Detector(imu_data[i - 4:i + 4, 1:8])
            if left_zv_state[i] > 0.5:
                lkf.measurement_function_zv(np.asarray((0, 0, 0)),
                                            np.diag((0.0001, 0.0001, 0.0001)))
        # kf.measurement_function_mag(imu_data[i,7:10],np.identity(3)*0.05)

        # print(kf.state_x)
        # print( i /)
        left_trace[i, :] = lkf.state[0:3]
        vel[i, :] = lkf.state[3:6]
        ang[i, :] = lkf.state[6:9]
        ba[i, :] = lkf.state[9:12]
        bg[i, :] = lkf.state[12:15]
        rate = i / left_imu_data.shape[0]
        iner_acc[i, :] = lkf.acc

        # print('finished:', rate * 100.0, "% ", i, imu_data.shape[0])

        if i < right_imu_data.shape[0]:
            rkf.state_transaction_function(right_imu_data[i, 1:7],
                                           np.diag((0.01, 0.01, 0.01,
                                                    0.01 * np.pi / 180.0,
                                                    0.01 * np.pi / 180.0,
                                                    0.01 * np.pi / 180.0)))

            dkf.state_transaction_function(left_imu_data[i, 1:7],
                                           right_imu_data[i, 1:7],
                                           np.diag((0.01, 0.01, 0.01,
                                                    0.01 * np.pi / 180.0,
                                                    0.01 * np.pi / 180.0,
                                                    0.01 * np.pi / 180.0)),
                                           np.diag((0.01, 0.01, 0.01,
                                                    0.01 * np.pi / 180.0,
                                                    0.01 * np.pi / 180.0,
                                                    0.01 * np.pi / 180.0)))

            if i > 5 and i < right_imu_data.shape[0] - 5:
                if right_zv_state[i] > 0.5:
                    rkf.measurement_function_zv(np.asarray((0.0, 0.0, 0.0)),
                                                np.diag((0.0001, 0.0001, 0.0001)))
                dkf.zv_update(left_zv_state[i], right_zv_state[i])

        right_trace[i, :] = rkf.state[0:3]

        dual_left_trace[i, :] = dkf.l_ekf.state[0:3]
        dual_right_trace[i, :] = dkf.r_ekf.state[0:3]

    end_time = time.time()
    print('totally time:', end_time - start_time, 'data time:', left_imu_data[-1, 0] - left_imu_data[0, 0])


    def aux_plot(data: np.ndarray, name: str):
        plt.figure()
        plt.title(name)
        plt.plot(left_zv_state * data.max() * 1.1, 'r-', label='zv state')
        for i in range(data.shape[1]):
            plt.plot(data[:, i], label=str(i))
        plt.grid()
        plt.legend()


    # #
    # aux_plot(left_imu_data[:, 1:4], 'acc')
    # aux_plot(left_imu_data[:, 4:7], 'gyr')
    # aux_plot(left_imu_data[:, 7:10], 'mag')
    # aux_plot(left_trace, 'trace')
    # aux_plot(vel, 'vel')
    # aux_plot(ang, 'ang')
    # aux_plot(ba, 'ba')
    # aux_plot(bg, 'bg')

    # aux_plot(iner_acc, 'inner acc')
    #
    # plt.figure()
    # plt.plot(left_trace[:, 0], left_trace[:, 1], '-+')
    # plt.grid()

    # plt.figure()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(left_trace[:, 0], left_trace[:, 1], left_trace[:, 2], '-+', label='left trace')
    ax.plot(right_trace[:, 0], right_trace[:, 1], right_trace[:, 2], '-+', label='right trace')
    ax.plot(dual_left_trace[:, 0], dual_left_trace[:, 1], dual_left_trace[:, 2], '-+', label='dual left trace')
    ax.plot(dual_right_trace[:, 0], dual_right_trace[:, 1], dual_right_trace[:, 2], '-+', label='dual right trace')
    ax.grid()
    ax.legend()
    plt.show()
