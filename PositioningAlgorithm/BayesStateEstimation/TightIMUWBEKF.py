# -*- coding:utf-8 -*-
# carete by steve at  2018 / 05 / 10　下午9:19
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

from numba import jit, prange, njit
# from numba import float64,jitclass

from AlgorithmTool.ImuTools import *


class TightIMUWBEKF:
    def __init__(self, initial_prob, uwb_number, beacon_set, local_g=-9.8, time_interval=0.01):
        '''

        :param initial_prob:
        :param local_g:
        :param time_interval:
        '''
        self.rotation_q = np.zeros([4])
        self.state = np.zeros([15 + uwb_number])
        self.prob_state = initial_prob
        self.local_g = local_g
        self.time_interval = time_interval

        self.F = np.zeros([self.state.shape[0], self.state.shape[0]])
        self.G = np.zeros([self.state.shape[0], 6])

        self.beacon_set = beacon_set * 1.0

        self.uwb_eta_dict = dict()
        self.dx_dict = dict()

        # self.uwb_eta_list = list()

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

        for i in range(15, self.state.shape[0]):
            self.state[i] = np.linalg.norm(self.state[0:3] - self.beacon_set[i - 15, :])
            self.prob_state[i, 0:3] = (self.state[0:3] - self.beacon_set[i - 15, :]) / self.state[i]
            print(self.state[i])

        # state
        self.last_x = self.state * 1.0
        self.last_P = self.prob_state * 1.0

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

        last_v = self.state[3:6] * 1.0
        last_p = self.state[0:3] * 1.0

        self.state[0:3] = self.state[0:3] + self.state[3:6] * self.time_interval
        self.state[3:6] = self.state[3:6] + acc * self.time_interval
        self.state[6:9] = dcm2euler(q2dcm(self.rotation_q))

        self.state, mF = update_uwb_measurement(self.state, last_v, last_p, self.time_interval, self.beacon_set)

        f_t = Rb2t.dot(imu_data[0:3])

        St = np.asarray((
            0.0, -f_t[2], f_t[1],
            f_t[2], 0.0, -f_t[0],
            -f_t[1], f_t[0], 0.0
        )).reshape([3, 3])

        aux_build_F_G(self.F, self.G, St, q2dcm(self.rotation_q), self.time_interval)
        # print('G shape:', self.G.shape)

        self.F[15:, :] = mF * 1.0

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

    def measurement_uwb_direct(self, uwb_measurement, beacon_set, cov_m):
        '''
        Uwb measurement direct observed uwb measurement in system state.
        :param uwb_measurement:
        :param beacon_set:
        :param cov_m:
        :return:
        '''
        # print(uwb_measurement)
        H = np.zeros(shape=(uwb_measurement.shape[0], self.state.shape[0]))
        Rk = np.zeros(shape=(uwb_measurement.shape[0], uwb_measurement.shape[0]))
        for i in range(uwb_measurement.shape[0]):
            Rk[i, i] = cov_m
            if uwb_measurement[i] > 0.0 and beacon_set[i, 0] < 5000.0:
                H[i, i + 15] = 1.0
        # print('uwb measurement H:', H)

        y = uwb_measurement - H.dot(self.state)

        K = (self.prob_state.dot(np.transpose(H))).dot(
            np.linalg.inv((H.dot(self.prob_state).dot(np.transpose(H))) + Rk)
        )

        self.prob_state = (np.identity(self.prob_state.shape[0]) - K.dot(H)).dot(self.prob_state)

        self.prob_state = 0.5 * self.prob_state + 0.5 * np.transpose(self.prob_state)

        dx = K.dot(y)
        # print('dx:', dx)

        self.state[0:6] = self.state[0:6] + dx[0:6]
        #
        self.rotation_q = quaternion_left_update(self.rotation_q, dx[6:9], -1.0)
        # self.rotation_q = quaternion_right_update(self.rotation_q, dx[6:9], 1.0)
        #
        self.state[6:9] = dcm2euler(q2dcm(self.rotation_q))

        self.state[9:] = self.state[9:] + dx[9:]

    def measurement_uwb_robust(self,
                               uwb_measurement,
                               beacon_set,
                               cov_m,
                               ka_squard,
                               Td):
        '''
        robust EKF with cov condition.
        :param uwb_measurement:
        :param beacon_set:
        :param cov_m:
        :return:
        '''

        # @jit(cache=True)
        def get_vk_eta(measurement, state, cov, P, beacon_id, ka_squard, Td):
            if self.uwb_eta_dict.get(beacon_id) is None:
                self.uwb_eta_dict[beacon_id] = list()
                # print('initial list',beacon_id)
            # else:
            #     print(self.uwb_eta_dict[beacon_id],type(self.uwb_eta_dict))
            z = np.zeros(1)
            y = np.zeros(1)
            z[0] = measurement
            y[0] = state[beacon_id + 15]

            H = np.zeros(shape=(1, state.shape[0]))
            H[0, beacon_id + 15] = 1.0

            R_k = np.asarray((cov[0] * 1.0)).reshape(-1)
            v_k = z - y
            # print(v_k)
            eta_k = np.zeros(1)

            robust_loop_flag = True
            first_time = True
            while robust_loop_flag:
                robust_loop_flag = False
                # print('while ', 'beacon id:', beacon_id)

                P_v = (H.dot(P)).dot(np.transpose(H)) + R_k

                eta_k[0] = (np.transpose(v_k).dot(np.linalg.inv(P_v))).dot(v_k)

                if first_time:
                    self.uwb_eta_dict[beacon_id].append(eta_k[0])
                    first_time = False

                if eta_k[0] > ka_squard:
                    self.uwb_eta_dict[beacon_id][-1] = eta_k[0]

                    serial_length = 5
                    if len(self.uwb_eta_dict[beacon_id]) > serial_length:
                        lambda_k = np.std(np.asarray(self.uwb_eta_dict[beacon_id][-serial_length:]))
                        # print(lambda_k)
                        if lambda_k > Td:
                            robust_loop_flag = True
                            R_k[0] = eta_k[0] / ka_squard * R_k[0]
                            # print(R_k[0])
            # print('R_k:', R_k, 'eta_k:', eta_k[0])
            return R_k[0]

        # print(uwb_measurement)
        R_list = list()
        H = np.zeros(shape=(uwb_measurement.shape[0], self.state.shape[0]))
        Rk = np.zeros(shape=(uwb_measurement.shape[0], uwb_measurement.shape[0]))
        for i in range(uwb_measurement.shape[0]):
            Rk[i, i] = cov_m
            if uwb_measurement[i] > 0.0 and beacon_set[i, 0] < 5000.0:
                H[i, i + 15] = 1.0
                Rk[i, i] = get_vk_eta(uwb_measurement[i],
                                      self.state,
                                      np.asarray((cov_m)).reshape(-1),
                                      self.prob_state, i, ka_squard, Td)
                R_list.append(Rk[i, i])

                # if abs(uwb_measurement[i]-self.state[i+15]) > 2.0:
                #     H[i, i + 15] = 0.0
        if np.std(np.asarray(R_list)) > 2.0:
            for i in range(uwb_measurement.shape[0]):
                Rk[i, i] = Rk[i, i] / np.mean(np.asarray(R_list))

        # print('uwb measurement H:', H)

        y = uwb_measurement - H.dot(self.state)

        K = (self.prob_state.dot(np.transpose(H))).dot(
            np.linalg.inv((H.dot(self.prob_state).dot(np.transpose(H))) + Rk)
        )

        self.prob_state = (np.identity(self.prob_state.shape[0]) - K.dot(H)).dot(self.prob_state)

        self.prob_state = 0.5 * self.prob_state + 0.5 * np.transpose(self.prob_state)

        dx = K.dot(y)
        # print('dx:', dx)

        self.state[0:6] = self.state[0:6] + dx[0:6]
        #
        self.rotation_q = quaternion_left_update(self.rotation_q, dx[6:9], -1.0)
        # self.rotation_q = quaternion_right_update(self.rotation_q, dx[6:9], 1.0)
        #
        self.state[6:9] = dcm2euler(q2dcm(self.rotation_q))

        self.state[9:] = self.state[9:] + dx[9:]

    def measurement_uwb_ite_robust(self, uwb_measurement, beacon_set, cov_m, ka_squard=6.0):
        '''

        :param uwb_measurement:
        :param beacon_set:
        :param cov_m:
        :param ka_squard:
        :return:
        '''
        # print(uwb_measurement)
        H = np.zeros(shape=(uwb_measurement.shape[0], self.state.shape[0]))
        Rk = np.zeros(shape=(uwb_measurement.shape[0], uwb_measurement.shape[0]))
        for i in range(uwb_measurement.shape[0]):
            Rk[i, i] = cov_m
            if uwb_measurement[i] > 0.0 and beacon_set[i, 0] < 5000.0:
                H[i, i + 15] = 1.0
        # print('uwb measurement H:', H)

        y = uwb_measurement - H.dot(self.state)
        pminus = self.prob_state * 1.0
        pplus = pminus * 1.0
        xminus = self.state

        xplus = self.state
        xop = self.state * 0.0

        dx = np.zeros(self.state.shape[0])

        ite_counter = 0
        mask = np.zeros(uwb_measurement.shape[0])

        while np.linalg.norm(xplus - xop) > 0.01 and ite_counter < 30:
            ite_counter += 1
            xop = xplus * 1.0
            y = np.linalg.norm(xop[0:3] - beacon_set, axis=1)
            # y = rou(y)
            H = np.zeros(shape=(uwb_measurement.shape[0], self.state.shape[0]))
            H[:, 0:3] = (xop[0:3] - beacon_set) / y.reshape(-1, 1)  # * d_rou(y.reshape(-1, 1))

            v = uwb_measurement - y

            index = np.argsort(np.abs(v))
            break_flag = False

            for i in range(index.shape[0]):
                if mask[index[i]] < 100.01:
                    pv = (H[index[i], :].dot(pplus)).dot(np.transpose(H[index[i], :])) + Rk[index[i], index[i]]
                    gamma = v[index[i]] * v[index[i]] / pv
                    # print(pv, v[index[i]])
                    # ka_squard = 7.0

                    if gamma < ka_squard:  # or i < np.floor(index.shape[0] / 2):
                        # break_flag=True
                        mask[index[i]] = 1.0
                        # if abs(v[i]) > np.linalg.norm(v) / float(v.shape[0]):
                        #     mask[index[i]] = 0.5
                        # mask[index[i]] = ka_squard / gamma * 1.0
                        # Rk[index[i],index[i]]=cov_m[0]
                    else:
                        # print('corrected Rk')
                        # mask[index[i]] = (ka_squard / gamma * 1.0)**4.0
                        mask[index[i]] = 0.2  # ka_squard/gamma
                        Rk[index[i], index[i]] = gamma / ka_squard * Rk[index[i], index[i]]
                        # mask[index[i]] = 1.0 / gamma
                    i = index.shape[0] + 1
            mask = mask / np.sum(mask) * float(mask.shape[0])

            K = (pminus.dot(np.transpose(H))).dot(
                np.linalg.inv(H.dot(pminus.dot(np.transpose(H))) + Rk))
            kh = K.dot(H)
            pplus = (np.identity(kh.shape[0]) - kh).dot(pminus)
            # if tp_plus
            dx = K.dot((uwb_measurement - y - H.dot(xminus - xop)) * mask)
            xplus = xminus + dx

        # print(ite_counter)
        self.prob_state = (np.identity(self.prob_state.shape[0]) - K.dot(H)).dot(self.prob_state)

        self.prob_state = 0.5 * self.prob_state + 0.5 * np.transpose(self.prob_state)

        # dx = K.dot(y)
        # print('dx:', dx)

        self.state[0:6] = self.state[0:6] + dx[0:6]
        #
        self.rotation_q = quaternion_left_update(self.rotation_q, dx[6:9], -1.0)
        # self.rotation_q = quaternion_right_update(self.rotation_q, dx[6:9], 1.0)
        #
        self.state[6:9] = dcm2euler(q2dcm(self.rotation_q))

        self.state[9:] = self.state[9:] + dx[9:]


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


@njit(nopython=True)
def update_uwb_measurement(x,
                           v,
                           p,
                           time_interval,
                           beacon_set):
    offset_num = 15
    mF = np.zeros(shape=(x.shape[0] - offset_num, x.shape[0]))
    D = v * time_interval
    if np.linalg.norm(D) < 1e-3:
        return x, mF
    for i in prange(mF.shape[0]):
        ddotbp = D[0] * (beacon_set[i, 0] - p[0]) + D[1] * (beacon_set[i, 1] - p[1]) + D[2] * (
                beacon_set[i, 2] - p[2])
        last_bp = np.linalg.norm(beacon_set[i, :] - p)
        last_m = x[i + offset_num]
        m = math.sqrt(np.linalg.norm(D) + last_m * last_m - 2.0 * last_m * ddotbp / last_bp)
        x[i + offset_num] = m
        mF[i, i + offset_num] = last_m / m - ddotbp / last_bp / m
        # print(m-last_m)
        for j in range(3):
            mF[i, j] = 0.5 / m * (
                    -1.0 * (ddotbp * (beacon_set[i, j] - p[j]) / (last_bp ** 3.0)) - 1.0 * D[0] / last_bp)
            mF[i, j + 3] = time_interval * (0.5 / m * (
                    D[j] / np.linalg.norm(D) - 2.0 * last_m * (beacon_set[i, j] - p[j]) / last_bp))

    return x, mF
