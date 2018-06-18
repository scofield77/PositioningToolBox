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
        '''

        :param initial_prob:
        :param local_g:
        :param time_interval:
        '''
        self.rotation_q = np.zeros([4])
        self.state = np.zeros([15])
        self.prob_state = initial_prob
        self.local_g = local_g
        self.time_interval = time_interval

        self.F = np.zeros([15, 15])
        self.G = np.zeros([15, 6])

        self.uwb_eta_dict = dict()
        self.dx_dict = dict()

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
        # print('G shape:', self.G.shape)

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

    def measurement_function_z_axis(self, m, cov_matrix):
        '''
        z-axis constraint.
        Suitable for ekf with more than 15 state model.
        :param m: actually is Vector3d(0,0,0)
        :param cov_matrix:
        :return:
        '''
        H = np.zeros([1, self.state.shape[0]])
        H[0, 2] = 1.0

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

    def iter_measurement_function_uwb(self, m, cov_matrix):
        xop = self.state
        xk = xop * 1.0
        while True:
            H = np.zeros([3, xop.shape[0]])

            H[0:3, 3:6] = np.identity(3)

            K = (self.prob_state.dot(np.transpose(H))).dot(
                np.linalg.inv((H.dot(self.prob_state)).dot(np.transpose(H)) + cov_matrix)
            )

            # xk = xk + K.dot(m-H.dot(xop)-)

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

        z[0] = measurement
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

        kh = self.K.dot(self.H)
        self.prob_state = (np.identity(kh.shape[0]) - kh).dot(self.prob_state)

    def measurement_uwb_robust(self, measurement,
                               cov_m,
                               beacon_pos,
                               beacon_id,
                               ka_squard=18.0,
                               T_d=15.0):
        '''
        Robust EKF .
        :param measurement:
        :param cov_m:
        :param beacon_pos:
        :param beacon_id:
        :param ka_squard:
        :param T_d:
        :return:
        '''
        if self.uwb_eta_dict.get(beacon_id) is None:
            self.uwb_eta_dict[beacon_id] = list()
        z = np.zeros(1)
        y = np.zeros(1)

        z[0] = measurement
        y[0] = np.linalg.norm(self.state[0:3] - beacon_pos)

        self.H = np.zeros(shape=(1, self.state.shape[0]))
        self.H[0, 0:3] = (self.state[0:3] - beacon_pos).transpose() / y[0]

        R_k = cov_m * 1.0
        P_v = (self.H.dot(self.prob_state)).dot(np.transpose(self.H)) + R_k;
        v_k = z - y
        eta_k = np.zeros(1)
        self.uwb_eta_dict[beacon_id].append(eta_k[0])
        # print(v_k)
        # if v_k[0] > 0.5:
        #     return

        robust_loop_flag = True
        first_time = True
        while robust_loop_flag:
            robust_loop_flag = False

            P_v = (self.H.dot(self.prob_state)).dot(np.transpose(self.H)) + R_k

            eta_k[0] = (np.transpose(v_k).dot(np.linalg.inv(P_v))).dot(v_k)
            # print(eta_k[0])
            #
            # if eta_k[0] > 1.0:
            #     return
            if first_time:
                self.uwb_eta_dict[beacon_id].append(eta_k[0])
                first_time = False
            if (eta_k[0] > ka_squard):
                # if first_time:
                #
                #     first_time=False
                # else:
                self.uwb_eta_dict[beacon_id][-1] = eta_k[0]

                # np.std()
                serial_length = 5
                if len(self.uwb_eta_dict[beacon_id]) > serial_length:
                    lambda_k = np.std(np.asarray(self.uwb_eta_dict[beacon_id][-serial_length:]))
                    # print(self.uwb_eta_dict[beacon_id][-serial_length:],lambda_k, R_k[0])
                    if lambda_k > T_d:
                        robust_loop_flag = True
                        R_k[0] = eta_k[0] / ka_squard * R_k[0]
                # self.uwb_eta_dict[beacon_id].pop()

        cov_m = R_k
        self.R_k = R_k
        # print('-------------')

        self.K = (self.prob_state.dot(np.transpose(self.H))).dot(
            np.linalg.inv((self.H.dot(self.prob_state)).dot(np.transpose(self.H)) + cov_m)
        )

        dx = self.K.dot(z - y)

        self.state = self.state + dx

        self.rotation_q = quaternion_left_update(self.rotation_q, dx[6:9], -1.0)

        self.state[6:9] = dcm2euler(q2dcm(self.rotation_q))

        kh = self.K.dot(self.H)
        self.prob_state = (np.identity(kh.shape[0]) - kh).dot(self.prob_state)

    def measurement_uwb_iterate_standard(self, measurement, cov_m, beacon_set, ref_trace, once_flag=False):
        '''
        Standard IEKF measurement function.
        :param measurement:
        :param cov_m:
        :param beacon_set:
        :param ref_trace:
        :return:
        '''

        pminus = self.prob_state * 1.0
        pplus = pminus * 1.0
        xminus = self.state

        xplus = self.state
        xop = self.state * 0.0

        # process and select measurement and beaconset
        measurement = measurement[np.where(beacon_set[:, 0] < 5000.0)] * 1.0
        beacon_set = beacon_set[np.where(beacon_set < 5000.0)] * 1.0
        beacon_set = beacon_set.reshape([-1, 3])

        m_index = np.where(measurement > 0.0)
        measurement = measurement[m_index] * 1.0
        beacon_set = beacon_set[m_index, :] * 1.0
        # print(measurement.shape, beacon_set.shape)
        measurement = measurement.reshape(-1)
        beacon_set = beacon_set.reshape([-1, 3])

        Rk = np.identity(measurement.shape[0], float) * cov_m[0]
        dx = np.zeros(self.state.shape[0])

        ite_counter = 0
        while np.linalg.norm(xplus - xop) > 0.01 and ite_counter < 30:
            ite_counter += 1
            xop = xplus * 1.0
            y = np.linalg.norm(xop[0:3] - beacon_set, axis=1)
            # y = rou(y)
            H = np.zeros(shape=(measurement.shape[0], self.state.shape[0]))
            H[:, 0:3] = (xop[0:3] - beacon_set) / y.reshape(-1, 1)  # * d_rou(y.reshape(-1, 1))

            K = (pminus.dot(np.transpose(H))).dot(
                np.linalg.inv(H.dot(pminus.dot(np.transpose(H))) + Rk))
            kh = K.dot(H)
            pplus = (np.identity(kh.shape[0]) - kh).dot(pminus)
            # if tp_plus
            dx = K.dot((measurement - y - H.dot(xminus - xop)))
            xplus = xminus + dx
            # print('it')
            if once_flag is True:
                break

        self.state = self.state + dx

        self.rotation_q = quaternion_left_update(self.rotation_q, dx[6:9], -1.0)

        self.state[6:9] = dcm2euler(q2dcm(self.rotation_q))

        self.prob_state = pplus

    def measurement_uwb_iterate(self, measurement, cov_m, beacon_set, ref_trace, ka_squard=10.0):
        '''
        Robust iekf based uwb measurement
        :param measurement:
        :param cov_m:
        :param beacon_set:
        :param ref_trace:
        :param ka_squard:
        :return:
        '''

        pminus = self.prob_state * 1.0
        pplus = pminus * 1.0
        xminus = self.state

        xplus = self.state
        xop = self.state * 0.0

        # process and select measurement and beaconset
        measurement = measurement[np.where(beacon_set[:, 0] < 5000.0)] * 1.0
        beacon_set = beacon_set[np.where(beacon_set < 5000.0)] * 1.0
        beacon_set = beacon_set.reshape([-1, 3])

        m_index = np.where(measurement > 0.0)
        measurement = measurement[m_index] * 1.0
        beacon_set = beacon_set[m_index, :] * 1.0
        # print(measurement.shape, beacon_set.shape)
        measurement = measurement.reshape(-1)
        beacon_set = beacon_set.reshape([-1, 3])

        Rk = np.identity(measurement.shape[0], float) * cov_m[0]
        dx = np.zeros(self.state.shape[0])

        # weight for dx.
        mask = np.zeros(measurement.shape[0])
        ite_counter = 0
        while np.linalg.norm(xplus - xop) > 0.001 and ite_counter < 30:
            ite_counter += 1
            xop = xplus * 1.0
            y = np.linalg.norm(xop[0:3] - beacon_set, axis=1)
            # y = rou(y)
            H = np.zeros(shape=(measurement.shape[0], self.state.shape[0]))
            H[:, 0:3] = (xop[0:3] - beacon_set) / y.reshape(-1, 1)  # * d_rou(y.reshape(-1, 1))

            v = measurement - y

            index = np.argsort(np.abs(v))
            break_flag = False

            for i in range(index.shape[0]):
                # if ite_counter is 1:
                #     break
                if mask[index[i]] < 100.01:
                    pv = (H[index[i], :].dot(pplus)).dot(np.transpose(H[index[i], :])) + Rk[index[i], index[i]]
                    gamma = v[index[i]] * v[index[i]] / pv
                    # Rk[index[i],index[i]] = cov_m[0]
                    # print(pv, v[index[i]])
                    # ka_squard = 7.0

                    if gamma < ka_squard:  # or i < np.floor(index.shape[0] / 2):
                        # break_flag=True
                        mask[index[i]] = 1.0
                        # Rk[index[i],index[i]]=cov_m[0]

                        Rk[index[i], index[i]] = 1.0 * Rk[index[i], index[i]]
                        # Rk[index[i], index[i]] = ((gamma / ka_squard)**0.01) * Rk[index[i], index[i]]
                        # Rk[index[i], index[i]] = ((gamma / ka_squard)) * cov_m[0]
                    else:
                        # print('corrected Rk')
                        mask[index[i]] = 1.0
                        # mask[index[i]] = ka_squard / gamma * 1.0
                        # mask[index[i]] = 0.5#ka_squard/gamma
                        # Rk[index[i], index[i]] = ((gamma / ka_squard)**0.1) * Rk[index[i], index[i]]
                        Rk[index[i], index[i]] = ((gamma / ka_squard)) * Rk[index[i], index[i]]
                        # Rk[index[i],index[i]] = ((gamma/ka_squard)) * cov_m[0]

                        # mask[index[i]] = 1.0 / gamma
                    i = index.shape[0] + 1
                    # print(gamma)

                    # i=index.shape[0]+1
            # if break_flag:
            #     break
            # print(mask)

            K = (pminus.dot(np.transpose(H))).dot(
                np.linalg.inv(H.dot(pminus.dot(np.transpose(H))) + Rk))
            kh = K.dot(H)
            pplus = (np.identity(kh.shape[0]) - kh).dot(pminus)
            # pminus = pplus * 1.0
            # if tp_plus
            dx = K.dot((measurement - y - H.dot(xminus - xop)) * mask)
            xplus = xminus + dx
            # print(np.linalg.norm(pplus[0:3, 0:3]))
            # break
            # print('it')
        # print('-----')
        # print(ite_counter)
        # print(Rk)
        rt = np.ones(Rk.shape[0])
        for i in range(Rk.shape[0]):
            rt[i] = Rk[i, i]
        #     print(Rk)
        # print('rt   :',rt)

        self.state = self.state + dx

        self.rotation_q = quaternion_left_update(self.rotation_q, dx[6:9], -1.0)

        self.state[6:9] = dcm2euler(q2dcm(self.rotation_q))

        self.prob_state = pplus

        self.R = Rk * 1.0
        self.R_k = Rk * 1.0

    def measurement_uwb_mc_itea(self, measurement, cov_m, beacon_set, ref_trace):
        '''
        Use particle simulator posterior distribution.
        uncompleted!!! and may be invalid in such situation.
        :param measurement:
        :param cov_m:
        :param beacon_set:
        :param ref_trace:
        :return:
        '''

        measurement = measurement[np.where(beacon_set[:, 0] < 5000.0)] * 1.0
        beacon_set = beacon_set[np.where(beacon_set < 5000.0)] * 1.0
        beacon_set = beacon_set.reshape([-1, 3])

        m_index = np.where(measurement > 0.0)
        measurement = measurement[m_index] * 1.0
        beacon_set = beacon_set[m_index, :] * 1.0
        # print(measurement.shape, beacon_set.shape)
        measurement = measurement.reshape(-1)
        beacon_set = beacon_set.reshape([-1, 3])

        if measurement.shape[0] < 3:
            # self.measurement_uwb_iterate(measurement, cov_m, beacon_set, np.zeros([10, 10]))
            self.R_k_all = np.identity(measurement.shape[0]) * cov_m[0]
            for i in range(measurement.shape[0]):
                self.measurement_uwb_robust(np.asarray(measurement[i]),
                                            cov_m,
                                            np.transpose(beacon_set[i, :]), i)
                self.R_k_all[i, i] = self.R_k[0]
            self.R_k = self.R_k_all * 1.0
            return
        # else:
        #     print('mc')

        R = np.identity(measurement.shape[0]) * cov_m[0]

        particles = np.zeros(shape=(9000, 3))
        w = np.ones(shape=particles.shape[0])
        w = w / w.sum()

        # rnd_p = np.random.normal(0.0, 1.0, size=particles.shape)
        # rnd_p  = np.random.multivariate_normal()
        def gaussian_distribution(x, miu, sigma):
            # print('sigma',sigma)
            a = 1.0 / sigma / math.sqrt(2.0 * 3.1415926)
            b = -1.0 * ((x - miu) * (x - miu) / 2.0 / sigma / sigma)
            # if math.isnan(a):
            #     print('a is nan', x, miu, sigma)
            # if math.isnan(b):
            #     print('b is nan', x, miu, sigma)
            # print(a * math.exp(b))#,a,b, x,miu,sigma)
            if math.isnan(math.exp(b)):
                print('error:', b)
                return -10000.0
            else:
                return math.log(a) * (b)

        gaussian_pdf_v = np.vectorize(gaussian_distribution)

        p_mean = self.state[0:3] * 1.0
        p_std = self.prob_state

        counter = 0

        while np.linalg.norm(np.mean(particles, axis=0) - p_mean) > 0.01 or counter is 0:
            p_mean = np.mean(particles, axis=0)
            counter += 1
            w = np.ones(shape=particles.shape[0])
            # w = w / w.sum()
            w = w - np.log(np.sum(np.exp(w)))

            # print('normal w:',np.mean(w),'ana w:', np.log(1/float(w.shape[0])))
            particles = np.random.multivariate_normal(self.state[0:3] * 1.0, self.prob_state[0:3, 0:3] * 150.0,
                                                      size=particles.shape[0])

            for j in range(beacon_set.shape[0]):
                w = w + gaussian_pdf_v(np.linalg.norm(particles - beacon_set[j, :], axis=1),
                                       np.ones_like(w) * measurement[j],
                                       np.ones_like(w) * R[j, j] * 50.0)
            s = np.sum(np.exp(w))

            for i in range(beacon_set.shape[0]):
                R[i, i] = np.sum(
                    np.abs(np.linalg.norm(particles - beacon_set[i, :], axis=1) - measurement[i]) * np.exp(w) / np.sum(
                        np.exp(w)),
                    axis=0) ** 2.0  # actually variance of noise.
                # R[i,i] = np.std(np.linalg.norm(particles-beacon_set[i,:],axis=1)*np.exp(w),axis=0)
        self.R_k = R * 1.0

        for i in range(beacon_set.shape[0]):
            self.measurement_uwb(np.asarray(measurement[i]),
                                 np.ones(1) * (R[i, i]),
                                 np.transpose(beacon_set[i, :]))
            if np.isnan(self.state).any():
                print('error', i, self.state)

    def measurement_uwb_sp(self, measurement, cov_m, beacon_set, ref_trace):
        '''
        Using Sigma points sample technology.
        :param measurement:
        :param cov_m:
        :param beacon_set:
        :param ref_trace:
        :return:
        '''

        measurement = measurement[np.where(beacon_set[:, 0] < 5000.0)] * 1.0
        beacon_set = beacon_set[np.where(beacon_set < 5000.0)] * 1.0
        beacon_set = beacon_set.reshape([-1, 3])

        m_index = np.where(measurement > 0.0)
        measurement = measurement[m_index] * 1.0
        beacon_set = beacon_set[m_index, :] * 1.0
        # print(measurement.shape, beacon_set.shape)
        measurement = measurement.reshape(-1)
        beacon_set = beacon_set.reshape([-1, 3])

        if measurement.shape[0] < 3:
            self.R_k_all = np.identity(measurement.shape[0])
            for i in range(measurement.shape[0]):
                self.measurement_uwb_robust(np.asarray(measurement[i]),
                                            cov_m,
                                            np.transpose(beacon_set[i, :]), i)
                self.R_k_all[i, i] = self.R_k[0]
            self.R_k = self.R_k_all * 1.0
            return

        particles = np.zeros(shape=(8, 3))
        w = np.ones(shape=particles.shape[0])
        w = w / w.sum()

        # sample
        particles[0, :] = self.state[0:3] * 1.0
        particles[1, :] = self.state[0:3] * 1.0
        L = sp.linalg.cholesky(self.prob_state[0:3, 0:3] * 1.0)
        for i in range(3):
            # particles[i * 2, :] = self.state[0:3] * 1.0 + self.prob_state[0:3, i]*10.0
            particles[i * 2+2, :] = self.state[0:3] * 1.0 + L[0:3, i] * 10.0
            # particles[i * 2 + 1, :] = self.state[0:3] * 1.0 - self.prob_state[0:3, i]*10.0
            particles[i * 2 + 1+2, :] = self.state[0:3] * 1.0 - L[0:3, i] * 10.0
            # w[i*2] = 1.2
            # w[i*2+1] = 1.2

        # measurement
        # @jit(nopython=True)
        def gaussian_distribution(x, miu, sigma):
            a = 1.0 / sigma / math.sqrt(2.0 * 3.1415926)
            b = -1.0 * ((x - miu) * (x - miu) / 2.0 / sigma / sigma)
            return a * math.exp(b)

        gaussian_pdf_v = np.vectorize(gaussian_distribution)

        for j in range(beacon_set.shape[0]):
            w = w * gaussian_pdf_v(np.linalg.norm(particles - beacon_set[j, :], axis=1),
                                   np.ones_like(w) * measurement[j],
                                   np.ones_like(w) * 25.0)
        w = w / w.sum()

        self.R_k = np.identity(measurement.shape[0])
        all_m_score = np.zeros_like(measurement)
        for i in range(beacon_set.shape[0]):
            all_m_score[i] = np.sum(
                np.abs(np.linalg.norm(particles - beacon_set[i, :], axis=1) - measurement[i]) ** 2.0 * w,
                axis=0)  # *float(particles.shape[0]-1)/float(particles.shape[0])
            m_diff = np.sum((np.linalg.norm(particles - beacon_set[i, :]) - measurement[i]) * w)
            self.R_k[i, i] = all_m_score[i] ** 2.0
            # if self.R_k[i, i] > 1.0:
            #     continue
            # else:
            self.measurement_uwb(np.asarray(measurement[i]),
                                 np.ones(1) * (all_m_score[i]),
                                 np.transpose(beacon_set[i, :]))
            # self.measurement_uwb(np.asarray(measurement[i]+m_diff),
            #                      np.ones(1) * (0.1+np.std((np.linalg.norm(particles-beacon_set[i,:])-measurement[i]))),
            #                      np.transpose(beacon_set[i, :]))

            # tm = np.linalg.norm(particles-beacon_set[i,:],axis=1)
            # avg = np.average(tm,weights=w)
            # all_m_score[i] = np.average((tm-avg)**2.0,weights=w)
            # all_m_score[i] = np.std(np.linalg.norm(particles-beacon_set[i,:],axis=1), weight=w)

    def measurement_uwb_mc(self, measurement, cov_m, beacon_set, ref_trace):
        '''
        Use particle simulator posterior distribution.
        uncompleted!!! and may be invalid in such situation.
        :param measurement:
        :param cov_m:
        :param beacon_set:
        :param ref_trace:
        :return:
        '''

        measurement = measurement[np.where(beacon_set[:, 0] < 5000.0)] * 1.0
        beacon_set = beacon_set[np.where(beacon_set < 5000.0)] * 1.0
        beacon_set = beacon_set.reshape([-1, 3])

        m_index = np.where(measurement > 0.0)
        measurement = measurement[m_index] * 1.0
        beacon_set = beacon_set[m_index, :] * 1.0
        # print(measurement.shape, beacon_set.shape)
        measurement = measurement.reshape(-1)
        beacon_set = beacon_set.reshape([-1, 3])

        if measurement.shape[0] < 3:
            self.R_k_all = np.identity(measurement.shape[0])
            for i in range(measurement.shape[0]):
                self.measurement_uwb_robust(np.asarray(measurement[i]),
                                            cov_m,
                                            np.transpose(beacon_set[i, :]), i)
                self.R_k_all[i, i] = self.R_k[0]
            self.R_k = self.R_k_all * 1.0
            return

        particles = np.zeros(shape=(10, 3))
        w = np.ones(shape=particles.shape[0])
        w = w / w.sum()

        # sample
        # particles = np.random.multivariate_normal(self.state[0:3], self.prob_state[0:3, 0:3] * 50000.0,
        #                                           size=particles.shape[0])

        particles = np.random.multivariate_normal(self.state[0:3], np.identity(3) * 5.0,
                                                  size=particles.shape[0])

        # plt.figure(10)
        # plt.clf()
        # plt.title('prior hist')
        # plt.hist2d(particles[:, 0], particles[:, 1], bins=50, weights=w)
        # plt.pause(0.1)
        # measurement
        # @jit(nopython=True)
        def gaussian_distribution(x, miu, sigma):
            a = 1.0 / sigma / math.sqrt(2.0 * 3.1415926)
            b = -1.0 * ((x - miu) * (x - miu) / 2.0 / sigma / sigma)
            return a * math.exp(b)

        gaussian_pdf_v = np.vectorize(gaussian_distribution)

        for j in range(beacon_set.shape[0]):
            w = w * gaussian_pdf_v(np.linalg.norm(particles - beacon_set[j, :], axis=1),
                                   np.ones_like(w) * measurement[j],
                                   np.ones_like(w) * 25.0)
        w = w / w.sum()

        # vote for each measurement
        # plt.figure(11)
        # plt.clf()
        # plt.hist(w*float(w.shape[0]))
        # plt.pause(0.1)

        # plt.figure(11)
        # plt.clf()
        # plt.title('posterior hist')
        # plt.hist2d(particles[:, 0], particles[:, 1], bins=50, weights=w)
        # plt.pause(0.1)

        self.R_k = np.identity(measurement.shape[0])
        all_m_score = np.zeros_like(measurement)
        for i in range(beacon_set.shape[0]):
            all_m_score[i] = np.sum(
                np.abs(np.linalg.norm(particles - beacon_set[i, :], axis=1) - measurement[i]) ** 2.0 * w,
                axis=0)  # *float(particles.shape[0]-1)/float(particles.shape[0])
            m_diff = np.sum((np.linalg.norm(particles - beacon_set[i, :]) - measurement[i]) * w)
            self.R_k[i, i] = all_m_score[i] ** 2.0
            # if self.R_k[i, i] > 1.0:
            #     continue
            # else:
            self.measurement_uwb(np.asarray(measurement[i] + m_diff),
                                 np.ones(1) * (all_m_score[i]),
                                 np.transpose(beacon_set[i, :]))
            # self.measurement_uwb(np.asarray(measurement[i]+m_diff),
            #                      np.ones(1) * (),
            #                      np.transpose(beacon_set[i, :]))

            # tm = np.linalg.norm(particles-beacon_set[i,:],axis=1)
            # avg = np.average(tm,weights=w)
            # all_m_score[i] = np.average((tm-avg)**2.0,weights=w)
            # all_m_score[i] = np.std(np.linalg.norm(particles-beacon_set[i,:],axis=1), weight=w)

    def measurement_uwb_robust_multi(self, measurement, cov_m, beacon_set, ka_squard):
        '''
        Uwb robust Measurement multi UWB
        :param measurement:
        :param cov_m:
        :param beacon_set:
        :param ka_squard:
        :return:
        '''

        # @jit()# @jit(nopython=True)
        def get_vk_eta(measurement, beacon_pos, state, cov, P, beacon_id):
            if self.uwb_eta_dict.get(beacon_id) is None:
                self.uwb_eta_dict[beacon_id] = list()
            z = np.zeros(1)
            y = np.zeros(1)
            z[0] = measurement
            y[0] = np.linalg.norm(state[0:3] - beacon_pos)

            H = np.zeros(shape=(1, state.shape[0]))
            H[0, 0:3] = (state[0:3] - beacon_pos).transpose() / y[0]

            R_k = cov * 1.0

            P_v = (H.dot(P).dot(np.transpose(H)) + R_k)
            v_k = z - y
            eta_k = np.zeros(1)

            robust_loop_flag = True
            first_time = True
            while robust_loop_flag:
                robust_loop_flag = False

                P_v = (H.dot(P)).dot(np.transpose(H)) + R_k

                eta_k[0] = (np.transpose(v_k).dot(np.linalg.inv(P_v))).dot(v_k)
                # print(eta_k[0])
                #
                # if eta_k[0] > 1.0:
                #     return
                if first_time:
                    self.uwb_eta_dict[beacon_id].append(eta_k[0])
                    first_time = False
                if (eta_k[0] > ka_squard):
                    # if first_time:
                    #
                    #     first_time=False
                    # else:
                    self.uwb_eta_dict[beacon_id][-1] = eta_k[0]

                    # np.std()
                    serial_length = 5
                    if len(self.uwb_eta_dict[beacon_id]) > serial_length:
                        lambda_k = np.std(np.asarray(self.uwb_eta_dict[beacon_id][-serial_length:]))
                        # print(self.uwb_eta_dict[beacon_id][-serial_length:],lambda_k, R_k[0])
                        if lambda_k > 1.0:
                            robust_loop_flag = True
                            R_k[0] = eta_k[0] / ka_squard * R_k[0]

            K = (P.dot(np.transpose(H))).dot(
                np.linalg.inv(((H.dot(P)).dot(np.transpose(H)) + R_k))
            )
            dx = K.dot(z - y)
            return v_k[0], R_k[0], H, K, dx

        v_k_list = list()
        R_k_list = list()
        H_list = list()
        K_list = list()
        dx_list = list()
        # if measurement.
        measurement = measurement.reshape(-1)

        for i in range(measurement.shape[0]):
            if self.dx_dict.get(i) is None:
                self.dx_dict[i] = list()
            if measurement[i] > 0.0 and beacon_set[i, 0] < 5000.0:
                tvk, trk, th, tk, tdx = get_vk_eta(measurement[i],
                                                   beacon_set[i, :].transpose(),
                                                   self.state, cov_m,
                                                   self.prob_state, i)
                if abs(tvk) < 100.0:  # or tvk > 10.0:
                    v_k_list.append(tvk)
                    R_k_list.append(cov_m[0])
                    H_list.append(th)
                    K_list.append(tk)
                    dx_list.append(tdx)
                    self.dx_dict[i].append(tdx)
            else:
                self.dx_dict[i].append(np.zeros_like(self.state))
        # print(len(v_k_list))

        # if len(v_k_list) > 4:
        #     # iter
        #     max_index = np.argmax(np.asarray(v_k_list))
        #     print('max:', v_k_list[max_index], R_k_list[max_index])
        #     if abs(v_k_list[max_index]) > 1.0:
        #         v_k_list[max_index] = 0.0
        #         R_k_list[max_index] = 1000000000.0
        #     max_index = np.argmax(np.asarray(v_k_list))
        #     print('max:', v_k_list[max_index], R_k_list[max_index])
        #     if abs(v_k_list[max_index]) > 1.0:
        #         v_k_list[max_index] = 0.0
        #         R_k_list[max_index] = 1000000000.0
        # elif len(v_k_list) is 4:
        #     max_index = np.argmax(np.asarray(v_k_list))
        #     if abs(v_k_list[max_index]) > 1.0:
        #         v_k_list[max_index] = 0.0
        #         R_k_list[max_index] = 1000000000.0

        # print('size:',len(R_k_list),R_k_list)
        # print('size:',len(R_k_list),v_k_list)
        assert len(v_k_list) == len(R_k_list) == len(H_list)
        R_matrix = np.zeros(shape=(len(v_k_list), len(v_k_list)))
        self.H = np.zeros(shape=(len(v_k_list), self.state.shape[0]))
        V = np.zeros(shape=(len(v_k_list), 1))
        # print(sorted(v_k_list))
        # print(v_k_list)
        # print(R_k_list)
        # print('-0---------------------------------')

        for i in range(len(v_k_list)):
            R_matrix[i, i] = R_k_list[i]
            self.H[i, :] = H_list[i]
            V[i, 0] = v_k_list[i]

        self.K = (self.prob_state.dot(np.transpose(self.H))).dot(
            np.linalg.inv((self.H.dot(self.prob_state)).dot(np.transpose(self.H)) + R_matrix)
        )

        dx = self.K.dot(V).reshape(-1)

        self.state = self.state + dx

        self.rotation_q = quaternion_left_update(self.rotation_q, dx[6:9], -1.0)

        self.state[6:9] = dcm2euler(q2dcm(self.rotation_q))

    def measurement_reftrace(self, cov_m, ref_f):
        z = np.ones(1) * 0.0
        y = np.ones(1) * ref_f.eval_point2d(self.state[0:2])

        H = np.zeros([1, 15])
        ts = self.state[0:2] * 1.0
        ts[0] += 0.1
        H[0, 0] = (ref_f.eval_point2d(ts) - y) / 0.1
        ts = self.state[0:2] * 1.0
        ts[1] += 0.1
        H[0, 1] = (ref_f.eval_point2d(ts) - y) / 0.1

        s = (H.dot(self.prob_state)).dot(np.transpose(H)) + np.identity(1) * cov_m

        K = (self.prob_state.dot(np.transpose(H))).dot(np.linalg.inv(s))

        dx = K.dot(z - y).reshape(-1)

        self.state = self.state + dx

        self.rotation_q = quaternion_left_update(self.rotation_q, dx[6:9], -1.0)

        self.state[6:9] = dcm2euler(q2dcm(self.rotation_q))

        self.prob_state = (np.identity(self.prob_state.shape[0]) - K.dot(H)).dot(self.prob_state)


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
