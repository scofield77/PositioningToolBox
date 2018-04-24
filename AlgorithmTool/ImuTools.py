# -*- coding:utf-8 -*-
# Created by steve @ 18-3-18 下午2:10
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

import numdifftools

from numba import jit

import math

import matplotlib.pyplot as plt


class settings:
    '''
    class contained all settings of imu .
    '''

    def __init__(self):
        self.dim_constraint = 2.0
        self.alpha_level = 0.95
        self.range_constraint = 1.0  # [m]
        self.min_rud_sep = 820

        self.init_heading1 = (-95.0 - 2.5) * np.pi / 180.0
        self.init_heading2 = (94.05 + 2.0) * np.pi / 180.0

        self.init_pos1 = np.zeros([1, 3])
        self.init_pos2 = np.zeros([1, 3])

        self.range_constraint_on = False

        self.altitude = 100
        self.latitude = 50

        self.Ts = 1.0 / 820.0

        self.init_heading = 0.0
        self.init_pos = np.zeros([3, 1])

        self.sigma_a = 0.035

        self.sigma_g = 0.35 * np.pi / 180.0

        self.gamma = 200.0

        self.sigma_acc = 4.0 * 0.7 * np.ones([3, 1])

        self.sigma_gyro = 4.0 * 10.0 * np.ones([3, 1]) * 0.1 * np.pi / 180.0

        self.sigma_vel = 5.0 * np.ones([1, 3]) * 0.01

        self.sigma_initial_pos = 1e-2 * 0.1 * np.ones([3, 1])
        self.sigma_initial_vel = 1e-5 * np.ones([3, 1])
        self.sigma_initial_att = (np.pi / 180.0 *
                                  np.array([0.1, 0.1, 0.001]).reshape(3, 1))

        self.sigma_initial_pos2 = 1e-2 * 0.1 * np.ones([3, 1])
        self.sigma_initial_vel2 = 1e-5 * np.ones([3, 1])
        self.sigma_initial_att2 = (np.pi / 180.0 *
                                   np.array([0.1, 0.1, 0.001]).reshape(3, 1))

        self.sigma_initial_range_single = 1.0

        # self.s_P = np.loadtxt("../Data/P.csv", dtype=float, delimiter=",")
        self.gravity = 9.8

        self.time_Window_size = 3

        #
        self.pose_constraint = True


class zero_velocity_tester:
    '''
    zero velocity detector by several different methods.
    '''

    def __init__(self, setting):
        self.setting = setting

    def tester(self, u, method_type='GLRT'):
        if method_type is 'GLRT':
            self.GLRT_Detector(u)

    # @jit
    def GLRT_Detector(self, u):
        g = self.setting.gravity

        sigma2_a = self.setting.sigma_a
        sigma2_g = self.setting.sigma_g
        sigma2_a = sigma2_a ** 2.0
        sigma2_g = sigma2_g ** 2.0

        W = self.setting.time_Window_size
        # W = u.

        N = u.shape[0]
        T = np.zeros([N - W + 1, 1])

        for k in range(N - W + 1):
            ya_m = np.mean(u[k:k + W - 1, 0:3], 0)
            # print(ya_m.shape)

            for l in range(k, k + W - 1):
                tmp = u[l, 0:3] - g * ya_m / np.linalg.norm(ya_m)
                T[k] = T[k] + (np.linalg.norm(u[l, 3:6]) ** 2.0) / sigma2_g + \
                       (np.linalg.norm(tmp) ** 2.0) / sigma2_a
        T = T / W

        zupt = np.zeros([u.shape[0], 1])
        # import matplotlib.pyplot as plt
        # plt.figure(1)
        # plt.plot(T)
        # plt.show()
        for k in range(T.shape[0]):
            if T[k] < self.setting.gamma:
                zupt[k:k + W] = np.ones([W, 1])

        return zupt

# class gravity

@jit
def quaternion_right_update(q, euler, rate):
    '''
    quaternion right update
    :param q:
    :param euler:
    :return:
    '''
    # Theta = cuda.local.array(shape=(4, 4), dtype=float64)
    Theta = np.zeros([4, 4])

    euler = euler * rate
    delta_euler = (euler[0] * euler[0] + euler[1] * euler[1] + euler[2] + euler[2])
    delta_euler = math.sqrt(delta_euler)

    if delta_euler > 1e-19:
        c = math.cos(delta_euler / 2.0)
        sdiv = math.sin(delta_euler / 2.0) / delta_euler
    else:
        c = 1.0
        sdiv = 0.0

    Theta[0, 0] = c
    Theta[0, 1] = -euler[0] * sdiv
    Theta[0, 2] = -euler[1] * sdiv
    Theta[0, 3] = -euler[2] * sdiv

    Theta[1, 0] = euler[0] * sdiv
    Theta[1, 1] = c
    Theta[1, 2] = euler[2] * sdiv
    Theta[1, 3] = -euler[1] * sdiv

    Theta[2, 0] = euler[1] * sdiv
    Theta[2, 1] = -euler[2] * sdiv
    Theta[2, 2] = c
    Theta[2, 3] = euler[0] * sdiv

    Theta[3, 0] = euler[2] * sdiv
    Theta[3, 1] = euler[1] * sdiv
    Theta[3, 2] = -euler[0] * sdiv
    Theta[3, 3] = c

    # tq = cuda.local.array(shape=(4), dtype=float64)
    # tq = q
    tq = np.zeros_like(q)
    for i in range(4):
        tq[i] = q[i] * 1.0
    norm_new_q = 0.0
    for i in range(4):
        q[i] = 0.0
        for j in range(4):
            q[i] += Theta[i, j] * tq[j]
        norm_new_q = norm_new_q + q[i] * q[i]
    norm_new_q = math.sqrt(norm_new_q)

    for i in range(4):
        q[i] = q[i] / norm_new_q

    return q


@jit
def euler2R(ang, R):
    '''
    :
    :param ang:
    :return:
    '''
    cr = math.cos(ang[0])
    sr = math.sin(ang[0])

    cp = math.cos(ang[1])
    sp = math.sin(ang[1])

    cy = math.cos(ang[2])
    sy = math.sin(ang[2])

    # R = np.array(
    #     [[cy * cp, sy * cp, -sp],
    #      [-sy * cr + cy * sp * sr, cy * cr + sy * sp * sr, cp * sr],
    #      [sy * sr + cy * sp * cr, -cy * sr + sy * sp * cr, cp * cr]]
    # )
    # return R
    R[0, 0] = cy * cp
    R[0, 1] = sy * cp
    R[0, 2] = -sp

    R[1, 0] = -sy * cr + cy * sp * sr
    R[1, 1] = cy * cr + sy * sp * sr
    R[1, 2] = cp * sr

    R[2, 0] = sy * sr + cy * sp * cr
    R[2, 1] = -cy * sr + sy * sp * cr
    R[2, 2] = cp * cr

    return R


@jit
def dcm2q(R):
    """
    http://www.ee.ucr.edu/~farrell/AidedNavigation/D_App_Quaternions/Rot2Quat.pdf
    [1] Farrell J A. Computation of the Quaternion from a Rotation Matrix[J]. 2008.
    Transform from rotation matrix to quanternions.
    :param R:old rotation matrix
    :param q: return value
    """
    T = 1.0 + R[0, 0] + R[1, 1] + R[2, 2]
    # print (T)

    # Really Big Change.
    # ToDo:Why there are some value is smallter than zero.
    if math.fabs(T) > 1e-3:
        S = 0.5 / math.sqrt(math.fabs(T))

        qw = 0.25 / S
        qx = (R[2, 1] - R[1, 2]) * S
        qy = (R[0, 2] - R[2, 0]) * S
        qz = (R[1, 0] - R[0, 1]) * S

    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0

            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S

        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0

            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    # quart = np.array(np.transpose([qx, qy, qz, qw]))

    # quart /= np.linalg.norm(quart)
    q_norm = 0.0
    # for i in range(4):
    #     q_norm =
    q_norm = qw * qw + qx * qx + qy * qy + qz * qz
    q_norm = math.sqrt(q_norm)
    q = np.zeros(4)
    q[0] = qw / q_norm
    q[1] = qx / q_norm
    q[2] = qy / q_norm
    q[3] = qz / q_norm
    return q



if __name__ == '__main__':
    '''
    Test all alogorithm in this file.
    '''

    # quaternion right update

    step_len = 0.1 * np.pi / 180.0

    # qx = np.zeros([4])
    # qy = np.zeros([4])
    # qz = np.zeros([4])
    # qx[0] = 1.0
    # qy[0] = 1.0
    # qz[0] = 1.0
    q_all = np.zeros([4, 3])
    euler_all = np.zeros([3, 3])
    q_all[0, :] += 1.0

    counter = math.floor(2.0 * np.pi / step_len)
    result_q_all = np.zeros([4, 3, counter])
    result_angle = np.zeros([3, 3, counter])
    for i in range(result_q_all.shape[2]):
        for k in range(3):
            euler = np.zeros([3])
            euler[k] = step_len
            q_all[:, k] = quaternion_right_update(q_all[:, k], euler, 1.0)
            euler_all[:, k] = dcm2euler(q2dcm(q_all[:, k]))

        result_q_all[:, :, i] = q_all * 1.0
        result_angle[:, :, i] = euler_all * 1.0

    plt.figure()
    for i in range(3):
        plt.subplot(310 + i + 1)
        plt.title('ax:' + str(i))
        plt.plot(result_q_all[:, i, :].transpose())
        plt.legend()
        plt.grid()

    plt.figure()
    for i in range(3):
        plt.subplot(310 + i + 1)
        plt.title('ax:' + str(i))
        plt.plot(result_angle[:, i, :].transpose())
        plt.legend()
        plt.grid()
    plt.show()