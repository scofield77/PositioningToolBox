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

# import numdifftools

from numba import jit, prange, float64, int32

import math

import matplotlib.pyplot as plt

from scipy.optimize import minimize


class settings:
    '''
    class contained all settings of imu (especially IMU for ZUPT-based location).
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


# @jit(cache=True)
@jit(float64[:](float64[:, :], float64, float64, float64, float64, int32),
     nopython=True)  # , parallel=True)#, cache=True)
def GLRT_Detector(u,
                  sigma_a=0.4,
                  sigma_g=0.4 * np.pi / 180.0,
                  gamma=280.0,
                  gravity=9.8,
                  time_Window_size=10):
    '''
    zero-velocity detect function.
    :param u: imu data  acc(m/m^2) gyr(rad/s)
    :param sigma_a:
    :param sigma_g:
    :param gamma:
    :param gravity:
    :param time_Window_size:
    :return:
    '''
    g = gravity

    sigma2_a = sigma_a
    sigma2_g = sigma_g
    sigma2_a = sigma2_a ** 2.0
    sigma2_g = sigma2_g ** 2.0

    W = time_Window_size
    # W = u.

    N = u.shape[0]
    T = np.zeros((N - W + 1, 1))

    for k in prange(N - W + 1):
        # ya_m = np.mean(u[k:k + W - 1, 0:3], 0)
        ya_m = np.zeros(3)
        count = 0
        for j in range(k, k + W - 1):
            # ya_m = ya_m + u[j, 0]
            # ya_m = ya_m + u[j, 1]
            # ya_m = ya_m + u[j, 2]
            for d in range(3):
                ya_m[d] = ya_m[d] + u[j, d]
            count += 1
        ya_m = ya_m / float(count)
        # print(ya_m.shape)

        for l in range(k, k + W - 1):
            # tmp = u[l, 0:3] - g * ya_m / np.linalg.norm(ya_m)
            tmp = np.zeros(3)
            for j in range(3):
                tmp[j] = u[l, j] - g * ya_m[j] / np.linalg.norm(ya_m)

            T[k] = T[k] + (np.linalg.norm(u[l, 3:6]) ** 2.0) / sigma2_g + \
                   (np.linalg.norm(tmp) ** 2.0) / sigma2_a

    T = T / W

    zupt = np.zeros(shape=(u.shape[0]))
    # import matplotlib.pyplot as plt
    # plt.figure(1)
    # plt.plot(T)
    # plt.show()
    for k in range(T.shape[0]):
        if T[k, 0] < gamma:
            # zupt[k:k + W] = np.ones([W, 1])
            for j in range(k, k + W):
                zupt[j] = 1.0

    return zupt


# @jit(cache=True)
@jit(float64[:](float64[:, :], float64, float64, float64, float64, int32),
     nopython=True)  # , parallel=True)#, cache=True)
def GLRT_Detector_prob(u,
                       sigma_a=0.4,
                       sigma_g=0.4 * np.pi / 180.0,
                       gamma=280.0,
                       gravity=9.8,
                       time_Window_size=10):
    '''
    zero-velocity detect function.
    :param u: imu data  acc(m/m^2) gyr(rad/s)
    :param sigma_a:
    :param sigma_g:
    :param gamma:
    :param gravity:
    :param time_Window_size:
    :return:
    '''
    g = gravity

    sigma2_a = sigma_a
    sigma2_g = sigma_g
    sigma2_a = sigma2_a ** 2.0
    sigma2_g = sigma2_g ** 2.0

    W = time_Window_size
    # W = u.

    N = u.shape[0]
    T = np.zeros((N - W + 1, 1))

    for k in prange(N - W + 1):
        # ya_m = np.mean(u[k:k + W - 1, 0:3], 0)
        ya_m = np.zeros(3)
        count = 0
        for j in range(k, k + W - 1):
            # ya_m = ya_m + u[j, 0]
            # ya_m = ya_m + u[j, 1]
            # ya_m = ya_m + u[j, 2]
            for d in range(3):
                ya_m[d] = ya_m[d] + u[j, d]
            count += 1
        ya_m = ya_m / float(count)
        # print(ya_m.shape)

        for l in range(k, k + W - 1):
            # tmp = u[l, 0:3] - g * ya_m / np.linalg.norm(ya_m)
            tmp = np.zeros(3)
            for j in range(3):
                tmp[j] = u[l, j] - g * ya_m[j] / np.linalg.norm(ya_m)

            T[k] = T[k] + (np.linalg.norm(u[l, 3:6]) ** 2.0) / sigma2_g + \
                   (np.linalg.norm(tmp) ** 2.0) / sigma2_a

    T = T / W

    zupt = np.zeros(shape=(u.shape[0]))
    # import matplotlib.pyplot as plt
    # plt.figure(1)
    # plt.plot(T)
    # plt.show()
    for k in range(T.shape[0]):
        # if T[k, 0] < gamma:
        # zupt[k:k + W] = np.ones([W, 1])
        for j in range(k, k + W):
            zupt[j] = np.log(1.0 + gamma / T[k, 0])
        # zupt[j] = T[k, 0] / gamma

    return zupt


# @jit('float64[:](float64[:],float64[:],float64)')
@jit(float64[:](float64[:], float64[:], float64), nopython=True, cache=True)
def quaternion_right_update(q, euler, rate):
    '''
    quaternion right update
    :param q:
    :param euler:
    :return:
    '''
    eta = euler * rate * 0.5
    eta_norm = np.linalg.norm(eta)

    mul_q = np.zeros(4)
    # qL = np.zeros_like([4, 4])

    if (eta_norm > 1e-18):
        mul_q[0] = math.cos(eta_norm)
        mul_q[1:4] = eta * math.sin(eta_norm) / eta_norm
    else:
        mul_q[0] = 1.0
        mul_q[1:4] = eta

    # qL = np.asarray((
    #     mul_q[0], -mul_q[1], -mul_q[2], -mul_q[3],
    #     mul_q[1], mul_q[0], mul_q[3], -mul_q[2],
    #     mul_q[2], -mul_q[3], mul_q[0], mul_q[1],
    #     mul_q[3], mul_q[2], -mul_q[1], mul_q[0]
    # )).reshape([4, 4])

    # tmp_q = qL.dot(q)
    # for i in range(4)
    tmp_q = np.zeros(4)
    tmp_q[0] = mul_q[0] * q[0] - mul_q[1] * q[1] - mul_q[2] * q[2] - mul_q[3] * q[3]
    tmp_q[1] = mul_q[1] * q[0] + mul_q[0] * q[1] + mul_q[3] * q[2] - mul_q[2] * q[3]
    tmp_q[2] = mul_q[2] * q[0] - mul_q[3] * q[1] + mul_q[0] * q[2] + mul_q[1] * q[3]
    tmp_q[3] = mul_q[3] * q[0] + mul_q[2] * q[1] - mul_q[1] * q[2] + mul_q[0] * q[3]

    tmp_q /= np.linalg.norm(tmp_q)
    return tmp_q


@jit(float64[:](float64[:], float64[:], float64), nopython=True, cache=True)
def quaternion_left_update(q, euler, rate):
    eta = euler * rate * 0.5
    eta_norm = np.linalg.norm(eta)

    mul_q = np.zeros(4)
    # qL = np.zeros_like([4, 4])

    if (eta_norm > 1e-18):
        mul_q[0] = math.cos(eta_norm)
        mul_q[1:4] = eta * math.sin(eta_norm) / eta_norm
    else:
        mul_q[0] = 1.0
        mul_q[1:4] = eta

    # qL = np.asarray((
    #     mul_q[0], -mul_q[1], -mul_q[2], -mul_q[3],
    #     mul_q[1], mul_q[0], -mul_q[3], mul_q[2],
    #     mul_q[2], mul_q[3], mul_q[0], -mul_q[1],
    #     mul_q[3], -mul_q[2], mul_q[1], mul_q[0]
    # )).reshape([4, 4])
    #
    # tmp_q = qL.dot(q)
    tmp_q = np.zeros(4)
    tmp_q[0] = mul_q[0] * q[0] - mul_q[1] * q[1] - mul_q[2] * q[2] - mul_q[3] * q[3]
    tmp_q[1] = mul_q[1] * q[0] + mul_q[0] * q[1] - mul_q[3] * q[2] + mul_q[2] * q[3]
    tmp_q[2] = mul_q[2] * q[0] + mul_q[3] * q[1] + mul_q[0] * q[2] - mul_q[1] * q[3]
    tmp_q[3] = mul_q[3] * q[0] - mul_q[2] * q[1] + mul_q[1] * q[2] + mul_q[0] * q[3]

    tmp_q /= np.linalg.norm(tmp_q)
    return tmp_q


@jit(float64[:](float64[:], float64[:]), nopython=True, cache=True)
def quaternion_product(ql, qr):
    '''
    q = ql * qr (quaternion product)
    :param ql:
    :param qr:
    :return:
    '''


# @jit
def euler2R(ang):
    '''
    euler to rotation matrix.ta
    :param ang:
    :return:
    '''

    cr = math.cos(ang[0])
    sr = math.sin(ang[0])

    cp = math.cos(ang[1])
    sp = math.sin(ang[1])

    cy = math.cos(ang[2])
    sy = math.sin(ang[2])

    R = np.zeros([3, 3])
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


# @jit
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
    # q_norm = 0.0
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


@jit(float64[:, :](float64[:]), nopython=True)
def q2dcm(q_in):
    """
    :param q:
    :return:
    """
    # p = np.zeros([6, 1])
    # p = cuda.local.array(shape=(6), dtype=float64)

    # p[0:4] = q.reshape(4, 1) ** 2.0
    # q_norm = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
    # q_norm = math.sqrt(q_norm)
    # for i in range(4):
    #     q[i] = q[i] / q_norm
    # q = np.asarray((q_in[1], q_in[2], q_in[3], q_in[0]))
    q = np.zeros(4)
    # q = np.zeros([4])
    q[0] = q_in[1]
    q[1] = q_in[2]
    q[2] = q_in[3]
    q[3] = q_in[0]

    q = q / np.linalg.norm(q)

    p = np.zeros(6)

    # for i in range(4):
    #     p[i] = q[i] * q[i]
    # qdq = np.vectorize(lambda x:x*x)
    # p[0:4] = qdq(q)
    # p[0:4] = q * q
    for i in range(4):
        p[i] = q[i] * q[i]

    p[4] = p[1] + p[2]

    if (math.fabs(p[0] + p[3] + p[4]) > 1e-25):
        p[5] = 2.0 / (p[0] + p[3] + p[4])
    else:
        p[5] = 0.0

    R = np.zeros(shape=(3, 3))

    R[0, 0] = 1.0 - p[5] * p[4]
    R[1, 1] = 1.0 - p[5] * (p[0] + p[2])
    R[2, 2] = 1.0 - p[5] * (p[0] + p[1])

    p[0] = p[5] * q[0]
    p[1] = p[5] * q[1]
    p[4] = p[5] * q[2] * q[3]
    p[5] = p[0] * q[1]

    R[0, 1] = p[5] - p[4]
    R[1, 0] = p[5] + p[4]

    p[4] = p[1] * q[3]
    p[5] = p[0] * q[2]

    R[0, 2] = p[5] + p[4]
    R[2, 0] = p[5] - p[4]

    p[4] = p[0] * q[3]
    p[5] = p[1] * q[2]

    R[1, 2] = p[5] - p[4]
    R[2, 1] = p[5] + p[4]

    return R


@jit(nopython=True, cache=True)
def dcm2euler(R):
    euler = np.zeros(3)
    euler[0] = math.atan2(R[2, 1], R[2, 2])
    euler[1] = math.atan2(R[2, 0], math.sqrt(1.0 - R[2, 0] * R[2, 0]))
    euler[2] = math.atan2(R[1, 0], R[0, 0])
    return euler


# @jit
def get_initial_rotation(imu_data,
                         initial_yaw):
    '''
    keep acc_z of static stage equal to norm of acc.
    :param imu_data: imu data contain at list ten times of measurements(recommend)
    :param initial_yaw: initial orientation
    :return: euler angle in three axis, quaternion represented the rotation.
    '''
    acc = imu_data.mean(axis=0)

    # @jit(nopython=True)
    def error_func(w: np.ndarray) -> float:
        # q = euler.euler2quat(w[0], w[1], initial_yaw)
        R = euler2R(np.asarray((w[0], w[1], initial_yaw)))
        rotated_acc = R.dot(acc)
        return np.linalg.norm(rotated_acc - np.asarray((0.0, 0.0, np.linalg.norm(acc))))

    result = minimize(error_func, np.asarray((0, 0)))
    # print(result)
    w = result.x
    euler = np.asarray((w[0], w[1], initial_yaw))
    q = dcm2q(euler2R(euler))

    print('acc:', acc, 'rotated acc:', euler2R(euler).dot(acc))

    return euler, q


# return state
def imu_plot_aux(data, time=None, title_name='default'):
    '''
    plot data at each col.
    :param data:
    :param time: time stamp of each row.
    :param title_name:
    :return:
    '''
    plt.figure()
    plt.title(title_name)

    if time is None:
        for i in range(1, 4):
            plt.plot(data[:, 0], data[:, i], label=str(i))
    else:
        for i in range(data.shape[1]):
            plt.plot(time, data[:, i], label=str(i))

    plt.grid()
    plt.legend()


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
