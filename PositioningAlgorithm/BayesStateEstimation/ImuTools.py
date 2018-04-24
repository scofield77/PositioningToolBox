# -*- coding:utf-8 -*-
# carete by steve at  2018 / 04 / 23　下午9:29
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

from matplotlib import pyplot as plt

import math
from numba import jit


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

    euler = euler * 0.5 * rate
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


@jit
def q2dcm(q):
    """
    :param q:
    :return:
    """
    # p = np.zeros([6, 1])
    # p = cuda.local.array(shape=(6), dtype=float64)

    # p[0:4] = q.reshape(4, 1) ** 2.0
    q_norm = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
    q_norm = math.sqrt(q_norm)
    for i in range(4):
        q[i] = q[i] / q_norm
    R = np.zeros([3, 3])

    # p[1] = q[1] * q[1]
    # p[2] = q[2] * q[2]
    # p[3] = q[3] * q[3]
    # p[0] = q[0] * q[0]
    #
    # p[4] = p[1] + p[2]
    #
    # if math.fabs(p[0] + p[3] + p[4]) > 1e-18:
    #     p[5] = 2.0 / (p[0] + p[3] + p[4])
    # else:
    #     p[5] = 0.0
    #
    # # R = np.zeros([3, 3])
    #
    # R[0, 0] = 1.0 - p[5] * p[4]
    # R[1, 1] = 1.0 - p[5] * (p[0] + p[2])
    # R[2, 2] = 1.0 - p[5] * (p[0] + p[1])
    #
    # p[0] = p[5] * q[0]
    # p[1] = p[5] * q[1]
    # p[4] = p[5] * q[2] * q[3]
    # p[5] = p[0] * q[1]
    #
    # R[0, 1] = p[5] - p[4]
    # R[1, 0] = p[5] + p[4]
    #
    # p[4] = p[1] * q[3]
    # p[5] = p[0] * q[2]
    #
    # R[0, 2] = p[5] + p[4]
    # R[2, 0] = p[5] - p[4]
    #
    # p[4] = p[0] * q[3]
    # p[5] = p[1] * q[2]
    #
    # R[1, 2] = p[5] - p[4]
    # R[2, 1] = p[5] + p[4]
    R[0, 0] = (q[0] * q[0] + q[1] * q[1] - 0.5) * 2.0
    R[0, 1] = (q[1] * q[2] - q[0] * q[3]) * 2.0
    R[0, 2] = (q[0] * q[2] + q[1] * q[3]) * 2.0

    R[1, 0] = (q[0] * q[3] + q[1] * q[2]) * 2.0
    R[1, 1] = (q[0] * q[0] + q[2] * q[2] - 0.5) * 2.0
    R[1, 2] = (q[2] * q[3] - q[0] * q[1]) * 2.0

    R[2, 0] = (q[1] * q[3] - q[0] * q[2]) * 2.0
    R[2, 1] = (q[0] * q[1] + q[2] * q[3]) * 2.0
    R[2, 2] = (q[0] * q[0] + q[3] * q[3] - 0.5) * 2.0
    return R


@jit
def dcm2euler(R):
    euler = np.zeros(3)
    euler[0] = math.atan2(R[2, 1], R[2, 2])
    euler[1] = math.atan2(R[2, 0], math.sqrt(1.0 - R[2, 0] * R[2, 0]))
    euler[2] = math.atan2(R[1, 0], R[0, 0])
    return euler


if __name__ == '__main__':
    '''
    Test all alogorithm in this file.
    '''

    # quaternion right update

    step_len = 1.0 * np.pi / 180.0

    # qx = np.zeros([4])
    # qy = np.zeros([4])
    # qz = np.zeros([4])
    # qx[0] = 1.0
    # qy[0] = 1.0
    # qz[0] = 1.0
    q_all = np.zeros([4, 3])
    euler_all = np.zeros([3, 3])
    q_all[0, :] += 1.0

    counter = math.floor(6.0 * np.pi / step_len)
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
