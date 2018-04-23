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
def quaternion_add_euler(q, euler, rate):
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
        tq[i] = q[i]
    norm_new_q = 0.0
    for i in range(4):
        q[i] = 0.0
        for j in range(4):
            q[i] += Theta[i, j] * tq[j]
        norm_new_q += q[i] * q[i]
    norm_new_q = math.sqrt(norm_new_q)

    for i in range(4):
        q[i] = q[i] / norm_new_q

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
    q_all = np.zeros([4,3])

    counter = math.floor(4.0 * np.pi / step_len)
    result_q_all = np.zeros([4,3,counter])
    for i in range(result_q_all.shape[2]):
        for k in range(3):
            euler = np.zeros([3])
            euler[k] = step_len



