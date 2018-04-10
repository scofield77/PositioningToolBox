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