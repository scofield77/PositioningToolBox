# -*- coding:utf-8 -*-
# carete by steve at  2018 / 05 / 06　下午4:48
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


class UwbRangeEKF:
    def __init__(self, sigma, beacon_set):
        self.m = np.asarray((-10000.0))
        self.cov = np.asarray((sigma)).reshape(-1)
        self.measurement = list()
        self.beacon_set = beacon_set * 1.0

        self.last_pose = np.asarray((0.0, 0.0, 0.0))
        self.last_pose_prob = np.identity(3)
        self.pos_list = list()
        self.pose_prob_list = list()

        self.last_eta = list()

    def initial_pose(self, m, pose, pose_prob):
        self.m = np.asarray((m))
        self.last_pose = pose
        self.last_pose_prob = pose_prob

    def state_transmition(self, pose, pose_prob):
        # dv = pose-self.last_pose
        dm = np.linalg.norm(pose - self.beacon_set) - np.linalg.norm(self.last_pose - self.beacon_set)

        self.m = self.m + np.asarray((dm))
        if self.m < 0.0:
            print(self.m, self.last_pose, pose, dm)

        G = np.zeros(shape=(1, 6))
        G[0, 0:3] = 2.0 * (pose - self.beacon_set).reshape(1,-1)
        G[0, 3:6] = -2.0 * (self.last_pose - self.beacon_set).reshape(1,-1)

        P = np.zeros(shape=[6, 6])
        P[0:3, 0:3] = self.last_pose_prob * 1.0
        P[3:6, 3:6] = pose_prob * 1.0

        self.cov[0] = self.cov[0] + (
            (G).dot(P)).dot(np.transpose(G))

        self.last_pose_prob = pose_prob * 1.0
        self.last_pose = pose * 1.0

    def measurement_func(self, measurement, cov_m, ka_squard=10.0, T_d=15.0):
        z = np.asarray((measurement))

        y = self.m

        self.H = np.ones(shape=(1, 1))

        R_k = np.asarray((cov_m))

        P_v = (self.H.dot(self.cov)).dot(np.transpose(self.H)) + R_k
        v_k = z - y
        eta_k = np.zeros(1)
        self.last_eta.append(eta_k[0])
        # print(v_k)
        # if v_k[0] > 0.5:
        #     return

        robust_loop_flag = True
        first_time = True
        while robust_loop_flag:
            robust_loop_flag = False

            P_v = (self.H.dot(self.cov)).dot(np.transpose(self.H)) + R_k

            eta_k[0] = v_k*v_k/P_v
            # print(eta_k[0])
            #
            # if eta_k[0] > 1.0:
            #     return
            if first_time:
                self.last_eta.append(eta_k[0])
                first_time = False
            if (eta_k[0] > ka_squard):
                # if first_time:
                #
                #     first_time=False
                # else:
                self.last_eta[-1] = eta_k[0]

                # np.std()
                serial_length = 5
                if len(self.last_eta) > serial_length:
                    lambda_k = np.std(np.asarray(self.last_eta[-serial_length:]))
                    # print(self.uwb_eta_dict[beacon_id][-serial_length:],lambda_k, R_k[0])
                    if lambda_k > T_d:
                        robust_loop_flag = True
                        R_k = eta_k / ka_squard * R_k
                # self.uwb_eta_dict[beacon_id].pop()

        cov_m = np.asarray((R_k))
        # print('-------------')

        self.m = np.asarray((self.m))

        self.K = (self.m.dot(np.transpose(self.H))).dot(
            np.linalg.inv((self.H.dot(self.m)).dot(np.transpose(self.H)) + cov_m)
        )

        dx = self.K.dot(z - y)

        # self.state = self.state + dx
        self.m = self.m + dx

        kh = self.K.dot(self.H)
        self.cov = (np.identity(kh.shape[0]) - kh).dot(self.m)
