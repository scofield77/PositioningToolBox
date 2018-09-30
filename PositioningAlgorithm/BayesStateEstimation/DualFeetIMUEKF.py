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


class DualImuEKFComplex:
    def __init__(self, initial_prob, local_g=-9.8, time_interval=0.01):
        '''

        :param initial_prob:
        :param local_g:
        :param time_interval:
        '''
        self.l_q = np.zeros([4])
        self.state = np.zeros([30])
        self.prob_state = initial_prob
        self.local_g = local_g
        self.time_interval = time_interval

        self.F = np.zeros([30, 30])
        self.G_left = np.zeros([30, 6])
        self.G_right = np.zeros([30, 6])

    def initial_state(self, left_imu_data: np.ndarray,
                      right_imu_data: np.ndarray,
                      left_pos=np.asarray((0.0, 0.0, 0.0)),
                      right_pos=np.asarray((0.0, 0.0, 0.0)),
                      ori: float = 0.0,
                      mag=np.asarray9(1.0, 0.0, 0.0)):
        assert left_imu_data.shape[0] > 10 and left_imu_data.shape[1] > 6
        assert right_imu_data.shape[0] > 10 and right_imu_data.shape[1] > 6

        # if np.linalg.norm(self.state)
        self.state[6:9], self.l_q = get_initial_rotation(left_imu_data[:, 0:3], ori)
