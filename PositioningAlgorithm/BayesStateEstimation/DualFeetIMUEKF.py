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
        self.l_ekf = ImuEKFComplex(initial_prob,local_g,time_interval)
        self.r_ekf = ImuEKFComplex(initial_prob,local_g,time_interval)


        self.whole_P = np.zeros([self.l_ekf.prob_state.shape[0]+self.r_ekf.prob_state.shape[0],
                                 self.l_ekf.prob_state.shape[1]+self.r_ekf.prob_state.shape[1]])

        first_index = self.l_ekf.state.shape[0]

        self.whole_P[0:first_index,0:first_index] = 1.0 * self.l_ekf.prob_state
        self.whole_P[first_index:,first_index:] = 1.0 * self.r_ekf.prob_state


    def initial_state(self, left_imu_data: np.ndarray,
                      right_imu_data: np.ndarray,
                      left_pos=np.asarray((0.0, 0.0, 0.0)),
                      right_pos=np.asarray((0.0, 0.0, 0.0)),
                      ori: float = 0.0,
                      mag=np.asarray9(1.0, 0.0, 0.0)):
        '''

        :param left_imu_data:
        :param right_imu_data:
        :param left_pos:
        :param right_pos:
        :param ori:
        :param mag:
        :return:
        '''
        assert left_imu_data.shape[0] > 10 and left_imu_data.shape[1] > 6
        assert right_imu_data.shape[0] > 10 and right_imu_data.shape[1] > 6


        # if np.linalg.norm(self.state)









