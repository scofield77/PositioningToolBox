# -*- coding:utf-8 -*-
# carete by steve at  2018 / 10 / 31　5:05 PM
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
from PositioningAlgorithm.BayesStateEstimation.DualFeetImu import *


class MadgwickFilter():
    def __init__(self, initial_ori, initial_gcc):
        self.initial_ori = initial_ori
        self.initial_gcc = initial_gcc


if __name__ == '__main__':
    print('dual feet test')

    import math
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib

    # from transforms3d import euler, quaternions

    from PositioningAlgorithm.BayesStateEstimation.KalmanFIlterBase import *

    from scipy.optimize import minimize

    from AlgorithmTool.ImuTools import *

    from PositioningAlgorithm.BayesStateEstimation.ImuEKF import *
    # from gr import pygr

    # from AlgorithmTool
    import time

    import mkl

    mkl.set_num_threads(6)
    # print(np.show_config())
    # print(mk)
    matplotlib.use('Qt5Agg')
    # matplotlib.rcParams['toolbar'] = 'toolmanager'
    start_time = time.time()
    # dir_name = '/home/steve/Data/FusingLocationData/0017/'
    # dir_name = '/home/steve/Data/NewFusingLocationData/0036/'
    # dir_name = 'C:/Data/NewFusingLocationData/0039/'
    dir_name = '/home/steve/Data/PDR/0011/'
    # dir_name = 'D:\\NewFusingLocationData\\0035\\'

    left_imu_data = np.loadtxt(dir_name + 'LEFT_FOOT.data', delimiter=',')
    # imu_data = np.loadtxt(dir_name + 'RIGHT_FOOT.data', delimiter=',')
    left_imu_data = left_imu_data[:, 1:]
    left_imu_data[:, 1:4] = left_imu_data[:, 1:4] * 9.81
    left_imu_data[:, 4:7] = left_imu_data[:, 4:7] * (np.pi / 180.0)

    right_imu_data = np.loadtxt(dir_name + 'RIGHT_FOOT.data', delimiter=',')
    right_imu_data = right_imu_data[:, 1:]
    right_imu_data[:, 1:4] = right_imu_data[:, 1:4] * 9.81
    right_imu_data[:, 4:7] = right_imu_data[:, 4:7] * (np.pi / 180.0)
