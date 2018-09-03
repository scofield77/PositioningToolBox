# -*- coding:utf-8 -*-
# carete by steve at  2018 / 09 / 03　上午10:21
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
from numba import jit

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

# from mayavi import mlab

if __name__ == '__main__':
    import mkl

    mkl.set_num_threads(6)
    # print(np.show_config())
    # print(mk)
    matplotlib.use('Qt5Agg')
    # matplotlib.rcParams['toolbar'] = 'toolmanager'
    start_time = time.time()
    # dir_name = '/home/steve/Data/FusingLocationData/0017/'
    dir_name = '/home/steve/Data/NewFusingLocationData/0035/'

    # dir_name = 'D:\\NewFusingLocationData\\0035\\'

    left_imu_data = np.loadtxt(dir_name + 'LEFT_FOOT.data', delimiter=',')
    right_imu_data = np.loadtxt(dir_name + 'RIGHT_FOOT.data', delimiter=',')

    left_imu_data = left_imu_data[:, 1:]
    left_imu_data[:, 1:4] = left_imu_data[:, 1:4] * 9.81
    left_imu_data[:, 4:7] = left_imu_data[:, 4:7] * (np.pi / 180.0)

    right_imu_data = right_imu_data[:, 1:]
    right_imu_data[:, 1:4] = right_imu_data[:, 1:4] * 9.81
    right_imu_data[:, 4:7] = right_imu_data[:, 4:7] * (np.pi / 180.0)

    t_phone_imu_data = np.loadtxt(dir_name + 'HAND_SMARTPHONE_IMU.data', delimiter=',')
    # phone_imu_data = phone_imu_data[:,1:]
    phone_imu_data = np.zeros([t_phone_imu_data.shape[0], t_phone_imu_data.shape[1] - 1])
    phone_imu_data[:, 0] = (t_phone_imu_data[:, 0] - t_phone_imu_data[0, 0]) * 1e-6 + t_phone_imu_data[
        0, 1]  # left_imu_data[0,0]
    phone_imu_data[:, 1:] = t_phone_imu_data[:, 2:]

    print(phone_imu_data[-1, 0] - phone_imu_data[0, 0])
    print(left_imu_data[-1, 0] - left_imu_data[0, 0])

    l_zv = GLRT_Detector(left_imu_data[:, 1:7], sigma_a=1.0, sigma_g=1.0 * np.pi / 180.0,
                         gamma=300,
                         gravity=9.8,
                         time_Window_size=5)
    r_zv = GLRT_Detector(right_imu_data[:, 1:7], sigma_a=1.0, sigma_g=1.0 * np.pi / 180.0,
                         gamma=300,
                         gravity=9.8,
                         time_Window_size=5)

    plt.figure()
    plt.subplot(211)
    # for i in range(1,4):
    #     plt.plot(phone_imu_data[:,0],phone_imu_data[:,i],'-+')
    plt.plot(phone_imu_data[:, 0], np.linalg.norm(phone_imu_data[:, 1:4], axis=1) / 10.0)
    plt.plot(left_imu_data[:, 0], l_zv, '--y')
    plt.plot(right_imu_data[:, 0], r_zv, '--r')

    plt.subplot(212)
    for i in range(1, 4):
        plt.plot(left_imu_data[:, 0], left_imu_data[:, i])

    plt.show()
