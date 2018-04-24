# -*- coding:utf-8 -*-
# carete by steve at  2018 / 04 / 23　下午9:18
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

from transforms3d import euler, quaternions

from PositioningAlgorithm.BayesStateEstimation.KalmanFIlterBase import *

from scipy.optimize import minimize

from AlgorithmTool.ImuTools import *


# from AlgorithmTool


if __name__ == '__main__':
    dir_name = '/home/steve/Data/NewFusingLocationData/0013/'
    imu_data = np.loadtxt(dir_name + 'RIGHT_FOOT.data', delimiter=',')
    imu_data = imu_data[:, 1:]
    imu_data[:, 1:4] *= 9.81
    imu_data[:, 4:7] *= (np.pi / 180.0)

    initial_state = get_initial_state(imu_data[:40, 1:4], np.asarray((0, 0, 0)), 0.0, 9)

    trace = np.zeros([imu_data.shape[0], 3])
    zv_state = np.zeros([imu_data.shape[0], 1])

    kf = KalmanFilterBase(9)
    kf.state_x = initial_state
    kf.state_prob = np.diag((0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001))

    set_setting = settings()
    set_setting.sigma_a *= 5.0
    set_setting.sigma_g *= 5.0

    z_tester = zero_velocity_tester(set_setting)
    zv_state = z_tester.GLRT_Detector(imu_data[:, 1:7])

    for i in range(imu_data.shape[0]):


        kf.state_transaction_function(imu_data[i, 1:7],
                                      np.diag((0.01, 0.01, 0.01, 0.02, 0.02, 0.02)),
                                      inter_imu)
        if (i > 5) and (i < imu_data.shape[0] - 5):
            # print('i:',i)
            # zv_state[i] = z_tester.GLRT_Detector(imu_data[i - 4:i + 4, 1:8])
            if zv_state[i] > 0.5:
                kf.measurement_function(np.asarray((0, 0, 0)),
                                        np.diag((0.0001, 0.0001, 0.0001)),
                                        zero_velocity_measurement,
                                        update_function)

        # print(kf.state_x)
        # print( i /)
        trace[i, :] = kf.state_x[0:3]
        rate = i / imu_data.shape[0]

        print('finished:', rate * 100.0, "% ", i, imu_data.shape[0])


    def aux_plot(data: np.ndarray, name: str):
        plt.figure()
        plt.title(name)
        plt.plot(zv_state, 'r-', label='zv state')
        for i in range(data.shape[1]):
            plt.plot(data[:, i], label=str(i))
        plt.grid()
        plt.legend()


    aux_plot(imu_data[:, 1:4], 'acc')
    aux_plot(imu_data[:, 4:7], 'gyr')
    aux_plot(imu_data[:, 7:10], 'mag')
    aux_plot(trace, 'trace')

    plt.figure()
    plt.plot(trace[:, 0], trace[:, 1], '-+')
    plt.grid()

    plt.figure()
    # plt.show()