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
    dir_name = '/home/steve/Data/FusingLocationData/0017/'
    # dir_name = '/home/steve/Data/FusingLocationData/0013/'

    imu_data = np.loadtxt(dir_name + 'RIGHT_FOOT.data', delimiter=',')
    imu_data = imu_data[:, 1:]
    imu_data[:, 1:4] =imu_data[:,1:4] *  9.81
    imu_data[:, 4:7] = imu_data[:,4:7] * (np.pi / 180.0)

    # initial_state = get_initial_state(imu_data[:40, 1:4], np.asarray((0, 0, 0)), 0.0, 9)

    trace = np.zeros([imu_data.shape[0], 3])
    vel = np.zeros([imu_data.shape[0], 3])
    ang = np.zeros([imu_data.shape[0], 3])
    ba = np.zeros([imu_data.shape[0], 3])
    bg = np.zeros([imu_data.shape[0], 3])

    iner_acc = np.zeros([imu_data.shape[0],3])

    zv_state = np.zeros([imu_data.shape[0]])

    # kf = KalmanFilterBase(9)
    # kf.state_x = initial_state
    # kf.state_prob = np.diag((0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001))

    average_time_interval = (imu_data[-1, 0] - imu_data[0, 0]) / float(imu_data.shape[0])
    print('average time interval ', average_time_interval)

    kf = ImuEKFComplex(np.diag((
        0.001, 0.001, 0.001,
        0.001, 0.001, 0.001,
        0.001 * np.pi / 180.0, 0.001 * np.pi / 180.0, 0.0001 * np.pi / 180.0,
        0.0001,
        0.0001,
        0.0001,
        0.0001 * np.pi / 180.0,
        0.0001 * np.pi / 180.0,
        0.0001 * np.pi / 180.0
    )),
        local_g=-9.81, time_interval=average_time_interval)

    kf.initial_state(imu_data[:50, 1:7])

    zv_state = GLRT_Detector(imu_data[:, 1:7],sigma_a=0.4,
                  sigma_g=0.4 * np.pi / 180.0,
                  gamma=280.0,
                  gravity=9.8,
                  time_Window_size=10)

    for i in range(imu_data.shape[0]):
        # print('i:',i)

        kf.state_transaction_function(imu_data[i, 1:7],
                                      np.diag((0.01, 0.01, 0.01,
                                               0.01 * np.pi / 180.0,
                                               0.01 * np.pi / 180.0,
                                               0.01 * np.pi / 180.0))
                                      )
        if (i > 5) and (i < imu_data.shape[0] - 5):
            # print('i:',i)
            # zv_state[i] = z_tester.GLRT_Detector(imu_data[i - 4:i + 4, 1:8])
            if zv_state[i] > 0.5:
                kf.measurement_function_zv(np.asarray((0, 0, 0)),
                                           np.diag((0.0001, 0.0001, 0.0001)))

        # print(kf.state_x)
        # print( i /)
        trace[i, :] = kf.state[0:3]
        vel[i, :] = kf.state[3:6]
        ang[i, :] = kf.state[6:9]
        ba[i, :] = kf.state[9:12]
        bg[i, :] = kf.state[12:15]
        rate = i / imu_data.shape[0]
        iner_acc[i,:] = kf.acc


        # print('finished:', rate * 100.0, "% ", i, imu_data.shape[0])


    end_time = time.time()
    print('totally time:',end_time-start_time,'data time:',imu_data[-1,0]-imu_data[0,0])
    def aux_plot(data: np.ndarray, name: str):
        plt.figure()
        plt.title(name)
        plt.plot(zv_state*data.max()*1.1, 'r-', label='zv state')
        for i in range(data.shape[1]):
            plt.plot(data[:, i], label=str(i))
        plt.grid()
        plt.legend()

    # #
    aux_plot(imu_data[:, 1:4], 'acc')
    aux_plot(imu_data[:, 4:7], 'gyr')
    aux_plot(imu_data[:, 7:10], 'mag')
    aux_plot(trace, 'trace')
    aux_plot(vel, 'vel')
    aux_plot(ang, 'ang')
    aux_plot(ba, 'ba')
    aux_plot(bg, 'bg')

    aux_plot(iner_acc,'inner acc')

    plt.figure()
    plt.plot(trace[:, 0], trace[:, 1], '-+')
    plt.grid()

    # plt.figure()
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot(trace[:,0],trace[:,1],trace[:,2],'-+',label='trace')
    ax.grid()
    ax.legend()
    plt.show()
