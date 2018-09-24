# -*- coding:utf-8 -*-
# carete by steve at  2018 / 09 / 19　下午5:46
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

from AlgorithmTool.ImuTools import *
from PositioningAlgorithm.BayesStateEstimation.ImuEKF import ImuEKFComplex


def imu_data_preprocess(imu_data):
    '''

    :param imu_data:
    :return: time(s),acc x-y-z(m/s/s), gyr x-y-z (rad/s), mag(?)
    '''
    imu_data = imu_data[:, 1:]
    imu_data[:, 1:4] = imu_data[:, 1:4] * 9.81
    imu_data[:, 4:7] = imu_data[:, 4:7] * (np.pi / 180.0)

    return imu_data * 1.0  #


if __name__ == '__main__':
    dir_name = '/home/steve/Data/PDR/0001/'

    left_imu = imu_data_preprocess(np.loadtxt(dir_name + 'LEFT_FOOT.data', delimiter=','))
    right_imu = imu_data_preprocess(np.loadtxt(dir_name + 'RIGHT_FOOT.data', delimiter=','))
    head_imu = imu_data_preprocess(np.loadtxt(dir_name + 'HEAD.data', delimiter=','))
    phone_imu = np.loadtxt(dir_name + 'SMARTPHONE2_IMU.data', delimiter=',')[:, 1:]
    # process time interval
    phone_imu_average_time_interval = (phone_imu[-1, 0] - phone_imu[0, 0]) / float(phone_imu.shape[0])
    for i in range(1, phone_imu.shape[0]):
        phone_imu[i, 0] = phone_imu[i - 1, 0] + phone_imu_average_time_interval

    initial_orientation = 80.0 * np.pi / 180.0  # 40
    kf = ImuEKFComplex(np.diag((
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001 * np.pi / 180.0, 0.001 * np.pi / 180.0, 0.001 * np.pi / 180.0,
        0.0001,
        0.0001,
        0.0001,
        0.0001 * np.pi / 180.0,
        0.0001 * np.pi / 180.0,
        0.0001 * np.pi / 180.0
    )),
        local_g=-9.81, time_interval=(left_imu[-1, 0] - left_imu[0, 0]) / float(left_imu.shape[0]))
    print('average time interval:', (left_imu[-1, 0] - left_imu[0, 0]) / float(left_imu.shape[0]))

    kf.initial_state(left_imu[:50, 1:7],
                     mag=left_imu[0, 7:10]
                     )

    zv_state = GLRT_Detector(left_imu[:, 1:7], sigma_a=1.,
                             sigma_g=1. * np.pi / 180.0,
                             gamma=200,
                             gravity=9.8,
                             time_Window_size=5)

    # imu_plot_aux(left_imu[:,1:4],left_imu[:,1:4],'acc')
    plt.figure()
    # plt.plot(left_imu[:,0],zv_state,'+-')
    plt.plot(np.linalg.norm(left_imu[:, 1:4], axis=1))
    plt.plot(zv_state * 10.0, 'r-')

    pos = np.zeros([left_imu.shape[0], 3])

    for i in range(left_imu.shape[0]):
        kf.state_transaction_function(left_imu[i, 1:7], np.diag((0.01, 0.01, 0.01,
                                                                 0.01 * np.pi / 180.0,
                                                                 0.01 * np.pi / 180.0,
                                                                 0.01 * np.pi / 180.0)))
        if zv_state[i] > 0.5:
            kf.measurement_function_zv(np.asarray((0, 0, 0)),
                                       np.diag((0.0001, 0.0001, 0.0001)))
        pos[i, :] = kf.state[:3]

    plt.figure()
    plt.plot(pos[:, 0], pos[:, 1], '-+')
    plt.grid()
    plt.savefig(dir_name + 'ref_trace.png')

    new_trace = np.zeros([pos.shape[0], 4])
    new_trace[:, 0] = left_imu[:, 0]
    new_trace[:, 1:] = pos
    np.savetxt(dir_name + '/ref_trace_with_time.csv', new_trace, delimiter=',')
    np.savetxt(dir_name + '/chronic_phone_data.csv', phone_imu, delimiter=',')

    plt.show()
