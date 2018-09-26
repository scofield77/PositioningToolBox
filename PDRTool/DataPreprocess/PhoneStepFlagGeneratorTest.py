# -*- coding:utf-8 -*-
# carete by steve at  2018 / 09 / 21　下午3:23
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

import argparse

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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
    dir_name = '/home/steve/Data/PDR/0003/'

    left_imu = imu_data_preprocess(np.loadtxt(dir_name + 'LEFT_FOOT.data', delimiter=','))
    right_imu = imu_data_preprocess(np.loadtxt(dir_name + 'RIGHT_FOOT.data', delimiter=','))
    head_imu = imu_data_preprocess(np.loadtxt(dir_name + 'HEAD.data', delimiter=','))
    phone_imu = np.loadtxt(dir_name + 'SMARTPHONE2_IMU.data', delimiter=',')[:, 1:]
    # process time interval
    phone_imu_average_time_interval = (phone_imu[-1, 0] - phone_imu[0, 0]) / float(phone_imu.shape[0])
    for i in range(1, phone_imu.shape[0]):
        phone_imu[i, 0] = phone_imu[i - 1, 0] + phone_imu_average_time_interval

    left_zv_state = GLRT_Detector_prob(left_imu[:, 1:7], sigma_a=1.,
                                       sigma_g=1. * np.pi / 180.0,
                                       gamma=300,
                                       gravity=9.8,
                                       time_Window_size=15)

    right_zv_state = GLRT_Detector_prob(right_imu[:, 1:7], sigma_a=1.,
                                        sigma_g=1. * np.pi / 180.0,
                                        gamma=300,
                                        gravity=9.8,
                                        time_Window_size=15)
    # plt.figure()
    # plt.plot(left_imu[:,0])
    # plt.plot(right_imu[:,0])
    # plt.plot(phone_imu[:,0])
    # plt.grid()
    from array import array

    flag_array = array('d')

    li = 1
    ri = 1

    while li < left_imu.shape[0] and ri < right_imu.shape[0]:
        # if left_zv_state[li]-right_zv_state[ri]>0 and
        pre_diff = left_zv_state[li - 1] - right_zv_state[ri - 1]
        diff = left_zv_state[li] - right_zv_state[ri]

        if pre_diff * diff < 0.0:
            flag_array.append(left_imu[li, 0])
            flag_array.append(15.0)

        if left_imu[li, 0] < right_imu[ri, 0]:
            li += 1
        else:
            ri += 1

    change_flag_array = np.frombuffer(flag_array, dtype=np.float).reshape([-1, 2])

    plt.figure()
    plt.plot(left_imu[:, 0], left_zv_state, label='left')
    plt.plot(right_imu[:, 0], right_zv_state, label='right')

    plt.legend()

    plt.figure()
    plt.title('foot imu AND step flag')
    plt.plot(left_imu[:, 0], np.linalg.norm(left_imu[:, 1:4], axis=1), label='left')
    plt.plot(right_imu[:, 0], np.linalg.norm(right_imu[:, 1:4], axis=1), label='right')
    plt.plot(change_flag_array[:, 0], change_flag_array[:, 1], '*', label='change')

    plt.grid()
    plt.legend()

    plt.figure()
    plt.title('phone imu AND step flag')
    plt.plot(phone_imu[:, 0], np.linalg.norm(phone_imu[:, 1:4], axis=1), label='phone')
    plt.plot(change_flag_array[:, 0], change_flag_array[:, 1], '*', label='change')

    plt.grid()
    plt.legend()

    flag_phone_array = np.zeros(phone_imu.shape[0])
    # t_index = 0
    # for i in range(flag_phone_array.shape[0]):
    # if change_flag_array[t_index]
    # if change_flag_array[t_index]<phone_imu[i,0]-0.02:
    #     t_index+=1
    full_flag_array = np.zeros([phone_imu.shape[0],2])
    full_flag_array[:,0] = phone_imu[:,0]*1.0

    for i in range(full_flag_array.shape[0]):
        full_flag_array[i,1] = (np.min(np.abs(full_flag_array[i,0]-change_flag_array[:,0])))
        # full_flag_array[i,1] = 1.0/(1.0+np.exp(-1.0*full_flag_array[i,1]))
    sigma = 0.01
    full_flag_array[:,1] = 1.0/sigma / np.sqrt(2.0 * np.pi) * \
                           np.exp(-1.0*(full_flag_array[:,1]*full_flag_array[:,1])/2.0/sigma/sigma)




    # f_change = interp1d(change_flag_array[:, 0], change_flag_array[:, 1])

    plt.figure()
    # plt.subplot(211)
    plt.plot(full_flag_array[:,0],full_flag_array[:,1],'-+')
    plt.grid()
    np.savetxt(dir_name+'flag_array.csv', full_flag_array[:,1],delimiter=',')

    # plt.subplot(212)
    plt.plot(phone_imu[:,0],np.linalg.norm(phone_imu[:,1:4],axis=1))


    plt.show()
