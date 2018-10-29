# -*- coding:utf-8 -*-
# carete by steve at  2018 / 10 / 29　7:20 PM
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
    dir_name = '/home/steve/Data/PDR/0013/'

    left_imu = imu_data_preprocess(np.loadtxt(dir_name + 'LEFT_FOOT.data', delimiter=','))
    right_imu = imu_data_preprocess(np.loadtxt(dir_name + 'RIGHT_FOOT.data', delimiter=','))
    head_imu = imu_data_preprocess(np.loadtxt(dir_name + 'HEAD.data', delimiter=','))
    phone_imu = np.loadtxt(dir_name + 'SMARTPHONE2_IMU.data', delimiter=',')[:, 1:]
    # process time interval
    phone_imu_average_time_interval = (phone_imu[-1, 0] - phone_imu[0, 0]) / float(phone_imu.shape[0])
    for i in range(1, phone_imu.shape[0]):
        phone_imu[i, 0] = phone_imu[i - 1, 0] + phone_imu_average_time_interval
    # phone_imu = head_imu * 1.0

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

    from array import array

    flag_array = array('d')

    li = 1
    ri = 1
    hi = 1

    flag_list = list()
    flag_list.append(0)

    last_time = -1.0

    # Find stage based on foot imu.
    while li < left_imu.shape[0] and \
            ri < right_imu.shape[0] and \
            hi < head_imu.shape[0]:
        pre_diff = left_zv_state[li - 1] - right_zv_state[ri - 1]
        diff = left_zv_state[li] - right_zv_state[ri]

        if pre_diff * diff < 0.0:
            # print('----------',head_imu[hi,0]-last_time)

            if head_imu[hi, 0] - last_time > 0.2:
                flag_array.append(head_imu[hi, 0])
                flag_array.append(15.0)
                flag_list.append(hi)
                last_time = head_imu[hi, 0]

        if left_imu[li, 0] < right_imu[ri, 0]:
            li += 1
        else:
            ri += 1

        if head_imu[hi, 0] < right_imu[ri, 0] and \
                head_imu[hi, 0] < left_imu[li, 0]:
            hi += 1
            # print(hi)
            # if pre_diff * diff < 0.0:
            #     # print('----------',head_imu[hi,0]-last_time)
            #
            #     if head_imu[hi, 0] - last_time > 0.005:
            #         flag_array.append(head_imu[hi, 0])
            #         flag_array.append(15.0)
            #         flag_list.append(hi)
            #         last_time = head_imu[hi, 0]
    tmp_flag = np.frombuffer(flag_array, dtype=np.float).reshape([-1, 2])

    print('length of hi flag:', len(flag_list))
    change_flag_array = np.zeros([head_imu.shape[0], 2])
    change_flag_array[:, 0] = head_imu[:, 0]

    # # Find peak... and max value (ERROR)
    for i in range(0, len(flag_list)-1):
        pre_i = flag_list[i - 1] + 1
        last_i = flag_list[i] + 1

        max_v = -100000.0
        max_index = pre_i - 1
        # print(last_i-pre_i)
        k = pre_i-30
        # for k in range(pre_i, last_i):
        while k < pre_i+30:
            if k > 0 and k < head_imu.shape[0]:
                break

            if np.linalg.norm(head_imu[k, 1:4]) > max_v :#and \
                    # np.linalg.norm(head_imu[k, 1:4]) >= np.linalg.norm(head_imu[k - 1, 1:4]) and \
                    # np.linalg.norm(head_imu[k, 1:4]) >= np.linalg.norm(head_imu[k + 1, 1:4]):
                max_v = np.linalg.norm(head_imu[k, 1:4])
                max_index = k
            k += 1

            # print(k)

        change_flag_array[max_index, 1] = max_v

    # def is_peak(data, k):
    #     if data[k] > data[k-1] and data[k] > data[k+1]:
    #         return True
    #     else:
    #         return False
    #
    # norm_value = np.linalg.norm(head_imu[:,1:4],axis=1)
    # for i in range(1, len(flag_list)):
    #     pre_i = flag_list[i-1]
    #     last_i = flag_list[i]
    #
    #     for k in range(last_i-pre_i):
    #         if is_peak(norm_value, pre_i+k):
    #             change_flag_array[pre_i+k,1] = norm_value[k]
    #             break
    #         if is_peak(norm_value,pre_i-k):
    #             change_flag_array[pre_i-k,1] = norm_value[k]
    #             break





    plt.figure()
    plt.plot(head_imu[:, 0], np.linalg.norm(head_imu[:, 1:4], axis=1), label='head norm')
    plt.plot(change_flag_array[:, 0], change_flag_array[:, 1], label='change')
    plt.plot(tmp_flag[:, 0], tmp_flag[:, 1], '*', label='tmp flag')

    plt.legend()
    plt.grid()

    plt.ylim([np.min(np.linalg.norm(head_imu[:, 1:4], axis=1)) - 10.0,
              np.max(np.linalg.norm(head_imu[:, 1:4], axis=1)) + 10.0])

    plt.show()
