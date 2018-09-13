# -*- coding:utf-8 -*-
# carete by steve at  2018 / 09 / 09　下午2:40
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

import matplotlib.pyplot as plt

import numpy as np
from numpy.linalg import *
import scipy as sp

from AlgorithmTool.ImuTools import *

if __name__ == '__main__':
    data = np.loadtxt('/home/steve/Data/phoneData/PDRUWBBLEMini/0006/SMARTPHONE3_IMU.data', delimiter=',')
    # step_detector = StepDetector(1.0, 0.8)
    # step_estimator = StepLengthEstimatorV()

    t = data[:, 0] - data[0, 0]
    data = data[:, 1:]
    data[:, 0] = t * 1.0

    acc = np.zeros([data.shape[0], 4])
    acc[:, 0] = data[:, 0]
    acc[:, 1:] = data[:, 2:5]
    gyr = np.zeros([data.shape[0], 4])
    mag = np.zeros([data.shape[0], 4])
    ori = np.zeros([data.shape[0], 4])

    gyr[:, 0] = data[:, 0]
    gyr[:, 1:] = data[:, 5:8]

    mag[:, 0] = data[:, 0]
    mag[:, 1:] = data[:, 8:11]

    ori[:, 0] = data[:, 0]
    ori[:, 1:] = data[:, 11:14]

    # imu_plot_aux()
    imu_plot_aux(acc, 'acc')

    # def peak_and_valley_detector(acc,rotation):
    flag_array = np.zeros(acc.shape[0])
    step_array = np.zeros(acc.shape[0])
    # low_pass_array[0] = np.linalg.norm(acc[0,1:])
    low_pass_alpha_list = [0.002, 0.01, 0.2, 0.5, 0.8]
    low_pass_alpha_list = [0.0001, 0.0005, 0.001]  # , 0.002, 0.01]# 0.2, 0.5, 0.8]
    low_pass_array = np.zeros([acc.shape[0], len(low_pass_alpha_list)])
    low_pass_array[0, :] = 0.0 + np.linalg.norm(acc[0, 1:])

    acc_std = np.std(np.linalg.norm(acc[:, 1:], axis=1))
    print('acc std:', acc_std)

    last_peak = 0
    last_valley = 0
    time_interval = 0.2
    small_time_interval = 0.1

    for i in range(1, flag_array.shape[0] - 1):
        for j, low_pass_alpha in enumerate(low_pass_alpha_list):
            low_pass_array[i, j] = (1.0 - low_pass_alpha) * low_pass_array[i - 1, j] + (
                low_pass_alpha) * np.linalg.norm(
                acc[i, 1:])

        ta = 0.3
        acc[i, 1:] = ta * acc[i, 1:] + (1.0 - ta) * acc[i - 1, 1:]

        if abs(np.linalg.norm(acc[i, 1:]) - low_pass_array[i, 0]) > acc_std * 1.0:
            if np.linalg.norm(acc[i, 1:]) > max(np.linalg.norm(acc[i - 1, 1:]), np.linalg.norm(acc[i + 1, 1:])) and \
                    acc[i, 0] - acc[last_peak, 0] > time_interval and acc[i, 0] - acc[
                last_valley, 0] > small_time_interval and last_peak <= last_valley:
                flag_array[i] = np.linalg.norm(acc[i, 1:])
                last_peak = i
            elif np.linalg.norm(acc[i, 1:]) < min(np.linalg.norm(acc[i - 1, 1:]), np.linalg.norm(acc[i + 1, 1:])) \
                    and acc[i, 0] - acc[last_valley, 0] > time_interval and acc[i, 0] - acc[
                last_peak, 0] > small_time_interval and last_valley < last_peak:
                flag_array[i] = np.linalg.norm(acc[i, 1:])
                step_array[i] = np.linalg.norm(acc[last_peak,1:])
                last_valley = i

    plt.plot(acc[:, 0], flag_array, '+')
    plt.plot(acc[:,0],step_array,'-+',label='step detector')
    # plt.plot(mag[:,0],np.linalg.norm(mag[:,1:],axis=1)/5.0)
    # plt.plot(gyr[:, 0], np.linalg.norm(gyr[:, 1:], axis=1) / 2.5)
    plt.legend()

    # plt.plot(acc[:,0],low_pass_array,'--',label='low_pass')
    plt.figure()
    for j in range(len(low_pass_alpha_list)):
        plt.plot(low_pass_array[:, j], label=str(low_pass_alpha_list[j]))
    plt.legend()

    plt.show()
