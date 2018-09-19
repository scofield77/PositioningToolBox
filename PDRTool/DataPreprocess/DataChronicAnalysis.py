# -*- coding:utf-8 -*-
# carete by steve at  2018 / 09 / 18　下午2:56
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

from AlgorithmTool.ImuTools import imu_plot_aux


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
    # phone_imu =imu_data_preprocess(np.)
    phone_imu = np.loadtxt(dir_name + 'SMARTPHONE2_IMU.data', delimiter=',')[:, 1:]
    # phone_imu[:, 0] = phone_imu[0, 0] + (phone_imu[-1, 0] - phone_imu[0, 0]) / float(phone_imu.shape[0])
    phone_imu_average_time_interval = (phone_imu[-1, 0] - phone_imu[0, 0]) / float(phone_imu.shape[0])
    for i in range(1,phone_imu.shape[0]):
        phone_imu[i,0] = phone_imu[i-1,0]+phone_imu_average_time_interval


    # plot data
    # plt.figure()
    # plt.plot(left_imu[:,1:4],'acc')
    # imu_plot_aux(left_imu[:, 1:4], time=left_imu[:, 0], title_name='acc')
    # imu_plot_aux(left_imu[:, 4:7], time=left_imu[:, 0], title_name='gyr')

    # imu_plot_aux(phone_imu[:,1:4],phone_imu[:,0],'acc')
    # imu_plot_aux(phone_imu[:,4:7],phone_imu[:,0],'gyr')

    # plot Time compare
    # plt.figure()
    # plt.title('time compare')
    # plt.plot(left_imu[:, 0], '+-', label='left')
    # plt.plot(right_imu[:, 0], '+-', label='right')
    # plt.plot(head_imu[:, 0], '+-', label='head')
    # plt.plot(phone_imu[:, 0], '+-', label='phone')
    #
    # plt.grid()
    # plt.legend()





    plt.show()
