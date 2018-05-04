# -*- coding:utf-8 -*-
# carete by steve at  2018 / 05 / 03　下午9:51
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
from scipy import interpolate
import matplotlib.pyplot as plt

from numba import jit

import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import time

from PositioningAlgorithm.OptimizationAlgorithm.UwbOptimizeLocation import UwbOptimizeLocation

if __name__ == '__main__':
    matplotlib.use('Qt5Agg')
    # matplotlib.rcParams['toolbar'] = 'toolmanager'
    start_time = time.time()
    # dir_name = '/home/steve/Data/FusingLocationData/0017/'
    # dir_name = '/home/steve/Data/FusingLocationData/0013/'
    dir_name = '/home/steve/Data/NewFusingLocationData/0032/'

    # uwb_data = np.loadtxt(dir_name + 'uwb_result.csv', delimiter=',')
    # beacon_set = np.loadtxt(dir_name + 'beaconSet.csv', delimiter=',')
    uwb_data = np.loadtxt(dir_name + 'uwb_data.csv', delimiter=',')
    beacon_set = np.loadtxt(dir_name + 'beaconset_no_mac.csv', delimiter=',')

    uol = UwbOptimizeLocation(beacon_set)
    uwb_trace = np.zeros([uwb_data.shape[0], 3])
    uwb_opt_res = np.zeros([uwb_data.shape[0]])

    uwb_trace_r = np.zeros([uwb_data.shape[0], 3])
    uwb_opt_res_r = np.zeros([uwb_data.shape[0]])

    for i in range(uwb_data.shape[0]):
        if i is 0:
            uwb_trace[i, :], uwb_opt_res[i] = \
                uol.positioning_function((0, 0, 0),
                                         uwb_data[i, 1:])
            uwb_trace_r[i, :], uwb_opt_res_r[i] = \
                uol.positioning_function_robust((0, 0, 0),
                                                uwb_data[i, 1:])
        else:
            uwb_trace[i, :], uwb_opt_res[i] = \
                uol.positioning_function(uwb_trace[i - 1, :],
                                         uwb_data[i, 1:])
            uwb_trace_r[i, :], uwb_opt_res_r[i] = \
                uol.positioning_function_robust(uwb_trace_r[i - 1, :],
                                                uwb_data[i, 1:])

    plt.figure()
    plt.title('trace 2d')
    plt.plot(uwb_trace[:, 0], uwb_trace[:, 1], label='uwb')
    plt.plot(uwb_trace_r[:, 0], uwb_trace_r[:, 1], label='uwb r')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.title('res')
    plt.plot(uwb_opt_res,label= 'uwb')
    plt.plot(uwb_opt_res_r, label='uwb r')
    plt.grid()
    plt.legend()

    fig = plt.figure()
    # plt.title('trace 3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('trace 3d')

    ax.plot(uwb_trace[:, 0], uwb_trace[:, 1], uwb_trace[:, 2], label='uwb')
    ax.plot(uwb_trace_r[:, 0], uwb_trace_r[:, 1], uwb_trace_r[:, 2], label='uwb r')
    ax.grid()
    ax.legend()

    plt.show()
