# -*- coding:utf-8 -*-
# carete by steve at  2018 / 05 / 02　下午4:22
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
    for i in range(uwb_data.shape[0]):
        if i is 0:
            uwb_trace[i, :], uwb_opt_res[i] = \
                uol.positioning_fucntion((0, 0, 0),
                                         uwb_data[i, 1:])
        else:
            uwb_trace[i, :], uwb_opt_res[i] = \
                uol.positioning_fucntion(uwb_trace[i - 1, :],
                                         uwb_data[i, 1:])

    ref_trace = np.loadtxt(dir_name + 'ref_trace.csv', delimiter=',')

    plt.figure()
    plt.plot(ref_trace[:, 1], ref_trace[:, 2], '-+', label='ref')
    plt.plot(uwb_trace[:, 0], uwb_trace[:, 1], '-+', label='uwb')
    plt.grid()

    print('uwb data:', uwb_data[:, 0].min(), uwb_data[:, 0].max())
    print('ref data:', ref_trace[:, 0].min(), ref_trace[:, 0].max())

    ref_range = np.zeros(shape=(ref_trace.shape[0], uwb_data.shape[1]))


    @jit
    def compute_ref_range(ref_range, beacon_set, ref_trace):

        for i in range(ref_trace.shape[0]):
            ref_range[i, 0] = ref_trace[i, 0]
            for j in range(1, uwb_data.shape[1]):
                ref_range[i, j] = np.linalg.norm(ref_trace[i, 1:] - beacon_set[j-1, :])
                if ref_range[i, j] > 1000.0:
                    ref_range[i, j] = 0.0


    compute_ref_range(ref_range, beacon_set, ref_trace)

    plt.figure()
    plt.title('uwb range')
    for i in range(1, ref_range.shape[1]):
        plt.plot(ref_range[:, 0], ref_range[:, i], label=str(i))
    plt.grid()
    plt.legend()

    plt.show()
