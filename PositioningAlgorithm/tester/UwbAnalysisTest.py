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

    for i in range(1, uwb_data.shape[1]):
        # plt.plot(ref_range[:, 0], ref_range[:, i], label=str(i))
        if np.max(uwb_data[:, i] > 0.0) and beacon_set[i - 1, 0] < 5000.0:
            plt.text(beacon_set[i-1,0],beacon_set[i-1,1],str(i))

    plt.grid()

    print('uwb data:', uwb_data[:, 0].min(), uwb_data[:, 0].max())
    print('ref data:', ref_trace[:, 0].min(), ref_trace[:, 0].max())

    ref_range = np.zeros(shape=(ref_trace.shape[0], uwb_data.shape[1]))


    @jit
    def compute_ref_range(ref_range, beacon_set, ref_trace):

        for i in range(ref_trace.shape[0]):
            ref_range[i, 0] = ref_trace[i, 0]
            for j in range(1, uwb_data.shape[1]):
                ref_range[i, j] = np.linalg.norm(ref_trace[i, 1:] - beacon_set[j - 1, :])
                if ref_range[i, j] > 1000.0:
                    ref_range[i, j] = 0.0


    compute_ref_range(ref_range, beacon_set, ref_trace)

    ref_range_f = list()
    ref_cal_range = np.zeros_like(uwb_data)
    ref_cal_range[:, 0] = uwb_data[:, 0]
    for i in range(1, ref_range.shape[1]):
        # print(i)
        f = interpolate.UnivariateSpline(ref_range[:, 0], ref_range[:, i], s=1.0)
        # ref_range_f.append(f)
        ref_cal_range[:, i] = f(uwb_data[:, 0]) * 1.0


    @jit(nopython=True)
    def process_ref_cal_range(ref_cal_range, ref_range):
        for i in range(1, ref_cal_range.shape[1]):
            for j in range(0, ref_cal_range.shape[0]):
                if ref_cal_range[j, 0] < ref_range[1, 0]:
                    ref_cal_range[j, i] = ref_range[1, i]
                if ref_cal_range[j, 0] > ref_range[-1, 0]:
                    ref_cal_range[j, i] = ref_range[-1, i]


    process_ref_cal_range(ref_cal_range, ref_range)

    plt.figure()
    plt.title('uwb range')
    for i in range(1, ref_range.shape[1]):
        # plt.plot(ref_range[:, 0], ref_range[:, i], label=str(i))
        if np.max(uwb_data[:, i] > 0.0) and beacon_set[i - 1, 0] < 5000.0:
            plt.plot(ref_cal_range[:, 0], ref_cal_range[:, i], label=str(i))
            plt.plot(uwb_data[:, 0], uwb_data[:, i], '*', label=str(i))
    plt.grid()
    plt.legend()

    r_error = np.zeros_like(ref_cal_range)
    r_error[:, 0] = ref_cal_range[:, 0]


    def cal_error(r_error, ref_cal_range, uwb_data):
        for i in range(uwb_data.shape[0]):
            for j in range(1, uwb_data.shape[1]):
                if uwb_data[i, j] > 0.0 and abs(uwb_data[i, j] - ref_cal_range[i, j]) < 1000.0:
                    r_error[i, j] = uwb_data[i, j] - ref_cal_range[i, j]


    cal_error(r_error, ref_cal_range, uwb_data)

    plt.figure()
    plt.title('uwb range')
    for i in range(1, ref_range.shape[1]):
        # plt.plot(ref_range[:, 0], ref_range[:, i], label=str(i))
        if np.max(uwb_data[:, i] > 0.0) and beacon_set[i - 1, 0] < 5000.0:
            # plt.plot(ref_cal_range[:, 0], ref_cal_range[:, i], label=str(i))
            # plt.plot(uwb_data[:, 0], uwb_data[:, i], '*', label=str(i))
            plt.plot(r_error[:, 0], r_error[:, i], label=str(i))
    plt.grid()
    plt.legend()

    plt.figure()
    plt.title('trace')
    plt.plot(ref_trace[:, 0], ref_trace[:, 1], label='x')
    plt.plot(ref_trace[:, 0], ref_trace[:, 2], label='y')
    plt.grid()
    plt.legend()

    plt.show()
