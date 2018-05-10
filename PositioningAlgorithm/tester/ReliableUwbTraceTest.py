# -*- coding:utf-8 -*-
# carete by steve at  2018 / 05 / 03　下午4:47
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
    dir_name = '/home/steve/Data/NewFusingLocationData/0038/'


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
                uol.iter_positioning((0, 0, 0),
                                         uwb_data[i, 1:])
        else:
            uwb_trace[i, :], uwb_opt_res[i] = \
                uol.iter_positioning(uwb_trace[i - 1, :],
                                         uwb_data[i, 1:])


    @jit(nopython=True, cache=True)
    def compute_z_ave(uwb_trace, low_b=2.0, high_b=3.0):
        counter = 0
        the_sum = 0.0
        for i in range(uwb_trace.shape[0]):
            if low_b < uwb_trace[i, 2] < high_b:
                counter += 1
                the_sum += uwb_trace[i, 2]

        return float(the_sum) / float(counter)


    average_high = compute_z_ave(uwb_trace)
    print(average_high)

    # write acceptable data to file
    t_file = open(dir_name+'selected_uwb_trace.csv','w')

    t_trace = np.zeros_like(uwb_trace)
    for i in range(uwb_trace.shape[0]):
        if (uwb_opt_res[i] > 0.2 or abs(uwb_trace[i,2]-average_high)>0.1)and i > -1:
            # t_trace[i,:] =
            t_trace[i, 0] = t_trace[i, 0]
        else:
            if i > 2 and i < uwb_trace.shape[0]-2:
                diff = uwb_trace[i+1,:]+uwb_trace[i-1,:]-2.0 * uwb_trace[i,:]
                if np.linalg.norm(diff) > 1.0:
                    continue


            t_trace[i, :] = uwb_trace[i, :]
            t_file.write("%15.15f,%15.15f,%15.15f,%15.15f\n"%(uwb_data[i,0],uwb_trace[i,0],uwb_trace[i,1],uwb_trace[i,2]))
    t_file.close()


    plt.figure()
    plt.title('measuremment & res')
    for i in range(1, uwb_data.shape[1]):
        if uwb_data[:, i].max() > 0.0:
            plt.plot(uwb_data[:, 0], uwb_data[:, i], '+', label=str(i))
    plt.plot(uwb_data[:, 0], uwb_opt_res, '-*', label='res')
    plt.legend()
    plt.grid()

    # plt.figure()
    # plt.title('z value')
    # plt.hist(uwb_trace[:, 2], bins=100)
    # plt.grid()

    plt.figure()
    plt.title('uwb trace')
    plt.plot(uwb_trace[:, 0], uwb_trace[:, 1], label='source uwb')
    plt.plot(t_trace[:, 0], t_trace[:, 1], '*', label='t trace')
    plt.grid()
    plt.legend()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("trace 3d")
    ax.plot(uwb_trace[:, 0], uwb_trace[:, 1], uwb_trace[:, 2], '-+', label='source uwb')
    ax.plot(t_trace[:, 0], t_trace[:, 1], t_trace[:, 2], '*', label='t trace\\beta')
    ax.grid()
    ax.legend()

    plt.show()
