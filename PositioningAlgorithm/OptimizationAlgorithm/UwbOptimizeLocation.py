# -*- coding:utf-8 -*-
# Created by steve @ 18-3-18 下午2:19
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
import scipy  as sp

from scipy.optimize import minimize
import matplotlib.pyplot as plt


from numba import jit
class UwbOptimizeLocation:
    def __init__(self, beacon_set):
        '''

        :param beacon_set:
        '''
        self.beacon_set = beacon_set
        self.measurements = np.zeros(beacon_set.shape[0])


    def position_error_function(self, pose):
        '''

        :param measurements:
        :return:
        '''
        dis_to_beacon = np.linalg.norm(pose - self.beacon_set, axis=1)
        return np.linalg.norm((dis_to_beacon - self.measurements)[
                                  np.where(np.logical_and(self.measurements > 0.0, self.measurements < 550.0))])

    def positioning_fucntion(self, initial_pose, measurements):
        '''

        :param initial_pose:
        :param measurements:
        :return:
        '''
        self.measurements = measurements

        result = minimize(self.position_error_function,
                          initial_pose, method='BFGS')

        # pose = result.x
        # res = result.fun
        # print(pose, res)
        # return pose, res
        return result.x, result.fun


if __name__ == '__main__':
    dir_name = '/home/steve/Data/XsensUwb/MTI700/0002/'
    beacon_data = np.loadtxt(dir_name + 'beaconset_no_mac.csv', delimiter=',')
    uwb_data = np.loadtxt(dir_name + 'uwb_data.csv', delimiter=',')

    uol = UwbOptimizeLocation(beacon_set=beacon_data)

    trace = np.zeros([uwb_data.shape[0], 3])
    res_error = np.zeros([uwb_data.shape[0]])

    for i in range(uwb_data.shape[0]):
        if i is 0:
            trace[i, :], res_error[i] = uol.positioning_fucntion((0, 0, 0), uwb_data[i, 1:])
        else:
            trace[i, :], res_error[i] = uol.positioning_fucntion(trace[i - 1, :], uwb_data[i, 1:])

    plt.figure()
    plt.plot(trace[:, 0], trace[:, 1], '-+', label='source')
    selected_trace = trace[np.where(res_error < 2.5)]
    plt.plot(selected_trace[:, 0], selected_trace[:, 1], '-*', label='selected')
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(res_error)
    plt.grid()

    plt.figure()
    plt.title('uwb source data')
    plt.plot(uwb_data[:,1:])
    plt.grid()

    plt.show()
