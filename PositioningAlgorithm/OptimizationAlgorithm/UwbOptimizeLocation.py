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
import math

from numba import jit, float64, vectorize


class UwbOptimizeLocation:
    def __init__(self, beacon_set):
        '''

        :param beacon_set:
        '''
        # self.beacon_set = beacon_set
        self.measurements = np.zeros(beacon_set.shape[0])
        self.use_index = np.where(beacon_set[:, 0] < 1000.0)
        self.beacon_set = beacon_set[self.use_index]


        self.counter = 0

    def position_error_function(self, pose):
        '''

        :param measurements:
        :return:
        '''

        # beacon_set = self.beacon_set[self.use_index]
        # measurement = self.measurements[]
        dis_to_beacon = np.linalg.norm(pose - self.beacon_set, axis=1)
        return np.linalg.norm((dis_to_beacon - self.measurements)[
                                  np.where(np.logical_and(self.measurements > 0.0, self.measurements < 550.0))])

    def positioning_function(self, initial_pose, measurements):
        '''

        :param initial_pose:
        :param measurements:
        :return:
        '''
        # print('beacon',self.beacon_set.shape[0])
        # print('use index:',self.use_index)
        self.measurements = measurements[self.use_index]
        self.beacon_set = self.beacon_set[self.use_index]

        result = minimize(self.position_error_function,
                          initial_pose, method='BFGS')

        return result.x, result.fun

    def position_error_robust_function(self, pose):
        # @vectorize([float64(float64)])
        def rou(u):
            # return 0.5 * u * u / (1.0 + u * u)

            u2 = u*u
            if u2 < 0.5:
                return 0.5 * u2
            else:
                return 2.0 * u2 /(1.0+u2)-0.5

        # self.measurements = measurements[self.use_index]
        # self.beacon_set = self.beacon_set[self.use_index]
        rou = np.vectorize(rou)
        dis_to_beacon = np.linalg.norm(pose - self.beacon_set, axis=1)

        return np.linalg.norm(rou(dis_to_beacon - self.measurements)[
                                  np.where(np.logical_and(self.measurements > 0.0, self.measurements < 550.0))])

    def positioning_function_robust(self, initial_pose, measurements, max_dis=2.0):
        '''
        Robust based on rou function( a simplify of M-Estimation method).
        :param initial_pose:
        :param measurements:
        :param max_dis:
        :return:
        '''
        # k = 0
        # dis_to_beacon
        # pose = initial_pose

        # while k < 20:
        #     dis_to_beacon=np.linalg.norm(pose-beacon_data)
        #
        #     dis_error = np.linalg
        initial_pose = np.asarray(initial_pose)
        self.measurements = measurements[self.use_index]
        # self.beacon_set = self.beacon_set[self.use_index]

        # result = minimize(self.position_error_function,
        #                   initial_pose,method='BFGS')

        result = minimize(self.position_error_robust_function,
                          initial_pose, method='BFGS')

        return result.x, result.fun

    def iter_positioning(self, initial_pose, measurements):
        self.measurements = measurements[self.use_index]
        measurements = measurements[self.use_index]
        beacon_backup = self.beacon_set * 1.0


        # initial result based on all measurements and normal error function.
        best_result = minimize(self.position_error_function,
                               initial_pose,method='BFGS')
        best_result.fun=100000.0
        min_index = 1000
        # if measurements.shape[0] <= 4:
        #     self.beacon_set = self.beacon_set[np.where(
        #         np.logical_and(measurements>0.0 , measurements<550.0)
        #     )]
        #     measurements = measurements[
        #         np.where(
        #             np.logical_and(measurements>0.0 , measurements<550)
        #         )
        #     ]
        #     self.beacon_set = beacon_backup
        #         # self.measurements = measurements*1.0
        #
        #     best_result = minimize(self.position_error_robust_function,
        #                            initial_pose,method='BFGS')
        #     print('out')
        #     return best_result.x, best_result.fun

        while measurements.shape[0] > 4:
            func_error = list()
            # if np.logical_and(measurements>0.0,measurements<550)
            self.beacon_set = self.beacon_set[np.where(
                np.logical_and(measurements>0.0 , measurements<550.0)
            )]
            measurements = measurements[
                np.where(
                    np.logical_and(measurements>0.0 , measurements<550)
                )
            ]
            # valid measurements less than 4, so use robust positioning result.
            if measurements.shape[0] < 4:
                self.beacon_set = beacon_backup
                # self.measurements = measurements*1.0

                best_result = minimize(self.position_error_robust_function,
                                       initial_pose,method='BFGS')
                # self.counter +=1
                # print(self.counter)
                # print('in')
                # if not self.c :
                #     self.c = 1
                # else:
                #     self.c += 1
                #     print(self.c)
                break



            for i in range(measurements.shape[0]):


                self.measurements=measurements*1.0
                self.measurements[i] = 10000.0

                result = minimize(self.position_error_function,
                                  initial_pose,method='BFGS')
                # if best_result is None:
                #     best_result = result
                # else:
                if result.fun < best_result.fun:
                    best_result = result
                    min_index = i+0
                func_error.append(result.fun)
                # print(result.fun,best_result.fun)
            # print(sorted(func_error))
            # break
            if np.std(np.asarray(func_error))<2.0:
                break
            else:
                if min_index< measurements.shape[0]:
                    measurements[min_index] = 100000
                else:
                    break
        # if measurements


        self.beacon_set= beacon_backup

        # print(best_result.x)

        # print('---------------------------------------------------')
        return best_result.x,best_result.fun




if __name__ == '__main__':
    dir_name = '/home/steve/Data/XsensUwb/MTI700/0002/'
    beacon_data = np.loadtxt(dir_name + 'beaconset_no_mac.csv', delimiter=',')
    uwb_data = np.loadtxt(dir_name + 'uwb_data.csv', delimiter=',')

    uol = UwbOptimizeLocation(beacon_set=beacon_data)

    trace = np.zeros([uwb_data.shape[0], 3])
    res_error = np.zeros([uwb_data.shape[0]])

    for i in range(uwb_data.shape[0]):
        if i is 0:
            trace[i, :], res_error[i] = uol.positioning_function((0, 0, 0), uwb_data[i, 1:])
        else:
            trace[i, :], res_error[i] = uol.positioning_function(trace[i - 1, :], uwb_data[i, 1:])

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
    plt.plot(uwb_data[:, 1:])
    plt.grid()

    plt.show()
