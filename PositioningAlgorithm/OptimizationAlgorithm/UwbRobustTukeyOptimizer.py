# -*- coding:utf-8 -*-
# carete by steve at  2019 / 01 / 09　3:44 PM
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
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

from numba import jit, float64

import time

from array import array


@jit(float64(float64, float64), nopython=True, cache=True)
def tukey_func(u, d):
    '''
    Tukey robust function.
    :param u:  value
    :param d: tukey parameter.
    :return:
    '''
    if abs(u < d):
        return d * d / 6.0 * (1.0 - (u * u / d / d) ** 3.0)
    else:
        return d * d / 6.0


def tukey_derivative(u, d):
    if abs(u < d):
        return tukey_func
    else:
        return 0.0


@jit()
def uwb_pos_derivative(pos, uwb_cov, range_array, beacon_array, eta):
    dis_array = np.zeros_like(range_array)
    derivative = np.zeros([range_array.shape[0], 3])
    for i in range(range_array.shape[0]):
        dis_array[i] = np.linalg.norm(pos - beacon_array[i, :])

        if range_array[i] > dis_array[i] + eta:
            derivative[i, :] = np.zeros(3)
        elif range_array[i] > dis_array[i]:
            derivative[i, :] = (1.0 - (dis_array[i] - range_array[i]) ** 2.0 / eta / eta) ** 2.0 \
                               * (dis_array[i] - range_array[i]) * (pos - beacon_array[i, :]) / \
                               dis_array[i] / uwb_cov[i]
        else:
            derivative[i, :] = (2.0 * (dis_array[i] - range_array[i]) * (pos - beacon_array[i, :]) /
                                dis_array[i]) / uwb_cov[i]

    # print(np.sum(np.abs(dis_array)))
    return np.sum(derivative, axis=0)


class UwbRobustTukeyOptimizer:
    def __init__(self, eta):
        self.eta = eta

    def RobustPositioning(self,
                          uwb_measurement,
                          uwb_cov,
                          uwb_beacon_set,
                          initial_pos=np.zeros(3)):
        pos = np.zeros(3)
        pos = initial_pos
        assert uwb_measurement.shape[0] == uwb_beacon_set.shape[0]
        range_buf = array('d')
        beacon_buf = array('d')
        for i in range(uwb_measurement.shape[0]):
            if (uwb_measurement[i] > 0.0 and -1e3 < uwb_beacon_set[i, 0] < 1e3):
                range_buf.append(uwb_measurement[i])
                for k in range(3):
                    beacon_buf.append(uwb_beacon_set[i, k])

        self.range_array = np.frombuffer(range_buf, dtype=np.float64).reshape([-1])
        self.beacon_array = np.frombuffer(beacon_buf, dtype=np.float64).reshape([-1, 3])

        itea_counter = 0
        update_step_length = 1.0
        latest_error = 1.0
        while itea_counter < 20 and update_step_length > 1e-10:
            itea_counter += 1
            derivative_pos = uwb_pos_derivative(pos, uwb_cov, self.range_array, self.beacon_array, self.eta)
            pos -= 0.1 * derivative_pos
            # pos[2] = 1.0

        return pos

    # def global_RobustPositioning(self, uwb_measurement, uwb_cov, uwb_beacon_set, initial_pos):


if __name__ == '__main__':
    # print(mk)
    matplotlib.use('Qt5Agg')
    # matplotlib.rcParams['toolbar'] = 'toolmanager'
    start_time = time.time()
    # dir_name = '/home/steve/Data/FusingLocationData/0017/'
    # dir_name = '/home/steve/Data/FusingLocationData/0036/'
    dir_name = '/home/steve/Data/NewFusingLocationData/0023/'
    # dir_name = '/home/steve/Data/VehicleUWBINS/0002/'
    # dir_name = '/home/steve/Data/19-1-12/0002/'

    # imu_data = np.loadtxt(dir_name + 'RIGHT_FOOT.data', delimiter=',')
    # imu_data = np.loadtxt(dir_name + 'HEAD.data', delimiter=',')
    # imu_data = imu_data[:, 1:]
    # imu_data[:, 1:4] = imu_data[:, 1:4] * 9.81
    # imu_data[:, 4:7] = imu_data[:, 4:7] * (np.pi / 180.0)

    # uwb_data = np.loadtxt(dir_name + 'uwb_result.csv', delimiter=',')
    # beacon_set = np.loadtxt(dir_name + 'beaconSet.csv', delimiter=',')
    uwb_data = np.loadtxt(dir_name + 'uwb_data.csv', delimiter=',')
    # uwb_strength_data = np.loadtxt(dir_name + 'uwb_signal_data.csv', delimiter=',')
    beacon_set = np.loadtxt(dir_name + 'beaconset_no_mac.csv', delimiter=',')
    # ref_trace = np.loadtxt(dir_name + 'ref_trace.csv', delimiter=',')

    uol = UwbRobustTukeyOptimizer(1.0)
    uwb_trace = np.zeros([uwb_data.shape[0], 3])
    uwb_opt_res = np.zeros([uwb_data.shape[0]])
    initial_pos = np.zeros(3)
    initial_pos[2] = 1.5
    for i in range(uwb_data.shape[0]):
        # if i is 0:
        #     uwb_trace[i, :], uwb_opt_res[i] = \
        #         uol.iter_positioning((0, 0, 0),
        #                              uwb_data[i, 1:])
        # else:
        #     uwb_trace[i, :], uwb_opt_res[i] = \
        #         uol.iter_positioning(uwb_trace[i - 1, :],
        #                              uwb_data[i, 1:])

        uwb_trace[i, :] = uol.RobustPositioning(uwb_data[i, 1:], np.ones(uwb_data.shape[1] - 1) * 0.5, beacon_set,
                                                initial_pos)
        initial_pos = uwb_trace[i, :]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(uwb_trace[:, 0], uwb_trace[:, 1], uwb_trace[:, 2], '-+', label='uwb trace')
    ax.grid()
    ax.legend()

    plt.figure()
    plt.title('uwb trace 2d')
    plt.plot(uwb_trace[:, 0], uwb_trace[:, 1], '-+')
    plt.grid()

    plt.show()
