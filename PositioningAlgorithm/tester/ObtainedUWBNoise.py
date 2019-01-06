# -*- coding:utf-8 -*-
# carete by steve at  2019 / 01 / 06　2:53 PM
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
from numba import jit, njit, prange

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

# from transforms3d import euler, quaternions

from PositioningAlgorithm.BayesStateEstimation.KalmanFIlterBase import *

from scipy.optimize import minimize

from AlgorithmTool.ImuTools import *

from PositioningAlgorithm.BayesStateEstimation.ImuEKF import *
# from gr import pygr

# from AlgorithmTool
import time

from PositioningAlgorithm.BayesStateEstimation.UwbMeasurementEKF import UwbRangeEKF
from PositioningAlgorithm.OptimizationAlgorithm.UwbOptimizeLocation import UwbOptimizeLocation

from AlgorithmTool.ReferTraceEvaluateTools import *

# from mayavi import mlab

if __name__ == '__main__':

    # print(mk)
    matplotlib.use('Qt5Agg')
    # matplotlib.rcParams['toolbar'] = 'toolmanager'
    start_time = time.time()
    # dir_name = '/home/steve/Data/FusingLocationData/0017/'
    # dir_name = '/home/steve/Data/FusingLocationData/0013/'
    # dir_name = '/home/steve/Data/NewFusingLocationData/0044/'
    # dir_name = 'D:/Data/NewFusingLocationData/0040/'
    # dir_name = 'D:/Data/NewFusingLocationData/0039/'
    dir_name = '/home/steve/Data/NewFusingLocationData/0040/'
    # dir_name = '/home/steve/Data/ZUPTPDR/0003/'
    # dir_name = 'C:/Data/NewFusingLocationData/0040/'

    # ref_score = Refscor(dir_name)
    # imu_data = np.loadtxt(dir_name + 'RIGHT_FOOT.data', delimiter=',')
    imu_data = np.loadtxt(dir_name + 'LEFT_FOOT.data', delimiter=',')
    # imu_data = np.loadtxt(dir_name + 'HEAD.data', delimiter=',')
    imu_data = imu_data[:, 1:]
    imu_data[:, 1:4] = imu_data[:, 1:4] * 9.81
    imu_data[:, 4:7] = imu_data[:, 4:7] * (np.pi / 180.0)

    # uwb_data = np.loadtxt(dir_name + 'uwb_result.csv', delimiter=',')
    # beacon_set = np.loadtxt(dir_name + 'beaconSet.csv', delimiter=',')
    uwb_data = np.loadtxt(dir_name + 'uwb_data.csv', delimiter=',')
    beacon_set = np.loadtxt(dir_name + 'beaconset_no_mac.csv', delimiter=',')
    uwb_strength_data = np.loadtxt(dir_name + 'uwb_signal_data.csv', delimiter=',')
    # beacon_set = np.loadtxt(dir_name + 'beaconset_fill.csv', delimiter=',')

    initial_pos = np.asarray((48.19834796,
                              44.89176719,
                              2.0))

    # for i in range(uwb_data.shape[0]):
    #     for j in range(1,uwb_data.shape[1]):
    #         if uwb_data[i,j] > 0.0 and uwb_strength_data[i,j] < 20:
    #             uwb_data[i,j] = -10.0

    '''
    Delete some beacon's data randomly.
    '''
    uwb_valid = list()
    for i in range(1, uwb_data.shape[1]):
        if uwb_data[:, i].max() > 0.0 and beacon_set[i - 1, 0] < 5000.0:
            uwb_valid.append(i)

    random_index = np.random.randint(0, len(uwb_valid) - 1, len(uwb_valid))
    # for i in range(min(random_index.shape[0], 6)):  # delete parts of beacons's data
    #     uwb_data[:, uwb_valid[random_index[i]]] *= 0.0
    #     uwb_data[:, uwb_valid[random_index[i]]] -= 10.0

    delet_index = [29]  # use 6 beacons
    # delet_index = [30, 33, 35, 36]  # use 3 beacons
    # delet_index = [30, 31, 33, 34, 35]  # use 2 beacons
    # print('delet index:', type(delet_index), delet_index)
    for i in range(len(delet_index)):
        print('deleted:', delet_index[i])
        uwb_data[:, delet_index[i]] *= 0.0
        uwb_data[:, delet_index[i]] -= 10.0

    after_valid_list = list()
    for i in range(1, uwb_data.shape[1]):
        if uwb_data[:, i].max() > 0.0 and beacon_set[i - 1, 0] < 5000.0:
            after_valid_list.append(i)
            print(i, beacon_set[i - 1, :])
    print('before valid:', len(uwb_valid), uwb_valid)
    print('after  valid:', len(after_valid_list), after_valid_list)
    '''
    END
    '''


    uwb_filter_list = list()
    for i in range(1, uwb_data.shape[1]):
        uwb_filter_list.append(UwbRangeEKF(1.0, beacon_set[i - 1, :].reshape(-1)))
        # print(uwb_filter_list[i-1])
    uwb_est_data = np.zeros_like(uwb_data)
    uwb_est_data[:, 0] = uwb_data[:, 0] * 1.0
    uwb_est_data[:, 1:] = uwb_est_data[:, 1:] - 10.0
    uwb_est_prob = np.zeros_like(uwb_est_data)

    uol = UwbOptimizeLocation(beacon_set)
    uwb_trace = np.zeros([uwb_data.shape[0], 3])
    uwb_opt_res = np.zeros([uwb_data.shape[0]])
    stime = time.time()


    # @njit( cache=True)
    def cal(uwb_trace):
        for i in range(uwb_data.shape[0]):
            if i is 0:
                uwb_trace[i, :], uwb_opt_res[i] = \
                    uol.positioning_function(initial_pos,
                                             uwb_data[i, 1:])
            else:
                uwb_trace[i, :], uwb_opt_res[i] = \
                    uol.positioning_function(uwb_trace[i - 1, :],
                                             uwb_data[i, 1:])
        return uwb_trace


    uwb_trace = cal(uwb_trace)
    uwb_ref_trace = np.zeros_like(uwb_trace)
    print('uwb cost time:', time.time() - stime)

    initial_pos = np.asarray(uwb_trace[0, :])
    # initial_pos = np.asarray((62.5,
    #                           21.0,
    #                           2.0))  # (32)

    # initial_orientation = 80.0 * np.pi / 180.0  # 38-45
    # initial_orientation = 50.0 * np.pi / 180.0  # 36
    # initial_orientation = 80.0 * np.pi / 180.0  # 38
    # initial_orientation = 80.0 * np.pi / 180.0  # 37
    # initial_orientation = 80.0 * np.pi / 180.0  # 39
    initial_orientation = 80.0 * np.pi / 180.0  # 40
    # initial_orientation = -110.0 * np.pi / 180.0  # 32
    # initial_orientation = 50.0 * np.pi / 180.0  # 44

    # ref_trace = np.loadtxt(dir_name + 'ref_trace.csv', delimiter=',')

    ref_trace = np.zeros(shape=(uwb_trace.shape[0], uwb_trace.shape[1] + 1))
    ref_trace[:, 1:] = uwb_trace * 1.0
    # initial_state = get_initial_state(imu_data[:40, 1:4], np.asarray((0, 0, 0)), 0.0, 9)

    trace = np.zeros([imu_data.shape[0], 3])
    ftrace = np.zeros([imu_data.shape[0], 3])
    rtrace = np.zeros([imu_data.shape[0], 3])
    ortrace = np.zeros([imu_data.shape[0], 3])
    dtrace = np.zeros_like(trace)
    vel = np.zeros([imu_data.shape[0], 3])
    ang = np.zeros([imu_data.shape[0], 3])
    ba = np.zeros([imu_data.shape[0], 3])
    bg = np.zeros([imu_data.shape[0], 3])

    iner_acc = np.zeros([imu_data.shape[0], 3])

    zv_state = np.zeros([imu_data.shape[0]])

    # kf = KalmanFilterBase(9)
    # kf.state_x = initial_state
    # kf.state_prob = np.diag((0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001))

    average_time_interval = (imu_data[-1, 0] - imu_data[0, 0]) / float(imu_data.shape[0])
    print('average time interval ', average_time_interval)

    kf = ImuEKFComplex(np.diag((
        0.001, 0.001, 0.001,
        0.001, 0.001, 0.001,
        0.001 * np.pi / 180.0, 0.001 * np.pi / 180.0, 0.0001 * np.pi / 180.0,
        0.0001,
        0.0001,
        0.0001,
        0.0001 * np.pi / 180.0,
        0.0001 * np.pi / 180.0,
        0.0001 * np.pi / 180.0
    )),
        local_g=-9.81,
        time_interval=average_time_interval)

    kf.initial_state(imu_data[:50, 1:7],
                     pos=initial_pos,
                     ori=initial_orientation)


    zv_state = GLRT_Detector(imu_data[:, 1:7],
                             sigma_a=1.0,
                             sigma_g=1.0 * np.pi / 180.0,
                             gamma=200.0,
                             gravity=9.8,
                             time_Window_size=10)


    uwb_index = 0

    for i in range(imu_data.shape[0]):
        # print('i:',i)

        kf.state_transaction_function(imu_data[i, 1:7],
                                      np.diag((0.01, 0.01, 0.01,
                                               0.01 * np.pi / 180.0,
                                               0.01 * np.pi / 180.0,
                                               0.01 * np.pi / 180.0))
                                      )

        if (i > 5) and (i < imu_data.shape[0] - 5):
            # print('i:',i)
            # zv_state[i] = z_tester.GLRT_Detector(imu_data[i - 4:i + 4, 1:8])
            if zv_state[i] > 0.5:

                kf.measurement_function_zv(np.asarray((0, 0, 0)),
                                           np.diag((0.0001, 0.0001, 0.0001)))

            if uwb_data[uwb_index, 0] < imu_data[i, 0]:
                if uwb_index < uwb_data.shape[0] - 1:
                    kf.measurement_uwb_iterate(np.asarray(uwb_data[uwb_index, 1:]),
                                                 np.ones(1) * 0.01,
                                                 beacon_set, ref_trace)
                    uwb_index += 1

        trace[i, :] = kf.state[0:3]
        vel[i, :] = kf.state[3:6]
        ang[i, :] = kf.state[6:9]
        ba[i, :] = kf.state[9:12]
        bg[i, :] = kf.state[12:15]
        rate = i / imu_data.shape[0]
        iner_acc[i, :] = kf.acc

        # ftrace[i, :] = fkf.state[0:3]
        # rtrace[i, :] = rkf.state[0:3]
        # ortrace[i, :] = orkf.state[0:3]
        # dtrace[i, :] = drkf.state[0:3]

        # print('finished:', rate * 100.0, "% ", i, imu_data.shape[0])

    end_time = time.time()
    print('all ekf totally time:', end_time - start_time, 'data time:', imu_data[-1, 0] - imu_data[0, 0])


    def aux_plot(data: np.ndarray, name: str):
        plt.figure()
        plt.title(name)
        plt.plot(zv_state * data.max() * 1.1, 'r-', label='zv state')
        for i in range(data.shape[1]):
            plt.plot(data[:, i], label=str(i))
        plt.grid()
        plt.legend()


    uwb_id_offset = 28

    color_dict = {
        'ref': 'gray',
        'UWB': 'lightblue',
        'Foot': 'darkorange',
        'Standard': 'seagreen',
        'REKF': 'blue',
        'RIEKF': 'red'
    }



    plt.figure()
    # plt.title('Trajectory')
    # for i in range(ref_vis.shape[0]):
    #     plt.plot([ref_vis[i, 0], ref_vis[i, 2]], [ref_vis[i, 1], ref_vis[i, 3]], '-', color=color_dict['ref'],
    #              alpha=0.5, lw='10')
    plt.plot(trace[:, 0], trace[:, 1], '-', color=color_dict['Standard'], label='Standard EKF')
    plt.plot(uwb_trace[:, 0], uwb_trace[:, 1], '+', color=color_dict['UWB'], label='uwb')
    # plt.plot(ref_trace[:, 1], ref_trace[:, 2], '-', label='ref')
    # for i in range(beacon_set.shape[0]):
    #     if uwb_data[i + 1, :].max() > 0 and beacon_set[i, 0] < 5000.0:
    #         plt.text(beacon_set[i, 0], beacon_set[i, 1], s=str(i + 1))

    plt.axis('equal')

    for i in range(len(uwb_valid)):
        if uwb_data[:, uwb_valid[i]].max() > 0:
            plt.text(beacon_set[uwb_valid[i] - 1, 0], beacon_set[uwb_valid[i] - 1, 1], s=str(i))
            plt.plot(beacon_set[uwb_valid[i] - 1, 0], beacon_set[uwb_valid[i] - 1, 1], 'r*')
    local_fsize = 12
    plt.xlabel('X/m', fontsize=local_fsize)
    plt.ylabel('Y/m', fontsize=local_fsize)
    plt.xticks(fontsize=local_fsize)
    plt.yticks(fontsize=local_fsize)
    plt.legend(fontsize=local_fsize)
    plt.grid()

    from mpl_toolkits.mplot3d import Axes3D

    # 3d plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trace[:, 0], trace[:, 1], trace[:, 2], '-', label='standard EKF')
    ax.plot(rtrace[:, 0], rtrace[:, 1], rtrace[:, 2], '-', label='Robust EKF')
    ax.plot(ortrace[:, 0], ortrace[:, 1], ortrace[:, 2], '-', label='or IEKF')
    ax.plot(uwb_trace[:, 0], uwb_trace[:, 1], uwb_trace[:, 2], '*', label='uwb')
    plt.legend()

    plt.figure()
    plt.subplot(211)
    # for i in range(beacon_set.shape[0]):
    #     if uwb_data[:,i+1].max() > 0 and beacon_set[i, 0] < 5000.0:
    #         plt.plot(uwb_data[:, 0] - uwb_data[0, 0], uwb_data[:, i+1],'+', label='id:' + str(i-uwb_id_offset))
    for i in range(len(uwb_valid)):
        plt.plot(uwb_data[:, 0] - uwb_data[0, 0], uwb_data[:, uwb_valid[i]], '+', label='id:' + str(i))
    plt.grid()
    plt.legend()

    plt.subplot(212)
    for i in range(len(uwb_valid)):
        plt.plot(uwb_data[:, 0] - uwb_data[0, 0], uwb_strength_data[:, uwb_valid[i]], '+', label='id:' + str(i))
    plt.grid()
    plt.legend()


    #
    plt.show()