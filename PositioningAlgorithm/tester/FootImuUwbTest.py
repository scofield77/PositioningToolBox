# -*- coding:utf-8 -*-
# carete by steve at  2018 / 04 / 25　下午3:51
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
from numba import jit

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

# from mayavi import mlab

if __name__ == '__main__':

    # print(mk)
    matplotlib.use('Qt5Agg')
    # matplotlib.rcParams['toolbar'] = 'toolmanager'
    start_time = time.time()
    # dir_name = '/home/steve/Data/FusingLocationData/0017/'
    # dir_name = '/home/steve/Data/FusingLocationData/0013/'
    dir_name = '/home/steve/Data/NewFusingLocationData/0039/'
    # dir_name = 'D:/Data/NewFusingLocationData/0033/'

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

    uwb_valid = list()
    for i in range(1, uwb_data.shape[1]):
        if uwb_data[:, i].max() > 0.0:
            uwb_valid.append(i)
    random_index = np.random.randint(0, len(uwb_valid) - 1, len(uwb_valid))
    # for i in range(min(random_index.shape[0], 1)):
    #     uwb_data[:, uwb_valid[random_index[i]]] *= 0.0
    #     uwb_data[:, uwb_valid[random_index[i]]] -= 10.0

    uwb_filter_list = list()
    for i in range(1, uwb_data.shape[1]):
        uwb_filter_list.append(UwbRangeEKF(1.0, beacon_set[i - 1, :].reshape(-1)))
        # print(uwb_filter_list[i-1])
    uwb_est_data = np.zeros_like(uwb_data)
    uwb_est_data[:, 0] = uwb_data[:, 0] * 1.0
    uwb_est_data[:, 1:] = uwb_est_data[:, 1:] - 10.0
    uwb_est_prob = np.zeros_like(uwb_est_data)

    # ref_trace = np.loadtxt(dir_name + 'ref_trace.csv', delimiter=',')

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

    initial_pos = ref_trace[0, 1:]

    ti = 1
    while np.linalg.norm(ref_trace[ti, 1:] - ref_trace[0, 1:]) < 5.0:
        ti += 1
    # initial_orientation = math.atan2(ref_trace[ti, 2] - ref_trace[0, 2],
    #                                  ref_trace[ti, 1] - ref_trace[0, 1]) - 10.0 * np.pi / 180.0  # 35
    # initial_orientation = math.atan2(ref_trace[ti, 2] - ref_trace[0, 2],
    #                                  ref_trace[ti, 1] - ref_trace[0, 1]) + 150.0 * np.pi / 180.0  # 32
    initial_orientation = 80.0 * np.pi / 180.0

    #  initial_orientation = 200.0 / 180.0 * np.pi

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
    fkf = ImuEKFComplex(np.diag((
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

    fkf.initial_state(imu_data[:50, 1:7],
                      pos=initial_pos,
                      ori=initial_orientation)

    rkf = ImuEKFComplex(np.diag((
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001 * np.pi / 180.0, 0.001 * np.pi / 180.0, 0.001 * np.pi / 180.0,
        0.0001,
        0.0001,
        0.0001,
        0.0001 * np.pi / 180.0,
        0.0001 * np.pi / 180.0,
        0.0001 * np.pi / 180.0
    )),
        local_g=-9.81, time_interval=average_time_interval)

    rkf.initial_state(imu_data[:50, 1:7],
                      pos=initial_pos,
                      ori=initial_orientation)
    orkf = ImuEKFComplex(np.diag((
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001 * np.pi / 180.0, 0.001 * np.pi / 180.0, 0.001 * np.pi / 180.0,
        0.0001,
        0.0001,
        0.0001,
        0.0001 * np.pi / 180.0,
        0.0001 * np.pi / 180.0,
        0.0001 * np.pi / 180.0
    )),
        local_g=-9.81, time_interval=average_time_interval)

    orkf.initial_state(imu_data[:50, 1:7],
                       pos=initial_pos,
                       ori=initial_orientation)
    drkf = ImuEKFComplex(np.diag((
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001 * np.pi / 180.0, 0.001 * np.pi / 180.0, 0.001 * np.pi / 180.0,
        0.0001,
        0.0001,
        0.0001,
        0.0001 * np.pi / 180.0,
        0.0001 * np.pi / 180.0,
        0.0001 * np.pi / 180.0
    )),
        local_g=-9.81, time_interval=average_time_interval)

    drkf.initial_state(imu_data[:50, 1:7],
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
        fkf.state_transaction_function(imu_data[i, 1:7],
                                       np.diag((0.01, 0.01, 0.01,
                                                0.01 * np.pi / 180.0,
                                                0.01 * np.pi / 180.0,
                                                0.01 * np.pi / 180.0))
                                       )
        rkf.state_transaction_function(imu_data[i, 1:7],
                                       np.diag((0.01, 0.01, 0.01,
                                                0.01 * np.pi / 180.0,
                                                0.01 * np.pi / 180.0,
                                                0.01 * np.pi / 180.0))
                                       )
        orkf.state_transaction_function(imu_data[i, 1:7],
                                        np.diag((0.01, 0.01, 0.01,
                                                 0.01 * np.pi / 180.0,
                                                 0.01 * np.pi / 180.0,
                                                 0.01 * np.pi / 180.0))
                                        )
        drkf.state_transaction_function(imu_data[i, 1:7],
                                        np.diag((0.01, 0.01, 0.01,
                                                 0.01 * np.pi / 180.0,
                                                 0.01 * np.pi / 180.0,
                                                 0.01 * np.pi / 180.0))
                                        )

        if (i > 5) and (i < imu_data.shape[0] - 5):
            # print('i:',i)
            # zv_state[i] = z_tester.GLRT_Detector(imu_data[i - 4:i + 4, 1:8])
            if zv_state[i] > 0.5:
                fkf.measurement_function_zv(np.asarray((0, 0, 0)),
                                            np.diag((0.0001, 0.0001, 0.0001)))

                kf.measurement_function_zv(np.asarray((0, 0, 0)),
                                           np.diag((0.0001, 0.0001, 0.0001)))
                rkf.measurement_function_zv(np.asarray((0, 0, 0)),
                                            np.diag((0.0001, 0.0001, 0.0001)))
                orkf.measurement_function_zv(np.asarray((0, 0, 0)),
                                             np.diag((0.0001, 0.0001, 0.0001)))
                drkf.measurement_function_zv(np.asarray((0, 0, 0)),
                                             np.diag((0.0001, 0.0001, 0.0001)))

            if uwb_data[uwb_index, 0] < imu_data[i, 0]:

                if uwb_index < uwb_data.shape[0] - 1:
                    # orkf.measurement_uwb_robust_multi(np.asarray(uwb_data[uwb_index, 1:]),
                    #                                   np.ones(1) * 0.1,
                    #                                   beacon_set,
                    #                                   6.0)
                    # rkf.measurement_uwb_mc(np.asarray(uwb_data[uwb_index,1:]),
                    #                        np.ones(1)*1.0,
                    #                        beacon_set, ref_trace)
                    orkf.measurement_uwb_iterate(np.asarray(uwb_data[uwb_index, 1:]),
                                                 np.ones(1) * 0.1,
                                                 beacon_set, ref_trace)
                    uwb_index += 1
                    for j in range(1, uwb_data.shape[1]):
                        # right

                        if uwb_filter_list[j - 1].m > -1000.0:
                            uwb_filter_list[j - 1].state_transmition_2d(drkf.state[0:3], drkf.prob_state[0:3, 0:3])
                            # uwb_filter_list[j - 1].state_estimate(rkf.state[0:3], rkf.prob_state[0:3, 0:3])
                            uwb_est_data[uwb_index, j] = uwb_filter_list[j - 1].m
                            uwb_est_prob[uwb_index, j] = uwb_filter_list[j - 1].cov

                        if uwb_data[uwb_index, j] > 0.0 and \
                                uwb_data[uwb_index, j] < 1000.0 and \
                                beacon_set[j - 1, 0] < 1000.0:

                            # initial uwb filter
                            if uwb_filter_list[j - 1].m < -1000.0:
                                uwb_filter_list[j - 1].initial_pose(uwb_data[uwb_index, j], rkf.state[0:3] * 1.0,
                                                                    rkf.prob_state[0:3, 0:3] * 1.0)
                            else:
                                uwb_filter_list[j - 1].measurement_func_robust(uwb_data[uwb_index, j], 10.0, 2.0, 1.0)
                                # uwb_filter_list[j - 1].measurement_func(uwb_data[uwb_index, j], 1.0, 3.0, 1.0)
                                # uwb_est_data[uwb_index, j] = uwb_filter_list[j-1].m
                                if np.linalg.norm(uwb_filter_list[j - 1].beacon_set - beacon_set[j - 1, :]) > 0.1:
                                    print('error', uwb_filter_list[j - 1].beacon_set, beacon_set[j - 1, :])

                            kf.measurement_uwb(np.asarray(uwb_data[uwb_index, j]),
                                               np.ones(1) * 1.0,
                                               np.transpose(beacon_set[j - 1, :]))
                            rkf.measurement_uwb_robust(uwb_data[uwb_index, j],
                                                       np.ones(1) * 0.1,
                                                       np.transpose(beacon_set[j - 1, :]),
                                                       j, 7.0, 1.0)
                            # if uwb_filter_list[j-1].cov<0.02:
                            #     rkf.measurement_uwb(uwb_filter_list[j - 1].m,
                            #                         uwb_filter_list[j - 1].cov,
                            #                         np.transpose(beacon_set[j - 1, :]))
                    drkf.measurement_uwb_iterate(np.asarray(uwb_est_data[uwb_index, 1:]),
                                                 np.ones(1) * 0.1,
                                                 beacon_set, ref_trace)
                    # for j in range(1,uwb_data.shape[1]):
                    #     if uwb_filter_list[j-1].m > -1000.0:
                    #         uwb_filter_list[j-1].state_estimate(drkf.state[0:3],drkf.prob_state[0:3,0:3])
                    #         uwb_est_data[uwb_index, j] = uwb_filter_list[j-1].m

                    # print(rkf.prob_state[0,0],rkf.prob_state[1,1],rkf.prob_state[2,2])
        #
        # print(kf.state_x)
        # print( i /)
        trace[i, :] = kf.state[0:3]
        vel[i, :] = kf.state[3:6]
        ang[i, :] = kf.state[6:9]
        ba[i, :] = kf.state[9:12]
        bg[i, :] = kf.state[12:15]
        rate = i / imu_data.shape[0]
        iner_acc[i, :] = kf.acc

        ftrace[i, :] = fkf.state[0:3]
        rtrace[i, :] = rkf.state[0:3]
        ortrace[i, :] = orkf.state[0:3]
        dtrace[i, :] = drkf.state[0:3]

        # print('finished:', rate * 100.0, "% ", i, imu_data.shape[0])

    end_time = time.time()
    print('totally time:', end_time - start_time, 'data time:', imu_data[-1, 0] - imu_data[0, 0])


    def aux_plot(data: np.ndarray, name: str):
        plt.figure()
        plt.title(name)
        plt.plot(zv_state * data.max() * 1.1, 'r-', label='zv state')
        for i in range(data.shape[1]):
            plt.plot(data[:, i], label=str(i))
        plt.grid()
        plt.legend()


    # plot dx list

    # if len(rkf.dx_dict) > 0:
    #     dx_matrix = np.zeros(shape=(len(rkf.dx_dict), len(rkf.dx_dict[1]), 15))
    #
    #     for i in range(dx_matrix.shape[0]):
    #         for j in range(dx_matrix.shape[1]):
    #             dx_matrix[i, j, :] = rkf.dx_dict[i][j]
    #
    #     plt.figure()
    #     plt.title('dx')
    #     for i in range(dx_matrix.shape[0]):
    #         # plt.plot(dx_matrix[i,:,0],dx_matrix[i,:,1],'-.',label=str(i))
    #         plt.plot(np.linalg.norm(dx_matrix[i, :, 0:3], axis=1), '-+', label=str(i))
    #     plt.grid()
    #     plt.legend()

    # #
    # aux_plot(imu_data[:, 1:4], 'acc')
    # aux_plot(imu_data[:, 4:7], 'gyr')
    # aux_plot(imu_data[:, 7:10], 'mag')
    # aux_plot(trace, 'trace')
    # aux_plot(vel, 'vel')
    # aux_plot(ang, 'ang')
    # aux_plot(ba, 'ba')
    # aux_plot(bg, 'bg')

    # aux_plot(iner_acc, 'inner acc')

    plt.figure()
    plt.plot(trace[:, 0], trace[:, 1], '-', label='fusing')
    plt.plot(ftrace[:, 0], ftrace[:, 1], '-', label='foot')
    plt.plot(rtrace[:, 0], rtrace[:, 1], '-', label='robust')
    plt.plot(ortrace[:, 0], ortrace[:, 1], '-', label='own robust')
    plt.plot(dtrace[:, 0], dtrace[:, 1], '-', label='d ekf')
    plt.plot(uwb_trace[:, 0], uwb_trace[:, 1], '+', label='uwb')
    # plt.plot(ref_trace[:, 1], ref_trace[:, 2], '-', label='ref')
    for i in range(beacon_set.shape[0]):
        plt.text(beacon_set[i, 0], beacon_set[i, 1], s=str(i + 1))
    plt.legend()
    plt.grid()

    plt.figure()
    # plt.plot(trace[:, 0], trace[:, 1], '-', label='fusing')
    # plt.plot(ftrace[:, 0], ftrace[:, 1], '-', label='foot')
    plt.plot(rtrace[:, 0], rtrace[:, 1], '-', label='robust')
    plt.plot(ortrace[:, 0], ortrace[:, 1], '-+', label='own robust')
    plt.plot(dtrace[:, 0], dtrace[:, 1], '-+', label='d ekf')
    plt.plot(uwb_trace[:, 0], uwb_trace[:, 1], '+', label='uwb')
    # plt.plot(ref_trace[:, 1], ref_trace[:, 2], '-', label='ref')
    for i in range(beacon_set.shape[0]):
        plt.text(beacon_set[i, 0], beacon_set[i, 1], s=str(i + 1))
    plt.legend()
    plt.grid()

    plt.figure()
    plt.subplot(411)
    plt.title('uwb estimated')
    for i in range(1, uwb_est_data.shape[1]):
        if uwb_data[:, i].max() > 0.0 and beacon_set[i - 1, 0] < 5000.0:
            plt.plot(uwb_est_data[:, 0], uwb_est_data[:, i], '-+', label=str(i))
    plt.legend()
    plt.grid()
    plt.subplot(412)
    plt.title('uwb')
    for i in range(1, uwb_est_data.shape[1]):
        if uwb_data[:, i].max() > 0.0 and beacon_set[i - 1, 0] < 5000.0:
            plt.plot(uwb_est_data[:, 0], uwb_data[:, i], '+', label=str(i))
    plt.legend()
    plt.grid()
    plt.subplot(413)
    plt.title('uwb dif')
    for i in range(1, uwb_est_data.shape[1]):
        if uwb_data[:, i].max() > 0.0 and beacon_set[i - 1, 0] < 5000.0:
            index_list = np.where(uwb_data[:, i] > 0.0)
            # print(index_list)
            plt.plot(uwb_est_data[index_list[0], 0], uwb_est_data[index_list[0], i] - uwb_data[index_list[0], i], '-+',
                     label=str(i))
    plt.legend()
    plt.grid()
    plt.subplot(414)
    plt.title('uwb prob')
    for i in range(1, uwb_est_data.shape[1]):
        if uwb_data[:, i].max() > 0.0 and beacon_set[i - 1, 0] < 5000.0:
            plt.plot(uwb_est_data[:, 0], uwb_est_prob[:, i], '+', label=str(i))
    plt.legend()
    plt.grid()

    # plt.figure()
    # plt.title('corrected result')
    # plt.subplot(211)
    # for i in range(1, uwb_est_data.shape[1]):
    #     if uwb_data[:, i].max() > 0.0 and beacon_set[i - 1, 0] < 5000.0:
    #         plt.plot(uwb_est_data[:, 0], uwb_est_data[:, i], '-+', label=str(i))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trace[:, 0], trace[:, 1], trace[:, 2], '-+', label='trace')
    ax.plot(rtrace[:, 0], rtrace[:, 1], rtrace[:, 2], '-+', label='robust')
    ax.plot(ortrace[:, 0], ortrace[:, 1], ortrace[:, 2], '-+', label='own robust')
    ax.plot(uwb_trace[:, 0], uwb_trace[:, 1], uwb_trace[:, 2], '+', label='uwb')
    ax.grid()
    ax.legend()

    # plt.figure()
    # plt.title('uwb')
    # for i in range(1, uwb_data.shape[1]):
    #     plt.plot(uwb_data[:, 0], uwb_data[:, i], '+-', label=str(i))
    # plt.plot(uwb_data[:, 0], uwb_opt_res, '+-', label='res error')
    # plt.grid()
    # plt.legend()

    plt.show()
