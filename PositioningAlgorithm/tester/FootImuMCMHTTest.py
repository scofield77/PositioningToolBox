# -*- coding:utf-8 -*-
# carete by steve at  2018 / 05 / 28　下午3:48
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
    # dir_name = '/home/steve/Data/NewFusingLocationData/0039/'
    # dir_name = '/home/steve/Data/NewFusingLocationData/0040/'
    # dir_name = 'C:/Data/NewFusingLocationData/0039/'
    # dir_name = 'D:/Data/NewFusingLocationData/0039/'
    # dir_name = 'D:/Data/NewFusingLocationData/0040/'
    # dir_name = 'D:/Data/NewFusingLocationData/0036/'
    dir_name = '/home/steve/Data/NewFusingLocationData/0033/'
    # dir_name = 'D:/Data/NewFusingLocationData/0037/'

    ref_score = Refscor(dir_name)
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
    # beacon_set = np.loadtxt(dir_name + 'beaconset_fill.csv', delimiter=',')

    ref_vis = np.loadtxt(dir_name + 'ref_vis.csv', delimiter=',')

    initial_pos = np.asarray((48.19834796,
                              44.89176719,
                              2.0))
    initial_pos = np.asarray((63.0,
                              22.0,
                              2.0))#(32)


    # initial_orientation = 80.0 * np.pi / 180.0  # 38-45
    # initial_orientation = 50.0 * np.pi / 180.0  # 36
    initial_orientation = -120.0 * np.pi / 180.0  # 32
    initial_orientation = -110.0 * np.pi / 180.0  # 33
    # initial_orientation = 80.0 * np.pi / 180.0  # 37
    # initial_orientation = 80.0 * np.pi / 180.0  # 38
    # initial_orientation = 80.0 * np.pi / 180.0  # 39
    # initial_orientation = 80.0 * np.pi / 180.0  # 40

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

    # delet_index = [ 29]  # use 6 beacons
    delet_index = [33, 35]  # use 5 beacons A
    # delet_index = [33, 34]  # use 5 beacons B
    # delet_index = [30, 33, 35]  # use 4 beacons A
    # delet_index = [30, 33, 36]  # use 4 beacons B
    # delet_index = [34, 33, 36]  # use 4 beacons C
    # delet_index = [31, 33, 36]  # use 4 beacons D
    # delet_index = [30, 33, 35, 36]  # use 3 beacons A
    # delet_index = [30, 33, 31, 36]  # use 3 beacons B
    # delet_index = [31, 33, 36, 37]  # use 3 beacons C
    # delet_index = [31, 33, 35, 37]  # use 3 beacons D
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
    uwb_ref_trace = np.zeros_like(uwb_trace)

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
    print('uwb cost time:', time.time() - stime)

    # ref_trace = np.loadtxt(dir_name + 'ref_trace.csv', delimiter=',')

    ref_trace = np.zeros(shape=(uwb_trace.shape[0], uwb_trace.shape[1] + 1))
    ref_trace[:, 1:] = uwb_trace * 1.0
    # initial_state = get_initial_state(imu_data[:40, 1:4], np.asarray((0, 0, 0)), 0.0, 9)

    trace = np.zeros([imu_data.shape[0], 3])
    ftrace = np.zeros([imu_data.shape[0], 3])
    rtrace = np.zeros([imu_data.shape[0], 3])
    ortrace = np.zeros([imu_data.shape[0], 3])
    dtrace = np.zeros_like(trace)
    reftrace = np.zeros_like(trace)
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

    refrkf = ImuEKFComplex(np.diag((
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

    refrkf.initial_state(imu_data[:50, 1:7],
                         pos=initial_pos,
                         ori=initial_orientation)

    zv_state = GLRT_Detector(imu_data[:, 1:7],
                             sigma_a=1.0,
                             sigma_g=1.0 * np.pi / 180.0,
                             gamma=200.0,
                             gravity=9.8,
                             time_Window_size=10)

    uwb_index = 0

    uwb_R_rekf = np.zeros_like(uwb_data)
    uwb_R_iekf = np.zeros_like(uwb_data)
    uwb_R_mckf = np.zeros_like(uwb_data)

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
        refrkf.state_transaction_function(imu_data[i, 1:7],
                                          np.diag((0.01, 0.01, 0.01,
                                                   0.01 * np.pi / 180.0,
                                                   0.01 * np.pi / 180.0,
                                                   0.01 * np.pi / 180.0))
                                          )

        if (i > 5) and (i < imu_data.shape[0] - 5):
            # print('i:',i)
            # zv_state[i] = z_tester.GLRT_Detector(imu_data[i - 4:i + 4, 1:8])
            refrkf.measurement_reftrace(0.01, ref_score)
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

                refrkf.measurement_function_zv(np.asarray((0, 0, 0)),
                                               np.diag((0.0001, 0.0001, 0.0001)))

            if uwb_data[uwb_index, 0] < imu_data[i, 0]:

                if uwb_index < uwb_data.shape[0] - 1:
                    # orkf.measurement_uwb_robust_multi(np.asarray(uwb_data[uwb_index, 1:]),
                    #                                   np.ones(1) * 0.1,
                    #                                   beacon_set,
                    #                                   6.0)
                    # rkf.measurement_uwb_mc(np.asarray(uwb_data[uwb_index,1:]),
                    #                        np.ones(1)*0.01,
                    #                        beacon_set, ref_trace)
                    # orkf.measurement_uwb_mc(np.asarray(uwb_data[uwb_index, 1:]),
                    #                         np.ones(1) * 0.01,
                    #                         beacon_set, ref_trace)
                    orkf.measurement_uwb_sp(np.asarray(uwb_data[uwb_index, 1:]),
                                            np.ones(1) * 0.01,
                                            beacon_set, ref_trace)
                    #
                    # orkf.measurement_uwb_sp_itea(np.asarray(uwb_data[uwb_index, 1:]),
                    #                              np.ones(1) * 0.01,
                    #                              beacon_set, ref_trace)
                    # orkf.measurement_uwb_mc_itea(np.asarray(uwb_data[uwb_index, 1:]),
                    #                              np.ones(1) * 0.01,
                    #                              beacon_set, ref_trace)

                    #
                    tmp_index = 0
                    for k in range(1, uwb_data.shape[1]):

                        if uwb_data[uwb_index, k] > 0.0 and beacon_set[k - 1, 0] < 5000.0:
                            uwb_R_mckf[uwb_index, k] = orkf.R_k[tmp_index, tmp_index] * 1.0

                            tmp_index += 1

                    refrkf.measurement_uwb_iterate(np.asarray(uwb_data[uwb_index, 1:]),
                                                   np.ones(1) * 0.01,
                                                   beacon_set, ref_trace)
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
                                uwb_filter_list[j - 1].measurement_func_robust(uwb_data[uwb_index, j], 0.01, 3.0, 1.0)
                                # uwb_filter_list[j - 1].measurement_func(uwb_data[uwb_index, j], 1.0, 3.0, 1.0)
                                # uwb_est_data[uwb_index, j] = uwb_filter_list[j-1].m
                                if np.linalg.norm(uwb_filter_list[j - 1].beacon_set - beacon_set[j - 1, :]) > 0.1:
                                    print('error', uwb_filter_list[j - 1].beacon_set, beacon_set[j - 1, :])

                            kf.measurement_uwb(np.asarray(uwb_data[uwb_index, j]),
                                               np.ones(1) * 0.5,
                                               np.transpose(beacon_set[j - 1, :]))
                            rkf.measurement_uwb_robust(uwb_data[uwb_index, j],
                                                       np.ones(1) * 0.5,
                                                       np.transpose(beacon_set[j - 1, :]),
                                                       j, 8.0, -10.0)
                            uwb_R_rekf[uwb_index, j] = rkf.R_k[0] * 1.0

                            # if uwb_filter_list[j-1].cov<0.02:
                            #     rkf.measurement_uwb(uwb_filter_list[j - 1].m,
                            #                         uwb_filter_list[j - 1].cov,
                            #                         np.transpose(beacon_set[j - 1, :]))
                    drkf.measurement_uwb_iterate(np.asarray(uwb_data[uwb_index, 1:]),
                                                 np.ones(1) * 0.01,
                                                 beacon_set, ref_trace)
                    # drkf.measurement_uwb_sp(np.asarray(uwb_data[uwb_index, 1:]),
                    #                         np.ones(1) * 0.01,
                    #                         beacon_set, ref_trace)
                    tmp_index = 0
                    for k in range(1, uwb_data.shape[1]):
                        if uwb_data[uwb_index, k] > 0.0 and beacon_set[k - 1, 0] < 5000.0:
                            uwb_R_iekf[uwb_index, k] = drkf.R_k[tmp_index, tmp_index] * 1.0
                            tmp_index += 1

                    uwb_ref_trace[uwb_index, :] = refrkf.state[0:3]

                    uwb_index += 1
                    # uwb_est_data[uwb_index, 1:] = np.linalg.norm(rkf.state[0:3] - beacon_set)
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
        reftrace[i, :] = refrkf.state[0:3]

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
    # plt.rc('text',usetex=True)

    # color_dict = {
    #     'ref': 'gray',
    #     'UWB': 'lightblue',
    #     'Foot': 'darkorange',
    #     'Standard': 'seagreen',
    #     'REKF': 'blue',
    #     'RIEKF': 'red'
    # }
    color_dict = {
        'ref': 'gray',
        'UWB': 'lightblue',
        'Foot': 'darkorange',
        'Standard': 'blue',
        'REKF': 'red',
        'RIEKF': 'seagreen'
    }
    plt.figure()
    # for i in range(beacon_set.shape[0]):
    #     if uwb_data[:,i+1].max() > 0 and beacon_set[i, 0] < 5000.0:
    #         plt.plot(uwb_data[:, 0] - uwb_data[0, 0], uwb_data[:, i+1],'+', label='id:' + str(i-uwb_id_offset))
    for i in range(len(uwb_valid)):
        plt.plot(uwb_data[:, 0] - uwb_data[0, 0], uwb_data[:, uwb_valid[i]], '+', label='id:' + str(i))
    # plt.grid()
    plt.legend()
    plt.xlabel('Time/s')
    plt.ylabel('Range/m')
    plt.title('UWB measurements')
    plt.tight_layout()
    plt.xlim(0.0, uwb_data[-1, 0] - uwb_data[0, 0] + 10.0)
    plt.ylim(0.0, np.max(uwb_data[:, 1:]) + 2.0)
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

    # plt.figure()
    # plt.plot(trace[:, 0], trace[:, 1], '-', label='fusing')
    # plt.plot(ftrace[:, 0], ftrace[:, 1], '-', label='foot')
    # plt.plot(rtrace[:, 0], rtrace[:, 1], '-', label='robust')
    # plt.plot(ortrace[:, 0], ortrace[:, 1], '-', label='own robust')
    # # plt.plot(dtrace[:, 0], dtrace[:, 1], '-', label='d ekf')
    # plt.plot(uwb_trace[:, 0], uwb_trace[:, 1], '+', label='uwb')
    # # plt.plot(ref_trace[:, 1], ref_trace[:, 2], '-', label='ref')
    # for i in range(beacon_set.shape[0]):
    #     plt.text(beacon_set[i, 0], beacon_set[i, 1], s=str(i + 1))
    # plt.legend()
    # plt.grid()

    # plt.figure()
    # plt.subplot(411)
    # plt.title('uwb R mc')
    # plt.plot(uwb_R_mckf[:, 1:])
    # plt.grid()
    #
    # plt.subplot(412)
    # plt.title('uwb R rekf')
    # plt.plot(uwb_R_rekf[:, 1:])
    # plt.grid()
    #
    # plt.subplot(413)
    # plt.title('uwb R iekf')
    # plt.plot(uwb_R_iekf[:, 1:])
    # plt.grid()
    #
    # plt.subplot(414)
    # plt.title('ref uwb')
    # ref_uwb_m = np.zeros_like(uwb_data[:, 1:])
    # diff_uwb_m = np.zeros_like(ref_uwb_m)
    # for i in range(ref_uwb_m.shape[1]):
    #     if beacon_set[i, 0] < 5000.0 and uwb_data[:, i + 1].max() > 0.0:
    #         ref_uwb_m[:, i] = np.linalg.norm(uwb_ref_trace - beacon_set[i, :], axis=1)
    #     # ref_uwb_m[uwb_data[:,i+1]>0.0,i] = 0.0
    #     for j in range(uwb_data.shape[0]):
    #         if uwb_data[j, i + 1] > 0.0 and beacon_set[i, 0] < 5000.0:
    #             diff_uwb_m[j, i] = ref_uwb_m[j, i] - uwb_data[j, i + 1]
    # plt.ylim([-10.0, 10.0])

    # plt.plot(ref_uwb_m)
    # plt.plot(uwb_data[:,1:])
    # plt.plot(diff_uwb_m)

    local_fsize = 12
    plt.figure()
    plt.subplot(121)
    plt.title('(a)')
    for i in range(ref_vis.shape[0]):
        plt.plot([ref_vis[i, 0], ref_vis[i, 2]], [ref_vis[i, 1], ref_vis[i, 3]], '-', color=color_dict['ref'],
                 alpha=0.5, lw='10')
    plt.plot(trace[:, 0], trace[:, 1], '-', color=color_dict['Standard'], label='Standard EKF')
    # plt.plot(ftrace[:, 0], ftrace[:, 1], '-', color=color_dict['Foot'], label='Foot')
    plt.plot(rtrace[:, 0], rtrace[:, 1], '-', color=color_dict['REKF'], label='R-EKF')
    plt.plot(ortrace[:, 0], ortrace[:, 1], '-', color=color_dict['RIEKF'], label='SP-EKF')
    # plt.plot(reftrace[:, 0], reftrace[:, 1], '-', label='ref_trace by rkf')
    # plt.plot(dtrace[:, 0], dtrace[:, 1], '-+', label='d ekf')
    # plt.plot(uwb_trace[:, 0], uwb_trace[:, 1], '+', color=color_dict['UWB'], label='UWB')
    # plt.plot(ref_trace[:, 1], ref_trace[:, 2], '-', label='ref')
    # for i in range(beacon_set.shape[0]):
    #     if uwb_data[i + 1, :].max() > 0 and beacon_set[i, 0] < 5000.0:
    #         plt.text(beacon_set[i, 0], beacon_set[i, 1], s=str(i + 1))

    for i in range(len(uwb_valid)):
        if uwb_data[:, uwb_valid[i]].max() > 0:
            plt.text(beacon_set[uwb_valid[i] - 1, 0], beacon_set[uwb_valid[i] - 1, 1], s=str(i))
            plt.plot(beacon_set[uwb_valid[i] - 1, 0], beacon_set[uwb_valid[i] - 1, 1], 'r*')
    plt.xlabel('X/m', fontsize=local_fsize)
    plt.ylabel('Y/m', fontsize=local_fsize)
    plt.xticks(fontsize=local_fsize)
    plt.yticks(fontsize=local_fsize)
    plt.legend(fontsize=local_fsize)
    plt.axis('equal')
    # plt.legend()
    # plt.grid()

    # plt.figure()
    # # for i in range(beacon_set.shape[0]):
    # #     if uwb_data[:,i+1].max() > 0 and beacon_set[i, 0] < 5000.0:
    # #         plt.plot(uwb_data[:, 0] - uwb_data[0, 0], uwb_data[:, i+1],'+', label='id:' + str(i-uwb_id_offset))
    # for i in range(len(uwb_valid)):
    #     plt.plot(uwb_data[:, 0] - uwb_data[0, 0], uwb_data[:, uwb_valid[i]], '+', label='id:' + str(i))
    # # plt.grid()
    # plt.legend()
    # plt.xlabel('Time/s')
    # plt.ylabel('Range/m')
    # plt.title('UWB measurements')
    # plt.tight_layout()
    # plt.xlim(0.0, uwb_data[-1, 0] - uwb_data[0, 0] + 10.0)
    # plt.ylim(0.0, np.max(uwb_data[:, 1:]) + 2.0)

    rs = Refscor(dir_name)
    # plt.figure()

    start_time = time.time()
    u_error = np.linalg.norm(uwb_trace[:, 0:2] - uwb_ref_trace[:, 0:2], axis=1)
    f_error = np.linalg.norm(ftrace[:, 0:2] - reftrace[:, 0:2], axis=1)
    t_error = np.linalg.norm(trace[:, 0:2] - reftrace[:, 0:2], axis=1)
    r_error = np.linalg.norm(rtrace[:, 0:2] - reftrace[:, 0:2], axis=1)
    d_error = np.linalg.norm(dtrace[:, 0:2] - reftrace[:, 0:2], axis=1)
    or_error = np.linalg.norm(ortrace[:, 0:2] - reftrace[:, 0:2], axis=1)
    # plt.plot(u_error, label='uwb')
    # plt.plot(t_error, '-', color=color_dict['Standard'], label='Standard EKF')
    # plt.plot(r_error, '-', color=color_dict['REKF'], label='Robust EKF')
    # plt.plot(or_error, '-', color=color_dict['RIEKF'], label='Robust mcEKF')
    # plt.plot(d_error, label='dtrace')
    # plt.plot(rs.eval_points(ref_trace[:,1:]), label='ref')
    # plt.grid()
    # plt.legend()
    # plt.xlabel('Time step')
    # plt.ylabel('MSE/m')
    # plt.xlim(0, trace.shape[0])
    # plt.ylim(ymin=0.0)
    # plt.title('MSE')

    print('dir name:', dir_name)
    print('uwb:', np.mean(u_error), np.std(u_error))
    print('foot:', np.mean(f_error), np.std(f_error))
    print('fusing:', np.mean(t_error), np.std(t_error))
    print('rtrace:', np.mean(r_error), np.std(r_error))
    print('mc rtrace:', np.mean(or_error), np.std(or_error))
    print('dtrace(riekf):', np.mean(rs.eval_points(dtrace)))
    # print('ref:', np.mean(rs.eval_points(ref_trace[:, 1:])))
    print('eval cost time:', time.time() - start_time)

    import statsmodels.api as sm

    #
    ecdf_f = sm.distributions.ECDF(f_error)
    ecdf_t = sm.distributions.ECDF(t_error)
    ecdf_r = sm.distributions.ECDF(r_error)
    ecdf_or = sm.distributions.ECDF(or_error)
    x = np.linspace(0.0, max(np.max(r_error), np.max(or_error)))
    # plt.figure()
    plt.subplot(122)
    plt.title('(b)')

    # plt.step(x, ecdf_f(x), color=color_dict['Foot'], label='Foot')
    plt.step(x, ecdf_t(x), color=color_dict['Standard'], label='Standard EKF')
    plt.step(x, ecdf_r(x), color=color_dict['REKF'], label='R-EKF')
    plt.step(x, ecdf_or(x), color=color_dict['RIEKF'], label='SP-EKF')
    #
    plt.legend(fontsize=local_fsize)
    plt.xlabel('RMSE/m', fontsize=local_fsize)
    plt.ylabel('cumulative probability', fontsize=local_fsize)
    plt.xticks(fontsize=local_fsize)
    plt.yticks(fontsize=local_fsize)
    plt.xlim(xmin=0.0, xmax=2.5)
    plt.ylim(0.0, 1.0)
    # plt.grid()

    plt.figure()
    plt.title('uwb')
    for i in range(1, uwb_data.shape[1]):
        plt.plot(uwb_data[:, 0], uwb_data[:, i], '+-', label=str(i))
    plt.plot(uwb_data[:, 0], uwb_opt_res, '+-', label='res error')
    plt.grid()
    plt.legend()
    plt.figure()
    plt.subplot(311)
    plt.title('(a)')
    plt.plot(uwb_R_mckf[:, 1:])
    plt.ylabel('sqrt(R)/m', fontsize=local_fsize)
    plt.ylim([0.0, 20.0])
    # plt.grid()

    plt.subplot(312)
    plt.title('(b)')
    plt.plot(uwb_R_rekf[:, 1:])
    plt.ylabel(r'sqrt(R)/m', fontsize=local_fsize)
    plt.ylim([0.0, 20.0])
    # plt.grid()

    # plt.subplot(413)
    # plt.title('uwb R iekf')
    # plt.plot(uwb_R_iekf[:, 1:])
    # plt.grid()

    plt.subplot(313)
    plt.title('(c)')
    ref_uwb_m = np.zeros_like(uwb_data[:, 1:])
    diff_uwb_m = np.zeros_like(ref_uwb_m)
    for i in range(ref_uwb_m.shape[1]):
        if beacon_set[i, 0] < 5000.0 and uwb_data[:, i + 1].max() > 0.0:
            ref_uwb_m[:, i] = np.linalg.norm(uwb_ref_trace - beacon_set[i, :], axis=1)
        # ref_uwb_m[uwb_data[:,i+1]>0.0,i] = 0.0
        for j in range(uwb_data.shape[0]):
            if uwb_data[j, i + 1] > 0.0 and beacon_set[i, 0] < 5000.0:
                diff_uwb_m[j, i] = ref_uwb_m[j, i] - uwb_data[j, i + 1]
                if diff_uwb_m[j, i] > 30.0:
                    diff_uwb_m[j, i] = 0.0
    # plt.ylim((-20.0,10.0))
    # plt.ylim(ymax=25.0)
    plt.ylabel('Diff/m', fontsize=local_fsize)
    plt.xlabel('Time Step', fontsize=local_fsize)
    plt.ylim([0.0, 20.0])
    plt.subplots_adjust(hspace=0.5)

    # plt.plot(ref_uwb_m)
    # plt.plot(uwb_data[:,1:])
    plt.plot(np.abs(diff_uwb_m))

    plt.figure()

    plt.subplot(211)
    for i in range(len(uwb_valid)):
        if np.max(uwb_data[:,uwb_valid[i]])>0:
            plt.plot(uwb_data[:, 0] - uwb_data[0, 0], uwb_data[:, uwb_valid[i]], '+', label='id:' + str(i))
        # plt.grid()
    plt.legend()
    plt.xlabel('Time/s')
    plt.ylabel('Range/m')
    plt.title('(a)')
    plt.tight_layout()
    # plt.xlim(0.0, uwb_data[-1, 0] - uwb_data[0, 0] + 10.0)
    plt.ylim(0.0, np.max(uwb_data[:, 1:]) + 2.0)

    plt.subplot(212)
    # plt.plot(uwb_data[:, 0] - uwb_data[0, 0], u_error, label='uwb')
    plt.plot(imu_data[:, 0] - uwb_data[0, 0], t_error, '-', color=color_dict['Standard'], label='Standard EKF')
    plt.plot(imu_data[:, 0] - uwb_data[0, 0], r_error, '-', color=color_dict['REKF'], label='R-EKF')
    plt.plot(imu_data[:, 0] - uwb_data[0, 0], or_error, '-', color=color_dict['RIEKF'], label='SP-EKF')
    # plt.plot(d_error, label='dtrace')
    # plt.plot(rs.eval_points(ref_trace[:,1:]), label='ref')
    plt.grid()
    plt.legend()
    plt.xlabel('Times/s')
    plt.ylabel('RMSE/m')
    # plt.xlim(0, trace.shape[0])
    plt.ylim(ymin=0.0)
    plt.title('(b)')

    plt.show()
