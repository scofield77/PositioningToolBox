# -*- coding:utf-8 -*-
# Created by steve @ 18-3-18 下午7:15
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


from PositioningAlgorithm.BayesStateEstimation.KalmanFIlterBase import *

from scipy.optimize import minimize

from AlgorithmTool.ImuTools import settings, zero_velocity_tester


# @jit
def imu_state_update(state: np.ndarray,
                     input: np.ndarray,
                     time_interval=0.01) -> np.ndarray:
    '''
    state transaction method base on transform3d library.
    :param state:
    :param input:
    :param time_interval:
    :return:
    '''
    q = euler.euler2quat(state[6], state[7], state[8])
    # r_q = euler.euler2quat(input[])
    w = input[3:6] * time_interval
    r_q = euler.euler2quat(w[0], w[1], w[2])
    q = q * r_q
    # q.normalize()
    # q = q.norm()
    quaternions.qmult(q, r_q)

    # acc = np.dot(quaternions.quat2mat(q), input[0:3]) + np.asarray((0, 0, -9.8))
    acc = quaternions.rotate_vector(input[0:3], q) + np.asarray((0, 0, -9.8))
    state[0:3] = state[0:3] + state[3:6] * time_interval
    state[3:6] = state[3:6] + acc * time_interval
    # state[6:9] = quaternions.quat2axangle(q)
    state[6:9] = euler.quat2euler(q)

    return state


# @jit
def zero_velocity_measurement(state):
    '''
    zero velocity measurement function.
    :param state:
    :return:
    '''
    return state[3:6]


# @jit
def update_function(state,
                    dx):
    '''
    update state according to delta x.
    :param state:
    :param dx:
    :return:
    '''
    state[:6] += dx[:6]
    q = euler.euler2quat(state[6], state[7], state[8])
    # r_q = euler.euler2quat(input[])
    # w = input[3:6] * time_interval
    w = dx[6:9]
    r_q = euler.euler2quat(w[0], w[1], w[2])
    r_q = r_q / quaternions.qnorm(r_q)
    # q = q * r_q
    q = quaternions.qmult(q, r_q)
    # q = q.fillpositive
    # q = q.norm()
    q = q / quaternions.qnorm(q)
    state[6:9] = euler.quat2euler(q)

    return state


# @jit("float[:](float[:],float[:],float,int)")
@jit
def get_initial_state(imu_data,
                      initial_pose,
                      initial_yaw,
                      state_num):
    state = np.zeros(state_num)
    if state_num is 9:
        state[0:3] = initial_pose
        state[3:6] = np.zeros(3)

        acc = imu_data.mean(axis=0)

        def error_func(w: np.ndarray) -> float:
            q = euler.euler2quat(w[0], w[1], initial_yaw)
            # print(quaternions.rotate_vector(acc, q))
            return np.linalg.norm(quaternions.rotate_vector(acc, q) - np.asarray((0, 0, np.linalg.norm(acc))))

        result = minimize(error_func, np.asarray((0, 0)))
        # print(result)
        w = result.x
        state[6] = w[0]
        state[7] = w[1]
        state[8] = initial_yaw

    return state


if __name__ == '__main__':
    dir_name = '/home/steve/Data/NewFusingLocationData/0013/'
    imu_data = np.loadtxt(dir_name + 'RIGHT_FOOT.data', delimiter=',')
    imu_data = imu_data[:, 1:]
    imu_data[:, 1:4] *= 9.81
    imu_data[:, 4:7] *= (np.pi / 180.0)

    initial_state = get_initial_state(imu_data[:40, 1:4], np.asarray((0, 0, 0)), 0.0, 9)

    trace = np.zeros([imu_data.shape[0], 3])
    zv_state = np.zeros([imu_data.shape[0], 1])

    kf = KalmanFilterBase(9)
    kf.state_x = initial_state
    kf.state_prob = np.diag((0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001))

    set_setting = settings()
    set_setting.sigma_a *= 5.0
    set_setting.sigma_g *= 5.0

    z_tester = zero_velocity_tester(set_setting)
    zv_state = z_tester.GLRT_Detector(imu_data[:, 1:7])

    for i in range(imu_data.shape[0]):
        def inter_imu(s, i):
            return imu_state_update(s, i, 0.01)


        kf.state_transaction_function(imu_data[i, 1:7],
                                      np.diag((0.01, 0.01, 0.01, 0.02, 0.02, 0.02)),
                                      inter_imu)
        if (i > 5) and (i < imu_data.shape[0] - 5):
            # print('i:',i)
            # zv_state[i] = z_tester.GLRT_Detector(imu_data[i - 4:i + 4, 1:8])
            if zv_state[i] > 0.5:
                kf.measurement_function(np.asarray((0, 0, 0)),
                                        np.diag((0.0001, 0.0001, 0.0001)),
                                        zero_velocity_measurement,
                                        update_function)

        # print(kf.state_x)
        # print( i /)
        trace[i, :] = kf.state_x[0:3]
        rate = i / imu_data.shape[0]

        print('finished:', rate * 100.0, "% ", i, imu_data.shape[0])


    def aux_plot(data: np.ndarray, name: str):
        plt.figure()
        plt.title(name)
        plt.plot(zv_state, 'r-', label='zv state')
        for i in range(data.shape[1]):
            plt.plot(data[:, i], label=str(i))
        plt.grid()
        plt.legend()


    aux_plot(imu_data[:, 1:4], 'acc')
    aux_plot(imu_data[:, 4:7], 'gyr')
    aux_plot(imu_data[:, 7:10], 'mag')
    aux_plot(trace, 'trace')

    plt.figure()
    plt.plot(trace[:, 0], trace[:, 1], '-+')
    plt.grid()

    plt.figure()
    # plt.show()
