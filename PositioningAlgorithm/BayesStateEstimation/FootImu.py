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
import math
import matplotlib.pyplot as plt

from transforms3d import euler, quaternions

from PositioningAlgorithm.BayesStateEstimation.KalmanFIlterBase import *

from scipy.optimize import minimize


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
    quaternions.qmult(q,r_q)


    # acc = np.dot(quaternions.quat2mat(q), input[0:3]) + np.asarray((0, 0, -9.8))
    acc = quaternions.rotate_vector(q,input[0:3]) + np.asarray((0, 0, -9.8))
    state[0:3] = state[0:3] + state[3:6] * time_interval
    state[3:6] = state[3:6] + acc * time_interval
    state[6:9] = quaternions.quat2axangle(q)

    return state


def zero_velocity_measurement(state: np.ndarray) -> np.ndarray:
    '''
    zero velocity measurement function.
    :param state:
    :return:
    '''
    return state[3:6]


def update_function(state: np.ndarray,
                    dx: np.ndarray) -> np.ndarray:
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
    r_q = r_q.norm()
    q = q * r_q
    # q = q.fillpositive
    q = q.norm()
    state[6:9] = quaternions.quat2axangle(q)

    return state


def get_initial_state(imu_data: np.ndarray,
                      initial_pose: np.ndarray,
                      initial_yaw: float,
                      state_num: int) -> np.ndarray:
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
        print(result)
        w = result.x
        state[6] = w[0]
        state[7] = w[1]
        state[8] = initial_yaw

    return state


if __name__ == '__main__':
    dir_name = '/home/steve/Data/NewFusingLocationData/0013/'
    imu_data = np.loadtxt(dir_name + 'LEFT_FOOT.data', delimiter=',')
    imu_data = imu_data[:, 1:]
    imu_data[:, 1:4] *= 9.81
    imu_data[:, 4:7] *= (np.pi / 180.0)

    initial_state = get_initial_state(imu_data[:40, 1:4], np.asarray((0, 0, 0)), 0.0, 9)

    trace = np.zeros([imu_data.shape[0], 3])

    kf = KalmanFilterBase(9)
    kf.state_x = initial_state
    kf.state_prob = np.diag((0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001))

    for i in range(imu_data.shape[0]):
        def inter_imu(s, i):
            return imu_state_update(s, i, 0.01)


        kf.state_transaction_function(imu_data[i, 1:7],
                                      np.diag((0.01, 0.01, 0.01, 0.02, 0.02, 0.02)),
                                      inter_imu)
        print(kf.state_x)


    def aux_plot(data: np.ndarray, name: str):
        plt.figure()
        plt.title(name)
        for i in range(data.shape[1]):
            plt.plot(data[:, i], label=str(i))
        plt.grid()
        plt.legend()


    aux_plot(imu_data[:, 1:4], 'acc')
    aux_plot(imu_data[:, 4:7], 'gyr')
    aux_plot(imu_data[:, 7:10], 'mag')
    plt.show()
