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

from PositioningAlgorithm.BayesStateEstimation.KalmanFIlterBase import  *

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
    q = q.norm()

    # acc = np.dot(quaternions.quat2mat(q), input[0:3]) + np.asarray((0, 0, -9.8))
    acc = q.rotate_vector(input[0:3]) + np.asarray((0, 0, -9.8))
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
                      state_num=9:int) -> np.ndarray:
    state = np.zeros(state_num)
    if state_num is 9:
        state[0:3] = initial_pose
        state[3:6] = np.zeros(3)

        acc = imu_data.mean(axis=0)

        def error_func(w:np.ndarray)->float:
            q = euler.euler2quat(w[0],w[1],initial_yaw)
            return np.linalg.norm(quaternions.rotate_vector(acc,q)-np.asarray(0,0,np.linalg.norm(acc)))

        result = minimize(error_func,(0,0))
        print(result)
        w = result.x
        state[6] = w[0]
        state[7] = w[1]
        state[8]  = initial_yaw




    return state


if __name__ == '__main__':
    imu_data =
