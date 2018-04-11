# -*- coding:utf-8 -*-
# carete by steve at  2018 / 04 / 11　下午8:56
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

import matplotlib.pyplot

import numpy as np
import scipy as sp

from numba import jit

from numba import cuda

from PositioningAlgorithm.BayesStateEstimation.FootImu import *
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32



@cuda.jit
def initial_state(state_array: cuda.devicearry,
                  initial_state: cuda.devicearray,
                  state_prob: cuda.devicearray,
                  rng_states):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.y
    bw = cuda.blockDim.x

    pos = tx + ty * bw
    if pos < state_array.shape[0]:
        # for i in range(state_num.shape[0])
        for i in range(6):
            state_array[i, pos] =initial_state[i] + xoroshiro128p_uniform_float32(rng_states, tx)



if __name__ == '__main__':
    dir_name = '/home/steve/Data/NewFusingLocationData/0013/'
    imu_data = np.loadtxt(dir_name + 'RIGHT_FOOT.data', delimiter=',')
    imu_data = imu_data[:, 1:]
    imu_data[:, 1:4] *= 9.81
    imu_data[:, 4:7] *= (np.pi / 180.0)

    initial_state = get_initial_state(imu_data[:40, 1:4], np.asarray((0, 0, 0)), 0.0, 9)

    trace = np.zeros([imu_data.shape[0], 3])
    zv_state = np.zeros([imu_data.shape[0], 1])

    # kf = KalmanFilterBase(9)
    # kf.state_x = initial_state
    # kf.state_prob = np.diag((0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001))
    initial_prob = np.diag((0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001))

    block_num = 1024
    thread_pre_block = 64

    particle_num = block_num * thread_pre_block
    print("particle num is", particle_num)
    state_num = 10 + 6 + 6
    input_num = 6

    rng_states = create_xoroshiro128p_states(block_num * thread_pre_block, seed=1)

    state_array = cuda.device_array([particle_num, state_num])
    input_array = cuda.device_array([particle_num, input_num])
    prob_array = cuda.device_array([particle_num, 1])

    initial_state_q = np.zeros(state_num)

    initial_state_q[:6] = initial_state[:6]

    # set_setting = settings()
    # set_setting.sigma_a *= 5.0
    # set_setting.sigma_g *= 5.0

    z_tester = zero_velocity_tester(set_setting)
    zv_state = z_tester.GLRT_Detector(imu_data[:, 1:7])

    for i in range(imu_data.shape[0]):
        # def inter_imu(s, i):
        #     return imu_state_update(s, i, 0.01)

        # kf.state_transaction_function(imu_data[i, 1:7],
        #                               np.diag((0.01, 0.01, 0.01, 0.02, 0.02, 0.02)),
        #                               inter_imu)

        if (i > 5) and (i < imu_data.shape[0] - 5):
            # print('i:',i)
            # zv_state[i] = z_tester.GLRT_Detector(imu_data[i - 4:i + 4, 1:8])
            if zv_state[i] > 0.5:
        # kf.measurement_function(np.asarray((0, 0, 0)),
        #                         np.diag((0.0001, 0.0001, 0.0001)),
        #                         zero_velocity_measurement,
        #                         update_function)

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
