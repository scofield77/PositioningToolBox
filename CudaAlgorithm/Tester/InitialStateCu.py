# -*- coding:utf-8 -*-
# carete by steve at  2018 / 04 / 11　下午9:47
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
import matplotlib.pyplot  as plt

import numpy as np
import scipy as sp

from numba import jit

from numba import cuda

from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_normal_float64
import numba

from numba import float32, float64

import math

# import pyculib
from pyculib import blas as cublas






if __name__ == '__main__':
    print(cuda.devices._runtime.gpus)
    dir_name = '/home/steve/Data/NewFusingLocationData/0013/'
    imu_data = np.loadtxt(dir_name + 'RIGHT_FOOT.data', delimiter=',')
    imu_data = imu_data[:, 1:]
    imu_data[:, 1:4] *= 9.81
    imu_data[:, 4:7] *= (np.pi / 180.0)

    '''
    Prepare cuda parameters
    '''
    print(numba.__version__)
    block_num = 512
    thread_pre_block = 1024

    particle_num = block_num * thread_pre_block
    print("particle num is", particle_num, 'block num:', block_num, 'thread pre block:', thread_pre_block)
    state_num = 10 + 6 + 6
    input_num = 6
    cuda.profile_start()

    rng_states = create_xoroshiro128p_states(block_num * thread_pre_block, seed=1)
    # rng_states =

    qt = np.zeros((4, particle_num))
    q_state = cuda.device_array([4, particle_num], dtype=np.float64)  # ,order='F')

    # q_state = cuda.devicearray.DeviceNDArray(shape=qt.shape, strides=qt.strides, dtype=qt.dtype,order)
    q_weight = cuda.device_array([particle_num], dtype=q_state.dtype)

    initial_unit_q[block_num, thread_pre_block](q_state)
    init_weight[block_num, thread_pre_block](q_weight)

    input_array = cuda.device_array(input_num - 3, dtype=q_state.dtype)
    input_array = cuda.to_device(np.zeros(input_num - 3))

    euler_array = cuda.device_array([3, particle_num], dtype=q_state.dtype)

    # sample from
    sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)

    ave_q = cuda.device_array([4], dtype=q_state.dtype)
    ave_q = cuda.to_device(np.zeros(4, dtype=q_state.dtype))
    ave_q_buffer = cuda.device_array([4, particle_num], dtype=q_state.dtype)

    average_quaternion_simple[block_num, thread_pre_block](q_state, q_weight, ave_q_buffer, ave_q)

    q_state_host = np.empty(shape=q_state.shape, dtype=q_state.dtype)
    q_weight_host = np.empty(shape=q_weight.shape, dtype=q_weight.dtype)
    ave_q_buffer_host = np.empty(shape=ave_q_buffer.shape, dtype=ave_q_buffer.dtype)

    q_state.copy_to_host(q_state_host)
    q_weight.copy_to_host(q_weight_host)
    ave_q_buffer.copy_to_host(ave_q_buffer_host)

    ave_q_host = ave_q.copy_to_host()

    print('ave q norm:', np.linalg.norm(ave_q_host))
    print("ave q:", ave_q_host, "ave q normlized", ave_q_host / np.linalg.norm(ave_q_host))

    print('sum:', q_state_host.sum())
    print('std q state host:', q_state_host.std(axis=1))
    print('sum of weight:', q_weight_host.sum())

    print('cpu ave q norm:', np.linalg.norm((q_state_host * q_weight_host).sum(axis=1)))
    print('cpu ave:', (q_state_host * q_weight_host).sum(axis=1))
    print('cpu ave normed:',
          (q_state_host * q_weight_host).sum(axis=1) / np.linalg.norm((q_state_host * q_weight_host).sum(axis=1)))

    '''
    Test new quaternion weighte average method in cpu 
    Such algorithm isn't better than gpu average with orientation .
    '''
    # all_q = q_state_host
    #
    # Q = all_q.dot(all_q.transpose()) / float(all_q.shape[0])
    #
    # QQt = Q.dot(Q.transpose())
    #
    # # eigen_val = np.linalg.qr(QQt)
    # w, v = np.linalg.eig(QQt)

    # max_vector =
    # print('w:', w)
    # print('v:', v)
    '''
    End Test new quaternion weighted average method in cpu
    '''

    cuda.profile_stop()

    # plt.figure()
    # plt.plot(q_weight_host)
    # plt.grid()
    #
    # plt.figure()
    # plt.plot(q_state_host.transpose(), label='q')
    # plt.plot(np.linalg.norm(q_state_host, axis=0), label='norm')
    # plt.legend()
    # plt.grid()
    #
    # plt.figure()
    # plt.plot(np.linalg.norm(q_state_host, axis=0), label='norm')
    # plt.grid()
    # plt.legend()
    #
    # plt.figure()
    # plt.title('ave q buffer')
    # plt.plot(ave_q_buffer_host.transpose())
    # plt.grid()
    #
    # plt.show()
