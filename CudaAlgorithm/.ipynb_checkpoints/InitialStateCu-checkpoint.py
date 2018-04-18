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

from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numba

from numba import float32

import math

# import pyculib
from pyculib import blas


@cuda.jit
def initial_unit_q(q_array):
    # tx = cuda.threadIdx.x
    # ty = cuda.blockIdx.x
    # bw = cuda.blockDim.x

    # pos = tx + ty * bw
    pos = cuda.grid(1)

    if pos < q_array.shape[1]:
        q_array[0, pos] = 1.0
        q_array[1, pos] = 0.0
        q_array[2, pos] = 0.0
        q_array[3, pos] = 0.0


@cuda.jit(device=True)
def quaternion_add_euler(q, euler):
    '''
    quaternion add function.
    :param q:
    :param euler:
    :return:
    '''
    Theta = cuda.local.array(shape=(4, 4), dtype=float32)

    delta_euler = (euler[0] * euler[0] + euler[1] * euler[1] + euler[2] + euler[2])
    delta_euler = math.sqrt(delta_euler)

    if delta_euler > 1e-19:
        c = math.cos(delta_euler / 2.0)
        sdiv = math.sin(delta_euler / 2.0) / delta_euler
    else:
        c = 1.0
        sdiv = 0.0

    Theta[0, 0] = c
    Theta[0, 1] = -euler[0] * sdiv
    Theta[0, 2] = -euler[1] * sdiv
    Theta[0, 3] = -euler[2] * sdiv

    Theta[1, 0] = euler[0] * sdiv
    Theta[1, 1] = c
    Theta[1, 2] = euler[2] * sdiv
    Theta[1, 3] = -euler[1] * sdiv

    Theta[2, 0] = euler[1] * sdiv
    Theta[2, 1] = -euler[2] * sdiv
    Theta[2, 2] = c
    Theta[2, 3] = euler[0] * sdiv

    Theta[3, 0] = euler[2] * sdiv
    Theta[3, 1] = euler[1] * sdiv
    Theta[3, 2] = -euler[0] * sdiv
    Theta[3, 3] = c

    tq = cuda.local.array(shape=(4), dtype=float32)
    # tq = q
    for i in range(4):
        tq[i] = q[i]
    norm_new_q = 0.0
    for i in range(4):
        q[i] = 0.0
        for j in range(4):
            q[i] += Theta[i, j] * tq[j]
        norm_new_q += q[i] * q[i]
    norm_new_q = math.sqrt(norm_new_q)

    for i in range(4):
        q[i] = q[i] / norm_new_q


@cuda.jit
def sample(q_array, input, sigma, rng):
    pos = cuda.grid(1)

    if pos < q_array.shape[1]:

        euler = cuda.local.array(shape=(3), dtype=float32)
        for i in range(euler.shape[0]):
            euler[i] = input[i] + (xoroshiro128p_uniform_float32(rng, pos) - 0.5)
        quaternion_add_euler(q_array[:, pos], euler)
    cuda.syncthreads()


@cuda.jit
def init_weight(q_weight):
    pos = cuda.grid(1)

    if pos < q_weight.shape[0]:
        q_weight[pos] = 1.0 / float32(q_weight.shape[0])


@cuda.jit
def average_quaternion(q_array, q_weight, q_array_buffer, average_q):
    pos = cuda.grid(1)
    if pos < q_array.shape[1]:
        for i in range(4):
            # q_array_buffer[i, pos] = q_array[i, pos] * q_weight[pos]
            t = q_array[i, pos] * q_weight[pos]
            cuda.atomic.add(average_q, i, t)
        cuda.syncthreads()






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
    block_num = 1024
    thread_pre_block = 320

    particle_num = block_num * thread_pre_block
    print("particle num is", particle_num, 'block num:', block_num, 'thread pre block:', thread_pre_block)
    state_num = 10 + 6 + 6
    input_num = 6
    cuda.profile_start()

    rng_states = create_xoroshiro128p_states(block_num * thread_pre_block, seed=1)

    q_state = cuda.device_array([4, particle_num], dtype=np.float32)
    q_weight = cuda.device_array([particle_num], dtype=np.float32)

    initial_unit_q[block_num, thread_pre_block](q_state)
    init_weight[block_num, thread_pre_block](q_weight)

    input_array = cuda.device_array(input_num - 3, dtype=np.float32)
    input_array = cuda.to_device(np.zeros(input_num - 3))

    euler_array = cuda.device_array([3, particle_num], dtype=np.float32)

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

    ave_q = cuda.device_array([4], dtype=np.float32)
    ave_q = cuda.to_device(np.zeros(4, dtype=np.float32))
    ave_q_buffer = cuda.device_array([4, particle_num], dtype=np.float32)

    average_quaternion[block_num, thread_pre_block](q_state, q_weight, ave_q_buffer, ave_q)

    q_state_host = np.empty(shape=q_state.shape, dtype=q_state.dtype)
    q_weight_host = np.empty(shape=q_weight.shape, dtype=q_weight.dtype)
    ave_q_buffer_host = np.empty(shape=ave_q_buffer.shape, dtype=ave_q_buffer.dtype)

    q_state.copy_to_host(q_state_host)
    q_weight.copy_to_host(q_weight_host)
    ave_q_buffer.copy_to_host(ave_q_buffer_host)

    ave_q_host = ave_q.copy_to_host()

    print("ave q:", ave_q_host, "ave q normlized", ave_q_host/np.linalg.norm(ave_q_host))

    print('sum:', q_state_host.sum())
    print('std q state host:',q_state_host.std(axis=1))
    print('sum of weight:', q_weight_host.sum())

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
