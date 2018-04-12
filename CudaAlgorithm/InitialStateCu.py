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


# @cuda.jit
# def quaternion_add_euler(q, euler):
#     Theta = cuda.shared.array([4, 4], dtype=np.float32)
#
#     delta_euler = (euler[0] * euler[0] + euler[1] * euler[1] + euler[2] + euler[2])
#     delta_euler = math.sqrt(delta_euler)
#
#     c = math.cos(delta_euler / 2.0)
#     sdiv = math.sin(delta_euler / 2.0) / delta_euler
#
#     Theta[0, 0] = c
#     Theta[0, 1] = -euler[0] * sdiv
#     Theta[0, 2] = -euler[1] * sdiv
#     Theta[0, 3] = -euler[2] * sdiv
#
#     Theta[1, 0] = euler[0] * sdiv
#     Theta[1, 1] = c
#     Theta[1, 2] = euler[2] * sdiv
#     Theta[1, 3] = -euler[1] * sdiv
#
#     Theta[2, 0] = euler[1] * sdiv
#     Theta[2, 1] = -euler[2] * sdiv
#     Theta[2, 2] = c
#     Theta[2, 3] = euler[0] * sdiv
#
#     Theta[3, 0] = euler[2] * sdiv
#     Theta[3, 1] = euler[1] * sdiv
#     Theta[3, 2] = -euler[0] * sdiv
#     Theta[3, 3] = c
#
#     tq = cuda.device_array(4)
#     tq = q
#     norm_new_q = 0.0
#     for i in range(4):
#         q[i] = 0.0
#         for j in range(4):
#             q[i] += Theta[i, j] * tq[j]
#         norm_new_q += q[i] * q[i]
#     norm_new_q = math.sqrt(norm_new_q)
#
#     for i in range(4):
#         q[i] = q[i] / norm_new_q


@cuda.jit
def sample(q_array, input, sigma, rng):
    pos = cuda.grid(1)

    euler = cuda.local.array(shape=(3), dtype=float32)

    if pos < q_array.shape[1]:
        for i in range(euler.shape[0]):
            euler[i] = input[i] + (xoroshiro128p_uniform_float32(rng, pos) - 0.5)
        # Theta = cuda.shared.array([4, 4], dtype=np.float32)
        Theta = cuda.local.array(shape=(4, 4), dtype=float32)

        delta_euler = (euler[0] * euler[0] + euler[1] * euler[1] + euler[2] + euler[2])
        delta_euler = math.sqrt(delta_euler)

        c = math.cos(delta_euler / 2.0)
        sdiv = math.sin(delta_euler / 2.0) / delta_euler

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

        tq = cuda.local.array(4, dtype=float32)
        tq = q_array[:,pos]
        norm_new_q = 0.0
        for i in range(4):
            q_array[i, pos] = 0.0

            for j in range(4):
                q_array[i, pos] += Theta[i, j] * tq[j]
            norm_new_q += q_array[i, pos] * q_array[i, pos]
        norm_new_q = math.sqrt(norm_new_q)

        norm_new_q = 0.5
        for i in range(3):
            q_array[i, pos] = q_array[i, pos] / norm_new_q


if __name__ == '__main__':
    dir_name = '/home/steve/Data/NewFusingLocationData/0013/'
    imu_data = np.loadtxt(dir_name + 'RIGHT_FOOT.data', delimiter=',')
    imu_data = imu_data[:, 1:]
    imu_data[:, 1:4] *= 9.81
    imu_data[:, 4:7] *= (np.pi / 180.0)
    print(numba.__version__)
    block_num = 1024
    thread_pre_block = 32

    particle_num = block_num * thread_pre_block
    print("particle num is", particle_num)
    state_num = 10 + 6 + 6
    input_num = 6
    cuda.profile_start()

    rng_states = create_xoroshiro128p_states(block_num * thread_pre_block, seed=1)

    q_state = cuda.device_array([4, particle_num], dtype=np.float32)

    initial_unit_q[block_num, thread_pre_block](q_state)
    print('after initial')

    input_array = cuda.device_array(input_num - 3, dtype=np.float32)
    input_array = cuda.to_device(np.zeros(input_num - 3))

    print('before create euler')
    euler_array = cuda.device_array([3, particle_num], dtype=np.float32)

    print('befoer sample')
    sample[block_num,thread_pre_block](q_state, input_array, 0.0, rng_states)
    print('after sample')
    # print(q_state.to_host())
    q_state_host = np.empty(shape=q_state.shape, dtype=q_state.dtype)
    # q_state_host = q_state.to_host()
    q_state.copy_to_host(q_state_host)
    plt.figure()
    plt.plot(q_state_host.transpose(),label='q')
    plt.legend()
    plt.show()
    cuda.profile_stop()
