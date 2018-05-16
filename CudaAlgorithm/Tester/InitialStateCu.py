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

from CudaAlgorithm.ParticleFilter.InitialQuaternion import *
from CudaAlgorithm.ParticleFilter.GeneralModel import *


def q2dcm(q):
    """
    :param q:
    :return:
    """
    q /= np.linalg.norm(q)
    # p = np.zeros([6, 1])
    # # p = cuda.local.array(shape=(6), dtype=float64)
    #
    # # p[0:4] = q.reshape(4, 1) ** 2.0
    # p[0] = q[1] * q[1]
    # p[1] = q[2] * q[2]
    # p[2] = q[3] * q[3]
    # p[3] = q[0] * q[0]
    #
    # p[4] = p[1] + p[2]
    #
    # if math.fabs(p[0] + p[3] + p[4]) > 1e-18:
    #     p[5] = 2.0 / (p[0] + p[3] + p[4])
    # else:
    #     p[5] = 0.0
    # #
    R = np.zeros([3, 3])
    # #
    # R[0, 0] = 1.0 - p[5] * p[4]
    # R[1, 1] = 1.0 - p[5] * (p[0] + p[2])
    # R[2, 2] = 1.0 - p[5] * (p[0] + p[1])
    #
    # p[0] = p[5] * q[0]
    # p[1] = p[5] * q[1]
    # p[4] = p[5] * q[2] * q[3]
    # p[5] = p[0] * q[1]
    #
    # R[0, 1] = p[5] - p[4]
    # R[1, 0] = p[5] + p[4]
    #
    # p[4] = p[1] * q[3]
    # p[5] = p[0] * q[2]
    #
    # R[0, 2] = p[5] + p[4]
    # R[2, 0] = p[5] - p[4]
    #
    # p[4] = p[0] * q[3]
    # p[5] = p[1] * q[2]
    #
    # R[1, 2] = p[5] - p[4]
    # R[2, 1] = p[5] + p[4]

    ##################
    R[0, 0] = (q[0] * q[0] + q[1] * q[1] - 0.5) * 2.0
    R[0, 1] = (q[1] * q[2] - q[0] * q[3]) * 2.0
    R[0, 2] = (q[0] * q[2] + q[1] * q[3]) * 2.0

    R[1, 0] = (q[0] * q[3] + q[1] * q[2]) * 2.0
    R[1, 1] = (q[0] * q[0] + q[2] * q[2] - 0.5) * 2.0
    R[1, 2] = (q[2] * q[3] - q[0] * q[1]) * 2.0

    R[2, 0] = (q[1] * q[3] - q[0] * q[2]) * 2.0
    R[2, 1] = (q[0] * q[1] + q[2] * q[3]) * 2.0
    R[2, 2] = (q[0] * q[0] + q[3] * q[3] - 0.5) * 2.0

    # R /= np.linalg.norm(q)

    # print("R:",R,"R*R^T", R.dot(R.transpose()))
    return R


if __name__ == '__main__':
    print(numba.__version__)
    print(cuda.devices._runtime.gpus)
    cuda.profile_start()
    # dir_name = '/home/steve/Data/NewFusingLocationData/0013/'
    dir_name = 'D:/Data/NewFusingLocationData/0013/'
    imu_data = np.loadtxt(dir_name + 'RIGHT_FOOT.data', delimiter=',')
    imu_data = imu_data[:, 1:]
    imu_data[:, 1:4] *= 9.81
    imu_data[:, 4:7] *= (np.pi / 180.0)
    # imu_data = imu_data.as
    imu_data = np.ascontiguousarray(imu_data)

    fist_line = imu_data[:100, :].mean(axis=0)
    imu_data[0, :] = fist_line

    imu_data_device = cuda.device_array_like(imu_data)
    stream = cuda.stream()
    with stream.auto_synchronize():
        imu_data_device = cuda.to_device(imu_data, stream=stream)

    # imu_data_device = cuda.to_device(imu_data)
    # cuda.to_device(imu_data,copy=True,to=imu_data_device)
    # imu_data_device =

    '''
    Prepare cuda parameters
    '''

    block_num = 512
    thread_pre_block = 1024

    particle_num = block_num * thread_pre_block
    print("particle num is", particle_num, 'block num:', block_num, 'thread pre block:', thread_pre_block)
    state_num = 10 + 6 + 6
    input_num = 6
    # cuda.profile_start()

    rng_states = create_xoroshiro128p_states(block_num * thread_pre_block, seed=1)
    # rng_states =

    qt = np.zeros((4, particle_num))
    q_state = cuda.device_array([4, particle_num], dtype=np.float64)
    buffer_array = cuda.device_array([4, particle_num], dtype=q_state.dtype)
    # q_state = cuda.devicearray.DeviceNDArray(shape=qt.shape, strides=qt.strides, dtype=qt.dtype,order)
    q_weight = cuda.device_array([particle_num], dtype=q_state.dtype)

    initial_unit_q[block_num, thread_pre_block](q_state)
    init_weight[block_num, thread_pre_block](q_weight)

    input_array = cuda.device_array(input_num - 3, dtype=q_state.dtype)
    input_array = cuda.to_device(np.zeros(input_num - 3))

    euler_array = cuda.device_array([3, particle_num], dtype=q_state.dtype)

    # sample from
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)
    # sample[block_num, thread_pre_block](q_state, input_array, 0.0, rng_states)

    # quaternion_evaluate[block_num, thread_pre_block](q_state, q_weight, imu_data_device[1, 1:4],buffer_array[:,0])
    # quaternion_evaluate[block_num,thread_pre_block](q_state,q_weight,imu_data_device[1,1:4])

    tbuffer = np.zeros(buffer_array.shape)
    # tbuffer = buffer_array.to_host()
    buffer_array.copy_to_host(tbuffer)
    print('acc:', imu_data[1, 1:4])
    print(tbuffer[0, 0])
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

    weight_buff = cuda.device_array([1], dtype=q_state.dtype)

    out_acc_data = np.zeros([200, 3])

    # average_quaternion_simple[block_num, thread_pre_block](q_state, q_weight, buffer_array, ave_q)
    for i in range(out_acc_data.shape[0]):
        sample[block_num, thread_pre_block](q_state, input_array, 10.0 * np.pi / 180.0, rng_states)
        quaternion_evaluate[block_num, thread_pre_block](q_state, q_weight, imu_data_device[0, 1:4], buffer_array[:, 0])
        average_quaternion_simple[block_num, thread_pre_block](q_state, q_weight, ave_q)
        rejection_resample[block_num, thread_pre_block](q_state, buffer_array, q_weight, rng_states, weight_buff)

        q_state_host = np.empty(shape=q_state.shape, dtype=q_state.dtype)
        q_weight_host = np.empty(shape=q_weight.shape, dtype=q_weight.dtype)
        ave_q_buffer_host = np.empty(shape=buffer_array.shape, dtype=buffer_array.dtype)
        #
        # q_state.copy_to_host(q_state_host)
        # q_weight.copy_to_host(q_weight_host)
        # buffer_array.copy_to_host(ave_q_buffer_host)

        ave_q_host = ave_q.copy_to_host()

        out_acc = q2dcm(ave_q_host).dot(imu_data[1, 1:4].transpose())
        out_acc_data[i, :] = out_acc
        print('in acc:', imu_data[1, 1:4], 'out acc:', out_acc, "ave q hos:", ave_q_host, 'q norm:',
              np.linalg.norm(ave_q_host))

        # print('ave q norm:', np.linalg.norm(ave_q_host))
        # print("ave q:", ave_q_host, "ave q normlized", ave_q_host / np.linalg.norm(ave_q_host))
        #
        # print('sum:', q_state_host.sum())
        # print('std q state host:', q_state_host.std(axis=1))
        # print('sum of weight:', q_weight_host.sum())
        #
        # print('cpu ave q norm:', np.linalg.norm((q_state_host * q_weight_host).sum(axis=1)))
        # print('cpu ave:', (q_state_host * q_weight_host).sum(axis=1))
        # print('cpu ave normed:',
        #       (q_state_host * q_weight_host).sum(axis=1) / np.linalg.norm((q_state_host * q_weight_host).sum(axis=1)))

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

    plt.figure()
    plt.title('out acc')
    for i in range(3):
        plt.plot(out_acc_data[:, i], label=str(i))
    plt.plot(np.linalg.norm(out_acc_data, axis=1), label='norm')
    plt.legend()
    plt.grid()
    plt.show()
