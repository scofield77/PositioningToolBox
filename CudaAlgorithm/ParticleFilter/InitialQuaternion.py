# -*- coding:utf-8 -*-
# carete by steve at  2018 / 04 / 22　下午7:24
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


from numba import cuda

from PositioningAlgorithm.tester.FootImu import *
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32


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
    Theta = cuda.local.array(shape=(4, 4), dtype=float64)

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

    tq = cuda.local.array(shape=(4), dtype=float64)
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

        euler = cuda.local.array(shape=(3), dtype=float64)
        for i in range(euler.shape[0]):
            # euler[i] = input[i] + (xoroshiro128p_uniform_float32(rng, pos) - 0.5)
            euler[i] = input[i] + (xoroshiro128p_normal_float64(rng, pos) * 0.1)
        quaternion_add_euler(q_array[:, pos], euler)
    cuda.syncthreads()


@cuda.jit
def init_weight(q_weight):
    pos = cuda.grid(1)

    if pos < q_weight.shape[0]:
        q_weight[pos] = 1.0 / float32(q_weight.shape[0])


@cuda.jit
def average_quaternion_simple(q_array, q_weight, q_array_buffer, average_q):
    pos = cuda.grid(1)
    tid = cuda.threadIdx.x

    sdata = cuda.shared.array(shape=(4, 1024), dtype=float64)
    if pos < q_array.shape[1]:

        if pos < 4:
            average_q[pos] = 0.0
        for i in range(4):
            sdata[i, tid] = q_array[i, pos] * q_weight[pos]

        cuda.syncthreads()
        s = cuda.blockDim.x >> 1
        while s > 0:

            if tid < s:
                for i in range(4):
                    sdata[i, tid] = sdata[i, tid] + sdata[i, tid + s]
            s = s >> 1
            cuda.syncthreads()

        if tid == 0:
            for i in range(4):
                cuda.atomic.add(average_q, i, sdata[i, 0])

        cuda.syncthreads()

        ### Calculate average consider about q or -q.
        t_dot = 0.0
        for i in range(4):
            t_dot += average_q[i] * q_array[i, pos]
        if t_dot > 0:
            t_dot = 1.0
        else:
            t_dot = -1.0
        cuda.syncthreads()
        if pos < 4:
            average_q[i] = 0.0
        cuda.syncthreads()
        for i in range(4):
            if t_dot > 0:
                sdata[i, tid] = q_array[i, pos] * q_weight[pos]
            else:
                sdata[i, tid] = q_array[i, pos] * -1.0 * q_weight[pos]
            # q_array[i, pos] = t_dot * q_array[i, pos]
        cuda.syncthreads()

        s = cuda.blockDim.x >> 1
        while s > 0:
            if tid < s:
                for i in range(4):
                    sdata[i, tid] = sdata[i, tid] + sdata[i, tid + s]
            s = s >> 1
            cuda.syncthreads()

        if tid == 0:
            for i in range(4):
                cuda.atomic.add(average_q, i, sdata[i, 0])

