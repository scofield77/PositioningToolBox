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
        q_weight[pos] = 1.0 / float64(q_weight.shape[0])


@cuda.jit
def average_quaternion_simple(q_array, q_weight, average_q):
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
            cuda.syncthreads()
        if pos == 0:
            q_norm = 0.0
            for i in range(4):
                q_norm += average_q[i] * average_q[i]
            q_norm = math.sqrt(q_norm)
            for i in range(4):
                average_q[i] = average_q[i] / q_norm
            cuda.syncthreads()


@cuda.jit(device=True, inline=True)
def normal_pdf(x, miu, sigma):
    return (x - miu) * (x - miu) / sigma / sigma


@cuda.jit
def Rt2b(self, ang, R):
    '''
    :
    :param ang:
    :return:
    '''
    cr = math.cos(ang[0])
    sr = math.sin(ang[0])

    cp = math.cos(ang[1])
    sp = math.sin(ang[1])

    cy = math.cos(ang[2])
    sy = math.sin(ang[2])

    # R = np.array(
    #     [[cy * cp, sy * cp, -sp],
    #      [-sy * cr + cy * sp * sr, cy * cr + sy * sp * sr, cp * sr],
    #      [sy * sr + cy * sp * cr, -cy * sr + sy * sp * cr, cp * cr]]
    # )
    # return R
    R[0, 0] = cy * cp
    R[0, 1] = sy * cp
    R[0, 2] = -sp

    R[1, 0] = -sy * cr + cy * sp * sr
    R[1, 1] = cy * cr + sy * sp * sr
    R[1, 2] = cp * sr

    R[2, 0] = sy * sr + cy * sp * cr
    R[2, 1] = -cy * sr + sy * sp * cr
    R[2, 2] = cp * cr


@cuda.jit
def dcm2q(self, R, q):
    """
    http://www.ee.ucr.edu/~farrell/AidedNavigation/D_App_Quaternions/Rot2Quat.pdf
    [1] Farrell J A. Computation of the Quaternion from a Rotation Matrix[J]. 2008.
    Transform from rotation matrix to quanternions.
    :param R:old rotation matrix
    :param q: return value
    """
    T = 1.0 + R[0, 0] + R[1, 1] + R[2, 2]
    # print (T)

    # Really Big Change.
    # ToDo:Why there are some value is smallter than zero.
    if math.fabs(T) > 1e-3:
        S = 0.5 / math.sqrt(math.fabs(T))

        qw = 0.25 / S
        qx = (R[2, 1] - R[1, 2]) * S
        qy = (R[0, 2] - R[2, 0]) * S
        qz = (R[1, 0] - R[0, 1]) * S

    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0

            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S

        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0

            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0

            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    # quart = np.array(np.transpose([qx, qy, qz, qw]))

    # quart /= np.linalg.norm(quart)
    q_norm = 0.0
    # for i in range(4):
    #     q_norm =
    q_norm = qw * qw + qx * qx + qy * qy + qz * qz
    q_norm = math.sqrt(q_norm)
    q[0] = qw / q_norm
    q[1] = qx / q_norm
    q[2] = qy / q_norm
    q[3] = qz / q_norm


@cuda.jit(device=True, inline=True)
def q2dcm(q, R):
    """
    :param q:
    :return:
    """
    # p = np.zeros([6, 1])
    p = cuda.local.array(shape=(6), dtype=float64)

    # p[0:4] = q.reshape(4, 1) ** 2.0
    p[0] = q[1] * q[1]
    p[1] = q[2] * q[2]
    p[2] = q[3] * q[3]
    p[3] = q[0] * q[0]

    p[4] = p[1] + p[2]

    if math.fabs(p[0] + p[3] + p[4]) > 1e-18:
        p[5] = 2.0 / (p[0] + p[3] + p[4])
    else:
        p[5] = 0.0

    # R = np.zeros([3, 3])

    R[0, 0] = 1.0 - p[5] * p[4]
    R[1, 1] = 1.0 - p[5] * (p[0] + p[2])
    R[2, 2] = 1.0 - p[5] * (p[0] + p[1])

    p[0] = p[5] * q[0]
    p[1] = p[5] * q[1]
    p[4] = p[5] * q[2] * q[3]
    p[5] = p[0] * q[1]

    R[0, 1] = p[5] - p[4]
    R[1, 0] = p[5] + p[4]

    p[4] = p[1] * q[3]
    p[5] = p[0] * q[2]

    R[0, 2] = p[5] + p[4]
    R[2, 0] = p[5] - p[4]

    p[4] = p[0] * q[3]
    p[5] = p[1] * q[2]

    R[1, 2] = p[5] - p[4]
    R[2, 1] = p[5] + p[4]


@cuda.jit(device=True, inline=True)
def gravity_error_function(q, acc):
    R = cuda.local.array(shape=(3, 3), dtype=float64)
    # for i in range(3):
    #     for j in range(3):
    #         R[i, j] = 0.0
    q2dcm(q, R)

    az = R[2, 0] * acc[0] + R[2, 1] * acc[1] + R[2, 2] * acc[2]
    # az = 10000000.0
    a_norm = (acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2])
    # return az*az / a_norm
    return normal_pdf(az * az, a_norm, 1.0)
    # return acc[0]+acc[1]+acc[2]
    # prob = az / a_norm
    # prob=0.0


@cuda.jit
def quaternion_evaluate(q_array, q_weight, acc, weight_sum):
    pos = cuda.grid(1)
    tid = cuda.threadIdx.x

    sdata = cuda.shared.array(shape=(1024), dtype=float64)
    if pos < q_array.shape[1]:
        if pos == 0:
            weight_sum[0] = 0.0
        # prob = 100.0
        prob = gravity_error_function(q_array[:, pos], acc[:])

        # compute new weight
        q_weight[pos] = q_weight[pos] * prob

        sdata[tid] = q_weight[pos]
        # normalize weight
        cuda.syncthreads()

        s = cuda.blockDim.x >> 1
        while s > 0:
            if tid < s:
                sdata[tid] = sdata[tid] + sdata[tid + s]
            s = s >> 1
            cuda.syncthreads()
        if tid == 0:
            cuda.atomic.add(weight_sum, 0, sdata[0])

        cuda.syncthreads()
        q_weight[pos] = q_weight[pos] / weight_sum[0]

    # array_buffer


@cuda.jit
def rejection_resample(state_array, state_buffer, weight):
    pos = cuda.grid(0)
    tid = cuda.threadIdx.x

    sdata = cuda.shared.array(shape=(1024),dtype=float64)
    if pos < state_array.shape[1]:
        state_buffer[:, pos] = state_array[:, pos]

        sdata[tid] = weight[pos]
        s = cuda.blockDim.x >> 1
        while s>0:
            if tid < s:
                sdata[tid] = max(sdata[tid],sdata[tid+s])
            s = s >> 1
            cuda.syncthreads()
        if tid==0:
            cuda.atomic.compare_and_swap()
