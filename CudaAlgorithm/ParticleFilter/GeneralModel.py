# -*- coding:utf-8 -*-
# carete by steve at  2018 / 04 / 23　下午7:20
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

from numba import cuda
import matplotlib.pyplot  as plt

import numpy as np
import scipy as sp

from numba import jit

from numba import cuda

from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_normal_float64, \
    xoroshiro128p_uniform_float64
import numba

from numba import float32, float64

import math

# import pyculib
from pyculib import blas as cublas


@cuda.jit
def rejection_resample(state_array, state_buffer, weight, rng, weight_max_array):
    '''
    General resample function for particle filter.
    In paper : Parallel Resampling in the Particle Filter (Journal of Computational and Graphical Statistics)
    :param state_array:
    :param state_buffer: buffer array, its size same to state_array
    :param weight:
    :param rng:random generator(pyculib)
    :param weight_max_array: buffer , its size same to weight.
    :return:
    '''
    pos = cuda.grid(1)
    tid = cuda.threadIdx.x

    sdata = cuda.shared.array(shape=(1024), dtype=float64)
    if pos < state_array.shape[1]:
        if pos == 0:
            weight_max_array[0] = 0.0
        # state_buffer[:, pos] = state_array[:, pos]
        for i in range(state_buffer.shape[0]):
            state_buffer[i, pos] = state_array[i, pos]

        sdata[tid] = weight[pos]
        s = cuda.blockDim.x >> 1
        while s > 0:
            if tid < s:
                sdata[tid] = max(sdata[tid], sdata[tid + s])
            s = s >> 1
            cuda.syncthreads()
        if tid == 0:
            cuda.atomic.max(weight_max_array, 0, sdata[0])

        j = pos
        u = xoroshiro128p_uniform_float64(rng, pos)
        counter = 0
        while u > weight[j] / weight_max_array[0] and counter < 1000:
            counter = counter + 1
            j = int(math.ceil(xoroshiro128p_uniform_float64(rng, pos) * state_array.shape[1]))
            u = xoroshiro128p_uniform_float64(rng, pos)
        # state_array[:, pos] = state_buffer[:, j]
        for i in range(state_buffer.shape[0]):
            state_array[i, pos] = state_buffer[i, j]
        weight[pos] = 1.0 / float64(state_array.shape[1])
        cuda.syncthreads()
