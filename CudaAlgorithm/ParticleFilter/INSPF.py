# -*- coding:utf-8 -*-
# carete by steve at  2018 / 06 / 03　下午9:21
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
import matplotlib.pyplot as plt

from numba import cuda, jit, float64

import math


class INSPF:
    def __init__(self, particle_number, initial_state, initial_cov, block_num=512):
        '''
        Initial particle filter, calculate initial state.
        It is worth to notice that particle number will be modified according to
        block number which is actually relate to device.
        :param particle_number:
        :param initial_state:
        :param initial_cov:
        :param block_num: 512 for GTX1080Ti.
        '''
        self.particle_num = particle_number

        # max_block_number = cuda.gpus
        self.block_num = block_num

        pre_thread_num = math.ceil(float(particle_number) / float(self.block_num))
        self.thread_num = int(pre_thread_num)
        self.particle_num = self.block_num * self.thread_num
        print('block num:', self.block_num,
              'thread num', self.thread_num,
              'particle num', self.particle_num)

        self.particles = cuda.device_array([self.particle_num,
                                            initial_state.shape[0]],
                                           dtype=float64)
        self.w = cuda.device_array([self.particle_num,1],
                                   dtype=float64)
