# -*- coding:utf-8 -*-
# carete by steve at  2018 / 09 / 24　5:10 PM
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

import json
import torch
import torch.utils.data as data

import numpy as np


class SimpleDataSet(data.Dataset):

    def __init__(self, data_x, data_y, cut_length, overlap_length):
        '''

        :param data_x: x,
        :param data_y: y,
        :param cut_length: cut time sequaence to sub-sequence
        :param overlap_length: overlap length between adjacent sequence.
        '''
        self.x_mean = np.mean(data_x, axis=0)
        data_x = data_x - self.x_mean
        self.x_std = np.std(data_x, axis=1)
        data_x = data_x / self.x_std

        from array import array

        x_array = array('d')
        y_array = array('d')

    def preprocess_other(self, other_x):
        '''
        use to preprocess data which will be adopted in valid process.
        :param other_x:
        :return:
        '''
        other_x = (other_x - self.x_mean) / self.x_std
        return other_x
