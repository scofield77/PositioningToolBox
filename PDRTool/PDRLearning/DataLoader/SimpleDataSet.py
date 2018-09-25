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
        self.x_std = np.std(data_x, axis=0)
        data_x = data_x / self.x_std

        # data_y =
        min_y = np.min(data_y)
        max_y = np.max(data_y)
        data_y = (data_y - min_y) / (max_y - min_y)

        self.whole_x = data_x * 1.0
        self.whole_y = data_y * 1.0

        from array import array

        x_array = array('d')
        y_array = array('d')

        i = 0
        while i < data_x.shape[0] - cut_length:
            for j in range(i, i + cut_length):
                # x_array.append()
                y_array.append(data_y[j])
                for k in range(data_x.shape[1]):
                    x_array.append(data_x[j, k])
            i = i + cut_length - overlap_length

        self.x_dataset = np.frombuffer(x_array, dtype=np.float).reshape([-1, cut_length, data_x.shape[1]])
        self.y_dataset = np.frombuffer(y_array, dtype=np.float).reshape([-1, cut_length, 1])

        # assert self.x_dataset.shape[0] == self.y_dataset.shape[0]
        # print(self.x_dataset.shape, self.y_dataset.shape)
        self.dataset_length = self.x_dataset.shape[0]

        self.input_size = self.x_dataset.shape[1]
        self.output_size = self.y_dataset.shape[1]

        self.cut_size = cut_length

    def __getitem__(self, idx):
        # print('size:', self.x_dataset[idx, :, :].shape[0])
        return self.x_dataset[idx, :, :].reshape([-1, self.whole_x.shape[1]]).transpose(), \
               self.y_dataset[idx, :, :].reshape([-1, 1]).transpose()

    def __len__(self):
        return self.dataset_length

    def preprocess_other(self, other_x):
        '''
        use to preprocess data which will be adopted in valid process.
        :param other_x:
        :return:
        '''
        other_x = (other_x - self.x_mean) / self.x_std
        return other_x
