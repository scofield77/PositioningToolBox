# -*- coding:utf-8 -*-
# Created by steve @ 18-3-14 下午9:30
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
import os

import re

import numpy as np

import matplotlib.pyplot as plt
import array


def mergeData(file_name):
    source_data = np.loadtxt(file_name, delimiter=',')

    target_basic_array = array.array('d')
    flag_array = np.zeros(source_data.shape[1]) - 10.0

    flag_batch_end = False
    for i in range(source_data.shape[0]):
        if flag_array[0] < 0.0:
            flag_array[0] = source_data[i, 0]
        for j in range(1, source_data.shape[1]):
            if source_data[i, j] > 0.0:
                if flag_array[j] > 0.0:
                    flag_batch_end = True
                    i = i - 1
                    # break
                flag_array[j] = source_data[i, j]

        if (i > 0 and source_data[i, 0] - source_data[i - 1, 0] > 0.2) or flag_batch_end:
            for k in range(flag_array.shape[0]):
                target_basic_array.append(flag_array[k])

    target_data = np.frombuffer(flag_array, dtype=np.float).reshape([-1, source_data.shape[1]])
    print('________________________________START____________________________________________')
    for i in range(target_data.shape[0]):
        print(target_data[i,:])
    print('---------------------------------END----------------------------------------------')

if __name__ == '__main__':
        file_name  = 'C:\\Users\\steve\\Documents\\Tencent Files\\551619855\\FileRecv\\uwb_result.csv'
        mergeData(file_name)