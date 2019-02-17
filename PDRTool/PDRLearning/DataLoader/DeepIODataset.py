# -*- coding:utf-8 -*-
# carete by steve at  2019 / 02 / 07　3:05 PM
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

import os
import sys


class DeepIODataset(data.Dataset):
    def __init__(self, dir_name):
        '''
        load file name
        :param dir_name:
        '''
        self.valid_dir_name_list = list()




        '''
        recursion searching directory named 'syn'
        '''
        def dg_search(current_dir_name):
            if current_dir_name[-1] != '/':
                current_dir_name = current_dir_name+'/'
            sub_dir_list = os.listdir(current_dir_name)
            for sub_dir in sub_dir_list:
                # print('searching', sub_dir_list)
                if os.path.isdir(current_dir_name + sub_dir):
                    # print('sub dir:', sub_dir)
                    if sub_dir == 'syn':
                        print(current_dir_name+sub_dir)
                        self.valid_dir_name_list.append(current_dir_name+sub_dir)

                    else:
                        dg_search(current_dir_name+sub_dir)

        # save all the sub directory named 'syn'
        dg_search(dir_name)

    def load_to_mem(self):
        '''
        Load data to memory.(save as list of dir)
        :return:
        '''
        for dir_name in self.valid_dir_name_list:
            print(dir_name)
            if 'vi1.csv' in os.listdir(dir_name):
                print(os.listdir(dir_name))
                if dir_name[-1] != '/':
                    dir_name  = dir_name + '/'
                for imu_file_name in os.listdir(dir_name):
                    if 'imu' in imu_file_name:
                        vi_file_name = imu_file_name.replace('imu','vi')
                        print(imu_file_name,vi_file_name)

                        vi_data = np.loadtxt(dir_name+vi_file_name,
                                             delimiter=',')
                        imu_data = np.loadtxt(dir_name+imu_file_name,
                                              delimiter=',')
                        print('imu shape:', imu_data.shape,
                              'vi shape:', vi_data.shape)
                        print('pos:',np.std(vi_data[:,2:5],axis=0))



if __name__ == '__main__':
    dir_name = "/home/steve/Data/Oxford Inertial Tracking Dataset/"
    import os
    import sys

    print(os.listdir(dir_name))

    DIODataset = DeepIODataset(dir_name)
    # for the_name in os.listdir(dir_name):
    # process .
    # print(the_name, os.path.isdir(dir_name+'/'+the_name))
    DIODataset.load_to_mem()
