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
        self.version


if __name__ == '__main__':
    dir_name = "/home/steve/Data/Oxford Inertial Tracking Dataset/"
    import os
    import sys

    print(os.listdir(dir_name))

    DIODataset = DeepIODataset(dir_name)
    # for the_name in os.listdir(dir_name):
    # process .
    # print(the_name, os.path.isdir(dir_name+'/'+the_name))
