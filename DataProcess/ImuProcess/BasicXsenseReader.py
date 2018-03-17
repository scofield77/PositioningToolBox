# -*- coding:utf-8 -*-
# Created by steve @ 18-3-14 下午9:22
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
import time
import datetime

import matplotlib.pyplot as plt
import scipy as sp
import numpy as np


class BasicXsenseReader:

    def __init__(self, file_name):
        '''

        :param file_name:
        '''
        self.file_name = file_name
        self.load()

    def load(self, is_debug=False):
        '''
        load data from file.
        :param is_debug: if is true, print the time stamp.
        :return:
        '''
        file_lines = open(self.file_name).readlines()

        self.data = np.zeros([len(file_lines) - 8, 13])

        # first line of data.
        for i in range(7, len(file_lines) - 1):

            # print(file_lines[i])
            matcher = re.compile('[-]{0,1}[0-9]{1,3}\.{0,1}[0-9]{0,15}')
            all_num = matcher.findall(file_lines[i])

            # print(tt)
            tt = datetime.datetime(int(all_num[2]), int(all_num[3]), int(all_num[4]), int(all_num[5]),
                                   int(all_num[6]),
                                   int(all_num[7]))

            if is_debug:
                print(tt.timestamp() + float(all_num[1]) * 1e-9)
            self.data[i - 7, 0] = tt.timestamp() + float(all_num[0]) * 1e-9

            # print(all_num)
            for j in range(11):
                self.data[i - 7, 1 + j] = float(all_num[j + len(all_num) - 11])
        print(self.data)

    def show(self):
        # plt.figure()
        # plt.imshow(self.data / self.data.std(axis=0))
        # plt.imshow(self.data)
        # plt.colorbar()
        plt.figure()
        for i in range(3):
            plt.plot(self.data[:, 0], self.data[:, i + 1], '-+', label='acc' + str(i))
        plt.legend()
        plt.grid()
        plt.figure()
        for i in range(3):
            plt.plot(self.data[:, 0], self.data[:, i + 4], '-+', label='gyr' + str(i))
        plt.legend()
        plt.grid()
        plt.figure()
        for i in range(3):
            plt.plot(self.data[:, 0], self.data[:, i + 7], '-+', label='mag' + str(i))
        plt.legend()
        plt.grid()
        plt.figure()
        for i in range(3):
            plt.plot(self.data[:, 0], self.data[:, i + 10], '-+', label='angle' + str(i))
        plt.legend()
        plt.grid()
        plt.show()

    def save(self, file_name):
        np.savetxt(file_name, self.data, delimiter=',')


if __name__ == '__main__':
    dir_name = '/home/steve/Data/XsensUwb/MTI700/0001/'

    bxr = BasicXsenseReader(dir_name + 'HEAD.txt')
    bxr.save(dir_name + 'imu.data')
    bxr.show()
