# -*- coding:utf-8 -*-
# Created by steve @ 18-3-26 上午8:12
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

# import numba
from numba import jit

import os
import re
import time
import datetime

import matplotlib.pyplot as plt
import scipy as sp
import numpy as np

# import numba
from numba import jit


class BasicXimuReader:
    def __init__(self, file_dir, key_word='', use_tmp=True):
        file_lists = os.listdir(file_dir)
        print(file_lists)
        print(type(file_lists))

        for file_name in file_lists:

            if 'Inertial' in file_name and key_word in file_name:
                # print(file_name)
                # data_file  = open(file_dir + "/" + file_name,'r')
                # data_file = data_file.readlines()
                # print("len:",len(data_file))
                '''
                Reference code:
                lines = np.loadtxt('iris.csv',delimiter=',',dtype='str')
                df = lines[1:,:4].astype('float')
                Reference code 2(quicker):
                lines = [line.split(',') for line in open('iris.csv')]
                df = [[float(x) for x in line[:4]] for line in lines[1:]]
                '''
                # self.data_index =  np.loadtxt(file_dir + "/" + file_name,delimiter=',',dtype='str')
                # self.data_index = self.data_index[1:,:].astype('float')

                the_lines = [line.split(',')[:-1] for line in open(file_dir + '/' + file_name)]

                # print(the_lines)
                the_lines = the_lines[1:]
                # print(the_lines)

                self.data_index = np.asarray(the_lines, dtype=float)
                # plt.figure(1)
                # plt.plot(self.data_index[1:,0]-self.data_index[0:-1,0],'r+-')
                # plt.show()
                # print(self.data_index.shape)
                # print(self.data_index[0,:])
                np.savetxt(file_dir + '/' + 'tmp_data_index.txt', self.data_index)

            elif 'Time' in file_name and key_word in file_name:
                # print("time",file_name)
                the_lines = [line.split(',') for line in open(file_dir + '/' + file_name)]
                # print(the_lines)

                the_lines = the_lines[1:]

                self.time_index = np.asarray(the_lines, dtype=int)
                np.savetxt(file_dir + '/' + 'tmp_time_index.txt', self.time_index.astype(int))

        '''
        For synchronic.
        '''
        # test time array to time that units is s.
        import time

        # self.time_index.astype(dtype='int')

        self.sec_index = np.zeros([self.time_index.shape[0], 2])
        for index in range(self.time_index.shape[0]):
            ISFORMAT = "%Y-%m-%d-%H-%M-%S"
            tmp_time_str = '{0}-{1}-{2}-{3}-{4}-{5}'.format(int(self.time_index[index, 1]),
                                                            int(self.time_index[index, 2]),
                                                            int(self.time_index[index, 3]),
                                                            int(self.time_index[index, 4]),
                                                            int(self.time_index[index, 5]),
                                                            int(self.time_index[index, 6]))

            self.sec_index[index, 0] = (time.mktime(time.strptime(tmp_time_str, ISFORMAT)))
            self.sec_index[index, 1] = self.time_index[index, 0]

        # print(self.sec_index)
        # np.savetxt('sec_index.txt',self.sec_index.astype('int'))

        # Test synchronic(index to time)

        last_second_point = 0
        first_use_point = 0

        for i in range(self.sec_index.shape[0]):
            if i == self.sec_index.shape[0] - 1 or self.sec_index[i, 0] < self.sec_index[i + 1, 0] - 0.8:
                # Don't use the first second's data.
                if i == self.sec_index.shape[0] - 1:
                    break

                if last_second_point == 0:
                    last_second_point = i + 1
                    first_use_point = last_second_point
                    continue
                index_offset = self.sec_index[i, 1] - self.sec_index[last_second_point, 1]

                # print(1/(index_offset))
                # print(self.data_index[i,1],self.data_index[last_second_point,1])
                time_step = 1 / (index_offset + 2.0)

                for j in range(last_second_point, i + 1):
                    self.sec_index[j, 0] += (self.sec_index[j, 1] - self.sec_index[last_second_point, 1]) * time_step

                last_second_point = i + 1
        self.sec_index = self.sec_index[first_use_point:last_second_point, :]

        # plt.figure(1)
        # plt.plot(self.sec_index[:,1],self.sec_index[:,0],'r+-')
        # plt.show()

        # Add time to acc and gyrl
        # print(self.data_index.shape)
        start_index = -10
        end_index = -100000
        speed_up_index = 0
        for i in range(self.data_index.shape[0]):
            index_tmp = self.data_index[i, 0]
            for j in range(speed_up_index, self.sec_index.shape[0]):
                if 0 < (index_tmp - self.sec_index[j, 1]) < 5:
                    # print("shot on.")
                    if start_index < 0:
                        start_index = i
                    else:
                        end_index = i

                    time_tmp = self.sec_index[j, 0] + \
                               (index_tmp - self.sec_index[j, 1]) / (self.sec_index[j, 1] - self.sec_index[j - 1, 1]) * \
                               (self.sec_index[j, 0] - self.sec_index[j - 1, 0])
                    self.data_index[i, 0] = time_tmp
                    speed_up_index = j - 1
                    continue
        self.data_index = self.data_index[start_index:end_index, :]
        #
        # plt.figure(1)
        # plt.plot(self.data_index[:,0],'r-+')
        # plt.show()
        # print(start_index,end_index)
        # print('MARK00',self.data_index[:,0])

        self.data = np.zeros_like(self.data_index)
        self.data[:, 0] = self.data_index[:, 0]
        self.data[:, 1:4] = self.data_index[:, 4:7] * 9.81
        self.data[:, 4:7] = self.data_index[:, 1:4] * (np.pi / 180.0)

        '''
        Left Right change
        '''
        # self.data_index[:,2] *= -1.0
        # self.data_index[:,5] *= -1.0

        # tmp = self.data_index.copy()
        #
        # self.data_index[:,1] = tmp[:,2]
        # self.data_index[:,2] = tmp[:,1]
        #
        # self.data_index[:,4] = tmp[:,5]
        # self.data_index[:,5] = tmp[:,4]

    def show(self):
        plt.figure()
        plt.plot(self.data_index[:, 0])

        plt.figure()
        plt.title('acc')
        plt.plot(self.data[:, 1:4])

        plt.figure()
        plt.title('gyr')
        plt.plot(self.data[:, 4:7])
        plt.show()

    def save(self, file_name):
        np.savetxt(file_name, self.data, delimiter=',')


if __name__ == '__main__':
    ximu = BasicXimuReader('/home/steve/Data/XsensUwb/x-imu/', 'LoggedData1')
    ximu.show()
