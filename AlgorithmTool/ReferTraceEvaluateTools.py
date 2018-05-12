# -*- coding:utf-8 -*-
# carete by steve at  2018 / 05 / 12　下午4:30
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

import  numpy as np
import scipy as sp

import matplotlib.pyplot as plt


import os
import re


class Refscor:
    def __init__(self, file_dir):
        for name in os.listdir(file_dir):
            rem = re.compile('[0-9\.\-]{1,100}npy$')
            if rem.match(name):
                self.score_data = np.load(dir_name+name)
                name = name.split('.npy')[0]
                self.map_range = np.asarray(
                    ([float(x) for x in name.split('-')])
                ).reshape(2,2)
                self.relution = (self.map_range[0,1]-self.map_range[0,0])/float(self.score_data.shape[0])
                # print(self.map_range)
                print(self.relution)
        self.max_score = np.max(self.score_data)
    def eval_point2d(self,point):
        point = point.reshape(-1)
        pos_x = int((point[0]-self.map_range[0,0])/self.relution)
        pos_y = int((point[1]-self.map_range[1,0])/self.relution)
        return self.score_data[pos_x,pos_y]




if __name__ == '__main__':
    dir_name = '/home/steve/Data/NewFusingLocationData/0038/'
    ref_trace  = np.loadtxt(dir_name+'ref_trace.csv',delimiter=',')

    # print(ref_trace)

    rs = Refscor(dir_name)
    score = np.zeros(shape=(ref_trace.shape[0]))
    for i in range(ref_trace.shape[0]):
        score[i] = rs.eval_point2d(ref_trace[i,1:3])


    plt.figure()
    plt.plot(score)
    plt.grid()
    plt.show()