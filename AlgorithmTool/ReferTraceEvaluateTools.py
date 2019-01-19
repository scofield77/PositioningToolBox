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

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import re

from numba import jit, prange, njit


class Refscor:
    '''
    Calculate score using a matrix(*.npy) which is represent a score in a 2-D plane.
    '''
    def __init__(self, file_dir):
        for name in os.listdir(file_dir):
            rem = re.compile('[0-9\.\-]{1,100}npy$')
            if rem.match(name):
                print(name)
                self.score_data = np.load(file_dir + name)
                name = name.split('.npy')[0]
                self.map_range = np.asarray(
                    ([float(x) for x in name.split('-')])
                ).reshape(2, 2)
                self.relution = (self.map_range[0, 1] - self.map_range[0, 0]) / float(self.score_data.shape[0])
                print(self.map_range)
                # print(self.relution)
        self.max_score = np.max(self.score_data)

    def eval_point2d(self, point):
        '''
        compute score based on save score distribution matrix.
        :param point:
        :return:
        '''

        point = point.reshape(-1)
        pos_x = int((point[0] - self.map_range[0, 0]) / self.relution)
        pos_y = int((point[1] - self.map_range[1, 0]) / self.relution)
        if -1 < pos_x < self.score_data.shape[0] and -1 < pos_y < self.score_data.shape[1]:
            return self.score_data[pos_x, pos_y] * 1.0
        else:
            return self.max_score

    def eval_points(self, points):
        '''
        eval score for whole trace
        :param points: points should be a array with 3-d or 2-d positioin.
        :return:
        '''
        scores = np.zeros(shape=(points.shape[0]))
        scores = scores + self.max_score

        # @jit(parallel=True, nopython=True)
        # @njit(parallel=True)
        def scores_cal(ss, pts, map_range, relution, sd,max):
            first_error = True
            for i in prange(ss.shape[0]):
                # ss[i] = rf.eval_point2d(pts[i, :2])
                if map_range[0, 0] < pts[i, 0] < map_range[0, 1] and\
                        map_range[1, 0] < pts[i, 1] < map_range[1, 1]:
                    ss[i] = sd[int((pts[i, 0] - map_range[0, 0]) / relution),
                               int((pts[i, 1] - map_range[1, 0]) / relution)]
                else:
                    if first_error:
                        print('err:', pts[i,:],map_range)
                        first_error=False
                    ss[i] =max
            return ss

        return scores_cal(ss=scores, pts=points,
                          map_range=self.map_range,
                          relution=self.relution,
                          sd=self.score_data,
                          max = self.max_score)


if __name__ == '__main__':
    # dir_name = '/home/steve/Data/NewFusingLocationData/0039/'
    # dir_name = 'D:/Data/NewFusingLocationData/0039/'
    dir_name = 'C:/Data/NewFusingLocationData/0039/'
    ref_trace = np.loadtxt(dir_name + 'ref_trace.csv', delimiter=',')

    # print(ref_trace)

    rs = Refscor(dir_name)
    score = np.zeros(shape=(ref_trace.shape[0]))
    for i in range(ref_trace.shape[0]):
        score[i] = rs.eval_point2d(ref_trace[i, 1:3])

    score2 = rs.eval_points(ref_trace[:, 1:])

    print(np.mean(score), np.std(score))
    print(np.mean(score2), np.std(score2))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ref_trace[:, 1], ref_trace[:, 2], score[:], '-+', label='score')
    ax.plot(ref_trace[:, 1], ref_trace[:, 2], score2[:], '-+', label='score2')
    ax.legend()
    ax.grid()

    plt.figure()
    plt.plot(score, label='score')
    plt.plot(score2, label='score2')
    plt.legend()
    plt.grid()

    plt.show()
