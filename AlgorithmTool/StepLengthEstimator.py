# -*- coding:utf-8 -*-
# carete by steve at  2018 / 09 / 03　上午8:31
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

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from AlgorithmTool.LogLoader import LogLoader
from AlgorithmTool.StepDetector import StepDetector


class StepLengthEstimatorV:
    def __init__(self):
        self.Ka = 0.412

    def step_length_estimate(self, diff_acc):
        return self.Ka * (diff_acc ** 0.25)


def test_simple_data():
    data = np.loadtxt('/home/steve/Data/pdr_imu.txt', delimiter=',')
    step_detector = StepDetector(5.0, 2.0)
    step_estimator = StepLengthEstimatorV()

    acc = np.zeros([data.shape[0], 4])
    acc[:, 0] = data[:, 0]
    acc[:, 1:] = data[:, 2:5]

    t_alpha = 0.2
    for i in range(1, acc.shape[0]):
        acc[i, 1:] = t_alpha * acc[i, 1:] + (1.0 - t_alpha) * acc[i - 1, 1:]

    plt.figure()
    # for i in range(1, 4):
    #     plt.plot(acc[:, 0], acc[:, i])
    plt.plot(acc[:, 0], np.linalg.norm(acc[:, 1:], axis=1))

    step_flag = np.zeros(acc.shape[0])
    for i in range(1, acc.shape[0] - 1):
        if step_detector.step_detection(acc[i - 1:i + 2, 1:], i, acc[i, 0]):
            step_flag[i] = step_estimator.step_length_estimate(step_detector.miu_alpha * 2.0) + 10.0
    plt.plot(acc[:, 0], step_flag, '-+r')

    plt.grid()
    plt.show()


if __name__ == '__main__':
    test_simple_data()
