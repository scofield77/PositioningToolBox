# -*- coding:utf-8 -*-
# carete by steve at  2018 / 10 / 29　4:56 PM
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


def show_vi_trace(data):
    plt.figure()
    for i in range(3):
        plt.plot(data[:, i + 2], '-+', label=str(i))
    plt.legend()
    plt.grid()


if __name__ == '__main__':
    data_dir = "/home/steve/Data/InertialTrackingDataset/handbag/data1/syn/"

    vi_data = np.loadtxt(data_dir + 'vi1.csv', delimiter=',')
    show_vi_trace(vi_data)
    plt.show()
