# -*- coding:utf-8 -*-
# carete by steve at  2018 / 09 / 18　下午2:56
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

from AlgorithmTool.ImuTools import imu_plot_aux

if __name__ == '__main__':
    dir_name = '/home/steve/Data/PDR/0001/'

    left_imu = np.loadtxt(dir_name + 'LEFT_FOOT.data', delimiter=',')

    left_imu = left_imu[:, 1:]

    plt.figure()
    # plt.plot(left_imu[:,1:4],'acc')
    imu_plot_aux(left_imu[:, 1:4], time=left_imu[:, 0], title_name='acc')
    imu_plot_aux(left_imu[:, 4:7], time=left_imu[:, 0], title_name='gyr')

    plt.show()
