# -*- coding:utf-8 -*-
# carete by steve at  2018 / 06 / 27　下午6:53
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


from numba  import jit,cuda,float64
import numba


import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

if __name__ == '__main__':
    print(numba.__version__)
    print(cuda.devices._runtime.gpus)
    cuda.profile_start()
    dir_name = '/home/steve/Data/NewFusingLocationData/0013/'
    # dir_name = 'D:/Data/NewFusingLocationData/0013/'
    imu_data = np.loadtxt(dir_name + 'RIGHT_FOOT.data', delimiter=',')
    imu_data = imu_data[:, 1:]
    imu_data[:, 1:4] *= 9.81
    imu_data[:, 4:7] *= (np.pi / 180.0)
    # imu_data = imu_data.as
    imu_data = np.ascontiguousarray(imu_data)

    fist_line = imu_data[:100, :].mean(axis=0)
    imu_data[0, :] = fist_line

    # imu_data_device = cuda.device_array_like(imu_data)
    # stream = cuda.stream()
    # with stream.auto_synchronize():
    #     imu_data_device = cuda.to_device(imu_data, stream=stream)

    print('imu data shape', imu_data.shape)
