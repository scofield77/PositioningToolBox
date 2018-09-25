# -*- coding:utf-8 -*-
# carete by steve at  2018 / 09 / 02　上午11:35
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
from AlgorithmTool.StepDetector import *
from AlgorithmTool.StepLengthEstimator import StepLengthEstimatorV


def try_simple_data():
    data = np.loadtxt('/home/steve/Data/pdr_imu.txt', delimiter=',')
    # data = np.loadtxt('/home/steve/Data/phoneData/PDRUWBBLEMini/0000/SMARTPHONE2_IMU.data', delimiter=',')
    step_detector = StepDetector(2.1, 0.8)
    step_estimator = StepLengthEstimatorV()

    acc = np.zeros([data.shape[0], 4])
    acc[:, 0] = data[:, 0]
    acc[:, 1:] = data[:, 2:5]
    gyr = np.zeros([data.shape[0], 4])
    mag = np.zeros([data.shape[0], 4])
    ori = np.zeros([data.shape[0], 4])

    gyr[:, 0] = data[:, 0]
    gyr[:, 1:] = data[:, 5:8]

    mag[:, 0] = data[:, 0]
    mag[:, 1:] = data[:, 8:11]

    ori[:, 0] = data[:, 0]
    ori[:, 1:] = data[:, 11:14]

    t_alpha = 0.2
    for i in range(1, acc.shape[0]):
        acc[i, 1:] = t_alpha * acc[i, 1:] + (1.0 - t_alpha) * acc[i - 1, 1:]

    plt.figure()
    # for i in range(1, 4):
    #     plt.plot(acc[:, 0], acc[:, i])
    plt.plot(acc[:, 0], np.linalg.norm(acc[:, 1:], axis=1))

    step_flag = np.zeros(acc.shape[0])
    step_ori = np.zeros_like(step_flag)

    import array

    pos_array = array.array('d')
    last_pos_x = 0.0
    last_pos_y = 0.0

    import math
    for i in range(1, acc.shape[0] - 1):
        if np.linalg.norm(mag[i, 1:3]) < 0.1:
            mag[i, 1:] = mag[i - 1, 1:]
        else:
            alpha = 0.2
            mag[i, 1:] = alpha * mag[i, 1:] + (1.0 - alpha) * mag[i, 1:]
        if step_detector.step_detection(acc[i - 1:i + 2, 1:], i, acc[i, 0]):
            step_flag[i] = step_estimator.step_length_estimate(step_detector.miu_alpha * 2.0) + 10.0
            step_length = step_estimator.step_length_estimate(step_detector.miu_alpha * 2.0)
            # if i < 10:

            if i < 50:
                simple_ori = ori[i, 1] * 3.0
            else:
                simple_ori = np.mean(ori[i - 50:i, 1]) * 3.0
            # simple_ori = ori[i,1]*3.0
            # H

            # simple_ori = math.atan2(mag[i,2],mag[i,1])
            # simple_ori = (np.arctan2(mag[i,1],mag[i,2])-np.pi)*2.0
            # simple_ori = math.aco
            step_ori[i] = simple_ori
            # else:
            #     simple_ori = math.atan2(np.mean(mag[i-10:i+1,2]),np.mean(mag[i-10:i+1,1]))

            #
            last_pos_x += step_length * math.cos(simple_ori)
            last_pos_y += step_length * math.sin(simple_ori)
            # last_pos_x += step_length * mag[i,1] / np.linalg.norm(mag[i,1:])
            # last_pos_y += step_length * mag[i,2] / np.linalg.norm(mag[i,1:])
            pos_array.append(last_pos_x)
            pos_array.append(last_pos_y)

    plt.plot(acc[:, 0], step_flag, '-+r')

    plt.grid()

    plt.figure()
    plt.title('trace')
    pos = np.frombuffer(pos_array, dtype=np.float).reshape([-1, 2])
    plt.plot(pos[:, 0], pos[:, 1], '--+')
    plt.grid()
    #
    # plt.figure()
    # plt.subplot(311)
    # plt.title('gyr')
    # for i in range(1,4):
    #     plt.plot(gyr[:,0],gyr[:,i])
    # plt.subplot(312)
    # plt.title('mag')
    # for i in range(1,4):
    #     plt.plot(mag[:,0],mag[:,i],'+',label=str(i))
    # plt.plot(mag[:,0],step_ori/np.pi * 180.0,'--+')
    # plt.legend()
    #
    # plt.subplot(313)
    # plt.title('ori')
    # for i in range(1,4):
    #     plt.plot(ori[:,0],ori[:,i]/np.pi * 180.0,'+')

    # plt.figure()
    # plt.plot(np.arctan2(mag[:,1],mag[:,2])/np.pi * 180.0)

    plt.show()


def search_simple_data(alpha, beta):
    '''
    use to search hyperparameters (alpha and beta) in step detection function.
    Plot the trace based on given parameters.
    :param alpha:
    :param beta:
    :return:
    '''
    data = np.loadtxt('/home/steve/Data/pdr_imu.txt', delimiter=',')
    # print('in1:',alpha,beta)
    step_detector = StepDetector(alpha, beta)
    step_estimator = StepLengthEstimatorV()
    # print('in2:',alpha,beta)

    acc = np.zeros([data.shape[0], 4])
    acc[:, 0] = data[:, 0]
    acc[:, 1:] = data[:, 2:5]
    gyr = np.zeros([data.shape[0], 4])
    mag = np.zeros([data.shape[0], 4])
    ori = np.zeros([data.shape[0], 4])

    gyr[:, 0] = data[:, 0]
    gyr[:, 1:] = data[:, 5:8]

    mag[:, 0] = data[:, 0]
    mag[:, 1:] = data[:, 8:11]

    ori[:, 0] = data[:, 0]
    ori[:, 1:] = data[:, 11:14]

    t_alpha = 0.2
    for i in range(1, acc.shape[0]):
        acc[i, 1:] = t_alpha * acc[i, 1:] + (1.0 - t_alpha) * acc[i - 1, 1:]

    step_flag = np.zeros(acc.shape[0])
    step_ori = np.zeros_like(step_flag)

    import array

    pos_array = array.array('d')
    last_pos_x = 0.0
    last_pos_y = 0.0

    import math
    for i in range(1, acc.shape[0] - 1):
        if np.linalg.norm(mag[i, 1:3]) < 0.1:
            mag[i, 1:] = mag[i - 1, 1:]
        else:
            the_alpha = 0.2
            mag[i, 1:] = the_alpha * mag[i, 1:] + (1.0 - the_alpha) * mag[i, 1:]
        if step_detector.step_detection(acc[i - 1:i + 2, 1:], i, acc[i, 0]):
            step_flag[i] = step_estimator.step_length_estimate(step_detector.miu_alpha * 2.0) + 10.0
            step_length = step_estimator.step_length_estimate(step_detector.miu_alpha * 2.0)
            # if i < 10:

            if i < 50:
                simple_ori = ori[i, 1] * 3.0
            else:
                simple_ori = np.mean(ori[i - 50:i, 1]) * 3.0
            # simple_ori = ori[i,1]*3.0
            # H

            # simple_ori = math.atan2(mag[i,2],mag[i,1])
            # simple_ori = (np.arctan2(mag[i,1],mag[i,2])-np.pi)*2.0
            # simple_ori = math.aco
            step_ori[i] = simple_ori
            # else:
            #     simple_ori = math.atan2(np.mean(mag[i-10:i+1,2]),np.mean(mag[i-10:i+1,1]))

            #
            last_pos_x += step_length * math.cos(simple_ori)
            last_pos_y += step_length * math.sin(simple_ori)
            # last_pos_x += step_length * mag[i,1] / np.linalg.norm(mag[i,1:])
            # last_pos_y += step_length * mag[i,2] / np.linalg.norm(mag[i,1:])
            pos_array.append(last_pos_x)
            pos_array.append(last_pos_y)

    # plt.plot(acc[:, 0], step_flag, '-+r')
    #
    # plt.grid()

    plt.figure()
    plt.title('trace')
    pos = np.frombuffer(pos_array, dtype=np.float).reshape([-1, 2])
    plt.plot(pos[:, 0], pos[:, 1], '--+')
    plt.grid()
    plt.savefig(
        '/home/steve/Data/tmp/' + str(alpha) + '-' + str(beta) + '.jpg'
    )
    print(alpha, beta)
    #
    # plt.figure()
    # plt.subplot(311)
    # plt.title('gyr')
    # for i in range(1,4):
    #     plt.plot(gyr[:,0],gyr[:,i])
    # plt.subplot(312)
    # plt.title('mag')
    # for i in range(1,4):
    #     plt.plot(mag[:,0],mag[:,i],'+',label=str(i))
    # plt.plot(mag[:,0],step_ori/np.pi * 180.0,'--+')
    # plt.legend()
    #
    # plt.subplot(313)
    # plt.title('ori')
    # for i in range(1,4):
    #     plt.plot(ori[:,0],ori[:,i]/np.pi * 180.0,'+')

    # plt.figure()
    # plt.plot(np.arctan2(mag[:,1],mag[:,2])/np.pi * 180.0)

    # plt.show()


def try_ipin_data():
    file_name = '/home/steve/Data/IPIN2017Data/Track3/01-Training/CAR/logfile_CAR_R02-2017_S4.txt'
    # file_name = '/home/steve/Data/IPIN2017Data/Track3/01-Training/CAR/logfile_CAR_R01-2017_S4MINI.txt'

    ll = LogLoader(file_name)

    acc = np.zeros([ll.acce.shape[0], 4])
    acc[:, 0] = ll.acce[:, 0]
    acc[:, 1:] = ll.acce[:, 2:5]

    # show time interval
    plt.figure()
    plt.title('time interval')
    plt.plot(acc[1:, 0] - acc[:-1, 0])
    time_interval_array = acc[1:, 0] - acc[:-1, 0]

    #
    step_detector = StepDetector(2.1, 0.8)

    plt.figure()
    # for i in range(1, 4):
    #     plt.plot(acc[:, 0], acc[:, i])
    plt.plot(acc[:, 0], np.linalg.norm(acc[:, 1:], axis=1), '-')

    step_flag = np.zeros(acc.shape[0])
    step_alpha = np.zeros(acc.shape[0])
    step_p = np.zeros_like(step_alpha)
    step_v = np.zeros_like(step_alpha)
    for i in range(1, acc.shape[0] - 1):
        if step_detector.step_detection(acc[i - 1:i + 2, 1:], i, acc[i, 0]):
            step_flag[i] = 10.0

    plt.plot(acc[:, 0], step_flag, '-+r')
    plt.grid()
    plt.show()


def try_Phonemini_data():
    # data = np.loadtxt('/home/steve/Data/pdr_imu.txt', delimiter=',')
    data = np.loadtxt('/home/steve/Data/phoneData/PDRUWBBLEMini/0006/SMARTPHONE3_IMU.data', delimiter=',')
    # step_detector = StepDetector(1.0, 0.8)
    step_detector = StepDetectorMannual(data[0, 0],
                                        1.4,
                                        0.4,
                                        0.8,
                                        0.9,
                                        0.1)

    # step_estimator = StepLengthEstimatorV()

    t = data[:, 0]
    data = data[:, 1:]
    data[:, 0] = t * 1.0

    acc = np.zeros([data.shape[0], 4])
    acc[:, 0] = data[:, 0]
    acc[:, 1:] = data[:, 2:5]
    gyr = np.zeros([data.shape[0], 4])
    mag = np.zeros([data.shape[0], 4])
    ori = np.zeros([data.shape[0], 4])

    gyr[:, 0] = data[:, 0]
    gyr[:, 1:] = data[:, 5:8]

    mag[:, 0] = data[:, 0]
    mag[:, 1:] = data[:, 8:11]

    ori[:, 0] = data[:, 0]
    ori[:, 1:] = data[:, 11:14]

    t_alpha = 0.2
    for i in range(1, acc.shape[0]):
        acc[i, 1:] = t_alpha * acc[i, 1:] + (1.0 - t_alpha) * acc[i - 1, 1:]

    plt.figure()
    # for i in range(1, 4):
    #     plt.plot(acc[:, 0], acc[:, i])
    plt.plot(acc[:, 0], np.linalg.norm(acc[:, 1:], axis=1))

    step_flag = np.zeros(acc.shape[0])
    step_ori = np.zeros_like(step_flag)

    import array

    pos_array = array.array('d')
    last_pos_x = 0.0
    last_pos_y = 0.0

    import math
    for i in range(1, acc.shape[0] - 1):
        if np.linalg.norm(ori[i, 1:]) < 1.0:
            ori[i, 1:] = ori[i - 1, 1:] * 1.0

        if np.linalg.norm(mag[i, 1:3]) < 0.1:
            mag[i, 1:] = mag[i - 1, 1:]
        else:
            alpha = 0.2
            mag[i, 1:] = alpha * mag[i, 1:] + (1.0 - alpha) * mag[i, 1:]
        # if step_detector.step_detection(acc[i - 1:i + 2, 1:], i, acc[i, 0]):
        if step_detector.step_detection(acc[i - 1:i + 2, 1:], acc[i, 0]):
            # step_flag[i] = step_estimator.step_length_estimate(step_detector.miu_alpha * 2.0) + 10.0
            # step_length = step_estimator.step_length_estimate(step_detector.miu_alpha * 2.0)
            step_flag[i] = 10.0
            step_length = 1.2
            # if i < 10:

            if i < 50:
                simple_ori = ori[i, 2]  # * 2.0
            else:
                simple_ori = np.mean(ori[i - 50:i, 2])  # * 2.0
            # simple_ori = ori[i,1]*3.0
            # H

            # simple_ori = math.atan2(mag[i,2],mag[i,1])
            # simple_ori = (np.arctan2(mag[i,1],mag[i,2])-np.pi)*2.0
            # simple_ori = math.aco
            step_ori[i] = simple_ori
            # else:
            #     simple_ori = math.atan2(np.mean(mag[i-10:i+1,2]),np.mean(mag[i-10:i+1,1]))

            #
            last_pos_x += step_length * math.cos(simple_ori)
            last_pos_y += step_length * math.sin(simple_ori)
            # last_pos_x += step_length * mag[i,1] / np.linalg.norm(mag[i,1:])
            # last_pos_y += step_length * mag[i,2] / np.linalg.norm(mag[i,1:])
            pos_array.append(last_pos_x)
            pos_array.append(last_pos_y)

    plt.plot(acc[:, 0], step_flag, '-+r')

    plt.grid()

    plt.figure()
    plt.title('trace')
    pos = np.frombuffer(pos_array, dtype=np.float).reshape([-1, 2])
    plt.plot(pos[:, 0], pos[:, 1], '--+')
    plt.grid()
    #
    plt.figure()
    plt.subplot(311)
    plt.title('gyr')
    for i in range(1, 4):
        plt.plot(gyr[:, 0], gyr[:, i])
    plt.subplot(312)
    plt.title('mag')
    for i in range(1, 4):
        plt.plot(mag[:, 0], mag[:, i], '+', label=str(i))
    plt.plot(mag[:, 0], step_ori / np.pi * 180.0, '--+')
    plt.legend()

    plt.subplot(313)
    plt.title('ori')
    for i in range(1, 4):
        plt.plot(ori[:, 0], ori[:, i] / np.pi * 180.0, '+')

    # plt.figure()
    # plt.plot(np.arctan2(mag[:,1],mag[:,2])/np.pi * 180.0)

    plt.show()


if __name__ == '__main__':
    # try_simple_data()
    try_ipin_data()
    # try_Phonemini_data()

    # value_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.6, 1.8, 2.1, 2.5, 2.8, 3.2, 3.8]
    # for alpha in value_list:
    #     for beta in value_list:
    #         search_simple_data(alpha,beta)
#
