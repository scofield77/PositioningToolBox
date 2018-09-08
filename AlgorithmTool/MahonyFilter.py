# -*- coding:utf-8 -*-
# carete by steve at  2018 / 09 / 06　下午2:57
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

from AlgorithmTool.ImuTools import *


class MahonyFilterBase:
    def __init__(self):
        self.name = 'base Mahony filter'

        self.rotation_q = np.zeros([4])


class AHRSEKFSimple:
    def __init__(self, initial_prob):
        self.state = np.zeros([4])
        self.prob_state = initial_prob

        self.rotation_q = np.zeros([4])

        self.F = np.zeros([3, 3])
        self.G = np.zeros([3, 3])

        self.ref_mag = np.zeros([3])

    def initial_state(self, mag_data):
        self.ref_mag = np.mean(mag_data, axis=0)
        print('mag sahpe:',self.ref_mag.shape)

    def initial_state_euler(self, ori):
        '''
        Transform from ori to quaternion
        :param ori:
        :return:
        '''
        self.state = ori  # (euler2R(ori))
        self.rotation_q = dcm2q(euler2R(ori))

    def state_transaction_function(self,gyr_data,noise_matrix,time_interval):
        '''

        :param gyr_data:
        :param noise_matrix:
        :return:
        '''
        self.rotation_q = quaternion_right_update(self.rotation_q,gyr_data,time_interval)

        Rb2t = q2dcm(self.rotation_q)
        self.F = Rb2t * time_interval

        self.G = -1.0 * time_interval * Rb2t

        self.prob_state = (self.F.dot(self.prob_state)).dot(np.transpose(self.F)) + (self.G.dot(noise_matrix)).dot(
            np.transpose(self.G))

        self.prob_state = 0.5 * self.prob_state + 0.5 * self.prob_state.transpose()


    # def measurement_function_mag(self,mag, cov_matrix):
    #     '''
    #
    #     :param mag:
    #     :param cov_matrix:
    #     :return:
    #     '''
    #     self.H =






def try_simple_data():
    from AlgorithmTool.StepDetector import StepDetector
    from AlgorithmTool.StepLengthEstimator import StepLengthEstimatorV

    data = np.loadtxt('/home/steve/Data/pdr_imu.txt', delimiter=',')
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


def try_simple_data_ori():
    from AlgorithmTool.StepDetector import StepDetector
    from AlgorithmTool.StepLengthEstimator import StepLengthEstimatorV

    # data = np.loadtxt('/home/steve/Data/pdr_imu.txt', delimiter=',')
    data = np.loadtxt('/home/steve/Data/phoneData/0003/HAND_SMARTPHONE_IMU.data', delimiter=',')
    data = np.loadtxt('/home/steve/Data/phoneData/0004/SMARTPHONE3_IMU.data', delimiter=',')
    # print('data.shape:',data.shape)
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

    plt.figure()
    plt.subplot(311)
    for i in range(1, 4):
        plt.plot(gyr[:, 0], gyr[:, i], label=str(i))
    plt.legend()
    plt.subplot(312)
    for i in range(1, 4):
        plt.plot(mag[:, 0], mag[:, i], label=str(i))
    plt.plot(mag[:, 0], np.arctan2(mag[:, 1], mag[:, 2]) / np.pi * 180.0)
    plt.legend()
    plt.subplot(313)
    for i in range(1, 4):
        plt.plot(ori[:, 0], ori[:, i], label=str(i))
    plt.legend()

    plt.show()


if __name__ == '__main__':
    try_simple_data_ori()
