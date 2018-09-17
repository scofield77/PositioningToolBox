# -*- coding:utf-8 -*-
# carete by steve at  2018 / 09 / 02　下午3:00
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
import scipy as sp
import numpy as np

from AlgorithmTool.LogLoader import LogLoader

import array


class StepDetector:

    def __init__(self, alpha=1.0, beta=1.0):
        self.counter = 0
        self.condidate_list = list()
        self.condidate_value = list()

        self.miu_alpha = 10.0
        self.sigma_alpha = 1.0
        self.alpha = alpha

        self.beta = beta

        self.alpha_p = 10.0
        self.alpha_v = 10.0

        self.last_index = 0
        self.last_type = 0
        self.last_time_p = -1.0
        self.last_time_v = -1.0

        self.Thp = 0.12
        self.Thv = 0.12

        self.p_interval_list = list()
        self.v_interval_list = list()

        self.acc_buffer = array.array('d')

    def detect_condidate(self, acc):
        '''
        detect standard
        :param acc:
        :return:
        '''
        if np.linalg.norm(acc[1, :]) > max(np.linalg.norm(acc[0, :]), np.linalg.norm(acc[2, :])) and \
                np.linalg.norm(acc[1, :]) > self.miu_alpha + self.sigma_alpha / self.alpha and \
                np.linalg.norm(acc[1, :]) > 10.5:
            return 1
        elif np.linalg.norm(acc[1, :]) < min(np.linalg.norm(acc[0, :]), np.linalg.norm(acc[2, :])) and \
                np.linalg.norm(acc[1, :]) < self.miu_alpha - self.sigma_alpha / self.alpha:
            return -1
        else:
            return 0

    def update_peak(self, acc, data_time):
        '''

        :param acc:
        :param data_time:
        :return:
        '''
        self.alpha_p = np.linalg.norm(acc[1, :])
        if len(self.p_interval_list) < 3:
            self.Thp = data_time - self.last_time_p - 0.05
            # print(self.Thp, self.Thv)
            self.Thp = 0.2
            self.p_interval_list.append(data_time - self.last_time_p)
        else:
            self.p_interval_list.append(data_time - self.last_time_p)
            self.Thp = data_time - self.last_time_p - np.std(np.asarray(self.p_interval_list)) / self.beta
            if len(self.p_interval_list) > 5:
                self.p_interval_list.pop(0)
            # print(self.Thp, self.Thv,'---')
        if self.Thp < 0.05:
            self.Thp = 0.12

        self.last_time_p = data_time
        # print(self.alpha_v,self.alpha_p)

    def update_valley(self, acc, data_time):
        '''

        :param acc:
        :param data_time:
        :return:
        '''
        self.alpha_v = np.linalg.norm(acc[1, :])
        if len(self.v_interval_list) < 3:
            # self.Thv = data_time - self.last_time_v - 0.1
            self.Thv = 0.2
            self.v_interval_list.append(data_time - self.last_time_v)
        else:
            self.v_interval_list.append(data_time - self.last_time_v)
            self.Thv = data_time - self.last_time_v - np.std(np.asarray(self.v_interval_list)) / self.beta
            if len(self.v_interval_list) > 5:
                self.v_interval_list.pop(0)
        if self.Thv < 0.05:
            self.Thv = 0.12

        self.last_time_v = data_time

    def step_detection(self, acc, data_index, data_time):
        '''

        :param acc: 3 * 3 matrix, each row represent the acc at a definite time point
        :param data_index:
        :param data_time:
        :return:
        '''
        tmp_flag = self.detect_condidate(acc)
        step_flag = False

        if tmp_flag is 1:  # peak
            if self.last_index is 0:  # initial first step
                self.last_type = tmp_flag  # set to peak
                self.update_peak(acc, data_time)

            elif self.last_type is -1 and data_time - self.last_time_p > self.Thp:
                self.last_type = tmp_flag
                self.update_peak(acc, data_time)
                # self.miu_alpha = 0.5 * (self.alpha_v + self.alpha_p)

            elif self.last_type is 1 and data_time - self.last_time_p < self.Thp \
                    and np.linalg.norm(acc[1, :]) > self.alpha_p:
                self.update_peak(acc, data_time)

        elif tmp_flag is -1:
            if self.last_type is 1 and data_time - self.last_time_v > self.Thv:
                self.last_type = -1
                self.update_valley(acc, data_time)
                self.counter += 1
                step_flag = True
                # self.miu_alpha = 0.5 * (self.alpha_p + self.alpha_v)
            elif self.last_type is -1 and data_time - self.last_time_v < self.Thv \
                    and np.linalg.norm(acc[1, :]) < self.alpha_v:
                self.update_valley(acc, data_time)

        if step_flag:
            if self.counter > 1:
                # all_acc = np.frombuffer(self.acc_buffer, dtype=np.float).reshape([-1, 3])
                self.miu_alpha = 0.5 * (self.alpha_p + self.alpha_v)

                # self.sigma_alpha = np.std(np.linalg.norm(all_acc, axis=1))
                # self.acc_buffer = array.array('d')
                # print(self.sigma_alpha)
                # self.sigma_alpha = 5.0

        else:
            self.acc_buffer.append(acc[1, 0])
            self.acc_buffer.append(acc[1, 1])
            self.acc_buffer.append(acc[1, 2])

            if np.frombuffer(self.acc_buffer, dtype=np.float).shape[0] > 40:
                all_acc = np.frombuffer(self.acc_buffer, dtype=np.float).reshape([-1, 3])
                self.acc_buffer = array.array('d')
                self.sigma_alpha = np.std(np.linalg.norm(all_acc, axis=1))

        return step_flag


class StepDetectorSimple:
    def __init__(self):
        self.step_time = 0.12

        self.pk_num = 0
        self.vy_num = 0
        self.step_num = 0
        self.pv_flg = list()

        self.pk_ti = list()
        self.pk_ti.append(self.last_p_time)
        self.vy_ti = list()
        self.vy_ti.append(self.last_v_time)

        self.last_pk_acc = 10.0
        self.last_vy_acc = 0.0

        self.last_p_time = -1.0
        self.last_v_time = -1.0

        self.step_counter = 0

    def step_detection(self, acc, time, index):
        '''

        :param acc:
        :param time:
        :param index:
        :return:
        '''
        step_flag = False


class StepDetectorMannual():
    def __init__(self, start_time,
                 pv_time_interval,
                 time_interval,
                 low_pass_alpha,
                 ref_low_pass_alpha,
                 acc_sigma,
                 initial_acc=9.81):
        '''

        :param start_time: start time of acc time stamp
        :param pv_time_interval: minimum time interval between peak and valley.
        :param time_interval: minimum time interval between adjacent peak points or valley points.
        :param low_pass_alpha: low pass filter parameter for source acc.
        :param ref_low_pass_alpha: low pass filter parameter for reference acc which should represent low frequency motion.
        :param acc_sigma: std of acc during the whole process
        :param initial_acc: for initial the acc algorithm
        '''

        self.last_peak_time = start_time
        self.last_valley_time = start_time

        # self.last_peak_value =

        self.acc_low_pass = initial_acc
        self.acc_norm = initial_acc

        self.pv_time_interval = pv_time_interval
        self.time_interval = time_interval
        self.low_pass_alpha = low_pass_alpha
        self.ref_low_pass_alpha = ref_low_pass_alpha
        self.acc_sigma = acc_sigma

    def step_detection(self, acc,time):
        '''

        :param acc:
        :return:
        '''
        self.acc_low_pass = (1.0 - self.low_pass_alpha) * self.acc_low_pass + \
                            self.low_pass_alpha * np.linalg.norm(acc[1, 1:])

        # self.acc_norm =
        flag = False
        state_flag = 0


        if abs(np.linalg.norm(acc[1, 1:]) - self.acc_low_pass) > self.acc_sigma * 1.0:
            if np.linalg.norm(acc[1, 1:]) > max(np.linalg.norm(acc[0, 1:]), np.linalg.norm(acc[2, 1:])) and \
                    acc[1, 0] - self.last_peak_time > self.time_interval and \
                    acc[1, 0] -self.last_valley_time > self.pv_time_interval and \
                    self.last_peak_time<=self.last_valley_time:
                # flag = True
                state_flag =1
                # last_peak = i
                self.last_peak_time = acc[1,0]
            elif np.linalg.norm(acc[1, 1:]) < min(np.linalg.norm(acc[0, 1:]), np.linalg.norm(acc[2, 1:])) \
                    and acc[1,0] - self.last_valley_time > self.pv_time_interval and \
                    acc[1, 0] - self.last_peak_time > self.time_interval and\
                    self.last_valley_time < self.last_peak_time:
                # flag_array[i] = np.linalg.norm(acc[i, 1:])
                # step_array[i] = np.linalg.norm(acc[last_peak,1:])
                # last_valley = i
                self.last_valley_time = acc[1,0]
                flag = True
                state_flag = 0
        return flag


def try_simple_data():
    data = np.loadtxt('/home/steve/Data/pdr_imu.txt', delimiter=',')
    step_detector = StepDetector(5.0, 2.0)
    # step_detector = StepDetectorMannual(data[0,0],1.4)

    acc = np.zeros([data.shape[0], 4])
    acc[:, 0] = data[:, 0]
    acc[:, 1:] = data[:, 2:5]

    # t_alpha = 0.2
    # for i in range(1, acc.shape[0]):
    #     acc[i, 1:] = t_alpha * acc[i, 1:] + (1.0 - t_alpha) * acc[i - 1, 1:]

    plt.figure()
    # for i in range(1, 4):
    #     plt.plot(acc[:, 0], acc[:, i])
    plt.plot(acc[:, 0], np.linalg.norm(acc[:, 1:], axis=1))

    step_flag = np.zeros(acc.shape[0])
    step_alpha = np.zeros(acc.shape[0])
    step_p = np.zeros_like(step_alpha)
    step_v = np.zeros_like(step_alpha)
    for i in range(1, acc.shape[0] - 1):
        if step_detector.step_detection(acc[i - 1:i + 2, 1:], i, acc[i, 0]):
            step_flag[i] = 10.0
        step_alpha[i] = step_detector.miu_alpha
        step_p[i] = step_detector.alpha_p
        step_v[i] = step_detector.alpha_v
    plt.plot(acc[:, 0], step_flag, '-+r')
    plt.plot(acc[:, 0], step_alpha, '--g')
    plt.plot(acc[:, 0], step_p, '--y')
    plt.plot(acc[:, 0], step_v, '--y')
    plt.grid()
    plt.show()


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
    step_detector = StepDetector(2.0, 1.0)

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
        step_alpha[i] = step_detector.miu_alpha
        step_p[i] = step_detector.alpha_p
        step_v[i] = step_detector.alpha_v
    plt.plot(acc[:, 0], step_flag, '-+r')
    plt.plot(acc[:, 0], step_alpha, '--g')
    plt.plot(acc[:, 0], step_p, '--y')
    plt.plot(acc[:, 0], step_v, '--y')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    try_ipin_data()
    # try_simple_data()
