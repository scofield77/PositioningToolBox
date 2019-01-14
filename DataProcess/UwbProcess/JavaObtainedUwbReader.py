# -*- coding:utf-8 -*-
# Created by steve @ 18-3-17 下午3:53
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

import re


class UwbProcess:
    def __init__(self, name, mac_file_name):
        self.file_name = name

        file = open(name)

        lines = file.readlines()

        mac_list = list()
        mac_file_lines = open(mac_file_name).readlines()
        beaconSet = np.zeros([len(mac_file_lines), 3])
        for mac_line in mac_file_lines:
            mac_list.append(mac_line.split(',')[0])
            for k in range(3):
                beaconSet[len(mac_list) - 1, k] = mac_line.split(',')[k + 1]

        print(mac_list)
        print(beaconSet)

        m_re = re.compile('\\{[0-9|A-Z]{8}:[\\.|0-9]{1,},[\\.|0-9]{1,},}')
        # for
        uwb_data = np.zeros([len(lines), len(mac_list) + 1])
        uwb_data = uwb_data - 10.0
        uwb_signal_data = np.zeros_like(uwb_data)
        uwb_signal_data = uwb_signal_data - 10.0
        for i in range(len(lines)):
            the_line = lines[i]
            uwb_data[i, 0] = float(the_line.split(',')[1])
            uwb_signal_data[i, 0] = float(the_line.split(',')[1])

            # print(m_re.findall(the_line))
            for m in m_re.findall(the_line):
                mac = m[1:m.find(':')]
                dis = m[m.find(':') + 1:m.find(',')]
                signal_strength = m[m.find(',') + 1:len(m) - 2]
                print('mac:', mac, ' dis:', dis, 'index:', mac_list.index(mac), 'strength:', signal_strength)
                uwb_data[i, 1 + mac_list.index(mac)] = dis
                uwb_signal_data[i, 1 + mac_list.index(mac)] = signal_strength

        self.uwb_data = uwb_data
        self.beaconSet = beaconSet
        self.uwb_signal_data = uwb_signal_data
        # plt.show()

    def show(self):
        plt.figure()
        plt.subplot(211)
        for i in range(1, self.uwb_data.shape[1]):
            if (np.max(self.uwb_data[:, i]) > 0):
                plt.plot(self.uwb_data[:, 0], self.uwb_data[:, i], '.', label=str(i))
        plt.legend()
        plt.grid()

        plt.subplot(212)
        for i in range(1, self.uwb_data.shape[1]):
            if (np.max(self.uwb_data[:, i]) > 0):
                plt.plot(self.uwb_signal_data[:, 0], self.uwb_signal_data[:, i], '.', label=str(i))
        plt.legend()
        plt.grid()

        plt.show()

        #

    def save(self, dir_name):
        np.savetxt(dir_name + 'uwb_data.csv', self.uwb_data, delimiter=',')
        np.savetxt(dir_name + 'beaconset_no_mac.csv', self.beaconSet, delimiter=',')
        np.savetxt(dir_name + 'uwb_signal_data.csv', self.uwb_signal_data, delimiter=',')


if __name__ == '__main__':
    # uwb_file_name = UwbProcess("/home/steve/Data/FusingLocationData/0013/HEAD_UWB.data",
    #                            '/home/steve/Data/FusingLocationData/mac.txt')
    # dir_name = '/home/steve/Data/XsensUwb/MTI700/0003/'
    # uwb_file_p = UwbProcess(dir_name + "HEAD_UWB.data",
    #                         dir_name + 'beaconSet.csv')

    dir_name = "/home/steve/Data/NewFusingLocationData/0048/"
    dir_name = "/home/steve/Data/ZUPTPDR/0007/"
    dir_name = "/home/steve/Data/VehicleUWBINS/0003/"
    dir_name = "/home/steve/Data/19-1-12/0007/"
    uwb_file_p = UwbProcess(dir_name + "HEAD_UWB.data",
                            dir_name + '../BeaconSet.csv')

    uwb_file_p.save(dir_name)
    # uwb_file_p.show()
