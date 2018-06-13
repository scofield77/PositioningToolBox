# -*- coding:utf-8 -*-
# carete by steve at  2018 / 06 / 06　20:28

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from PositioningAlgorithm.OptimizationAlgorithm import UwbStaticLocation

if __name__ == '__main__':
    # dir_name = '/home/steve/Data/NewFusingLocationData/0044/'
    base_dir_name = 'D:/Data/NewFusingLocationData/'

    test_dir_name = base_dir_name + "0058/"


    def plot_static_measurement(dir_name):
        uwb_data = np.loadtxt(dir_name + 'uwb_data.csv', delimiter=',')
        beacon_set = np.loadtxt(dir_name + 'beaconset_no_mac.csv', delimiter=',')
        usl = UwbStaticLocation.UwbStaticLocation(beacon_set)
        pose = usl.calculate_position(uwb_data[:, 1:])
        plt.figure(int(dir_name.split('/')[-2]))
        valid_count = 0
        for i in range(beacon_set.shape[0]):
            if beacon_set[i,0] < 1000.0 and np.max(uwb_data[:,i+1])>0.0:
                valid_count+=1
        valid_index = 0
        for i in range(beacon_set.shape[0]):
            if beacon_set[i, 0] < 1000.0 and np.max(uwb_data[:, i + 1]) > 0.0:
                valid_index+=1
                plt.subplot(100 * valid_count+10+valid_index)
                plt.title('label:' + str(i))

                tmp_uwb_data = uwb_data[:, i + 1] * 1.0
                plt.hist(tmp_uwb_data[tmp_uwb_data > 0.0],
                         label=str(tmp_uwb_data[tmp_uwb_data > 0].shape[0]) + '/' + str(tmp_uwb_data.shape[0]))
                plt.axvline(np.linalg.norm(pose-beacon_set[i,:]),color='k',linestyle='dashed',linewidth=1)

                plt.legend()
        plt.figure(111)
        plt.title('beacon')
        i_list = list()
        for i in range(beacon_set.shape[0]):
            if beacon_set[i, 0] < 1000.0 and np.max(uwb_data[:, i + 1]) > 0.0:
                plt.text(beacon_set[i, 0], beacon_set[i, 1],  str(i-29))
                i_list.append(i)
        plt.plot(beacon_set[i_list, 0], beacon_set[i_list, 1], 'r*')

        plt.plot(pose[0], pose[1], 'r+')

        plt.text(pose[0], pose[1], 'p' + dir_name.split('/')[-2])

        plt.grid()
        print(dir_name, np.linalg.norm(pose - beacon_set[32, :]))


    # plot_static_measurement(test_dir_name)
    for i in range(53,63):
        plot_static_measurement(base_dir_name + '%04d/' % (i))

    plt.show()
