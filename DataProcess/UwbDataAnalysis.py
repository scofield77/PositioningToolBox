# -*- coding:utf-8 -*-
# carete by steve at  2018 / 06 / 06　20:28

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

if __name__ == '__main__':
    dir_name = '/home/steve/Data/NewFusingLocationData/0044/'
    base_dir_name = 'D:/Data/NewFusingLocationData/'

    test_dir_name = base_dir_name + "0057/"


    def plot_static_measurement(dir_name):
        uwb_data = np.loadtxt(dir_name + 'uwb_data.csv', delimiter=',')
        beacon_set = np.loadtxt(dir_name + 'beaconset_no_mac.csv', delimiter=',')

        # plt.figure()
        for i in range(beacon_set.shape[0]):
            if beacon_set[i, 0] < 1000.0 and np.max(uwb_data[:, i + 1]) > 0.0:
                plt.figure()
                plt.title('label:' + str(i))

                tmp_uwb_data = uwb_data[:, i + 1] * 1.0
                plt.hist(tmp_uwb_data[tmp_uwb_data > 0.0],
                         label=str(tmp_uwb_data[tmp_uwb_data > 0].shape[0]) + '/' + str(tmp_uwb_data.shape[0]))

                plt.legend()
        plt.figure()
        plt.title('beacon')
        i_list = list()
        for i in range(beacon_set.shape[0]):
            if beacon_set[i, 0] < 1000.0 and np.max(uwb_data[:, i + 1]) > 0.0:
                plt.text(beacon_set[i, 0], beacon_set[i, 1], 'label:' + str(i))
                i_list.append(i)
        plt.plot(beacon_set[i_list, 0], beacon_set[i_list, 1], '*')
        plt.grid()
        print(np.linalg.norm(beacon_set[29, :] - beacon_set[32, :]))


    plot_static_measurement(test_dir_name)

    plt.show()