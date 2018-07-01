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


    def plot_static_measurement(dir_name, p_i, total_num):
        uwb_data = np.loadtxt(dir_name + 'uwb_data.csv', delimiter=',')
        beacon_set = np.loadtxt(dir_name + 'beaconset_no_mac.csv', delimiter=',')
        usl = UwbStaticLocation.UwbStaticLocation(beacon_set)
        pose = usl.calculate_position(uwb_data[:, 1:])
        # plt.figure(int(dir_name.split('/')[-2]))
        plt.figure(13)

        valid_count = 0
        for i in range(beacon_set.shape[0]):
            if beacon_set[i, 0] < 1000.0 and np.max(uwb_data[:, i + 1]) > 0.0:
                valid_count += 1
        valid_index = 0
        for i in range(beacon_set.shape[0]):
            # Plot each beacon's measurement.
            if beacon_set[i, 0] < 1000.0 and np.max(uwb_data[:, i + 1]) > 0.0:
                print(valid_index, p_i, valid_index)
                num = 100 * valid_count + 10 * total_num + valid_index + ((p_i - 1) * 6) + 1
                print(num)
                # plt.subplot(num)
                plt.subplot2grid([valid_count + 1, total_num + 1], [valid_index, p_i])
                # plt.subplot(100*)
                # plt.title(str(i - 29))

                tmp_uwb_data = uwb_data[:, i + 1] * 1.0
                plt.hist(tmp_uwb_data[tmp_uwb_data > 0.0],
                         label=str(tmp_uwb_data[tmp_uwb_data > 0].shape[0]) + '/' + str(tmp_uwb_data.shape[0]))
                # plt.axvline(np.linalg.norm(pose - beacon_set[i, :]), color='k', linestyle='dashed', linewidth=1)
                # plt.tight_layout()

                if p_i == 1:
                    # plt.title(str(i - 29), loc='left')
                    # plt.ylabel('Number')
                    plt.ylabel('Beacon:' + str(valid_index) + '\nNumber')

                if valid_index == valid_count - 1:
                    plt.xlabel('dist/m')

                if valid_index == 0:
                    plt.title('T' + str(p_i))

                valid_index += 1
                plt.legend()
        # plt.tight_layout()
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.figure(111)
        plt.title('Beacons and Test points')
        i_list = list()
        for i in range(beacon_set.shape[0]):
            if beacon_set[i, 0] < 1000.0 and np.max(uwb_data[:, i + 1]) > 0.0:
                if i- 29 < 3:
                    plt.text(beacon_set[i, 0], beacon_set[i, 1], str(i - 29))
                else:
                    plt.text(beacon_set[i, 0], beacon_set[i, 1], str(i - 29-1))
                i_list.append(i)
        plt.plot(beacon_set[i_list, 0], beacon_set[i_list, 1], 'r*')

        plt.plot(pose[0], pose[1], 'b+')

        plt.text(pose[0] , pose[1], 'T'+ str(p_i),horizontalalignment='center', fontsize=10)

        plt.grid()
        print(dir_name, np.linalg.norm(pose - beacon_set[29, :]))


    # plot_static_measurement(test_dir_name)

    start_i = 56
    total_num = 3
    total_num = 6
    # total_num = 9
    for i in range(start_i, total_num + start_i):
        plot_static_measurement(base_dir_name + '%04d/' % (i), i - start_i + 1, total_num)

    plt.show()
