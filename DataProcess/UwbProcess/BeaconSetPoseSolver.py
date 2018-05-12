# -*- coding:utf-8 -*-
# carete by steve at  2018 / 05 / 10　下午3:38
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

from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from numba import jit

if __name__ == '__main__':
    beacon_set = np.loadtxt('/home/steve/Data/NewFusingLocationData/0044/beaconset_no_mac.csv', delimiter=',')

    know_beacon = beacon_set[29:37, :]
    # 0 34483C38, 58.37, 40.35, 2.17
    # 1 34483945, 66.67, 46.48, 2.17
    # 2 34483935, 40.27, 46.10, 2.17
    # 3 3448452F, 63.65, 32.15, 2.17
    # 4 34483C36, 40.17, 40.44, 2.17
    # 5 34483C31, 58.93, 16.69, 2.17
    # 6 34483946, 57.99, 46.01, 2.17
    # 7 3448384D, 63.55, 16.69, 2.17
    print(know_beacon.shape)
    print(know_beacon)

    unknow_beacon = np.zeros(shape=(6, 3))
    # 0 b40
    # 1 932
    # 2 b39
    # 3 b25
    # 4 c25
    # 5 null

    unknow_matrix = np.asarray((
        0.0, -10.0, 26.10, 14.38, 14.10, 20.37,
        -10.0, 0.0, 14.81, 25.91, -10.0, 11.96,
        -10.0, -10.0, 0.0, -10.0, 12.0, 19.11,
        -10.0, -10.0, -10.0, 0.0, -10.0, 13.99,
        -10.0, -10.0, -10.0, -10.0, 0.0, 14.96,
        -10.0, -10.0, -10.0, -10.0, -10.0, 0.0
    )).reshape(6, 6)
    print(unknow_matrix.shape)

    # know unknow distance
    uk_matrix = np.asarray((
        4, 2, 9.18,
        4, 1, 23.22,
        0, 2, 11.06,
        0, 1, 6.54,
        6, 5, 14.06,
        6, 3, 27.5,
        3, 0, 20.28,
        3, 3, 13.38
    )).reshape(-1, 3)


    def error_func(unknow_b):
        error = 0.0
        unknow_b = unknow_b.reshape(-1, 3)
        for i in range(unknow_matrix.shape[0]):
            for j in range(unknow_matrix.shape[1]):
                if unknow_matrix[i, j] > 0.1:
                    # print('i,j', i, j, unknow_matrix[i, j])
                    error += abs(np.linalg.norm(unknow_b[i, :] - unknow_b[j, :]) - unknow_matrix[i, j])
        for i in range(uk_matrix.shape[0]):
            k = int(uk_matrix[i, 0])
            u = int(uk_matrix[i, 1])
            d = float(uk_matrix[i, 2])
            error += abs(np.linalg.norm(know_beacon[k, 0:2] - unknow_b[u, 0:2]) - d)

        error += np.std(unknow_b[:, 2]) + abs(np.mean(unknow_b[:, 2]) - 1.4)
        return error


    res = minimize(error_func, x0=unknow_beacon)

    unknow_beacon = res.x.reshape(-1, 3)
    print(unknow_beacon)
    print('error:', res.fun)

    cal_dis_m = np.zeros_like(unknow_matrix)
    for i in range(cal_dis_m.shape[0]):
        for j in range(cal_dis_m.shape[1]):
            cal_dis_m[i, j] = np.linalg.norm(unknow_beacon[i, :] - unknow_beacon[j, :])
    print('ref matrix \n:', cal_dis_m)
    print('---------')
    print(unknow_matrix)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(unknow_beacon[:, 0], unknow_beacon[:, 1], unknow_beacon[:, 2], 'r*')
    for i in range(unknow_beacon.shape[0]):
        ax.text(unknow_beacon[i, 0], unknow_beacon[i, 1], unknow_beacon[i, 2], s=str(i))

    plt.figure()
    plt.plot(unknow_beacon[:, 0], unknow_beacon[:, 1], 'r*')

    for i in range(unknow_beacon.shape[0]):
        plt.text(unknow_beacon[i, 0], unknow_beacon[i, 1], s=str(i))

    plt.plot(know_beacon[:, 0], know_beacon[:, 1], 'b*')

    for i in range(know_beacon.shape[0]):
        plt.text(know_beacon[i, 0], know_beacon[i, 1], s='know-' + str(i))
    plt.grid()


    @jit(nopython=True)
    def p2line(p, p0, p1):
        dis = ((p0[1] - p1[1]) * p[0] + (p1[0] - p0[0]) * p[1] + p0[0] * p1[1] - p1[0] * p0[1]) / np.linalg.norm(
            p1 - p0)
        dis = abs(dis)

        c = np.linalg.norm(p1 - p0)
        a = np.linalg.norm(p - p0)
        b = np.linalg.norm(p - p1)
        if a ** 2.0 > c ** 2.0 + b ** 2.0:
            dis = b
        if b ** 2.0 > a ** 2.0 + c ** 2.0:
            dis = a

        return dis


    def generate_score_matrix():
        # line_pare = np.zeros(shape=(7, 2))
        line_pare = np.asarray((
            1, 2,
            1, 5,
            2, 4,
            4, 5,
            4, 0,
            0, 3,
            5, 3
        ), dtype=np.int32).reshape(-1, 2)

        map_range = np.asarray(
            (40.0, 65.0,
             15.0, 50.0)
        ).reshape(-1, 2)

        relution = 1.0 / 20.0
        # map_matrix = np.zeros()

        import array

        surf_array = array.array('d')

        x_pos = map_range[0, 0]
        y_pos = map_range[1, 0]
        print(map_range[0, 1] - map_range[0, 0], map_range[1, 1] - map_range[1, 0])
        full_array = np.zeros(shape=(int((map_range[0, 1] - map_range[0, 0]) * 1.0 / relution),
                                     int((map_range[1, 1] - map_range[1, 0]) * 1.0 / relution)))

        while x_pos < map_range[0, 1]:
            y_pos = map_range[1, 0]
            while y_pos < map_range[1, 1]:
                surf_array.append(x_pos * 1.0)
                surf_array.append(y_pos * 1.0)

                all_dis = np.zeros(line_pare.shape[0])

                for i in range(all_dis.shape[0]):
                    all_dis[i] = p2line(np.asarray((x_pos, y_pos)),
                                        unknow_beacon[line_pare[i, 0], :2],
                                        unknow_beacon[line_pare[i, 1], :2])
                surf_array.append(np.min(all_dis))
                # print(x_pos, y_pos, (all_dis))
                full_array[int((x_pos - map_range[0, 0]) * 1.0 / relution), int(
                    (y_pos - map_range[1, 0]) * 1.0 / relution)] = np.min(
                    all_dis)
                y_pos += relution
            x_pos += relution

        surf_mat = np.frombuffer(surf_array, dtype=np.float64).reshape(-1, 3)
        plt.figure()
        plt.imshow(full_array.transpose())
        # plt.axes(((map_range[0,0],map_range[0,1]),(map_range[1,0],map_range[1,1])))
        plt.colorbar()

        return line_pare, surf_mat, full_array, map_range


    lp, surf_mat, full_array, map_range = generate_score_matrix()
    print(surf_mat.shape)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(surf_mat[:, 0], surf_mat[:, 1], surf_mat[:, 2])
    # ax.scatter(surf_mat[:, 0], surf_mat[:, 1], surf_mat[:, 2])
    # ax.grid()
    dir_name = '/home/steve/Data/NewFusingLocationData/'
    beacon_set[14, :] = unknow_beacon[0, :]
    beacon_set[37:, :] = unknow_beacon[1:-1, :]

    print(beacon_set)


    def save_file(dir_name):
        import os
        import re

        for sub_dir in os.listdir(dir_name):
            dm = re.compile('^[0-9]{4}$')
            # print(sub_dir)
            if dm.match(sub_dir):
                # print(dir_name + sub_dir + '/beaconset_fill.csv')
                np.savetxt(dir_name + sub_dir + '/beaconset_fill.csv', beacon_set, delimiter=',')
                np.save(dir_name + sub_dir + '/{0}-{1}-{2}-{3}'.format(map_range[0, 0],
                                                                      map_range[0, 1],
                                                                      map_range[1, 0],
                                                                      map_range[1, 1]),
                        full_array)


    save_file(dir_name)

    plt.figure()
    plt.plot(unknow_beacon[:, 0], unknow_beacon[:, 1], 'r*')

    for i in range(unknow_beacon.shape[0]):
        plt.text(unknow_beacon[i, 0], unknow_beacon[i, 1], s=str(i))

    plt.plot(know_beacon[:, 0], know_beacon[:, 1], 'b*')

    for i in range(know_beacon.shape[0]):
        plt.text(know_beacon[i, 0], know_beacon[i, 1], s='know-' + str(i))

    # lp = generate_score_matrix()
    for i in range(lp.shape[0]):
        plt.plot(unknow_beacon[lp[i, :], 0], unknow_beacon[lp[i, :], 1], label='line' + str(i))

    plt.legend()

    plt.grid()

    plt.show()
