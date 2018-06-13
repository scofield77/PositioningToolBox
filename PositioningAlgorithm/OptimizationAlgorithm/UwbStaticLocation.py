# -*- coding:utf-8 -*-
# carete by steve at  2018 / 06 / 11　16:58

import numpy as np
import scipy as sp

from scipy.optimize import minimize
import math
from numba import jit, float64, vectorize

import matplotlib.pyplot as plt


class UwbStaticLocation:
    def __init__(self, beacon_set):
        '''
        Initial and set beaconset
        :param beacon_set:
        '''
        self.measurements = np.zeros([beacon_set.shape[0], 1])
        self.weight = np.zeros(self.measurements.shape[0])
        self.beacon_set = beacon_set * 1.0

    def positioning_error_robust_function(self, pose):
        '''

        :param pose:
        :return:
        '''

        def rou(u):
            # return 0.5 * u * u / (1.0 + u * u)

            u2 = u * u
            if u2 < 0.5:
                return u2
            else:
                return 0.5

        # @jit(nopython=True)
        def compute_error(pose, measurement, beaconset, m_w):
            # print(measurement.shape)
            error_sum = 0
            for i in range(measurement.shape[0]):
                for j in range(measurement.shape[1]):
                    if measurement[i, j] > 0.0 and beaconset[j, 0] < 5000.0 and m_w[j] > 0.95:
                        error_sum += rou((measurement[i, j] - np.linalg.norm(pose - beaconset[j, :])) ** 2.0)# / (
                                    # measurement[i, j] ** 0.5)
            return error_sum ** 0.5

        return compute_error(pose, self.measurements, self.beacon_set, self.m_weight)

    def positioning_error_function(self, pose):
        '''

        :param pose:
        :return:
        '''

        # @jit(nopython=True)
        def compute_error(pose, measurement, beaconset):
            # print(measurement.shape)
            error_sum = 0
            for i in range(measurement.shape[0]):
                for j in range(measurement.shape[1]):
                    if measurement[i, j] > 0.0 and beaconset[j, 0] < 5000.0:
                        error_sum += ((measurement[i, j] - np.linalg.norm(pose - beaconset[j, :])) ** 2.0)
            return error_sum ** 0.5

        return compute_error(pose, self.measurements, self.beacon_set)

    def calculate_position(self, measurement):
        self.measurements = measurement * 1.0

        self.m_weight = np.zeros(self.measurements.shape[1])

        for i in range(self.measurements.shape[1]):
            if np.max(self.measurements[:, i]) > 0.0 and self.beacon_set[i,0] < 5000.0:
                tmp_measurement = self.measurements[:, i] * 1.0
                valid_number = (tmp_measurement[tmp_measurement > 0.0]).shape[0]
                print(np.mean(tmp_measurement[tmp_measurement>0.0]),
                      np.std(tmp_measurement[tmp_measurement>0.0]),
                      float(tmp_measurement[tmp_measurement>0.0].shape[0])/float(tmp_measurement.shape[0]))
                self.m_weight[i] = float(tmp_measurement[tmp_measurement>0.0].shape[0])/float(tmp_measurement.shape[0])

        initial_pose = np.zeros(3)
        result = minimize(self.positioning_error_function,
                          initial_pose,
                          method='BFGS')
        initial_pose = result.x
        result = minimize(self.positioning_error_robust_function,
                          initial_pose,
                          method='BFGS')
        print('minimize result', result.x)
        return result.x
