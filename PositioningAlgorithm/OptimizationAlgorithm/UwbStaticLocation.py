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
        self.beacon_set = beacon_set * 1.0

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
                        error_sum += (measurement[i,j] - np.linalg.norm(pose - beaconset[j, :])) ** 2.0
            return error_sum ** 0.5

        return compute_error(pose, self.measurements, self.beacon_set)

    def calculate_position(self, measurement):
        self.measurements = measurement * 1.0

        initial_pose = np.zeros(3)
        result = minimize(self.positioning_error_function,
                          initial_pose,
                          method='BFGS')
        print('minimize result', result.x)
        return result.x
