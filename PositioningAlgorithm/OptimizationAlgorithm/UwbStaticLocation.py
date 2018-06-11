# -*- coding:utf-8 -*-
# carete by steve at  2018 / 06 / 11　16:58

import numpy as np
import scipy as sp

from scipy.optimize import minimize
import math
from numba import jit,float64,vectorize

import matplotlib.pyplot as plt



class UwbStaticLocation:
    def __init__(self,beacon_set):
        '''
        Initial and set beaconset
        :param beacon_set:
        '''
        self.measurements = np.zeros([beacon_set.shape[0],1])
        self.beacon_set = beacon_set * 1.0





