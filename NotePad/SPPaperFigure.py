# -*- coding:utf-8 -*-
# carete by steve at  2018 / 07 / 05　15:30

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

import math


def gaussian_distribution(x, miu, sigma):
    # print('sigma',sigma)
    a = 1.0 / sigma / math.sqrt(2.0 * 3.1415926)
    b = -1.0 * ((x - miu) * (x - miu) / 2.0 / sigma / sigma)
    # if math.isnan(a):
    #     print('a is nan', x, miu, sigma)
    # if math.isnan(b):
    #     print('b is nan', x, miu, sigma)
    # print(a * math.exp(b))#,a,b, x,miu,sigma)
    if math.isnan(math.exp(b)):
        print('error:', b)
        return -10000.0
    else:
        return a * math.exp(b)


if __name__ == '__main__':
    gaussian_pdf_v = np.vectorize(gaussian_distribution)

    x = np.linspace(-10.0, 10.0, 1000)
    plt.figure()
    plt.plot(x, gaussian_pdf_v(x, 0.0, 1.0))

    z = x + 1.0
    plt.figure()
    plt.plot(x, np.abs(np.log(gaussian_pdf_v(z, 0.0, 1.0) / gaussian_pdf_v(x, 0.0, 1.0))))

    plt.show()
