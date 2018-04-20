# -*- coding:utf-8 -*-
# carete by steve at  2018 / 04 / 18　下午3:29
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

from PositioningAlgorithm.tester.FootImu import *


def q_add(q,w,t):
    '''

    :param q: w x y z
    :param w: x y z
    :param t:
    :return:
    '''

if __name__ == '__main__':
    dir_name = '/home/steve/Data/NewFusingLocationData/0013/'
    imu_data = np.loadtxt(dir_name + 'RIGHT_FOOT.data', delimiter=',')
    imu_data = imu_data[:, 1:]
    imu_data[:, 1:4] *= 9.81
    imu_data[:, 4:7] *= (np.pi / 180.0)

    initial_state = get_initial_state(imu_data[:40, 1:4], np.asarray((0, 0, 0)), 0.0, 9)

    time_interval = (imu_data[-1,0]-imu_data[0,0])/float(imu_data.shape[0])
    print('time interval :', time_interval)

    trace = np.zeros([imu_data.shape[0], 3])
    zv_state = np.zeros([imu_data.shape[0], 1])




    q = np.asarray((1.0,0.0,0.0,0.0))

    for i in range(200):
        print(i)

