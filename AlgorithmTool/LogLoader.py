# -*- coding:utf-8 -*-
# carete by steve at  2018 / 09 / 01　下午4:52
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


from AlgorithmTool.MapLoadV import MapLoader
import numpy as np

import  array


class LogLoader:
    def __init__(self, file_name, wifi_mac_list= 'None', ble_mac_list = 'NONE'):
        '''
        load log file
        :param file_name:
        '''
        file = open(file_name)

        file_lines = file.readlines()

        posi_array = array.array('d')
        acce_array = array.array('d')
        gyro_array = array.array('d')
        magn_array = array.array('d')
        pres_array = array.array('d')



        for line in file_lines:
            if line[0] is '%':
                continue
            if 'POSI' in line:
                # print(line)
                unit = line.split(';')
                # print(unit[1:])
                for i in range(1,len(unit)):
                    posi_array.append(float(unit[i]))

            if 'ACCE' in line:
                unit = line.split(';')
                if len(unit) < 6:
                    print("error of data :",line)
                    continue

                for i in range(1,len(unit)):
                    # print(line)
                    acce_array.append(float(unit[i]))

            if 'GYRO' in line:
                unit = line.split(';')
                # print(line)
                for i in range(1,len(unit)):
                    gyro_array.append(float(unit[i]))

            if 'MAGN' in line:
                unit = line.split(';')
                for i in range(1,len(unit)):
                    magn_array.append(float(unit[i]))

            if 'PRES' in line:
                unit = line.split(';')
                for i in range(1,len(unit)):
                    pres_array.append(float(unit[i]))




        self.posi = np.frombuffer(posi_array,dtype=np.float).reshape([-1,6])
        self.acce = np.frombuffer(acce_array,dtype=np.float).reshape([-1,6])
        self.gyro = np.frombuffer(gyro_array,dtype=np.float).reshape([-1,6])
        self.magn = np.frombuffer(magn_array,dtype=np.float).reshape([-1,6])
        self.pres = np.frombuffer(pres_array,dtype=np.float).reshape([-1,4])


    # def set_ref_frame(self,long):


if __name__ == '__main__':
    # file_name = '/home/steve/Data/IPIN2017Data/Track3/01-Training/CAR/logfile_CAR_R01-2017_A5.txt'
    file_name = '/home/steve/Data/IPIN2017Data/Track3/01-Training/CAR/logfile_CAR_R02-2017_S4.txt'

    ll = LogLoader(file_name)

    ml = MapLoader('/home/steve/Data/IPIN2017Data/Track3/Map/CAR')
    # print(ml.picture_info_array)


    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.plot(ll.posi[:,2],ll.posi[:,3],'--*')
    plt.grid()
    # plt.show()

    axis_pos = np.zeros([ll.posi.shape[0],2])

    for i in range(axis_pos.shape[0]):
        axis_pos[i,0], axis_pos[i,1] = ml.wgs84_project(ll.posi[i,3],ll.posi[i,2])
    # axis_pos = axis_pos - np.mean(axis_pos,axis=0)

    plt.figure(2)
    plt.plot(axis_pos[:,0],axis_pos[:,1],'--+r')
    plt.grid()
    # plt.show()


    plt.figure()
    plt.title('acce')
    plt.plot(ll.acce[:,2:5])
    plt.grid()

    plt.figure()
    plt.title('gyro')
    plt.plot(ll.gyro[:,2:5])
    plt.grid()

    plt.figure()
    plt.title('magn')
    plt.plot(ll.magn[:,2:5])
    plt.grid()

    plt.figure()
    plt.title('pres')
    plt.plot(ll.pres[:,2])
    plt.grid()

    plt.show()

