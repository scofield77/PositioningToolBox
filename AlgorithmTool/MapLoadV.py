# -*- coding:utf-8 -*-
# carete by steve at  2018 / 09 / 01　下午4:24
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


import matplotlib.pyplot as plt
import numpy as np

import os

from scipy.misc import imread

import pyproj

class MapLoader:
    def __init__(self, file_dir):
        if file_dir[-1] == '/':
            self.file_dir = file_dir
        else:
            self.file_dir = file_dir + '/'


        # Read calibration file which contained picture name and relative parameters.
        self.calib_file_name  = 'None'

        for file_name in os.listdir(self.file_dir):
            if '.cal' in file_name:
                self.calib_file_name = self.file_dir + file_name

        if self.calib_file_name is 'None':
            print('Can not find out a calib file for build up image, the file dir is :\n',
                  self.file_dir,
                  '\n the files in such directory are:\n',
                  os.listdir(self.file_dir),
                  'current value of self.calib_file_name is :\n',
                  self.calib_file_name)


        # Load picture and parameters
        self.picture_info_array = np.loadtxt(self.calib_file_name,comments='%',dtype=np.str).transpose().reshape([-1,7])
        '''
        % A building is composed of several floors. Each floor has 6 calibration data:
        % 1) the filename of the bitmap
        % 2) the floor number (e.g. -2, 0, 3)
        % 3) the Building number (e.g. 1, 2, 3)
        % 4) Latidude (in degrees) of image center, 
        % 5) Longitude (in degrees) of image center, 
        % 6) Rotation (in degrees) of image to be aligned to the geometric north
        % 7) Scale (meters/pixel).

        '''
        # print(self.picture_info_array)

        self.picture_list = list() # picture represented map(numpy.narray)

        for i in range(self.picture_info_array.shape[0]):
            tmp_pic = imread(self.file_dir + self.picture_info_array[i,0])
            self.picture_list.append(tmp_pic)


        # initial frame transformation parameters

        import  math
        def convert_wgs_to_utm(lon, lat):
            utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
            if len(utm_band) == 1:
                utm_band = '0' + utm_band
            if lat >= 0:
                epsg_code = '326' + utm_band
            else:
                epsg_code = '327' + utm_band
            return epsg_code
        #
        # setup your projections

        self.centre_lon = float(self.picture_info_array[0,4])
        self.centre_lat = float(self.picture_info_array[0,3])

        utm_code = convert_wgs_to_utm(self.centre_lon, self.centre_lat)
        crs_wgs = pyproj.Proj(init='epsg:4326')  # assuming you're using WGS84 geographic
        crs_utm = pyproj.Proj(init='epsg:{0}'.format(utm_code))

        # then cast your geographic coordinates to the projected system, e.g.
        self.centre_x, self.centre_y = pyproj.transform(crs_wgs, crs_utm, self.centre_lon, self.centre_lat)
        # print( x,y)
        # x,y  =proj.transform(crs_wgs,crs_utm,input_lon+0.1,input_lat+0.1)
        # print(x,y)

    def wgs84_project(self,lon, lat):
        x = 0
        y = 0
        # cx = 0
        # cy = 0

        def convert_wgs_to_utm(lon, lat):
            import  math
            utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
            if len(utm_band) == 1:
                utm_band = '0' + utm_band
            if lat >= 0:
                epsg_code = '326' + utm_band
            else:
                epsg_code = '327' + utm_band
            return epsg_code

        # setup your projections
        utm_code = convert_wgs_to_utm(lon, lat)
        crs_wgs = pyproj.Proj(init='epsg:4326')  # assuming you're using WGS84 geographic
        crs_utm = pyproj.Proj(init='epsg:{0}'.format(utm_code))


        x,y = pyproj.transform(crs_wgs,crs_utm, lon, lat)
        # print(x,y, self.centre_x,self.centre_y)

        return x - self.centre_x, y - self.centre_y



if __name__ == '__main__':
    # ml = MapLoader('/home/steve/Data/IPIN2017Data/Track3/Map/CAR')
    # ml = MapLoader('/home/steve/Data/IPIN2017Data/Track3/Map/UJITI')
    ml = MapLoader('/home/steve/Data/IPIN2017Data/Track3/Map/UJIUB')
    # print(ml.picture_info_array)

    # plt.figure()
    # plt.imshow(ml.picture_list[0])
    # plt.show()
    for i, pic in enumerate(ml.picture_list):
        plt.figure(i+1)
        plt.imshow(pic)
    plt.show()




