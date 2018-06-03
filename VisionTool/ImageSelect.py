# -*- coding:utf-8 -*-
# carete by steve at  2018 / 06 / 03　下午3:17
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


import os
import shutil

if __name__ == '__main__':
    dir_name = '/media/steve/cb325f68-9e23-45ae-8fc1-0f7c0ee58131/ImageData/100GOPRO/park1/'
    new_dir_name ='/media/steve/cb325f68-9e23-45ae-8fc1-0f7c0ee58131/ImageData/100GOPRO/park1_cut/'

    # os.mkdir(new_dir_name)

    for file_name in os.listdir(dir_name):
        if 'png' in file_name:
            code = int(file_name.split('.')[0])

            if code % 3 is 0:
                shutil.copyfile(dir_name+file_name,new_dir_name+file_name)
