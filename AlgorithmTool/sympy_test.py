# -*- coding:utf-8 -*-
# Created by steve @ 18-3-26 上午9:36
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

import sympy
from sympy import *

if __name__ == '__main__':
    x, y, z, vx, vy, vz, wx, wy, wz = symbols('x y z vx vy vz wx wy wz')
    ax, ay, az, gx, gy, gz = symbols('ax ay az gx gy gz')

    p = Matrix([[x], [y], [z]])
    v = Matrix([[vx], [vy], [vz]])
    w = Matrix([[wx], [wy], [wz]])
    X = Matrix([[p], [v], [w]])
    acc = Matrix([[ax], [ay], [az]])
    gyr = Matrix([[gx], [gy], [gz]])

    dt = Symbol('dt')

    # p = v * dt + 0.5 * acc * dt * dt

    x = vx * dt + 0.5 * ax * dt * dt

    print(diff(x, vx), diff(x, ax), diff(x, dt))
