# -*- coding:utf-8 -*-
# carete by steve at  2018 / 04 / 11　下午9:47
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
import matplotlib.pyplot  as plt

import numpy as np
import scipy as sp

from numba import jit

from numba import cuda

from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32


@cuda.jit
def initial_unit_q(q_array):
    # tx = cuda.threadIdx.x
    # ty = cuda.blockIdx.x
    # bw = cuda.blockDim.x

    # pos = tx + ty * bw
    pos = cuda.grid(1)

    if pos < q_array.shape[1]:
        q_array[0, pos] = 1.0
        q_array[1, pos] = 0.0
        q_array[2, pos] = 0.0
        q_array[3, pos] = 0.0

@cuda.jit
def quaternion_add_euler(q,euler):
    Theta = cuda.device_array([4,4],dtype=np.float32)
    Theta[0,0] = 0.0
    Theta[0,1] = -euler[0]
    Theta[0,2] = -euler[1]
    Theta[0,3] = -euler[2]

    Theta[1,0] = euler[0]
    Theta[1,1] = 0.0



@cuda.jit
def sample(q_array,input,sigma,rng):
    pos = cuda.grid(1)



if __name__ == '__main__':
    dir_name = '/home/steve/Data/NewFusingLocationData/0013/'
    imu_data = np.loadtxt(dir_name + 'RIGHT_FOOT.data', delimiter=',')
    imu_data = imu_data[:, 1:]
    imu_data[:, 1:4] *= 9.81
    imu_data[:, 4:7] *= (np.pi / 180.0)

    block_num = 1024
    thread_pre_block = 32

    particle_num = block_num * thread_pre_block
    print("particle num is", particle_num)
    state_num = 10 + 6 + 6
    input_num = 6
    cuda.profile_start()

    rng_states = create_xoroshiro128p_states(block_num * thread_pre_block, seed=1)

    q_state = cuda.device_array([4, particle_num],dtype=np.float32)
    initial_unit_q[block_num, thread_pre_block](q_state)

    # print(q_state.to_host())
    q_state_host = np.empty(shape=q_state.shape,dtype=q_state.dtype)
    # q_state_host = q_state.to_host()
    q_state.copy_to_host(q_state_host)
    plt.figure()
    plt.plot(q_state_host.transpose())
    # plt.colorbar()
    plt.show()
    cuda.profile_stop()
