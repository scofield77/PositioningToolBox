# -*- coding:utf-8 -*-
# carete by steve at  2018 / 09 / 25　10:55 AM
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

import torch
import torch.nn as nn
# import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms

import tensorboardX

import matplotlib.pyplot as plt
import scipy   as sp
import numpy as np

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from PDRTool.PDRLearning.DataLoader import SimpleDataSet

    dir_name = '/home/steve/Data/PDR/0003/'
    phone_imu = np.loadtxt(dir_name + 'SMARTPHONE2_IMU.data', delimiter=',')[:, 1:]
    phone_imu_average_time_interval = (phone_imu[-1, 0] - phone_imu[0, 0]) / float(phone_imu.shape[0])
    full_flag_array = np.loadtxt(dir_name + 'flag_array.csv', delimiter=',')

    data_set = SimpleDataSet.SimpleDataSet(phone_imu[:, 1:10], full_flag_array, 600, 300)

    train_loader = torch.utils.data.DataLoader(dataset=data_set,
                                               batch_size=1000,
                                               shuffle=True)

    dir_name = '/home/steve/Data/PDR/0002/'
    phone_imu2 = np.loadtxt(dir_name + 'SMARTPHONE2_IMU.data', delimiter=',')[:, 1:]
    full_flag_array2 = np.loadtxt(dir_name + 'flag_array.csv', delimiter=',')

    valid_x = data_set.preprocess_other(phone_imu2[:, 1:10])
    ymin = np.min(full_flag_array2)
    ymax = np.max(full_flag_array2)

    valid_y = (full_flag_array2 - ymin) / (ymax - ymin) - 0.5

    from PDRTool.PDRLearning.Model import SimpleLSTM

    model = SimpleLSTM.SimpleLSTM(data_set.input_size,
                                  data_set.output_size,
                                  128,
                                  6).to(device)

    # model = nn.Sequential(
    #     nn.Linear(10*data_set.cut_size,40),
    #     nn.PReLU(),
    #     nn.Linear(40,40),
    #     nn.PReLU(),
    #     nn.Linear(40,1),
    #     nn.Softmax()
    # ).to(device)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(model.parameters())
    from array import array

    cost_array = array('d')
    model.train()

    for epoch in range(50000):
        for i, (dx, dy) in enumerate(train_loader):
            # print(i, dx.shape, dy.shape)
            dx = np.transpose(dx, (2, 0, 1))
            dy = np.transpose(dy, (2, 0, 1))
            # dx = dx.reshape([10,-1]).to(device).float()
            # dy = dy.reshape([10,-1]).to(device).float()
            dx = dx.to(device).float()
            dy = dy.to(device).float()
            # print('dx:', dx.size())
            # print('dy:', dy.size())

            output = model(dx)
            # print(output.size())
            # for i in range(1):
            #     loss = criterion(output[:, i, :], dy[:, i, :])

            loss = criterion(output, dy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cost_array.append(loss.item())

            if epoch % 100 == 0:
                print('epos:', epoch, 'step:', i, 'loss: {:.4f}'.format(loss.item()))

    with torch.no_grad():
        model.eval()
        # y = model(torch.tensor())
        # print('data set x shape', data_set.whole_x.reshape((-1, 1, 6)).shape)
        x_tensor = torch.from_numpy(data_set.whole_x.reshape((-1, 1, data_set.input_size))).float().to(device)
        # x = model(data_set.whole_x.reshape((-1,1,6)).to(device).float())
        y = model(x_tensor).to(torch.device('cpu'))

        plt.figure()
        plt.title('train data')
        plt.plot(data_set.whole_y, label='ref')
        plt.plot(y.numpy().reshape([-1]), label='result')
        plt.legend()

        plt.figure()
        plt.title('cost')

        plt.plot(np.frombuffer(cost_array, dtype=np.float).reshape([-1]))

        x_t = torch.from_numpy(valid_x.reshape((-1, 1, data_set.input_size))).float().to(device)
        yt = model(x_t).to(torch.device('cpu'))

        plt.figure()
        plt.title('valid data')
        plt.plot(valid_y, label='ref')
        plt.plot(yt.numpy().reshape([-1]), label='result')
        plt.legend()

        plt.show()
