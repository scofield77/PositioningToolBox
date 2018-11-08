# -*- coding:utf-8 -*-
# carete by steve at  2018 / 09 / 24　4:44 PM
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

import json
import torch
import torch.autograd.variable as Variable
import torch.nn as nn
import torch.utils.data as data


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(SimpleLSTM, self).__init__()
        self.lstm_group = nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  batch_first=True)
        # bidirectional=True)

        self.fc1 = nn.Linear(hidden_size, 20)
        self.dp = nn.Dropout(0.5)
        self.ac1 = nn.Tanh()
        self.fc2 = nn.Linear(20, output_size)

        # self.softmax = nn.Softmax(output_size)
        self.softmax = nn.Sigmoid()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        # h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        h0 = (torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(torch.device('cuda'))
        c0 = (torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(torch.device('cuda'))
        # h0 = (torch.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) for _ in range(self.hidden_size))
        # c0 = (torch.Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) for _ in range(self.hidden_size))
        out, _ = self.lstm_group(x, (h0, c0))

        # out = self.softmax(self.fc(out))
        # for
        real_out = torch.zeros([out.size(0), out.size(1), 1]).to(torch.device('cuda'))
        # print('out size:', out.size(), 'real out size:',real_out.size())
        for i in range(out.size(1)):
            # real_out[:, i, :] = self.softmax(self.fc(out[:, i, :]))
            # real_out[:, i, :] = self.fc(out[:, i, :])
            real_out[:, i, :] = self.fc2(self.ac1(self.dp(self.fc1(out[:, i, :]))))
            # real_out[:, i, :] = self.softmax(self.fc2(self.ac1(self.fc1(out[:, i, :]))))
        # for batch_i in range(out.size(1)):
        #     for time_i in range(out.size(0)):
        #         print(self.softmax(self.fc(out[time_i, batch_i, :].reshape([-1]))))
        #         real_out[time_i, batch_i, :] = self.softmax(self.fc(out[time_i, batch_i, :].reshape([-1])))
        # real_out = self.softmax(self.fc(out))
        return real_out
