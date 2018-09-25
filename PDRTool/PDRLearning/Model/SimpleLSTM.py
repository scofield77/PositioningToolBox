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
import torch.nn as nn
import torch.utils.data as data


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(SimpleLSTM, self).__init__()
        self.lstm_group = nn.LSTM(input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(output_size)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(torch.cuda.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(torch.cuda.device)

        out, _ = self.lstm_group(x, (h0, c0))

        # out = self.softmax(self.fc(out))
        # for
        for i in range(out.size(1)):
            out[:, i, :] = self.softmax(self.fc(out[:, i, :]))
        return out
