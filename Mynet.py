# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/27 16:45
# @Author  : Jis-Baos
# @File    : Mynet.py

import torch
import torch.nn as nn


class MyCRNN(nn.Module):
    def __init__(self, num_classes):
        """
        网络的输入为：W*32的灰度图
        """
        super(MyCRNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.activation1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.activation2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.activation4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1)
        self.batch_norm5 = nn.BatchNorm2d(num_features=512)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1)
        self.batch_norm6 = nn.BatchNorm2d(num_features=512)
        self.activation6 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(2, 2), stride=1, padding=0)

        # input_size: embedding_dim
        # hidden_size: hidden_layer's size
        self.rnn = nn.LSTM(input_size=512, hidden_size=256, num_layers=3, batch_first=True, bidirectional=True)

        self.transcript = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        """

        :param x: [B, 1, W, 32]
        :return: [B, W,num_classes]
        """
        # CNN
        output = self.conv1(x)
        output = self.activation1(output)
        # print(output.size())

        output = self.conv2(output)
        output = self.activation2(output)
        # print(output.size())

        output = self.conv3(output)
        # print(output.size())

        output = self.conv4(output)
        output = self.activation4(output)
        # print(output.size())

        output = self.conv5(output)
        output = self.batch_norm5(output)
        # print(output.size())

        output = self.conv6(output)
        output = self.batch_norm6(output)
        output = self.activation6(output)
        # print(output.size())

        output = self.conv7(output)
        print("CNN's output: ", output.size())

        # RNN
        output = torch.squeeze(output, dim=-2)
        # batch, embedding_dim, seq_len -> batch, seq_len, embedding_dim
        output = output.permute(0, 2, 1)
        print("RNN's input: ", output.size())
        output, (h1, h2) = self.rnn(output)
        print("RNN's output: ", output.size())
        print(output[0][0])
        h1 = h1.permute(1, 0, 2)
        h2 = h2.permute(1, 0, 2)
        print("RNN's h1: ", h1.size())
        print("RNN's h2: ", h2.size())

        # Transcript
        output = self.transcript(output)
        output = self.softmax(output)
        print("Transcript's output: ", output.size())

        return output


if __name__ == '__main__':
    x = torch.rand(size=(4, 1, 32, 164), dtype=torch.float)
    model = MyCRNN(num_classes=52)
    output = model(x)
    print(output.size())
    print(output[0][0].size())
    print(output[0][0])
