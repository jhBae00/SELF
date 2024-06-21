import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:,:,:-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        #res = x if self.downsample is None else self.downsample(x)
        return out#self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, in_channel, dilation):
        super(TCN, self).__init__()
        self.tcn = TemporalBlock(in_channel, in_channel, kernel_size=6, stride=1, dilation=9, padding=45, dropout=0.5)
        self.batch = nn.BatchNorm1d(in_channel)

    def forward(self, inputs):
        y1 = self.tcn(inputs)
        y1 = self.batch(y1)
        return y1


adj = torch.ones((8,8)) / 8
class GNN(nn.Module):
    def __init__(self, in_channel, out_channel): # in_channel, out_channel is the feature value
        super(GNN, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel)
        self.adj_weight = torch.nn.Parameter(torch.randn(8,8))
        self.batch = torch.nn.BatchNorm2d(out_channel)

    def forward(self, x):
        # graph generation
        graph = adj.to(x.device) + F.sigmoid(self.adj_weight)
        x = torch.transpose(x, 2, 3)
        output = torch.matmul(x, graph) # x shape should ends with 8 (channel)
        output = torch.transpose(x, 2, 3)
        output = torch.transpose(output, 1, 3)
        output = self.linear(output) # this linear need to change 2 dimensional feature
        output = torch.transpose(output, 1, 3)
        output = self.batch(output)
        return output

class STCB(nn.Module):
    def __init__(self, in_channel, out_channel, dilation):
        super(STCB, self).__init__()
        self.gnn = GNN(in_channel, out_channel)
        self.tcn = TCN(out_channel, dilation)
        self.residual = nn.Linear(in_channel, out_channel)

    def forward(self, x):
        x = torch.transpose(x, 1, 3)
        residual = self.residual(x)
        residual = torch.transpose(residual, 1, 3)
        x = torch.transpose(x, 1, 3)
        gnn = self.gnn(x)
        # x = gnn
        x = F.relu(residual + gnn)
        x = torch.transpose(x, 1, 2)
        B, channel, _, time = x.shape
        # print(x.shape)
        # exit(0)
        x = x.reshape(B*channel, -1, 100)
        x = self.tcn(x)
        x = x.reshape(B, channel, -1, time)
        x = torch.transpose(x, 1, 2)
        return residual + x

class ARMBANDGNN(nn.Module):
    def __init__(self, num_gestures):
        super(ARMBANDGNN, self).__init__()
        self.STCB1 = STCB(1, 8, 1)
        self.STCB2 = STCB(8, 16, 3)
        self.STCB3 = STCB(16, 32, 5)
        self.STCB4 = STCB(32, num_gestures, 7)
        self.linear = nn.Linear(100*8, 1)
        self.drop = nn.Dropout(0.4)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = F.relu(self.STCB1(x))
        x = F.relu(self.STCB2(x))
        x = F.relu(self.STCB3(x))
        x = F.relu(self.STCB4(x))# B *G(feature) * T(time) * N(channel)
        B, num_label, channel, time = x.shape
        x = x.reshape(B, num_label, -1)
        # x = torch.mean(x, dim=2) # B * G
        # print(x.shape)
        # exit(0)
        x = self.linear(x)
        x = torch.squeeze(x, dim=2)
        # x = self.drop(x)
        return F.log_softmax(x, dim=1)

# class ARMBANDGNN(nn.Module):
#     # input_size, output_size, num_channels, kernel_size, dropout
#     #args.channel_electrode, args.num_label, [32, 32, 64, 128], 3,0.05
#     def __init__(self, num_channels, channels, num_classes, input_feature_dim):
#         super(ARMBANDGNN, self).__init__()
#         self.tcn = TCN(num_channels, num_classes, channels, 3, 0.05)
#         self.spat = GNN(input_feature_dim, input_feature_dim)
#         self.spat2 = GNN(input_feature_dim, 100)
#         self.linear = nn.Linear(100*channels[-1], num_classes)
#         self.batchnorm1 = nn.BatchNorm1d(channels[-1])
#         self.batchnorm = nn.BatchNorm1d(num_channels)
#
#
#     def forward(self, x):
#         # x = self.spat(x)
#         #x = self.batchnorm(x)
#         #x = F.normalize(x)
#         #x = self.batchnorm(x)
#         # print('first', x.shape)
#         x = self.tcn(x)
#         # print('second', x.shape)
#         #x = F.relu(x)
#         #x = self.batchnorm1(x)
#         x = self.spat2(x)
#         #x = self.spat2(x)
#         b, _, _ = x.shape
#         x = x.reshape(b, -1)
#         x = self.linear(x)
#         return F.log_softmax(x, dim=1)
