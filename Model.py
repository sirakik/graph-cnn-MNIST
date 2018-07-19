# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module

class GCN(nn.Module):
    def __init__(self,
                 adj1,
                 adj2,
                 device):
        super(GCN,self).__init__()

        self.device = device

        self.adj1 = torch.from_numpy(adj1).float().to(self.device)
        self.adj2 = torch.from_numpy(adj2).float().to(self.device)
        self.V1 = self.adj1.size(0)
        self.V2 = self.adj2.size(0)
        self.adj1 = self.adj1.view(-1,self.V1,self.V1)
        self.adj2 = self.adj2.view(-1,self.V2,self.V2)

        self.gcn1 = GraphConv(self.adj1,num_node=784,in_channel=1,out_channel=50)
        self.gcn2 = GraphConv(self.adj2,num_node=196,in_channel=50,out_channel=100)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4900,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = x.view(-1,1,784)
        x = self.gcn1(x)
        x = self.pool1(x)
        x = F.dropout(x,0.25)

        x = x.view(-1,50,196)
        x = self.gcn2(x)
        x = self.pool2(x)
        x = F.dropout(x,0.25)

        x = x.view(-1,4900)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x,0.25)
        x = self.fc2(x)

        return x

class GraphConv(Module):
    def __init__(self,adj,num_node,in_channel,out_channel):
        super(GraphConv, self).__init__()
        self.num_node = num_node
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.adj = adj
        self.sides = math.sqrt(num_node)

        kernel_size = 1
        stride = 1

        self.conv = nn.Conv2d(self.in_channel,
                              self.out_channel,
                              kernel_size=(kernel_size, 1),
                              padding=(int((kernel_size - 1) / 2), 0),
                              stride=(stride, 1),
                              bias = True)

    def forward(self,x):
        '''
        for i,a in enumerate(self.adj):
            x = x.view(-1,self.num_node)
            xa = torch.mm(x,a)
            xa = xa.view(-1,self.in_channel,self.sides,self.sides)
            if i == 0:
                output = self.conv(xa)
            else:
                output = output + self.conv(xa)
        '''
        x = x.view(-1, self.num_node)
        x = torch.mm(x, self.adj)
        x = x.view(-1, self.in_channel, self.sides, self.sides)
        output = self.conv(x)
        print('aaaa')
        print(output.size())

        output = F.relu(output)

        return output
