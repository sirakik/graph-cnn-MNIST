# -*- coding: utf-8 -*-
#!/usr/bin/env python
import sys
sys.path.append(' /usr/local/lib/python3.6/site-packages')
import argparse
import numpy as np
from tqdm import tqdm
#torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset,DataLoader
from torchvision import datasets, transforms

from tool import generate_adjacency_matrix
from Model import GCN

class Processor():
    def __init__(self):
        self.get_argparser()
        self.load_data()
        self.adjacency_matrix()
        self.load_model()
        self.load_optimizer()

    def get_argparser(self):
        parser = argparse.ArgumentParser(description='Graph MNIST')
        parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
        parser.add_argument('--test-batch-size', type=int, default=1000)
        parser.add_argument('--lr', type=int, default=0.01)
        parser.add_argument('--momentum', type=int, default=0.5)
        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--log-interval', type=int, default=10)
        parser.add_argument('--use-gpu', action='store_true',default=False,help='use gpu True or False')
        parser.add_argument('--gpu', type=int, default=3, help='GPU device')
        #graph conv
        parser.add_argument('--num_class',type=int,default=10)
        parser.add_argument('--pool_size',default=(2,2))

        self.args = parser.parse_args()

        if self.args.use_gpu:
            if self.args.gpu == 0:
                self.device = torch.device('cuda:0')
            if self.args.gpu == 1:
                self.device = torch.device('cuda:1')
            if self.args.gpu == 2:
                self.device = torch.device('cuda:2')
            if self.args.gpu == 3:
                self.device = torch.device('cuda:3')
        else:
            self.device = torch.device('cpu')

    def load_data(self):
        self.train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data',
                                                                       train=True,
                                                                       download=True,
                                                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                                                     transforms.Normalize((0.1307,),(0.3081,))
                                                                                                     ])
                                                                       ),
                                                        batch_size=self.args.batch_size,
                                                        shuffle=True,
                                                        num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data',
                                                                      train=False,
                                                                      transform=transforms.Compose([transforms.ToTensor(),
                                                                                                    transforms.Normalize((0.1307,),(0.3081,))
                                                                                                    ])
                                                                      ),
                                                        batch_size=self.args.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=0)

    def load_model(self):
        self.model = GCN(adj1 = self.adjacency_mat1,
                         adj2 = self.adjacency_mat2,
                         device = self.device).to(self.device)
        self.loss = nn.CrossEntropyLoss().to(self.device)

    def load_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)

    def train(self, epoch):
        self.model.train()
        for batch_idx,(data,label) in enumerate(tqdm(self.train_loader,desc='batch',position=2)):
            #get data
            data = Variable(data.to(self.device))
            label = Variable(label.to(self.device))
            # forward
            output = self.model(data)
            loss = self.loss(output, label)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #print
            if batch_idx % self.args.log_interval == 0:
                self.print_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.item()),print_tf=False)

    def eval(self,epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        for batch_idx, (data, label) in enumerate(self.test_loader):
            data = Variable(data.to(self.device))
            label = Variable(label.to(self.device))

            output = self.model(data)
            test_loss += self.loss(output, label).item()
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)
        #print
        self.print_log('Test: Epoch[{}/{}], Accuracy: {}/{} ({:.2f}%)'
                       .format(epoch,self.args.epochs, correct, len(self.test_loader.dataset), 100.*correct/len(self.test_loader.dataset))
                       ,print_tf = True)

    def start(self):
        for epoch in tqdm(range(1, self.args.epochs + 1), desc='epoch',position=1):
            self.train(epoch)
            self.eval(epoch)

    def adjacency_matrix(self):
        self.adjacency_mat1 = generate_adjacency_matrix(28)
        self.adjacency_mat2 = generate_adjacency_matrix(14)

    def print_log(self,str,print_tf):
        #log
        with open('{}/log.txt'.format('.'), 'a') as f:
            f.write('\n' + str)
        #print
        if print_tf:
            sys.stdout.write('\r{}'.format(str))
            sys.stdout.flush()

if __name__ == '__main__':
    with open('{}/log.txt'.format('.'), 'w') as f:  # log.txt
        f.write('Graph MNIST\n')
    processor = Processor()
    processor.start()


