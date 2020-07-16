import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class graphNNet(nn.Module):
    def __init__(self, game, agent, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        if agent == 1:
            self.action_size = game.getActionSizeRunner()
        else:
            self.action_size = game.getActionSizeBlocker()
        self.args = args

        super(graphNNet, self).__init__()

        self.conv1 = GraphConv(3, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)

        self.lin1 = nn.Linear(256, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, self.action_size)
        self.lin4 = nn.Linear(64, 1)

        # super(gridNNet, self).__init__()
        # self.conv1 = nn.Conv2d(6, args.num_channels, 3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        # self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        #
        # self.bn1 = nn.BatchNorm2d(args.num_channels)
        # self.bn2 = nn.BatchNorm2d(args.num_channels)
        # self.bn3 = nn.BatchNorm2d(args.num_channels)
        # self.bn4 = nn.BatchNorm2d(args.num_channels)
        #
        # self.fc1 = nn.Linear(args.num_channels*(self.board_x-4)*(self.board_y-4), 1024)
        # self.fc_bn1 = nn.BatchNorm1d(1024)
        #
        # self.fc2 = nn.Linear(1024, 512)
        # self.fc_bn2 = nn.BatchNorm1d(512)
        #
        # self.fc3 = nn.Linear(512, self.action_size)
        #
        # self.fc4 = nn.Linear(512, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))

        pi = self.lin3(x)
        v = self.lin4(x)

        # #                                                           s: batch_size x board_x x board_y
        # s = s.view(-1, 6, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        # s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        # s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        # s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        # s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        # s = s.view(-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4))
        #
        # s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        # s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512
        #
        # pi = self.fc3(s)                                                                         # batch_size x action_size
        # v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)
