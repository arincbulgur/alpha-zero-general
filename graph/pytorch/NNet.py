import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim
from torch_geometric.data import Data, DataLoader

from .graphNNet import graphNNet as onnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})


class NNetWrapper(NeuralNet):
    def __init__(self, game, agent):
        self.board_x, self.board_y = game.getBoardSize()
        if agent == 1:
            self.action_size = game.getActionSizeRunner()
        else:
            self.action_size = game.getActionSizeBlocker()

        self.device = torch.device('cuda' if args.cuda else 'cpu')
        self.nnet = onnet(game, agent, args).to(self.device)

    def train(self, examples):

        optimizer = optim.Adam(self.nnet.parameters(), lr=0.0005)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                graphs = self.convert2graph(boards, pis, vs)
                loader = DataLoader(graphs, batch_size=len(graphs))

                for data in loader:
                    data = data.to(self.device)

                    # compute output
                    out_pi, out_v = self.nnet(data)
                    l_pi = self.loss_pi(data.pi, out_pi)
                    l_v = self.loss_v(data.v, out_v)
                    total_loss = l_pi + l_v

                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

    # def train(self, examples):
    #     """
    #     examples: list of examples, each example is of form (board, pi, v)
    #     """
    #     optimizer = optim.Adam(self.nnet.parameters())
    #
    #     for epoch in range(args.epochs):
    #         print('EPOCH ::: ' + str(epoch + 1))
    #         self.nnet.train()
    #         pi_losses = AverageMeter()
    #         v_losses = AverageMeter()
    #
    #         batch_count = int(len(examples) / args.batch_size)
    #
    #         t = tqdm(range(batch_count), desc='Training Net')
    #         for _ in t:
    #             sample_ids = np.random.randint(len(examples), size=args.batch_size)
    #             boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
    #             boards = torch.FloatTensor(np.array(boards).astype(np.float64))
    #             target_pis = torch.FloatTensor(np.array(pis))
    #             target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
    #
    #             # predict
    #             if args.cuda:
    #                 boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
    #
    #             # compute output
    #             out_pi, out_v = self.nnet(boards)
    #             l_pi = self.loss_pi(target_pis, out_pi)
    #             l_v = self.loss_v(target_vs, out_v)
    #             total_loss = l_pi + l_v
    #
    #             # record loss
    #             pi_losses.update(l_pi.item(), boards.size(0))
    #             v_losses.update(l_v.item(), boards.size(0))
    #             t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)
    #
    #             # compute gradient and do SGD step
    #             optimizer.zero_grad()
    #             total_loss.backward()
    #             optimizer.step()

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        graph = self.convert2graph_predict(board)
        loader = DataLoader([graph], batch_size=1)
        # board = torch.FloatTensor(board.astype(np.float64))
        # if args.cuda: board = board.contiguous().cuda()
        # board = board.view(6, self.board_x, self.board_y)
        self.nnet.eval()
        for data in loader:
            data = data.to(self.device)
            with torch.no_grad():
                pi, v = self.nnet(data)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])

    def convert2graph(self, boards, pis ,vs):
        data_list = []
        for (b,p,val) in zip(boards,pis,vs):
            edge_dep = []
            edge_arr = []
            node_label = []
            for i in range(self.board_x):
                for j in range(self.board_y):
                    if b[0][i][j]==0:
                        edge_dep.append(int(i*self.board_y+j))
                        if i < self.board_x-1:
                            edge_arr.append(int((i+1)*self.board_y+j))
                        else:
                            edge_arr.append(int(i*self.board_y+j))
                    if b[1][i][j]==0:
                        edge_dep.append(int(i*self.board_y+j))
                        if j > 0:
                            edge_arr.append(int(i*self.board_y+j-1))
                        else:
                            edge_arr.append(int(i*self.board_y+j))
                    if b[2][i][j]==0:
                        edge_dep.append(int(i*self.board_y+j))
                        if i > 0:
                            edge_arr.append(int((i-1)*self.board_y+j))
                        else:
                            edge_arr.append(int(i*self.board_y+j))
                    if b[3][i][j]==0:
                        edge_dep.append(int(i*self.board_y+j))
                        if j < self.board_y-1:
                            edge_arr.append(int(i*self.board_y+j+1))
                        else:
                            edge_arr.append(int(i*self.board_y+j))
                    if b[4][i][j]==1:
                        node_label.append([0,1,0])
                    elif b[5][i][j]==1:
                        node_label.append([0,0,1])
                    else:
                        node_label.append([1,0,0])
            edge_index = torch.tensor([edge_dep,edge_arr], dtype=torch.long)
            x = torch.tensor(node_label, dtype=torch.float)
            pi = torch.tensor([p], dtype=torch.float)
            v = torch.tensor([val], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, pi=pi, v=v)
            data_list.append(data)
        return data_list

    def convert2graph_predict(self, b):
        edge_dep = []
        edge_arr = []
        node_label = []
        for i in range(self.board_x):
            for j in range(self.board_y):
                if b[0][i][j]==0:
                    edge_dep.append(int(i*self.board_y+j))
                    if i < self.board_x-1:
                        edge_arr.append(int((i+1)*self.board_y+j))
                    else:
                        edge_arr.append(int(i*self.board_y+j))
                if b[1][i][j]==0:
                    edge_dep.append(int(i*self.board_y+j))
                    if j > 0:
                        edge_arr.append(int(i*self.board_y+j-1))
                    else:
                        edge_arr.append(int(i*self.board_y+j))
                if b[2][i][j]==0:
                    edge_dep.append(int(i*self.board_y+j))
                    if i > 0:
                        edge_arr.append(int((i-1)*self.board_y+j))
                    else:
                        edge_arr.append(int(i*self.board_y+j))
                if b[3][i][j]==0:
                    edge_dep.append(int(i*self.board_y+j))
                    if j < self.board_y-1:
                        edge_arr.append(int(i*self.board_y+j+1))
                    else:
                        edge_arr.append(int(i*self.board_y+j))
                if b[4][i][j]==1:
                    node_label.append([0,1,0])
                elif b[5][i][j]==1:
                    node_label.append([0,0,1])
                else:
                    node_label.append([1,0,0])
        edge_index = torch.tensor([edge_dep,edge_arr], dtype=torch.long)
        x = torch.tensor(node_label, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)

        return data
