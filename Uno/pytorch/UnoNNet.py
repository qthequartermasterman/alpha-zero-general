import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

class UnoNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(UnoNNet, self).__init__()
        self.fc1 = nn.Linear(self.board_x * self.board_y, 128)
        list_of_modules = []
        for _ in range(args.num_channels):
            list_of_modules += [nn.Linear(128, 128), nn.ReLU()]
        self.sequence = nn.Sequential(*list_of_modules)

        self.fc3 = nn.Linear(128, self.action_size)

        self.fc4 = nn.Linear(128, 1)

        #self.b_fc1 = nn.Linear(self.board_x*self.board_y)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, self.board_x*self.board_y)                # batch_size x 1 x board_x x board_y
        #s = s.flatten()
        s = nn.ReLU()(self.fc1(s))
        s = self.sequence(s)

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=0), torch.tanh(v)
