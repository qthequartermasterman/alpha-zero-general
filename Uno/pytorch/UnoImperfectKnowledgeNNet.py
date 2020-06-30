import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UnoImperfectKnowledgeNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(UnoImperfectKnowledgeNNet, self).__init__()

        self.b_fc1 = nn.Linear(3*self.action_size, 256)
        self.b_relu1 = nn.ReLU()
        self.b_fc2 = nn.Linear(256, 256)
        self.b_relu2 = nn.ReLU()

        self.hc_conv1d1 = nn.Conv1d(1, 64, kernel_size=1, padding=1)
        self.hc_relu1 = nn.ReLU()
        self.hc_conv1d2 = nn.Conv1d(64, 64, kernel_size=5, padding=0)
        self.hc_relu2 = nn.ReLU()
        self.hc_fc1 = nn.Linear(64*3, 256)

        self.combined_fc_pi = nn.Linear(256, self.action_size)
        self.combined_fc_v = nn.Linear(256, 1)
        self.combined_ReLU = nn.ReLU()

    def extract_allowed_information(self, board):
        # Extract only the information the NNet should know
        # The only cards it can know are the cards in the discard pile and the current player's hand
        # It also can know the number of cards in each player's hands
        num_rows, num_columns = board.shape[-2], board.shape[-1]
        num_players = num_rows-3
        known_cards = board[..., 1:4, :]
        card_counts_every_container = torch.sum(board, axis=-1)
        player_ids = [-2 % num_players,  # 2 players back
                      -1 % num_players,  # 1 player forward
                      0,                 # Current Player
                      1 % num_players,   # 1 player forward
                      2 % num_players]   # 2 players forward
        hand_counts_list = [card_counts_every_container[..., 3 + id] for id in player_ids]
        #hand_counts = torch.FloatTensor(hand_counts_list)
        hand_counts = torch.stack(hand_counts_list).T
        hand_counts = 1 / num_columns * hand_counts
        hand_counts = hand_counts.unsqueeze(-2)

        return known_cards, hand_counts

    def forward(self, board):
        if board.ndim == 2:
            board = board.unsqueeze(0)

        known_cards, hand_counts = self.extract_allowed_information(board)

        known_cards = known_cards.view(-1, self.action_size*3)
        known_cards = self.b_fc1(known_cards)
        known_cards = self.b_relu1(known_cards)
        known_cards = self.b_fc2(known_cards)
        known_cards = self.b_relu2(known_cards)

        # hand_counts = hand_counts.view(1, 1, 5)
        hand_counts = self.hc_conv1d1(hand_counts)
        hand_counts = self.hc_relu1(hand_counts)
        hand_counts = self.hc_conv1d2(hand_counts)
        hand_counts = self.hc_relu2(hand_counts)
        hand_counts = hand_counts.view(-1, 64*3)
        hand_counts = self.hc_fc1(hand_counts)

        combined = known_cards*hand_counts
        pi = self.combined_fc_pi(combined)
        v = self.combined_fc_v(combined)

        return F.log_softmax(pi, dim=0), torch.tanh(v)
