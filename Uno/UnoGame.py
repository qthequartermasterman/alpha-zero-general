import sys
sys.path.append('..')
from Game import Game
from .UnoLogic import UnoBoard as Board
from .DeckOfCards import size_of_deck
import numpy as np


class UnoGame(Game):
    def __init__(self, num_players=2, cards_dealt_per_player=7):
        self.n = num_players  # Default to a two player game of Uno
        self.board = Board(self.n, cards_dealt_per_player)
        # The MCTS crashes for some reason when we have random initial points,
        # so instead, we'll just feed it the same initial position each time
        self.constant_uno_board = Board(num_players, cards_dealt_per_player)

    def getInitBoard(self):
        # return initial board (numpy board)
        b = self.board
        return self.constant_uno_board.get_board_numpy()

    def getBoardSize(self):
        # (a,b) tuple
        # There are self.n + 2 rows since for the deck, discard, discard top, and self.n (# players) hands
        return self.n + 3, size_of_deck + 1

    def getActionSize(self):
        # return number of actions
        return size_of_deck + 1   # 1 more for drawing a card

    def getNextState(self, board, player_of_2, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = self.board
        b.from_numpy(board)
        return b.execute_move_2_player(action, player_of_2)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        b = self.board
        b.from_numpy(board)
        return b.get_valid_moves_2_player_numpy(player)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = self.board
        b.from_numpy(board)
        if b.has_empty_hand_2_player(player):
            return 1
        if b.has_empty_hand_2_player(-player):
            return -1
        return 0

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        if player == 1:
            return np.copy(board)
        else:
            temp = np.copy(board)
            temp[[3, 4]] = temp[[4, 3]]
            # diff = temp-board
            return temp

    def getSymmetries(self, board, pi):
        # There are two types of symmetries in Uno:
        # 1. Equivalent cards (i.e. all wild cards act the same, regardless of ID)
        # 2. Color interchangeability (i.e. swap all red with yellow and vice versa, and the game would play the same)
        sym = [(board, pi)]
        b = self.board
        b.from_numpy(board)

        # Equivalent wild and draw 4 cards (deck dependent)
        wild_ids = [100, 102, 104, 106]  # list of card ids that are standard wild cards
        draw_four_ids = [101, 103, 105, 107]  # list of card ids that are standard draw four cards
        first_wild_id = wild_ids.pop()
        first_draw_four_id = draw_four_ids.pop()
        for id in wild_ids:
            new_board = np.copy(board)
            new_board.T[[id, first_wild_id]] = new_board.T[[first_wild_id, id]]
            sym.append((new_board, pi))
            #duplicate_pairs.append((id, first_wild_id))
        for id in draw_four_ids:
            new_board = np.copy(board)
            new_board.T[[id, first_draw_four_id]] = new_board.T[[first_draw_four_id, id]]
            sym.append((new_board, pi))
            #duplicate_pairs.append((id, first_wild_id))

        # Equivalent numbered/draw 2 cards (deck dependent)
        # I couldn't find a fast way to check for identical cards, so their IDs are hard coded.
        duplicate_pairs = []
        first_blue_one = 1
        first_green_one = 27
        first_red_one = 51
        first_yellow_one = 76
        blue_special_duplicates = [(19, 20), (21, 22), (23, 24)]
        green_special_duplicates = [(44, 45), (46, 47), (48, 49)]
        red_special_duplicates = [(69, 70), (71, 72), (73, 74)]
        yellow_special_duplicates = [(96, 97), (98, 99), (94, 95)]
        number_of_each_color_with_duplicates = 9  # 1-9 have duplicates, 0 does not
        for i in range(number_of_each_color_with_duplicates):
            duplicate_pairs += [(first_blue_one+i,      first_blue_one+number_of_each_color_with_duplicates+i),
                                (first_green_one+i,     first_green_one+number_of_each_color_with_duplicates+i),
                                (first_red_one+i,       first_red_one+number_of_each_color_with_duplicates+i),
                                (first_yellow_one+i,    first_yellow_one+number_of_each_color_with_duplicates+i)]
        duplicate_pairs += blue_special_duplicates
        duplicate_pairs += green_special_duplicates
        duplicate_pairs += red_special_duplicates
        duplicate_pairs += yellow_special_duplicates
        # Make a new board with all duplicate card swapped. This is faster than a new board for each pair-wise swap
        new_board = np.copy(board)
        for pair in duplicate_pairs:
            new_board.T[[pair[0], pair[1]]] = new_board.T[[pair[1], pair[0]]]
        sym.append((new_board, pi))

        # Color interchangeability
        # For example, exchange all red cards with corresponding green cards
        first_blue = 0
        first_green = 25
        first_red = 50
        first_yellow = 75
        combinations = [(first_blue,    first_green),
                        (first_blue,    first_red),
                        (first_blue,    first_yellow),
                        (first_green,   first_red),
                        (first_green,   first_yellow),
                        (first_red,     first_yellow)]
        for combo in combinations:
            # Swap all cards with corresponding card of different color
            duplicate_pairs = [(combo[0]+i, combo[1]+i) for i in range(25)]
            # Make a new board with all duplicate card swapped. This is faster than a new board for each pair-wise swap
            new_board = np.copy(board)
            for pair in duplicate_pairs:
                new_board.T[[pair[0], pair[1]]] = new_board.T[[pair[1], pair[0]]]
            sym.append((new_board, pi))

        return sym

    def stringRepresentation(self, board):
        # 8x8 numpy array (canonical board)
        #return board.__repr__()
        flat_board = board.flatten().astype(int)
        str_list = map(str, flat_board.tolist())
        bin_string = ''.join(str_list)
        return hex(int(bin_string, 2))

    def getScore(self, board, player):
        return None

    def getCard(self, id):
        return self.board.get_card_with_id(id)

    def print_board_human_friendly(self, board):
        self.board.from_numpy(board)
        return str(self.board)

    def get_imperfect_knowledge_board(self, board, player_id):
        self.board.from_numpy(board)
        known_cards = np.stack([board[1], board[2], board[3+player_id]])
        player_ids = [self.board.get_next_player(player_id,skip=True, reverse=True),    # 2 players back
                      self.board.get_next_player(player_id,skip=False, reverse=True),   # 1 player forward
                      player_id,                                                        # Current Player
                      self.board.get_next_player(player_id,skip=False, reverse=False),  # 1 player forward
                      self.board.get_next_player(player_id,skip=True, reverse=False)]   # 2 players forward

        card_counts_every_container = np.sum(board, axis=1)
        card_counts_list = [card_counts_every_container[3+id] for id in player_ids]
        card_counts = np.array(card_counts_list)
        card_counts = 1/self.getActionSize() * card_counts

        return known_cards, card_counts

def display(board):
    b = Board()
    b.from_numpy(board)
    print(b)
