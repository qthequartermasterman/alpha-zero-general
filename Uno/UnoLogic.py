from .DeckOfCards import *
import random
import numpy as np


class UnoBoard:
    def __init__(self, number_of_players=2, cards_dealt_per_player=7):
        """Setup from the UNO (R) card game instructions
        1. Each player draws a card; the player that draws the highest number deals (count any card with a symbol as 0)
        2. The dealer shuffles and deals each player 7 cards
        3. Place the remainder of the deck facedown to form a DRAW pile
        4. The top card of the DRAW pile is turned over to begin a DISCARD pile.
            NOTE: If any of the Action Cards (symbol) are turned over to start the DISCARD pile,
            see FUNCTIONS OF ACTION CARDS for special instructions.
        """
        self.deck = Deck()
        self.discard = Discard()
        self.players = [Hand(self.deck, self.discard) for _ in range(number_of_players)]
        self.direction_of_play = 1
        # 1 means that we travel "left", i.e. players[0] to players[1] to players[2], etc.
        # -1 means that we travel "right", as if a reverse card was played
        self.curPlayer = 1

        # Initialize game set up instructions
        # 1. We ignore implementing step 1 for now, since all the players will be bots for training
        # 2. Deal the player 7 cards each
        for _ in range(cards_dealt_per_player):
            for player in self.players:
                player.draw_card_from_deck()
        # 3. step 3 is already done.
        # 4. Top card of the draw pile is turned over to begin discard, dealing with special cards if needed
        self.initial_draw()

    def initial_draw(self):
        game_started = False
        while not game_started:
            self.discard.append_card(self.deck.draw_card())
            # Draw 2 cards don't do anything special at the start
            # Reverse cards reverse the direction immediately
            if self.discard.top_card().symbol == 'r':
                self.direction_of_play = -1
                self.curPlayer = 0
                game_started = True
            # Skip cards at the start skip the player to left of dealer
            elif self.discard.top_card().symbol == 's':
                self.curPlayer = 2
                game_started = True
            # Wild Cards are fine
            elif self.discard.top_card().symbol == 'W':
                game_started = True
            # Wild Draw 4
            elif self.discard.top_card().draw_number == 4:
                # Place the card
                old_card = self.discard.cards.pop()
                self.deck.cards.insert(0, old_card)
                game_started = False  # We will draw a new card and start this process over
            # Number card
            else:
                game_started = True

    def get_next_player(self, current_player):
        number_of_players_to_move = 2 if self.discard.top_card().skip else 1
        # Move in the direction the number of players to move, cycling over if needed.
        return (current_player + self.direction_of_play * number_of_players_to_move) % len(self.players)

    def play_card(self, player: int, card):
        """"""
        self.players[player].play_card(card)
        if self.discard.top_card().reverse:
            # Reverse card was played
            self.direction_of_play *= -1
        next_player = self.get_next_player(player)
        for _ in range(self.discard.top_card().draw_number):
            self.reload_deck_if_needed()
            self.players[next_player].draw_card_from_deck()
        # if self.discard.top_card().wild:
        #    # self.discard.cards[-1].color = self.players[player].best_color_for_wild()
        #    self.discard.top_card().color = self.players[player].best_color_for_wild()
        self.curPlayer = next_player

    def reload_deck_if_needed(self):
        if len(self.deck) == 0:
            all_but_top_card_in_discard = self.discard.get_all_but_top_card()
            self.deck.reload_deck_from_pile(all_but_top_card_in_discard)

    def __str__(self):
        deck_string = f'Deck ({len(self.deck)}): {self.deck}\n'
        discard_string = f'Discard ({len(self.discard)}): {self.discard}\n'
        player_strings = [f'{"*****" if i == self.curPlayer else ""}Player {i} ({len(self.players[i])}): {self.players[i]}\n' for i in range(len(self.players))]
        all_strings = [deck_string, discard_string] + player_strings
        return ''.join(all_strings)

    def get_board_numpy(self):
        list_of_np_arrays = [self.deck.numpy(), self.discard.numpy(), self.discard.top_numpy()]
        list_of_np_arrays += [player.numpy() for player in self.players]
        return np.stack(list_of_np_arrays)

    def from_numpy(self, x):
        self.deck.from_numpy(x[:1])
        self.discard.from_numpy(x[1:2], x[2:3])
        self.players.clear()
        hands_np = x[3:]
        for hand in hands_np:
            hand_object = Hand(self.deck, self.discard)
            hand_object.from_numpy(hand)
            self.players.append(hand_object)

    def player_in_list_to_2_player(self, player_in_list):
        return 1 if player_in_list == 0 else -1

    def player_2_to_player_in_list(self, player_of_2):
        return 0 if player_of_2 == 1 else 1

    def execute_move(self, action_id, player):
        #print(f'Trying move with id: {action_id}, with player {player} in list.')
        #print(f'Player {player} in list\'s hand: {self.players[player]}')
        #print(f'Player {(player+1)% len(self.players)} in list\'s hand: {self.players[(player+1)% len(self.players)]}')
        self.curPlayer = player
        self.reload_deck_if_needed()
        if action_id not in [-1, size_of_deck]:
            # -1 or size_of_deck mean the best option is either to draw or there are no good options
            #print(f'Playing card id: {action_id} which is card: {master_list_of_cards[action_id]}.')
            #print(f'Player inside execute_move: {player}')
            old_size = len(self.discard)
            self.players[player].play_card_by_id(action_id)
            new_size = len(self.discard)
            #if new_size - old_size <= 0:
                #print('ERROR ERROR: CARD NOT STORED IN DISCARD')
        else:
            #print(f'Drawing a new card.')
            self.players[player].draw_card_from_deck()
        return self.get_board_numpy(), self.get_next_player(player)

    def execute_move_2_player(self, action_id, player_of_2):
        """Executes a move in a two player game, where player_of_2 is either 1->Player 1 or -1->Player2"""
        """Our uno implementation naturally lists zero from 0 to n"""
        """This function maps   player_of_2->player_in_list """
        """                      1         -> 0             """
        """                     -1         -> 1             """
        """The AlphaZero library that we're forked from uses this formatting"""
        player_in_list = self.player_2_to_player_in_list(player_of_2)
        new_board, new_player_in_list = self.execute_move(action_id, player_in_list)
        new_player_of_two = self.player_in_list_to_2_player(new_player_in_list)
        return new_board, new_player_of_two

    def get_valid_moves_numpy(self, player_in_list):
        return self.players[player_in_list].get_valid_moves_numpy()

    def get_valid_moves_2_player_numpy(self, player_of_2):
        player_in_list = self.player_2_to_player_in_list(player_of_2)
        return self.get_valid_moves_numpy(player_in_list)

    def has_legal_moves(self, player_in_list):
        return len(self.players[player_in_list].get_valid_moves()) > 0

    def has_legal_moves_2_player(self, player_of_2):
        player_in_list = self.player_2_to_player_in_list(player_of_2)
        return self.has_legal_moves(player_in_list)

    def has_empty_hand(self, player_in_list):
        return not bool(len(self.players[player_in_list]))

    def has_empty_hand_2_player(self, player_of_2):
        player_in_list = self.player_2_to_player_in_list(player_of_2)
        return self.has_empty_hand(player_in_list)

    def get_card_with_id(self, id):
        return master_list_of_cards[id]




'''
board = UnoBoard(4)
print(board)

num_cards = [len(player) for player in board.players]
while min(num_cards) > 0:
    current_player = board.curPlayer
    valid_moves = board.players[current_player].get_valid_moves()
    if len(valid_moves) > 0:
        chosen_move = random.choice(valid_moves)
        board.play_card(current_player, chosen_move)
    else:
        board.players[current_player].draw_card_from_deck()
        board.curPlayer = board.get_next_player(current_player)
    if len(board.deck) == 0:
        board.reload_deck_if_needed()
    num_cards = [len(player) for player in board.players]
    print(board)

print(board.get_board_numpy().shape)
'''