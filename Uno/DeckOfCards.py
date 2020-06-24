import random
import numpy as np

size_of_deck: int
colors = ['B', 'G', 'R', 'Y']


class UnoCard:
    def __init__(self, symbol, color, draw_number=0, reverse=False, skip=False, wild=False):
        self.color = color
        self.symbol = symbol
        self.draw_number = draw_number
        self.reverse = reverse
        self.skip = skip
        self.wild = wild
        self.id = None

    def __repr__(self):
        return f'{self.color if self.color is not None else ""}{self.symbol}'

    def __eq__(self, other):
        if not isinstance(other, UnoCard):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class DrawTwo(UnoCard):
    def __init__(self, color):
        super(DrawTwo, self).__init__('+2', color=color, draw_number=2)


class DrawFour(UnoCard):
    def __init__(self):
        super(DrawFour, self).__init__('W+4', color=None, draw_number=4, wild=True)


class Wild(UnoCard):
    def __init__(self):
        super(Wild, self).__init__('W', color=None, wild=True)


class Skip(UnoCard):
    def __init__(self, color):
        super(Skip, self).__init__('⊘', color=color, skip=True)


class Reverse(UnoCard):
    def __init__(self, color):
        super(Reverse, self).__init__('⇅', color=color, reverse=True)


def id_all_cards(list_of_cards):
    temp_list = list_of_cards[:]
    id = 0
    for card in temp_list:
        # Python guarantees this to be ordered.
        # We can safely assume that every card will be properly ID'ed
        # since we initialize every deck in the proper order.
        card.id = id
        id += 1
    return temp_list


# Make the master list of cards that we will use to generate our decks
master_list_of_cards = []  # List of cards
for color in colors:
    # Numbered Cards
    master_list_of_cards += [UnoCard(str(i), color) for i in range(10)] + [UnoCard(str(i), color) for i in range(1, 10)]
    # Draw Two Cards
    master_list_of_cards += [DrawTwo(color), DrawTwo(color)]
    # Reverse Cards
    master_list_of_cards += [Reverse(color), Reverse(color)]
    # Skip Cards
    master_list_of_cards += [Skip(color), Skip(color)]
# Four wild and wild draw four card
master_list_of_cards += [Wild(), DrawFour(), Wild(), DrawFour(), Wild(), DrawFour(), Wild(), DrawFour()]
# ID all the cards so we can later make a one-hot representation
size_of_deck = len(master_list_of_cards)
master_list_of_cards = id_all_cards(master_list_of_cards)



class Deck:
    """A Collection of un-drawn Cards and the methods necessary for game play"""
    def __init__(self):
        """Initial deck, taken from the UNO (R) card game instructions
        108 cards as follow:
        19 Blue cards--0-9 [one 0 and two 1-9]
        19 Green cards--0-9 [one 0 and two 1-9]
        19 Red cards--0-9 [one 0 and two 1-9]
        19 Yellow cards--0-9 [one 0 and two 1-9]
        8 Draw Two cards--2 each in blue, green, red, and yellow
        8 Reverse cards--2 each in blue, green, red, and yellow
        8 Skip cards--2 each in blue, green, red, and yellow
        4 Wild cards
        4 Wild Draw Four cards
        """
        self.cards = master_list_of_cards[:]
        # Shuffle cards upon initialization
        self.shuffle_cards()

    def __str__(self):
        return str(self.cards)

    def __len__(self):
        return len(self.cards)

    def shuffle_cards(self):
        """Shuffle self.cards so that they are in a random order"""
        #random.shuffle(self.cards)
        # We really don't need to shuffle the entire deck when using the MCTS, since the shuffle is lost each turn
        # Indeed, we only need to move at most 5 random cards to the top of the deck (in case of a draw four card and regular draw)
        if len(self.cards) >0:
            number_of_cards_to_move_to_top = min(5, len(self.cards))
            rand_indices = [random.randrange(len(self.cards)) for _ in range(number_of_cards_to_move_to_top)]
            for i in range(number_of_cards_to_move_to_top):
                rand_index = rand_indices[i]
                self.cards[-i], self.cards[rand_index] = self.cards[rand_index], self.cards[-i]

    def draw_card(self):
        card = self.cards.pop()
        self.shuffle_cards()
        return card

    def reload_deck_from_pile(self, pile_of_cards):
        self.cards = pile_of_cards[:]
        for card in self.cards:
            if card.wild:
                card.color = None
        self.shuffle_cards()

    def id_all_cards(self):
        id = 0
        for card in self.cards:
            # Python guarantees this to be ordered.
            # We can safely assume that every card will be properly ID'ed
            # since we initialize every deck in the proper order.
            card.id = id
            id += 1

    def numpy(self) -> np.array:
        present_ids = [card.id for card in self.cards]
        array = np.zeros(size_of_deck+1)
        array[present_ids] = 1
        return array

    def from_numpy(self, present_ids_array) -> None:
        self.cards = [master_list_of_cards[int(id)] for id in np.where(present_ids_array.squeeze())[0] if id < size_of_deck]
        self.shuffle_cards()


class Discard:
    def __init__(self):
        self.cards = []

    def __len__(self):
        return len(self.cards)

    def __str__(self):
        return str(self.cards)

    def append_card(self, card):
        self.cards.append(card)

    def top_card(self) -> UnoCard:
        return self.cards[-1]

    def get_all_but_top_card(self):
        all_but_top = self.cards[:-1]
        self.cards = self.cards[-1:]
        return all_but_top

    def numpy(self) -> np.array:
        present_ids = [card.id for card in self.cards[:-1]]
        array = np.zeros(size_of_deck+1)
        array[present_ids] = 1
        return array

    def top_numpy(self) -> np.array:
        array = np.zeros(size_of_deck+1)
        array[self.cards[-1].id] = 1
        num = np.sum(array)
        return array

    def from_numpy(self, present_ids_array_bottom, present_ids_array_top) -> None:
        bottoms = [master_list_of_cards[int(id)] for id in np.where(present_ids_array_bottom.squeeze())[0] if id < size_of_deck]
        top = [master_list_of_cards[int(id)] for id in np.where(present_ids_array_top.squeeze())[0] if id < size_of_deck]
        self.cards = bottoms+top
        pass


class Hand:
    """A collection Cards representing those in a player's hand and methods for game play"""
    def __init__(self, deck, discard):
        self.cards = []  # type: [UnoCard]
        self.deck = deck  # The deck that we will draw from
        self.discard = discard  # The discard will will play on

    def __len__(self):
        return len(self.cards)

    def __str__(self):
        return str(self.cards)

    def draw_card_from_deck(self):
        drawn_card = self.deck.draw_card()
        self.cards.append(drawn_card)

    def get_valid_moves(self):
        top_card_on_discard = self.discard.cards[-1]
        valid_moves = []
        if top_card_on_discard.wild:
            # I haven't figured out a good way to pass colored wilds into the neural network,
            # and that causes crashes and stalls during self-play.
            # A temporary work around is simply allow any card to be valid if the top card is a wild
            valid_moves = self.cards[:]
        for card in self.cards:
            if card.color == top_card_on_discard.color or card.symbol == top_card_on_discard.symbol or card.wild:
                # The conditions were placed in this order to take advantage of evaluation short-circuiting.
                # It's more likely that two cards have the same color than symbol than wild
                valid_moves.append(card)

        return valid_moves

    def get_valid_moves_numpy(self):
        valid_cards = self.get_valid_moves()
        valid_vector = [0]*(size_of_deck+1)
        for card in valid_cards:
            valid_vector[card.id] = 1
        if len(self.deck) > 0 or len(self.discard) > 1:
            valid_vector[-1] = 1  # We can always draw a card if there is a card to draw
        return np.array(valid_vector)

    def play_card(self, card):
        valid_moves = self.get_valid_moves()
        if card in valid_moves:
            self.cards.remove(card)
            if card.wild:
                card.color = self.best_color_for_wild()
            self.discard.append_card(card)
        else:
            print(f'Desired card {card} not a valid move. Valid moves are {valid_moves}')
            print(f'The deck currently looks like:')
            print('\t', self.deck)
            print(f'The hand currently looks like:')
            print('\t', self)
            print(f'The discard currently looks like:')
            print('\t', self.discard)
            print(f'The discard top card currently looks like:')
            print('\t', self.discard.top_card())

    def play_card_by_id(self, card_id):
        card = None
        for c in self.cards:
            if c.id == card_id:
                card = c
                break
        if card is not None:
            self.play_card(card)
        else:
            print(f'Card {card_id} not found in hand: {self}')

    def best_color_for_wild(self):
        colors_of_cards = [card.color for card in self.cards]
        if len(colors_of_cards):
            return max(set(colors_of_cards), key=colors_of_cards.count)
        else:
            return random.choice(colors)

    def numpy(self) -> np.array:
        present_ids = [card.id for card in self.cards]
        array = np.zeros(size_of_deck+1)
        array[present_ids] = 1
        array[-1] = 1  # Drawing a card is always an option
        return array

    def from_numpy(self, present_ids_array) -> None:
        self.cards = [master_list_of_cards[int(id)] for id in np.where(present_ids_array.squeeze())[0] if id < size_of_deck]
