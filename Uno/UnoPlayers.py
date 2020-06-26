import numpy as np
from .DeckOfCards import master_list_of_cards


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanUnoPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                # print(int(i/self.game.n), int(i%self.game.n))
                try:
                    print('Card ID:', i, 'Card:', master_list_of_cards[i])
                except:
                    print('Card ID:', i, 'Draw a new card')
                    pass
        print('Choose card ID: ')
        while True:
            a = input()

            #x,y = [int(x) for x in a.split(' ')]
            #a = self.game.n * x + y if x != -1 else self.game.n ** 2
            a = int(a)
            if valid[a]:
                break
            else:
                print('Invalid')

        return a


class GreedyUnoPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]
