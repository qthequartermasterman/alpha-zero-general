from Coach import Coach
from Uno.UnoGame import UnoGame as Game
from utils import *
import os

"""
Before using multiprocessing, please check 2 things before use this script.
1. The number of PlayPool should not over your CPU's core number.
2. Make sure all Neural Network which each process created can store in VRAM at same time. Check your NN size before use this.
"""
args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.51,
    'maxlenOfQueue': 200000,
    'numMCTSSims':  10,
    'cpuct': 1,
    'multiGPU': False,
    'setGPU': '0',
    # The total number of games when self-playing is:
    # Total = numSelfPlayProcess * numPerProcessSelfPlay
    'numSelfPlayProcess': 4,
    'numPerProcessSelfPlay': 15,
    # The total number of games when against-playing is:
    # Total = numAgainstPlayProcess * numPerProcessAgainst
    'numAgainstPlayProcess': 6,
    'numPerProcessAgainst': 10,
    'checkpoint': 'temp/Uno/',
    'numItersForTrainExamplesHistory': 9,

    'dirichletAlpha': 1.75     # α = {0.3, 0.15, 0.03} for chess, shogi and Go respectively, scaled in inverse proportion to the approximate number of legal moves in a typical position
})

if __name__=="__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    g = Game(cards_dealt_per_player=2)
    c = Coach(g, args)
    c.learn()
