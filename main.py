# -*- coding: utf-8 -*-
import logging

import coloredlogs

from Coach import Coach
from graph.graphGame import graphGame as Game
from graph.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 50,              # Number of complete self-play games to simulate during a new iteration.
    'maxlenOfEps': 40,          # Max number of steps in an episode
    'removal':10,               # Blocker's removal limit
    'remCost':0,                # Cost of removing one edge
    'tempThreshold': 15,        #
    'updateThreshold': 0.5,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp0716/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


def main():
    log.info('Loading %s...', Game.__name__)
    goal_position = [9,9]
    g = Game(10,10,goal_position)

    log.info('Loading %s...', nn.__name__)
    rnnet = nn(g,1)             # Neural network for the runner
    bnnet = nn(g,-1)             # Neural network for the blocker

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, rnnet, bnnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
