import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm
import operator
from itertools import starmap

from Arena import Arena
from MCTS import MCTS

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, rnnet, bnnet, args):
        self.game = game
        self.rnnet = rnnet
        self.bnnet = bnnet
        self.rpnet = self.rnnet.__class__(self.game,1)  # the competitor network
        self.bpnet = self.bnnet.__class__(self.game,-1)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.rnnet, self.bnnet, self.args)
        self.trainExamplesHistoryRunner = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.trainExamplesHistoryBlocker = []
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamplesRunner = []
        trainExamplesBlocker = []
        graph = self.game.getInitGraph()
        self.curPlayer = 1
        episodeStep = 0
        removal = self.args.removal

        while True:
            episodeStep += 1
            canonicalGraph = self.game.getCanonicalForm(graph, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalGraph, self.curPlayer, episodeStep, temp=temp)
            # sym = self.game.getSymmetries(canonicalBoard, pi)
            # for b, p in sym:
            #     trainExamples.append([b, self.curPlayer, p, None])
            # p = list(pi.ravel())
            if self.curPlayer == 1:
                trainExamplesRunner.append([graph, self.curPlayer, pi, None])
            else:
                trainExamplesBlocker.append([graph, self.curPlayer, pi, None])

            action = np.random.choice(len(pi), p=pi)
            graph, self.curPlayer = self.game.getNextState(graph, self.curPlayer, action)

            resStepsRun = (self.args.maxlenOfEps-(episodeStep+1-2))//2
            r = self.game.getGameEnded(graph, self.curPlayer, resStepsRun)*self.curPlayer

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamplesRunner],[(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamplesBlocker]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamplesRunner = deque([], maxlen=self.args.maxlenOfQueue)
                iterationTrainExamplesBlocker = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.rnnet, self.bnnet, self.args)  # reset search tree
                    iterationTrainExamplesRunner, iterationTrainExamplesBlocker = starmap(operator.iconcat, zip((iterationTrainExamplesRunner,iterationTrainExamplesBlocker),(self.executeEpisode())))

                # save the iteration examples to the history
                self.trainExamplesHistoryRunner.append(iterationTrainExamplesRunner)
                self.trainExamplesHistoryBlocker.append(iterationTrainExamplesBlocker)

            if len(self.trainExamplesHistoryRunner) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistoryRunner)}")
                self.trainExamplesHistoryRunner.pop(0)
                self.trainExamplesHistoryBlocker.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamplesRunner = []
            for e in self.trainExamplesHistoryRunner:
                trainExamplesRunner.extend(e)
            shuffle(trainExamplesRunner)

            trainExamplesBlocker = []
            for e in self.trainExamplesHistoryBlocker:
                trainExamplesBlocker.extend(e)
            shuffle(trainExamplesBlocker)

            # training new network, keeping a copy of the old one
            self.rnnet.save_checkpoint(folder=self.args.checkpoint, filename='temprunner.pth.tar')
            self.rpnet.load_checkpoint(folder=self.args.checkpoint, filename='temprunner.pth.tar')
            self.bnnet.save_checkpoint(folder=self.args.checkpoint, filename='tempblocker.pth.tar')
            self.bpnet.load_checkpoint(folder=self.args.checkpoint, filename='tempblocker.pth.tar')
            rpbpmcts = MCTS(self.game, self.rpnet, self.bpnet, self.args)

            self.rnnet.train(trainExamplesRunner)
            self.bnnet.train(trainExamplesBlocker)
            rnbnmcts = MCTS(self.game, self.rnnet, self.bnnet, self.args)
            rnbpmcts = MCTS(self.game, self.rnnet, self.bpnet, self.args)
            rpbnmcts = MCTS(self.game, self.rpnet, self.bnnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena1 = Arena(lambda x,y: np.argmax(rpbpmcts.getActionProb(x, 1, y, temp=0)),
                          lambda x,y: np.argmax(rpbpmcts.getActionProb(x, -1, y, temp=0)), self.game, self.args.maxlenOfEps)
            rpwins1, bpwins1, draws1 = arena1.playGames(self.args.arenaCompare)

            arena2 = Arena(lambda x,y: np.argmax(rpbnmcts.getActionProb(x, 1, y, temp=0)),
                          lambda x,y: np.argmax(rpbnmcts.getActionProb(x, -1, y, temp=0)), self.game, self.args.maxlenOfEps)
            rpwins2, bnwins1, draws2 = arena2.playGames(self.args.arenaCompare)

            arena3 = Arena(lambda x,y: np.argmax(rnbpmcts.getActionProb(x, 1, y, temp=0)),
                          lambda x,y: np.argmax(rnbpmcts.getActionProb(x, -1, y, temp=0)), self.game, self.args.maxlenOfEps)
            rnwins1, bpwins2, draws3 = arena3.playGames(self.args.arenaCompare)

            arena4 = Arena(lambda x,y: np.argmax(rnbnmcts.getActionProb(x, 1, y, temp=0)),
                          lambda x,y: np.argmax(rnbnmcts.getActionProb(x, -1, y, temp=0)), self.game, self.args.maxlenOfEps)
            rnwins2, bnwins2, draws4 = arena4.playGames(self.args.arenaCompare)

            log.info('PRERUN/PREBLO WINS : %d / %d' % (rpwins1, bpwins1))
            log.info('PRERUN/NEWBLO WINS : %d / %d' % (rpwins2, bnwins1))
            log.info('NEWRUN/PREBLO WINS : %d / %d' % (rnwins1, bpwins2))
            log.info('NEWRUN/NEWBLO WINS : %d / %d' % (rnwins2, bnwins2))
            if i==1:
                sfile = open(os.path.join(self.args.checkpoint, "score.txt"), "w+")
            sfile.write('Iteration '+str(i)+'\n')
            sfile.write('PRERUN/PREBLO WINS : %d / %d\n' % (rpwins1, bpwins1))
            sfile.write('PRERUN/NEWBLO WINS : %d / %d\n' % (rpwins2, bnwins1))
            sfile.write('NEWRUN/PREBLO WINS : %d / %d\n' % (rnwins1, bpwins2))
            sfile.write('NEWRUN/NEWBLO WINS : %d / %d\n' % (rnwins2, bnwins2))
            if rpwins1 + rpwins2 + rnwins1 + rnwins2 == 0:
                log.info('ZERO RUNNER WIN')
                sfile.write('ZERO RUNNER WIN\n')
                self.rnnet.load_checkpoint(folder=self.args.checkpoint, filename='temprunner.pth.tar')
            elif float(rnwins1 + rnwins2) / (rpwins1 + rpwins2 + rnwins1 + rnwins2) < self.args.updateThreshold:
                log.info('REJECTING NEW RUNNER MODEL')
                sfile.write('REJECTING NEW RUNNER MODEL\n')
                self.rnnet.load_checkpoint(folder=self.args.checkpoint, filename='temprunner.pth.tar')
            else:
                log.info('ACCEPTING NEW RUNNER MODEL')
                sfile.write('ACCEPTING NEW RUNNER MODEL\n')
                self.rnnet.save_checkpoint(folder=self.args.checkpoint, filename=('runner_' + self.getCheckpointFile(i)))
                self.rnnet.save_checkpoint(folder=self.args.checkpoint, filename='runnerbest.pth.tar')

            if bpwins1 + bpwins2 + bnwins1 + bnwins2 == 0:
                log.info('ZERO BLOCKER WIN')
                sfile.write('ZERO BLOCKER WIN\n')
                sfile.write('\n')
                self.bnnet.load_checkpoint(folder=self.args.checkpoint, filename='tempblocker.pth.tar')
            elif float(bnwins1 + bnwins2) / (bpwins1 + bpwins2 + bnwins1 + bnwins2) < self.args.updateThreshold:
                log.info('REJECTING NEW BLOCKER MODEL')
                sfile.write('REJECTING NEW BLOCKER MODEL\n')
                sfile.write('\n')
                self.bnnet.load_checkpoint(folder=self.args.checkpoint, filename='tempblocker.pth.tar')
            else:
                log.info('ACCEPTING NEW BLOCKER MODEL')
                sfile.write('ACCEPTING NEW BLOCKER MODEL\n')
                sfile.write('\n')
                self.bnnet.save_checkpoint(folder=self.args.checkpoint, filename=('blocker_' + self.getCheckpointFile(i)))
                self.bnnet.save_checkpoint(folder=self.args.checkpoint, filename='blockerbest.pth.tar')
            if i%20 == 0:
                arena5 = Arena(lambda x,y: np.argmax(rnbnmcts.getActionProb(x, 1, y, temp=0)),
                          lambda x,y: np.argmax(rnbnmcts.getActionProb(x, -1, y, temp=0)), self.game, self.args.maxlenOfEps)
                rnwinslast, bnwinslast, drawslast = arena5.playGames(5, True, i)

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename_run = os.path.join(folder, "runner_" + self.getCheckpointFile(iteration) + ".examples")
        with open(filename_run, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistoryRunner)
        f.closed
        filename_blo = os.path.join(folder, "blocker_" + self.getCheckpointFile(iteration) + ".examples")
        with open(filename_blo, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistoryBlocker)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
