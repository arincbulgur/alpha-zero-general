import logging
import math

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, rnnet, bnnet, args):
        self.game = game
        self.rnnet = rnnet
        self.bnnet = bnnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, curPlayer, episodeStep, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.Es = {}
            self.search(canonicalBoard, curPlayer, episodeStep)

        s = self.game.stringRepresentation(canonicalBoard)
        if curPlayer == 1:
            actionSize = self.game.getActionSizeRunner()
        else:
            actionSize = self.game.getActionSizeBlocker()
        counts = [self.Nsa[(s, curPlayer, a)] if (s, curPlayer, a) in self.Nsa else 0 for a in range(actionSize)]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard, curPlayer, episodeStep):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)
        resStepsRun = (self.args.maxlenOfEps-(episodeStep-2))//2

        gameEnding = self.game.getGameEnded(canonicalBoard, 1, resStepsRun)
        if gameEnding != 0:
            # terminal node
            return -gameEnding*curPlayer

        if (s, curPlayer) not in self.Ps:
            # leaf node
            if curPlayer == 1:
                self.Ps[(s, curPlayer)], v = self.rnnet.predict(canonicalBoard)
            else:
                self.Ps[(s, curPlayer)], v = self.bnnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, curPlayer)
            self.Ps[(s, curPlayer)] = self.Ps[(s, curPlayer)] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[(s, curPlayer)])
            if sum_Ps_s > 0:
                self.Ps[(s, curPlayer)] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[(s, curPlayer)] = self.Ps[(s, curPlayer)] + valids
                self.Ps[(s, curPlayer)] /= np.sum(self.Ps[(s, curPlayer)])

            expCost = 0
            if curPlayer == -1:
                cost = self.args.remCost
                expCost = np.inner(self.Ps[(s,curPlayer)],np.array([0,0,0,0,cost,cost,cost,cost,0]))
            self.Vs[(s, curPlayer)] = valids
            self.Ns[(s, curPlayer)] = 0
            return -v+expCost

        valids = self.Vs[(s, curPlayer)]
        cur_best = -float('inf')
        best_act = -1
        if curPlayer == 1:
            actionSize = self.game.getActionSizeRunner()
        else:
            actionSize = self.game.getActionSizeBlocker()

        # pick the action with the highest upper confidence bound
        for a in range(actionSize):
            if valids[a]:
                if (s, curPlayer, a) in self.Qsa:
                    u = self.Qsa[(s, curPlayer, a)] + self.args.cpuct * self.Ps[(s,curPlayer)][a] * math.sqrt(self.Ns[(s,curPlayer)]) / (
                            1 + self.Nsa[(s, curPlayer, a)])
                else:
                    u = self.args.cpuct * self.Ps[(s,curPlayer)][a] * math.sqrt(self.Ns[(s,curPlayer)] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player= self.game.getNextState(canonicalBoard, curPlayer, a)

        # print(episodeStep, self.game.getGameEnded(canonicalBoard, 1, resStepsRun))
        v = self.search(next_s,next_player,episodeStep+1)

        if (s, curPlayer, a) in self.Qsa:
            self.Qsa[(s, curPlayer, a)] = (self.Nsa[(s,curPlayer, a)] * self.Qsa[(s, curPlayer, a)] + v) / (self.Nsa[(s,curPlayer, a)] + 1)
            self.Nsa[(s,curPlayer, a)] += 1

        else:
            self.Qsa[(s, curPlayer, a)] = v
            self.Nsa[(s, curPlayer, a)] = 1

        self.Ns[(s,curPlayer)] += 1
        return -v
