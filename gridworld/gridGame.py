from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .gridLogic import Board
import numpy as np

class gridGame(Game):
    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }

    @staticmethod
    def getSquarePiece(piece):
        return OthelloGame.square_content[piece]

    def __init__(self, n_row, n_col, goal):
        self.n_row = n_row
        self.n_col = n_col
        self.goal = goal

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n_row, self.n_col, self.goal)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n_row, self.n_col)

    def getActionSizeRunner(self):
        # return number of actions
        return 5

    def getActionSizeBlocker(self):
        # return number of actions for blocker
        return 9

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if (player == 1 and action == self.getActionSizeRunner()-1) or (player == -1 and action == self.getActionSizeBlocker()-1):
            return (board, -player)
        b = Board(self.n_row,self.n_col, self.goal)
        b.pieces = np.copy(board)
        if player == 1:
            runner = np.where(board[4] == 1)
            b.r_row = runner[0][0]
            b.r_col = runner[1][0]
        else:
            blocker = np.where(board[5] == 1)
            b.b_row = blocker[0][0]
            b.b_col = blocker[1][0]
        b.execute_move(action, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        if player == 1:
            valids = [0]*self.getActionSizeRunner()
        else:
            valids = [0]*self.getActionSizeBlocker()
        b = Board(self.n_row,self.n_col, self.goal)
        b.pieces = np.copy(board)
        runner = np.where(board[4] == 1)
        blocker = np.where(board[5] == 1)
        b.r_row = runner[0][0]
        b.r_col = runner[1][0]
        b.b_row = blocker[0][0]
        b.b_col = blocker[1][0]
        legalMoves =  b.get_legal_moves(player)
        if len(legalMoves)==0:
            valids[-1]=1
            return np.array(valids)
        for x in legalMoves:
            valids[x]=1
        return np.array(valids)

    def getGameEnded(self, board, player, resStepsRun):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n_row, self.n_col, self.goal)
        b.pieces = np.copy(board)
        runner = np.where(board[4] == 1)
        blocker = np.where(board[5] == 1)
        b.r_row = runner[0][0]
        b.r_col = runner[1][0]
        b.b_row = blocker[0][0]
        b.b_col = blocker[1][0]
        if not b.reachability(resStepsRun, self.goal):
            return -1
        if b.atgoal(self.goal):
            return 1
        return 0

    def getCanonicalForm(self, board, player):
        return board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.countDiff(player)

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                print(OthelloGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")
