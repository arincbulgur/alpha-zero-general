'''
Author: Eric P. Nichols
Date: Feb 8, 2008.
Board class.
Board data:
  1=white, -1=black, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
'''
import random

class Board():

    # list of all 8 directions on the board, as (x,y) offsets
    __directions = [(1,0),(0,-1),(-1,0),(0,1)]

    def __init__(self, n_nodes, goal):
        "Set up initial board configuration."

        self.n_nodes = n_nodes
        self.goal = goal
        # Create the empty graph array.
        self.pieces = [None]*3 # 3 stands for binary feature planes (1 connectivity, 2 agents)
        for i in range(3):
            self.pieces[i] = [None]*self.n_nodes
            for j in range(self.n_nodes):
                self.pieces[i][j] = [0]*self.n_nodes

        # Set up for random connectivity and initial positions of runner and blocker
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                self.pieces[0][i][j] = random.choice(range(0,2))
        self.r_node = random.choice([i for i in range(0,self.n_nodes) if i not in [self.goal])
        self.b_node = random.choice([i for i in range(0,self.n_nodes) if i not in [self.r_node]])
        self.pieces[1][self.r_node][self.r_node] = 1
        self.pieces[2][self.b_node][self.b_node] = 1

    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.pieces[index]

    def countDiff(self, color):
        """Counts the # pieces of the given color
        (1 for white, -1 for black, 0 for empty spaces)"""
        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==color:
                    count += 1
                if self[x][y]==-color:
                    count -= 1
        return count

    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        moves = set()  # stores the legal moves.

        if color == 1:
            for i in range(self.n_nodes):
                if self[0][self.r_node][i] == 1:
                    moves.add(i)
        else:
            for i in range(self.n_nodes):
                if self[0][self.b_node][i] == 1:
                    moves.add(i)
                    moves.add(i+self.n_nodes)
        return list(moves)

    def has_legal_moves(self, color):
        for y in range(self.n_col):
            for x in range(self.n_row):
                if color == 1:
                    if self[4][x][y] == 1:
                        newmoves = self.get_moves_for_square((x,y))
                    if len(newmoves)>0:
                        return True
        return False

    def get_moves_for_square(self, square):
        """Returns all the legal moves that use the given square as a base.
        That is, if the given square is (3,4) and it contains a black piece,
        and (3,5) and (3,6) contain white pieces, and (3,7) is empty, one
        of the returned moves is (3,7) because everything from there to (3,4)
        is flipped.
        """
        (x,y) = square

        # determine the color of the piece.
        color = self[x][y]

        # skip empty source squares.
        if color==0:
            return None

        # search all possible directions.
        moves = []
        for direction in self.__directions:
            move = self._discover_move(square, direction)
            if move:
                # print(square,move,direction)
                moves.append(move)

        # return the generated move list
        return moves

    def execute_move(self, action, color):
        """Perform the given move on the board; flips pieces as necessary.
        color gives the color pf the piece to play (1=white,-1=black)
        """

        if color == 1:
            self[1][action][action] = 1
            self[1][self.r_node][self.r_node] = 0
        else:
            self[2][action%self.n_nodes][action%self.n_nodes] = 1
            self[2][self.b_node][self.b_node] = 0
            if action >= self.n_nodes:
                self[0][self.b_node][action%self.n_nodes] = 0

    def reachability(self,resStepsRun, goal):
        reachSet = set()
        currentSet = {goal}
        for i in range(resStepsRun):
            newSet = set()
            for j in currentSet:
                for k in range(self.n_nodes):
                    if self[0][k][j] == 1:
                        if not k in list(set().union(reachSet,newSet)):
                            newSet.add(k)
            currentSet = newSet - reachSet
            reachSet = set().union(reachSet,newSet)
            if self.r_node in reachSet:
                return True
        return False

    def atgoal(self,goal):
        if goal == self.r_node:
            return True
        return False

    def _discover_move(self, origin, direction):
        """ Returns the endpoint for a legal move, starting at the given origin,
        moving by the given increment."""
        x, y = origin
        color = self[x][y]
        flips = []

        for x, y in Board._increment_move(origin, direction, self.n):
            if self[x][y] == 0:
                if flips:
                    # print("Found", x,y)
                    return (x, y)
                else:
                    return None
            elif self[x][y] == color:
                return None
            elif self[x][y] == -color:
                # print("Flip",x,y)
                flips.append((x, y))

    def _get_flips(self, origin, direction, color):
        """ Gets the list of flips for a vertex and direction to use with the
        execute_move function """
        #initialize variables
        flips = [origin]

        for x, y in Board._increment_move(origin, direction, self.n):
            #print(x,y)
            if self[x][y] == 0:
                return []
            if self[x][y] == -color:
                flips.append((x, y))
            elif self[x][y] == color and len(flips) > 0:
                #print(flips)
                return flips

        return []

    @staticmethod
    def _increment_move(move, direction, n):
        # print(move)
        """ Generator expression for incrementing moves """
        move = list(map(sum, zip(move, direction)))
        #move = (move[0]+direction[0], move[1]+direction[1])
        while all(map(lambda x: 0 <= x < n, move)):
        #while 0<=move[0] and move[0]<n and 0<=move[1] and move[1]<n:
            yield move
            move=list(map(sum,zip(move,direction)))
            #move = (move[0]+direction[0],move[1]+direction[1])
