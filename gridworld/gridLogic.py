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

    def __init__(self, n_row, n_col):
        "Set up initial board configuration."

        self.n_row = n_row
        self.n_col = n_col
        # Create the empty board array.
        self.pieces = [None]*6 # 6 stands for binary feature planes (4 edges, 2 agents)
        for i in range(6):
            self.pieces[i] = [None]*self.n_row
            for j in range(self.n_row):
                self.pieces[i][j] = [0]*self.n_col

        # Set up for random initial positions of runner and blocker
        self.r_row = random.randint(0,self.n_row-1)
        self.r_col = random.randint(0,self.n_col-1)
        self.b_row = random.choice([i for i in range(0,self.n_row) if i not in [self.r_row]])
        self.b_col = random.choice([i for i in range(0,self.n_col) if i not in [r_col]])
        self.pieces[4][r_row][r_col] = 1
        self.pieces[5][b_row][b_col] = 1

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
            for i in range(4):
                if self[i][self.r_row][self.r_col] == 0 and tuple(map(sum,zip((self.r_row,self.r_col),self.__directions[i])))!=(self.b_row,self.b_col):
                    moves.update(i)
        else:
            for i in range(4):
                if self[i][self.b_row][self.b_col] == 0 and tuple(map(sum,zip((self.b_row,self.b_col),self.__directions[i])))!=(self.r_row,self.r_col):
                    moves.update(i)
                    moves.update(i+4)
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
            move = tuple(map(sum,zip((self.r_row,self.r_col),self.__directions[action])))
            self[4][self.r_row][self.r_col] = 0
            self[4][move[0]][move[1]] = 1
        else:
            move = tuple(map(sum,zip((self.r_row,self.r_col),self.__directions[action%4])))
            self[5][self.b_row][self.b_col] = 0
            self[5][move[0]][move[1]] = 1
            if action >= 4:
                self[action%4][self.b_row][self.b_col] = 1

    def reachability(self,resStepsRun, goal):
        reachSet = set()
        currentSet = {(goal[0],goal[1])}
        for i in range(resStepsRun):
            newSet = set()
            for j in currentSet:
                for k in range(4):
                    if self[k][j[0]][j[1]] == 0:
                        move = tuple(map(sum,zip(j,self.__directions[k])))
                        if move[0] >= 0 and move[0] < self.n_row and move[1] >= 0 and move[1] < self.n_col:
                            if not move in list(set().union(reachSet,newSet)):
                                newSet.append(move)
            currentSet = newSet - reachSet
            reachSet = set().union(reachSet,newSet)
            if (self.r_row,self.r_col) in reachSet:
                return True
        return False

    def atgoal(self,goal):
        if goal == [self.r_row,self.r_col]:
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
