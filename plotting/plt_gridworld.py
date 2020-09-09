from random import choice

import numpy as np
import pygame
from collections import deque

BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
GREEN = (0, 255,0)
RED = (255, 0, 0)
BLUE = (0, 0 , 255)
YELLOW = (255, 255, 0)
WINDOW_HEIGHT = 700                                                             #Set the size of the board
WINDOW_WIDTH = 700
blockSize = 50                                                                  #Set the size of the grid block


def main():
    #Initialization
    global SCREEN, CLOCK
    board = np.load('gameboard141.npy')
    pygame.init()
    SCREEN = pygame.display.set_mode((WINDOW_HEIGHT, WINDOW_WIDTH))
    CLOCK = pygame.time.Clock()
    SCREEN.fill(BLACK)

    #Drawing
    while True:
        for i in range(board.shape[0]):
            SCREEN.fill(BLACK)
            drawGrid(board[i])
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            pygame.display.update()
            input("Press Enter to continue...")
            a = False


def drawGrid(board):
    for x in range(int(WINDOW_WIDTH/blockSize)):
        for y in range(int(WINDOW_HEIGHT/blockSize)):
            rect = pygame.Rect(x*blockSize, y*blockSize,
                               blockSize, blockSize)
            pygame.draw.rect(SCREEN, WHITE, rect, 1)
            if board[0][y][x]==0:
                pygame.draw.circle(SCREEN, GREEN, (x*blockSize+blockSize//2,y*blockSize+(blockSize-10)),3)
            else:
                pygame.draw.circle(SCREEN, RED, (x*blockSize+blockSize//2,y*blockSize+(blockSize-10)),3)
            if board[1][y][x]==0:
                pygame.draw.circle(SCREEN, GREEN, (x*blockSize+10,y*blockSize+blockSize//2),3)
            else:
                pygame.draw.circle(SCREEN, RED, (x*blockSize+10,y*blockSize+blockSize//2),3)
            if board[2][y][x]==0:
                pygame.draw.circle(SCREEN, GREEN, (x*blockSize+blockSize//2,y*blockSize+10),3)
            else:
                pygame.draw.circle(SCREEN, RED, (x*blockSize+blockSize//2,y*blockSize+10),3)
            if board[3][y][x]==0:
                pygame.draw.circle(SCREEN, GREEN, (x*blockSize+(blockSize-10),y*blockSize+blockSize//2),3)
            else:
                pygame.draw.circle(SCREEN, RED, (x*blockSize+(blockSize-10),y*blockSize+blockSize//2),3)
            runner = np.where(board[4] == 1)
            pygame.draw.circle(SCREEN, YELLOW, (runner[1][0]*blockSize+blockSize//2,runner[0][0]*blockSize+blockSize//2),6)
            blocker = np.where(board[5] == 1)
            pygame.draw.circle(SCREEN, BLUE, (blocker[1][0]*blockSize+blockSize//2,blocker[0][0]*blockSize+blockSize//2),6)

if __name__ == "__main__":
    main()
