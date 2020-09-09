from random import choice
from random import random

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque

def main():
    #Initialization
    board = np.load('gameboard9602.npy')                                        #Loading the graph sequences in numpy array format
    plt.ion()                                                                   #Enabling interactive plotting
    G = nx.MultiDiGraph()
    node_pose = {}
    for i in range(board.shape[2]):
        node_pose[i] = (10 + 30*(i%4),10 + 30*(i//4))
        G.add_node(i, pos=node_pose[i])                                         #Adding nodes into the graph with their position

    #Drawing
    while True:
        for i in range(board.shape[0]):
            drawGraph(board[i],G,node_pose)                                     #Drawing graphs before or after transitions
            if i < board.shape[0]-1:
                drawEdge(board[i],board[i+1],G,node_pose)                       #Drawing graphs during transitions

def drawGraph(board,G,node_pose):
    #Node labeling
    color_map = []
    for i in range(board.shape[1]):
        if board[1,i,i] == 1 and board[2,i,i] == 1:
            color_map.append('green')
        elif board[1,i,i] == 1:
            color_map.append('yellow')
        elif board[2,i,i] == 1:
            color_map.append('blue')
        elif i == 14:
            color_map.append('orange')
        else:
            color_map.append('red')

    #Edge searching
    edges = []
    for i in range(board.shape[1]):
        for j in range(board.shape[2]):
            if board[0,i,j] == 1:
                edges.append((i,j))

    #Plotting
    G = nx.create_empty_copy(G)                                                 #Remove all edges before the new plot
    nx.draw(G,pos=node_pose,node_color=color_map)                               #Draw merely nodes
    layout = dict((n, G.node[n]["pos"]) for n in G.nodes())                     #Get position data for each node
    ax = plt.gca()                                                              #Curved edge plotting
    for edge in edges:
        ax.annotate("",
                    xy=layout[edge[1]], xycoords='data',
                    xytext=layout[edge[0]], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="0",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=-0.3",
                                    ),
                    )

    plt.margins(0.2)                                                            #Leaving margins around the plot

    green_patch = mpatches.Patch(color='green', label='Both agents')            #Legending
    blue_patch = mpatches.Patch(color='blue', label='Blocker')
    yellow_patch = mpatches.Patch(color='yellow', label='Runner')
    orange_patch = mpatches.Patch(color='orange', label='Goal')
    plt.legend(handles=[blue_patch,yellow_patch,green_patch,orange_patch])

    plt.show()
    input("Press Enter to continue...")
    plt.clf()

def drawEdge(board,next_board,G,node_pose):
    #Node labeling
    color_map = []
    for i in range(board.shape[1]):
        if board[1,i,i] == 1 and board[2,i,i] == 1:
            color_map.append('green')
        elif board[1,i,i] == 1:
            color_map.append('yellow')
        elif board[2,i,i] == 1:
            color_map.append('blue')
        elif i == 14:
            color_map.append('orange')
        else:
            color_map.append('red')
        if board[1,i,i] == 1:
            runner = i
        if next_board[1,i,i] == 1:
            next_runner = i
        if board[2,i,i] == 1:
            blocker = i
        if next_board[2,i,i] == 1:
            next_blocker = i

    #Edge and transition searching
    edges = []
    transitions = []
    for i in range(board.shape[1]):
        for j in range(board.shape[2]):
            if board[0,i,j] == 1:
                edges.append((i,j))
    if runner != next_runner:
        edges.remove((runner,next_runner))
        transitions.append((runner,next_runner))
    elif blocker != next_blocker:
        edges.remove((blocker,next_blocker))
        transitions.append((blocker,next_blocker))

    #Plotting
    G = nx.create_empty_copy(G)                                                 #Remove all edges before the new plot
    nx.draw(G,pos=node_pose,node_color=color_map)                               #Draw merely nodes
    layout = dict((n, G.node[n]["pos"]) for n in G.nodes())                     #Get position data for each node
    ax = plt.gca()                                                              #Curved edge and transition plotting
    for edge in edges:
        ax.annotate("",
                    xy=layout[edge[1]], xycoords='data',
                    xytext=layout[edge[0]], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="0",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=-0.3",
                                    ),
                    )
    for transition in transitions:
        ax.annotate("",
                    xy=layout[transition[1]], xycoords='data',
                    xytext=layout[transition[0]], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="0", lw=3,
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=-0.3",
                                    ),
                    )

    plt.margins(0.2)                                                            #Leaving margins around the plot

    green_patch = mpatches.Patch(color='green', label='Both agents')            #Legending
    blue_patch = mpatches.Patch(color='blue', label='Blocker')
    yellow_patch = mpatches.Patch(color='yellow', label='Runner')
    orange_patch = mpatches.Patch(color='orange', label='Goal')
    plt.legend(handles=[blue_patch,yellow_patch,green_patch,orange_patch])

    plt.show()
    input("Press Enter to continue...")
    plt.clf()

if __name__ == "__main__":
    main()
