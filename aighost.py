import copy as cp
import numpy as np
from termcolor import colored
import random as rd

class AI:
    def __init__(self, rmap, cmap, map, slham, ham, pacman, slghost, ghost):
        self.num_action = 4
        self.list_action = np.array(np.identity(self.num_action, dtype=int).tolist())
        self.location = np.array([[0, 1, 0, -1], [-1, 0, 1, 0]], dtype=int)

        self.r = rmap
        self.c = cmap
        self.map = map
        self.sl_ham = slham
        self.cuple_ham = ham
        self.pacman = pacman
        self.sl_ghost = slghost
        self.ghost = ghost
        # print(self.ghost)

    def math(self):
        print("AI")

    def next_location(self, n, r, c):
        nt = np.dot(self.location, self.list_action[n])
        return r+nt[0], c+nt[1]

    def check(self):
        print(self.pacman)

    def view(self):
        for i in range(self.r):
            for j in range(self.c):
                check = False
                for k in range(self.sl_ghost):
                    if i == self.ghost[k]['r']-1 and j == self.ghost[k]['c']-1:
                        print(colored(self.map[i][j], 'green', 'on_red'), end='')
                        check = True
                if check:
                    continue
                if i == self.pacman['r']-1 and j == self.pacman['c']-1:
                    print(colored(self.map[i][j], 'green', 'on_yellow'), end='')
                else:
                    print(self.map[i][j], end='')
            print()

class simple_ai(AI):
    def math(self):
        for i in range(self.sl_ghost):
            move = self.lcan_move(self.ghost[i]['r'], self.ghost[i]['c'])
            r = rd.randint(0, len(move)-1)
            # print('Can: {} - Selected {}'.format(len(move), move[r]))
            self.ghost[i]['r'] = move[r][0]
            self.ghost[i]['c'] = move[r][1]

    def lcan_move(self, r, c):
        move = []
        for i in range(4):
            r_m, c_m = self.next_location(i, r, c)
            if self.map[r_m-1][c_m-1] != '#':
                move.append((r_m, c_m))

        return move
