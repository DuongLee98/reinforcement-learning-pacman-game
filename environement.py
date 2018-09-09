import numpy as np
import copy as cp
import aighost
from termcolor import colored
import queue
import random
import time

class Evironment_PaMaCup:

    def __init__(self, name_map=""):
        self.Step = -1
        self.Lose = -35
        # self.Win = 1000
        self.Wall = -100000
        self.Dot = 12
        self.Guess = 8
        self.numState = 60*pow(2, 13)
        self.info = np.zeros([10], dtype=int)
        self.num_action = 4
        self.name_evironemt = "Moi truong train Pacman PAMA-CUP"
        self.name_map = name_map
        self.list_action = np.array(np.identity(self.num_action, dtype=int).tolist())
        self.location = np.array([[0, 1, 0, -1], [-1, 0, 1, 0]], dtype=int)
        self.init()
        self.aighost = aighost.simple_ai(self.r, self.c, self.map, self.sl_ham, self.cuple_ham, self.pacman, self.sl_ghost, self.ghost)


    def init(self):
        namefile = "gamepama"+self.name_map+".txt"
        file = open(namefile, "r")
        info_size_map = file.readline()
        split_info_size_map = info_size_map.split(" ", 1)
        self.r = int(split_info_size_map[0])
        self.c = int(split_info_size_map[1])

        # print(self.r, self.c)
        self.rmb_r = cp.deepcopy(self.r)
        self.rmb_c = cp.deepcopy(self.c)


        self.map = []
        for i in range(self.r):
            str_tmp = file.readline().split()
            line_map = np.array(list(str_tmp[0]), dtype=str)
            self.map.append(line_map)

        self.map = np.array(self.map, dtype=str)

        self.sldot = np.count_nonzero(self.map == '.')
        self.rmb_sldot = cp.deepcopy(self.sldot)

        # print(self.map)
        self.rmb_map = cp.deepcopy(self.map)

        self.sl_ham = (int)(file.readline())

        # print(self.sl_ham)
        self.rmb_sl_ham = cp.deepcopy(self.sl_ham)

        self.cuple_ham = []
        for i in range(self.sl_ham):
            str_tmp = file.readline().split()
            self.cuple_ham.append(str_tmp)
        self.cuple_ham = np.array(self.cuple_ham, dtype=int)

        # print(self.cuple_ham)
        self.rmb_cuple_ham = cp.deepcopy(self.cuple_ham)

        self.sl_ghost = (int)(file.readline())

        # print(self.sl_ghost)
        self.rmb_sl_ghost = cp.deepcopy(self.sl_ghost)

        self.ghost = []
        for i in range(self.sl_ghost):
            ghost_tmp = dict()
            tmp = file.readline().split(" ", maxsplit=1)
            ghost_tmp['r'] = int(tmp[0])
            ghost_tmp['c'] = int(tmp[1])
            self.ghost.append(ghost_tmp)

        # print(self.ghost)
        self.rmb_ghost = cp.deepcopy(self.ghost)

        self.pacman = dict()
        tmp = file.readline().split(" ", maxsplit=1)
        self.pacman['r'] = int(tmp[0])
        self.pacman['c'] = int(tmp[1])

        # print(self.pacman)
        self.rmb_pacman = cp.deepcopy(self.pacman)

        file.close()


    def reset(self):
        self.r = cp.deepcopy(self.rmb_r)
        self.c = cp.deepcopy(self.rmb_c)
        self.map = cp.deepcopy(self.rmb_map)
        self.sl_ham = cp.deepcopy(self.rmb_sl_ham)
        self.cuple_ham = cp.deepcopy(self.rmb_cuple_ham)
        self.sl_ghost = cp.deepcopy(self.rmb_sl_ghost)
        self.ghost = cp.deepcopy(self.rmb_ghost)
        self.pacman = cp.deepcopy(self.rmb_pacman)
        self.aighost = aighost.simple_ai(self.r, self.c, self.map, self.sl_ham, self.cuple_ham, self.pacman,
                                         self.sl_ghost, self.ghost)
        self.sldot = cp.deepcopy(self.rmb_sldot)
        return self.setInfo(done=False)


    def step(self, action):
        ghost_bf = cp.deepcopy(self.ghost)
        self.aighost.math()

        pacman_bf = cp.deepcopy(self.pacman)
        next_location = np.dot(self.location, self.list_action[action])
        self.pacman['r'] = self.pacman['r']+next_location[0]
        self.pacman['c'] = self.pacman['c']+next_location[1]

        done = False
        reward = 0
        for i in range(self.sl_ghost):
            if self.pacman['r'] == self.ghost[i]['r'] and self.pacman['c'] == self.ghost[i]['c']:
                reward += self.Lose
                done = True
            if self.pacman['r'] == ghost_bf[i]['r'] and self.pacman['c'] == ghost_bf[i]['c'] and pacman_bf['r'] == self.ghost[i]['r'] and pacman_bf['c'] == self.ghost[i]['c']:
                reward += self.Lose
                done = True

        if self.map[self.pacman['r']-1][self.pacman['c']-1] == '.':
            reward += self.Dot
            self.map[self.pacman['r']-1][self.pacman['c']-1] = '+'
            self.sldot -= 1
            if self.sldot == 0:
                # reward += self.Win
                done = True
        elif self.map[self.pacman['r']-1][self.pacman['c']-1] == '#':
            reward += self.Wall
            done = True
        else:
            reward += self.Step

        state = self.setInfo(done)

        # print('Reward: ', reward)
        # print('Dot ', self.sldot)
        # print('Done ', done)
        # print('State ', state)
        return state, reward, done, self.sldot

    def setInfo(self, done=False):
        self.info = np.zeros([14], dtype=int)

        for i in range(4):
            r_m = self.pacman['r'] + self.location[0][i]
            c_m = self.pacman['c'] + self.location[1][i]
            if r_m >= 1 and r_m <= self.r and c_m >= 1 and c_m <= self.c:
                if self.map[r_m - 1][c_m - 1] == '#':
                    self.info[i] = 1
            else:
                self.info[i] = 1

        nex = 4

        self.init_BFS()
        self.BFS(self.pacman, [])
        for i in range(self.sl_ghost):
            d = self.dem[self.ghost[i]['r']-1][self.ghost[i]['c']-1]
            if d is not None:
                r, c = self.truyvet(self.ghost[i]['r'], self.ghost[i]['c'], self.pacman['r'], self.pacman['c'])
                if r is not None and c is not None:
                    dr = self.derection(r, c, self.pacman['r'], self.pacman['c'])
                    if d <= self.Guess:
                        if dr is not None:
                            self.info[nex+dr] = 1

        nex = 9
        dmin = self.r*self.c
        rm = []
        cm = []
        for i in range(self.r):
            for j in range(self.c):
                if self.dem[i][j] is not None:
                    if self.map[i][j] == '.':
                        if self.dem[i][j] < dmin:
                            dmin = self.dem[i][j]
                            rm.clear()
                            cm.clear()
                            rm.append(i+1)
                            cm.append(j+1)
                        elif self.dem[i][j] == dmin:
                            rm.append(i + 1)
                            cm.append(j + 1)

        for i in range(len(rm)):
            r, c = self.truyvet(rm[i], cm[i], self.pacman['r'], self.pacman['c'])
            if r is not None and r is not None:
                dr = self.derection(r, c, self.pacman['r'], self.pacman['c'])
                self.info[nex + dr] = 1

        nex = 13
        if dmin != self.r*self.c:
            self.info[nex] = dmin

        nex = 8
        self.init_BFS()
        self.BFS(self.pacman, self.ghost)
        f = True
        for i in range(self.r):
            for j in range(self.c):
                if self.dem[i][j] is not None and self.dem[i][j] > self.Guess:
                    f = False
                    break
            if f == False:
                break

        if f:
            self.info[nex] = 1


        stt = self.info[13]*pow(2, 13)
        state = 0
        for i in range(13):
            state += self.info[12-i]*pow(2, i)

        return stt+state


    def derection(self, r, c, r_, c_):
        dr = r - r_
        dc = c - c_
        for i in range(4):
            if self.location[0][i] == dr and self.location[1][i] == dc:
                return i
        return None

    def truyvet(self, r, c, r_d, c_d):
        while self.mark[r-1][c-1] is not None:
            tmp = cp.deepcopy(self.mark[r-1][c-1])
            if tmp['r'] == r_d and tmp['c'] == c_d:
                return r, c
            else:
                r = tmp['r']
                c = tmp['c']
        return None, None

    def init_BFS(self):
        self.dem = np.empty([self.r, self.c], dtype=object)
        self.check = np.zeros([self.r, self.c], dtype=int)
        self.mark = np.empty([self.r, self.c], dtype=object)

    def BFS(self, st, list):
        u = cp.deepcopy(st)
        illegal = np.array(list)
        q = queue.Queue()
        q.put(u)
        self.dem[u['r']-1][u['c']-1] = 0
        self.check[u['r']-1][u['c']-1] = 1
        self.mark[u['r']-1][u['c']-1] = None
        while q.empty()==False:
            f = q.get()
            # self.spin_location()
            for i in range(4):
                r_m, c_m = self.next_location(i, f['r'], f['c'], self.location)
                if self.can_move(r_m, c_m, illegal) and self.check[r_m-1][c_m-1] == 0:
                    tmp = dict()
                    tmp['r'] = r_m
                    tmp['c'] = c_m
                    q.put(tmp)
                    self.dem[r_m-1][c_m-1] = self.dem[f['r']-1][f['c']-1] + 1
                    self.check[r_m-1][c_m-1] = 1
                    self.mark[r_m-1][c_m-1] = cp.deepcopy(f)

    def inside(self, r, c):
        if r>=1 and r<=self.r and c>=1 and c<=self.c:
            return True
        return False

    def can_move(self, r, c, illg):
        if self.inside(r, c):
            if self.map[r-1][c-1] != '#':
                for i in range(len(illg)):
                    if r == illg[i]['r'] and c == illg[i]['c']:
                        return False
                return True
        return False

    def spin_location(self):
        self.new_loca = np.zeros([2, 4], dtype=int)
        tmp_location = np.array(self.location, dtype=int)
        for i in range(4):
            r = random.randint(0, 3-i)
            r_ = tmp_location[0][r]
            c_ = tmp_location[1][r]
            tmp_location = np.delete(tmp_location, r, 1)
            self.new_loca[0][i] = r_
            self.new_loca[1][i] = c_


    def next_location(self, n, r, c, location):
        nt = np.dot(location, self.list_action[n])
        return r+nt[0], c+nt[1]

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


if __name__ == "__main__":
    evn = Evironment_PaMaCup(name_map="")
    state = evn.reset()
    evn.view()
    print(evn.info)
    print(state)
