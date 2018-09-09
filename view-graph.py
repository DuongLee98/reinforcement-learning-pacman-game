import environement
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import time
import cv2
from threading import Thread

done = False
done2 = False

def view(name):
    global done
    global done2
    name1 = 'Q'
    name2 = 'S'

    show().start()
    lenbd1 = 0
    lenbd2 = 0
    while True:
        if done2 == True:
            break
        episodes1 = np.load('E-'+name1+'.npy')
        lenend1 = len(episodes1)
        dots1 = np.load(name+'-'+name1+'.npy')


        episodes2 = np.load('E-' + name2 + '.npy')
        lenend2 = len(episodes2)
        dots2 = np.load(name+'-' + name2 + '.npy')

        if lenend1 > lenbd1 or lenend2 > lenbd2:
            saveimg(episodes1, dots1, episodes2, dots2)
            lenbd1 = lenend1
            lenbd2 = lenend2
            done = True
        time.sleep(10)
        done = False

def saveimg(episodes1, dots1, episodes2, dots2):
    with plt.style.context('Solarize_Light2'):
        # fg = plt.figure(num=None, figsize=(23.86, 7), dpi=100, facecolor='w', edgecolor='k')
        fg = plt.figure(num=None, figsize=(10, 7), dpi=100, facecolor='w', edgecolor='k')
        plt.plot(episodes1, dots1, '.-', color='red')
        plt.plot(episodes2, dots2, '.-', color='blue')
        plt.legend(('QL', 'Sarsa'))
        plt.ylabel('Average Dot', fontsize=12, color='k')
        plt.xlabel('Episode', fontsize=12, color='k')
        plt.title('Q Learning vs Sarsa')
        fg.savefig('Graph.png', bbox_inches='tight')
        plt.close(fg)

class show(Thread):
    def run(self):
        global done
        global done2
        img_tmp = cv2.imread('Graph.png')
        while True:
            if done == True:
                img = cv2.imread('Graph.png')
                img_tmp = img
            # print(arr.shape)
            else:
                img = img_tmp
            cv2.imshow('View', img)
            if cv2.waitKey(1) == ord('q'):
                done2 = True
                break

view('D')
# check(1)