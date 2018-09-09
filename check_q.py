import environement
import numpy as np
import time

def check(ave, st):
    if st == 'Q':
        name = 'Q-Table.npy'
    elif st == 'S':
        name = 'Q-Table-sarsa.npy'
    evn2 = environement.Evironment_PaMaCup()
    evn2.reset()
    sl_dot2 = evn2.sldot
    max_step2 = (evn2.r * evn2.c)
    qtable2 = np.load(name)[1]

    total_dot = []
    total_step = []
    total_reward = []

    for episode2 in range(ave):
        state2 = evn2.reset()
        evn2.view()
        lstep2 = 0
        ldot2 = 0
        rewards2 = 0
        done2 = False
        for step2 in range(max_step2):
            time.sleep(0.5)
            action2 = np.argmax(qtable2[state2, :])
            new_state2, reward2, done2, info2 = evn2.step(action2)
            evn2.view()
            rewards2 += reward2
            lstep2 = step2
            ldot2 = info2
            if done2:
                break
            state2 = new_state2

        total_dot.append(sl_dot2 - ldot2)
        total_step.append(lstep2)
        total_reward.append(rewards2)
    print('Max dot: ', max(total_dot), ' Min dot: ', min(total_dot))
    return sum(total_dot)/ave, sum(total_step)/ave, sum(total_reward)/ave

check(1, 'S')