import environement
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import time
import config

Init = config.INIT_Q_Sarsa

def check(ave, q):
    evn2 = environement.Evironment_PaMaCup()
    evn2.reset()
    sl_dot2 = evn2.sldot
    max_step2 = (evn2.r * evn2.c)
    qtable2 = q

    total_dot = []
    total_step = []
    total_reward = []

    for episode2 in range(ave):
        state2 = evn2.reset()
        # evn2.view()
        lstep2 = 0
        ldot2 = 0
        rewards2 = 0
        done2 = False
        for step2 in range(max_step2):
            # time.sleep(0.5)
            action2 = np.argmax(qtable2[state2, :])
            new_state2, reward2, done2, info2 = evn2.step(action2)
            # evn2.view()
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

evn = environement.Evironment_PaMaCup()
action_size = len(evn.list_action)
state_action = evn.numState
max_sldot = evn.sldot

qtable = np.zeros([state_action, action_size])

if Init == False:
    Best_Q = np.load('Best-S.npy')
    max_ave = Best_Q[0][0]
    best_q = Best_Q[1]
    print(max_ave)
    print(best_q)
else:
    max_ave = 0

sl_dot = evn.sldot
total_episodes = config.total_episodes
learning_rate = config.learning_rate
max_step = (evn.r*evn.c)
gamme = config.gamme
episodes_check = config.episodes_check

epsilon = config.epsilon
max_epsilon = config.max_epsilon
min_epsilon = config.min_epsilon
decay_rate = config.decay_rate

tranning = True

if tranning:
    if Init == False:
        info = np.load('Q-Table-sarsa.npy')
        qtable = info[1]
        episodest = info[0][0]
        epsilon = info[0][1]
        print(qtable)
    else:
        episodest = 0
        np.save('Q-Table-sarsa', np.array([[0, epsilon], qtable]))

    episodes = []
    rewards = []
    dots = []
    steps = []

    for i in range(total_episodes):
        episode = episodest+i
        state = evn.reset()
        step = 0
        done = False
        total_rewards = 0
        sldot = 0
        for step in range(max_step):
            exp = rd.uniform(0, 1)
            if exp > epsilon:
                action = np.argmax(qtable[state, :])
            else:
                same = np.where(qtable[state, :] == np.max(qtable[state, :]))
                action = rd.choice(same[0])

            new_state, reward, done, info = evn.step(action)
            sldot = info
            qtable[state, action] = qtable[state, action]+learning_rate*(reward+gamme*np.max(qtable[new_state, :]) - qtable[state, action])
            total_rewards += reward
            state = new_state
            if done:
                break
        print('Episode {}: rate: {} - dot: {} - rewards: {}'.format(episode, epsilon, max_sldot-sldot, total_rewards))
        if (episode+1) % episodes_check == 0 or episode == 0:
            np.save('Q-Table-sarsa', np.array([[episode+1, epsilon], qtable]))
            print('Saved Qtable-sarsa')


            tdot, tstep, treward = check(3, qtable)
            print('Ave D= ', tdot, ' S=', tstep, ' ')
            if tdot > max_ave:
                max_ave = tdot
                np.save('Best-S', np.array([[max_ave], qtable]))

            if Init == False:
                episodes = np.load('E-S.npy')
                episodes = np.hstack((episodes, np.array([episode])))
                dots = np.load('D-S.npy')
                dots = np.hstack((dots, np.array([tdot])))
                steps = np.load('S-S.npy')
                steps = np.hstack((steps, np.array([tstep])))
                rewards = np.load('R-S.npy')
                rewards = np.hstack((rewards, np.array([treward])))

                np.save('E-S', np.array(episodes))
                np.save('D-S', np.array(dots))
                np.save('S-S', np.array(steps))
                np.save('R-S', np.array(rewards))
            else:
                np.save('E-S', np.array([]))
                np.save('D-S', np.array([]))
                np.save('S-S', np.array([]))
                np.save('R-S', np.array([]))
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

    print(qtable)

else:
    print(check(1))
    # plt.figure()
    # episodes = np.load('E-Q.npy')
    # rewards = np.load('R-Q.npy')
    # plt.plot(episodes, rewards)
    # plt.title('Rewards-Q')


    plt.figure()
    episodes = np.load('E-Q.npy')
    dots = np.load('D-Q.npy')
    plt.plot(episodes, dots)
    plt.title('Dots-Q')


    plt.figure()
    episodes = np.load('E-Q.npy')
    steps = np.load('S-Q.npy')
    plt.plot(episodes, steps)
    plt.title('Steps-Q')
    plt.show()


