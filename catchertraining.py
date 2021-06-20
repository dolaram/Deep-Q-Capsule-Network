##---------------------------------------------------------------------------
# Importinting the packages
##---------------------------------------------------------------------------
import tensorflow as tf
import cv2
import sys
import time 
import random
import numpy as np
from collections import deque
from matplotlib import pyplot as plt
import pygame
from ple import PLE
from capsulenetwork import *
##---------------------------------------------------------------------------
# Setting the Game environment 
##---------------------------------------------------------------------------
from ple.games.catcher import Catcher
#game = Catcher()
game = Catcher(width=256, height=256)
game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)


##---------------------------------------------------------------------------
# Setting the RL environment Variable
##---------------------------------------------------------------------------
GAME  = "CATCHER"
ACTIONS = 3 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 500. # timesteps to observe before training
EXPLORE = 500. # frames over which to anneal epsilon
FINAL_EPSILON =  0.05 # final value of epsilon
INITIAL_EPSILON = 1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

##---------------------------------------------------------------------------
# Parameters for Capsule Network
##---------------------------------------------------------------------------
epsilon = 1e-9
iter_routing = 2
train_freq = 10
tf.reset_default_graph()

##---------------------------------------------------------------------------
#Function for training of CapsNet Agent for Flappy bird
##---------------------------------------------------------------------------

def TrainNetwork4Catcher(s, coeff, readout, sess):
    tick = time.time()
    # define the cost function
    a = tf.placeholder("float", shape=(None, ACTIONS))
    y = tf.placeholder("float", shape= (None))
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

    ## open up a game state to communicate with emulator
    # game_state = game.GameState()
    p =  PLE(game,display_screen=True)
    actions = p.getActionSet()
    p.init()
    reward = p.act(p.NOOP)
    
    # store the previous observations in replay memory
    D = deque()
    
    # get the first state by doing nothing and preprocess the image to 80x80x4
    
    action_index =  np.random.randint(0, 3)
    reward = p.act(actions[action_index])
    if(reward != 0):
        if(reward < -1):
            r_0 = -1
        else:
            r_0 = reward
        terminal = 1
        if(np.max(readout_t) > qmax):
            qmax = np.max(readout_t) 
        print("time_step", t,"SCORE ", score," rt ", r_t," Q_MAX %e" % qmax)
        qmax = -10000
    else:
        terminal = 0
        r_0 = reward
            
    x_t = p.getScreenRGB()
    x_t = cv2.cvtColor(cv2.resize(x_t, (28, 28)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    sess.run(tf.global_variables_initializer())
    # saving and loading networks
    saver = tf.train.Saver()
    saved_networks = GAME + "_saved_networks"
    checkpoint = tf.train.get_checkpoint_state(saved_networks)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    b_IJ1 = np.zeros((1, 32, 10, 1, 1)).astype(np.float16) # batch_size=1
    b_IJ2 = np.zeros((BATCH, 32, 10, 1, 1)).astype(np.float16) # batch_size=BATCH
    epsilon = INITIAL_EPSILON
    t = 0
    score = 0
    qmax = -100000
    tick = time.time()
    while True:
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict = {s:s_t.reshape((1,28,28,4)), coeff:b_IJ1})
        
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if random.random() <= epsilon or t <= OBSERVE:
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        reward = p.act(actions[action_index])
        if(reward != 0):
            if(reward < -1):
                r_t = -1
            else:
                r_t = reward
            terminal = 1
            if(np.max(readout_t) > qmax):
                qmax = np.max(readout_t) 
            print("time_step", t," SCORE ", score," rt ", r_t," Q_MAX %e" % qmax)
            qmax = -10000
        else:
            terminal = 0
            r_t = reward
        # run the selected action and observe next state and reward
        x_t1_colored = p.getScreenRGB()
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (28, 28)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (28, 28, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        
        # only train if done observing
        if t > OBSERVE and t%train_freq==0:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s:s_j1_batch, coeff:b_IJ2 })
            #readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                # if terminal only equals reward
                if minibatch[i][4]:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch,
                coeff: b_IJ2})
            loss = cost.eval(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch,
                coeff: b_IJ2})

        # update the old values
        s_t = s_t1
        t += 1
        
        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, saved_networks + '/' + GAME + '-dqn', global_step = t)

        score += reward
        if p.game_over():
            p.reset_game()
            #print("time_step:",t,"score:",score)
            score = 0

