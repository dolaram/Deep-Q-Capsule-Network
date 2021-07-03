##---------------------------------------------------------------------------
# Importinting the packages
##---------------------------------------------------------------------------
import cv2
import sys
import time 
import random
import numpy as np
from collections import deque
from matplotlib import pyplot as plt
sys.path.append("training4bigcapsnet/")
##---------------------------------------------------------------------------
# Import PyGame
##---------------------------------------------------------------------------
import pygame
from ple import PLE
##---------------------------------------------------------------------------
# Selecting the network and Import the DEEP_Q NETWORK modules
##---------------------------------------------------------------------------
import tensorflow as tf
network = 'CAPSULENETWORK'
#network = 'CNN'
if(network == 'CAPSULENETWORK'):
    from capsulenetwork4bignet import *
elif(network == 'CNN'):
    from CNN4bignet import *
else:
    print("Select one of the network as a deep Q Network")


##---------------------------------------------------------------------------
# Importting the trading module for each game 
#//one game of pygame is imported at a time to avoid screen splitting and frame overlapping 
##---------------------------------------------------------------------------
# Select the Game for which you want to train Deep-Q Capsule Network
GAME  = "CATCHER"
#GAME = 'FLAPPYBIRD'
#GAME = 'PONG'
if(GAME == "CATCHER"):
    from catchertraining4bignet import TrainNetwork4Catcher
    ACTIONS = 3 # number of valid actions for Catcher Game
elif(GAME == "FLAPPYBIRD"):
    from flappybirdtraining4bignet import TrainNetwork4FlappyBird
    ACTIONS = 2 # number of valid actions Flappy Bird
elif(GAME == "PONG"):
    from pongtraining4bignet import TrainNetwork4Pong
    ACTIONS = 6 # number of valid actions for Pong Game
else: 
    print("Select the appropriate Pygame for the capsule Network Training")

##---------------------------------------------------------------------------
# Create the Deep Q Capsule Network
##---------------------------------------------------------------------------
tf.reset_default_graph()
sess = tf.InteractiveSession()
s, coeff, readout = createNetwork(ACTIONS)

##---------------------------------------------------------------------------
# Training of the Deep Q capsule Network for the Pygames 
##---------------------------------------------------------------------------
if(GAME == "CATCHER"):
    TrainNetwork4Catcher(s, coeff, readout, sess)
elif(GAME == "FLAPPYBIRD"):
    TrainNetwork4FlappyBird(s, coeff, readout, sess)
elif(GAME == "PONG"):
    TrainNetwork4Pong(s, coeff, readout, sess)
else: 
    print("Select the appropriate Pygame for the capsule Network Training")
