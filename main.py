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
from catchertraining import TrainNetwork4Catcher
from flappybirdtraining import TrainNetwork4FlappyBird
from pongtraining import TrainNetwork4Pong

##---------------------------------------------------------------------------
# Training of DQCN for the Catcher Pygame 
##---------------------------------------------------------------------------
GAME  = "CATCHER"
#GAME = 'FLAPPYBIRD'
#GAME = 'PONG'
if(GAME == "CATCHER"):
    ACTIONS = 3 # number of valid actions for Catcher Game
elif(GAME == "FLAPPYBIRD"):
    ACTIONS = 2 # number of valid actions Flappy Bird
elif(GAME == "PONG"):
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
