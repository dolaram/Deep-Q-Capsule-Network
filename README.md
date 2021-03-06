# Deep-Q-Capsule-Network: Deep-Reinforcement-Learning-using-Capsule-Network
# Capsule Network
A Capsule Neural Network (CapsNet) is a type of artificial neural network (ANN) that can be used to better model hierarchical relationships. The approach is an attempt to more closely mimic biological neural organization.
The idea is to add structures called capsules to a convolutional neural network (CNN) and to reuse output from several of those capsules to form more stable (with respect to various perturbations) representations for higher order capsules. The output is a vector consisting of the probability of observation, and a pose for that observation. This vector is similar to what is done for example when classifying with localization in CNNs.
Among other benefits, Capsule Network addresses the "Picasso problem" in image recognition: images that have all the right parts but that are not in the correct spatial relationship (e.g., in a "face", the positions of the mouth and one eye are switched). For image recognition, Capsule Network exploits the fact that while viewpoint changes have nonlinear effects at the pixel level, they have linear effects at the part/object level. This can be compared to inverting the rendering of an object of multiple parts.

1. https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b
2. https://towardsdatascience.com/a-simple-and-intuitive-explanation-of-hintons-capsule-networks-b59792ad46b1
3. https://www.youtube.com/watch?v=rTawFwUvnLE  --Geoffrey Hinton talk on Capsule Neural Network

# Dynamic Routing
The fact that the output of a capsule is a vector makes it possible to use a powerful dynamic routing
mechanism to ensure that the output of the capsule gets sent to an appropriate parent in the layer
above.
1. [Sara Sabour, Nicholas Frosst, Geoffrey E Hinton. Dynamic Routing Between Capsules. NIPS 2017](https://arxiv.org/abs/1710.09829)   

# Deep Q-Learning
Deep Q Learning is a branch of Reinforcement Learning that deals with problems using CNN as a function approximator, we have implemented Deep Q-Learning for Pong game-based application where we have trained the network to play against an opponent. We have used the Capsule Network as the Deep Q Network instead of CNN for the Deep Q learning. We have found that Capsule Network works very well for reinforcement learning. 
Various experiments were previously done on the Capsule Network for RL can be found in https://github.com/dolaram/Deep-Reinforcement-Learning-using-Capsule-Network 

## Installation

**Clone this repository to local.**
```
git clone https://github.com/dolaram/Deep-Q-Capsule-Network.git Deep-Q-Capsule-Network
cd Deep-Q-Capsule-Network
```

**Install  tensorflow-gpu and CUDA Toolkit**
```
pip install tensorflow-gpu==1.2 # GPU
```
You can follow the steps given in https://www.tensorflow.org/install/gpu to install the CUDA Toolkit for your GPU. 
You can use the Anaconda to install appropriate version of tensorflow-gpu for your gpu.

**Python Game environment**
```
pip install pygame  # https://pypi.org/project/pygame/
# Installing PLE
#  https://pygame-learning-environment.readthedocs.io/en/latest/user/home.html#installation
git clone https://github.com/ntasfi/PyGame-Learning-Environment  
cd PyGame-Learning-Environment/
sudo pip install -e .
```

**opencv-python**
```
pip install opencv-python
# https://pypi.org/project/opencv-python/
```


**Training Deep-Q Capsule Network(DQCN) for Deep Q learning**  

Training with default settings for the Small Capsule Network Deep Q learning:
```
python main.py
# Select one of the 3 games for training by editing main.py. You need to give at a sufficiently long time for training the Capsule Network to get good results.
```
**Training Deep-Q Network(DQCN) for Deep Q learning**  

Training for the Big CNN and Capsule Network for Deep Q learning:
```
python main4bigcapsnet.py
# Select CNN or Capsule Network as the Deep Q Network and Select one of the 3 games for training by editing main4bigcapsnet.py. You need to give at a sufficiently long long time for training the Capsule Network to get good results. 
```
