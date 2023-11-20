import numpy as np
import gym
from gym import spaces
from lunar_heist_env import LunarHeistEnv 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# TODO: Import any additional libraries you need for the DQN agent
#Feel free to change these methods/remove or add them
class QLearningAgent:
    def __init__(self, action_space, learning_rate, gamma, epsilon, discretize_bins):
        # Initialize attributes
        # ...
        self.a = learning_rate
        self.e = epsilon 
        self.count = 0
        self.g = gamma
        self.n_actions = int(action_space.n)
        self.action_space = np.arange(self.n_actions)
        self.n_obs = 4
        self.disc_bins = discretize_bins
        s = len(discretize_bins)
        # TODO: Initialize the Q-table here
        self.q_table = np.ones((s,s,s,s,self.n_actions))  # Placeholder for student code

    def discretize_state(self, state):
        # TODO: Discretize the state
        # we have 4 observations:         
        #The horizontal coordinate (x) [-1,1]
        #The vertical coordinate (y) [0,1]
        #The horizontal velocity (vx) [-1,1]
        #The vertical velocity (vy) [-1,1]
        #let's say I want each of them discretazing in 100 possible values
        state = np.array(state)
        #print(state)
        state = (1 + np.round(state,1))*10 -1
        
        return state.astype(int)



    def choose_action(self, state):
        # TODO: Implement the action selection method
        
        state = self.discretize_state(state)
        p = self.e/(self.n_actions-1)*np.ones(self.n_actions) #vector to store the probabilities for the actions, on that particular state
        #search for the greedy action:
        greedy = np.argmax(self.q_table[state[0],state[1],state[2],state[3],:])
        p[greedy] = 1 - self.e
        action = np.random.choice(self.action_space, p=p)
        return action


    def update(self, state, action, reward, next_state, done):
        # TODO: Implement the Q-table update method
        #discretize the state
        state = self.discretize_state(state)
        next_state = self.discretize_state(next_state)
        # chose action with greatest value for that state
        if done != True:
            max_action = np.max(self.q_table[next_state[0],next_state[1],next_state[2],next_state[3],:])
            self.q_table[state[0],state[1],state[2],state[3],int(action)] = (
            self.q_table[state[0],state[1],state[2],state[3],int(action)] + 
            self.a*(reward + self.g*max_action - self.q_table[state[0],state[1],state[2],state[3],int(action)]))
            self.count +=1
        else:
            self.q_table[state[0],state[1],state[2],state[3],int(action)] = 0
    # Optional: implement methods to update epsilon and alpha



# class DQNAgent:
#     def __init__(self, state_size, action_size, learning_rate, gamma):
#         # Initialize attributes
#         # ...

#         # TODO: Create the model
#         self.model = self._build_model()  # Placeholder for student code

#     def _build_model(self):
#         # TODO: Build the Neural Network model here
#         # ...

#     def choose_action(self, state):
#         # TODO: Implement the action selection method
#         # ...

#     def remember(self, state, action, reward, next_state, done):
#         # TODO: Implement the method to store experience in memory
#         # ...

#     def replay(self, batch_size):
#         # TODO: Implement method to train the network with replay buffer
#         # ...

#     def load(self, name):
#         # TODO: Implement the method to load the model
#         # ...

#     def save(self, name):
#         # TODO: Implement the method to save the model
#         # ...


# # TODO: You may create additional classes/functions as needed.
