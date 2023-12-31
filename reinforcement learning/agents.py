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
from icecream import ic
import pandas as pd
# TODO: Import any additional libraries you need for the DQN agent
#Feel free to change these methods/remove or add them
class QLearningAgent:
    def __init__(self, action_space, learning_rate, gamma, epsilon,min_epsilon,decay_rate, discretize_bins):
        # Initialize attributes
        # ...
        self.a = learning_rate
        self.e = epsilon
        self.min_e = min_epsilon
        self.decay = decay_rate 
        self.g = gamma
        self.n_actions = int(action_space.n)
        self.action_space = np.arange(self.n_actions)
        self.n_obs = 4
        self.disc_bins = discretize_bins
        
        sx = len(discretize_bins[0])
        sy = len(discretize_bins[1])
        svx = len(discretize_bins[2])
        svy = len(discretize_bins[3])
        
        # TODO: Initialize the Q-table here
        self.q_table = 0.1*np.ones((sx,sy,svx,svy,self.n_actions))  # Placeholder for student code
        self.step = 0
        #ic(sx)
    def discretize_state(self, state):
        # TODO: Discretize the state
        # we have 4 observations:         
        #The horizontal coordinate (x) [-1,1]
        #The vertical coordinate (y) [0,1]
        #The horizontal velocity (vx) [-1,1]
        #The vertical velocity (vy) [-1,1]
        state = np.array(state)

        #arranging the states in the bins
        df = pd.DataFrame({"x":[state[0]], "y":[state[1]],"vx":[state[2]],"vy":[state[3]]})
        df['x_bin']=pd.cut(x = df['x'],
                        bins = self.disc_bins[0], 
                        labels = np.arange(0,len(self.disc_bins[0])-1,1).tolist())
        df['y_bin']=pd.cut(x = df['y'],
                        bins = self.disc_bins[1], 
                        labels = np.arange(0,len(self.disc_bins[1])-1,1).tolist())
        df['vx_bin']=pd.cut(x = df['vx'],
                        bins = self.disc_bins[2], 
                        labels = np.arange(0,len(self.disc_bins[2])-1,1).tolist())
        df['vy_bin']=pd.cut(x = df['vy'],
                        bins = self.disc_bins[3], 
                        labels = np.arange(0,len(self.disc_bins[3])-1,1).tolist())
        
        #returning a vector with the discretized states
        state[0] = int(df['x_bin'].loc[0])
        state[1] = int(df['y_bin'].loc[0])
        state[2] = int(df['vx_bin'].loc[0])
        state[3] = int(df['y_bin'].loc[0])
        return state.astype(int)

    def add_step(self):
        self.step += 1

    def choose_action(self, state):
        # TODO: Implement the action selection method
        
        state = self.discretize_state(state)
        if self.step%10==0 and self.e > self.min_e: #decay epsilon each 10 time steps
            self.e = self.e*self.decay

        sample = random.random()
        if sample <=self.e:
            action = np.random.choice(self.action_space)
        else:
        #search for the greedy action:
            action = np.argmax(self.q_table[int(state[0]),int(state[1]),int(state[2]),int(state[3]),:])

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
        else:
            self.q_table[state[0],state[1],state[2],state[3],int(action)] = 0
    # Optional: implement methods to update epsilon and alpha



class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size, learning_rate, gamma, min_epsilon,epsilon,decay_rate):
        # Initialize attributes
        self.gamma = gamma
        self.alpha = learning_rate
        self.action_size = action_size
        self.action_space = np.arange(4)
        self.state_size = state_size
        self.e = epsilon
        self.e_end = min_epsilon
        self.e_decay = decay_rate
        self.target_up_freq = 1000
        buffer_size = 800000
        self.buffer_replay = deque(maxlen=buffer_size)
        self.buffer_reward = deque([0,0], maxlen=1000)

        super().__init__()
        self.online_nn = NNetwork(self.state_size, self.action_size) #behavior policy
        self.target_nn = NNetwork(self.state_size, self.action_size) #target policy

       
        self.target_nn.load_state_dict(self.online_nn.state_dict())

        self.grad = optim.AdamW(self.online_nn.parameters(), lr=self.alpha, amsgrad=True) #gradient method

        self.step = 1 #step counter to update target network
        # TODO: Create the model
        #self.model = self._build_model()  # Placeholder for student code

    def _build_model(self):
        super(NNetwork, self).__init__()
        self.online_nn = NNetwork(self.state_size, self.action_size) #behavior policy
        self.target_nn = NNetwork(self.state_size, self.action_size) #target policy

    def add_step(self):
        self.step += 1

    def choose_action(self, state):
        # TODO: Implement the action selection method

        if self.step%10==0 and self.e > self.e_end: #decay epsilon at each 10 steps
            self.e = self.e*self.e_decay
        sample = random.random()
        if sample <=self.e_end:
            action = np.random.choice(self.action_space)
        else:
            action = self.online_nn.act(state)

        return action
    
    def remember(self, state, action, reward, next_state, done):
        # TODO: Implement the method to store experience in memory
        experience = (state, action, reward, next_state, done)
        self.buffer_replay.append(experience)


    def replay(self, batch_size):
        # TODO: Implement method to train the network with replay buffer
        if batch_size > len(self.buffer_replay):
            return
        experience = random.sample(self.buffer_replay, batch_size) #sampling from experience buffer

        #converting actions to tensor
        state = np.asarray([exp[0] for exp in experience])
        action = np.asarray([exp[1] for exp in experience])
        reward = np.asarray([exp[2] for exp in experience])
        next_state = np.asarray([exp[3] for exp in experience])
        done = np.asarray([exp[4] for exp in experience])

        state = torch.as_tensor(state,dtype = torch.float32)
        action = torch.as_tensor(action,dtype = torch.int64).unsqueeze(-1)
        reward = torch.as_tensor(reward,dtype = torch.float32).unsqueeze(-1)
        next_state = torch.as_tensor(next_state, dtype = torch.float32)
        done = torch.as_tensor(done, dtype = torch.float32).unsqueeze(-1)

        target_qs = self.target_nn(next_state) #computes the target values for all the possible actions that lead to next_state
        q_greedy = target_qs.max(dim=1, keepdim=True)[0] # gets the action value for the greedy action
        q_target = reward + self.gamma*(1-done)*q_greedy #updates the target using the greedy action value

        #optmizing the online network

        q = self.online_nn(state) #uses the online nn to estimate the taken action value
        q_action = torch.gather(input = q, dim=1,index = action)
        loss = nn.functional.smooth_l1_loss(q_action, q_target)

        # Gradient
        self.grad.zero_grad()
        loss.backward()#loss computation
        self.grad.step()

        #updating the target network each target_up_freq steps
        if self.step%self.target_up_freq == 0:

            self.target_nn.load_state_dict(self.online_nn.state_dict())


        



    def load(self, name):
        self.online_nn.load_state_dict(torch.load(name))
        self.online_nn.eval()
        ic('model loaded')
    def save(self, name):
        # TODO: Implement the method to save the model
        ic('saving the model')
        torch.save(self.target_nn.state_dict(), name)
        ic('model saved')

# # TODO: You may create additional classes/functions as needed.

class NNetwork(nn.Module):
    #neural network class
    def __init__(self, state_size, actions_n):
        super(NNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size,128),
            nn.Tanh(),
            nn.Linear(128,actions_n)
        )

    def forward(self,x):

        return self.network(x)    
    
    def act(self, state):
        state_t = torch.as_tensor(state, dtype = torch.float32)
        #with torch.no_grad():
        q = self(torch.unsqueeze(state_t,0))
        q_greedy = torch.argmax(q, dim=1)[0]
        action = q_greedy.detach().item()
        return action