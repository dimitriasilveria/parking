import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
#import pygame
import pandas as pd

# The LunarHeist class will inherit from gym.Env and will need to implement the following methods:
# __init__, step, reset, render, close, and seed.
class LunarHeistEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(LunarHeistEnv, self).__init__()

        # Constants for the environment
        self.GRAVITY = -0.003 #-0.0025
        self.THRUST_MAIN = 0.004 #0.002
        self.THRUST_SIDE = 0.002 #0.0005
        self.MAX_Y = 1.0
        self.MIN_Y = 0.0
        self.LANDING_THRESHOLD = 0.2
        self.INITIAL_Y = 1.0
        self.LANDING_PAD_X_RANGE = (0.30, 0.60)  # Define a range on the x-axis for the landing pad

        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # Actions: Do nothing, fire left engine, main engine, right engine
        self.observation_space = spaces.Box(low=np.array([-1.0, 0.0, -1.0, -1.0], dtype=np.float32), high=np.array([1.0, self.MAX_Y, 1.0, 1.0], dtype=np.float32), 
    dtype=np.float32)  #This sets up an observation space that only includes four variables:

        #The horizontal coordinate (x)
        #The vertical coordinate (y)
        #The horizontal velocity (vx)
        #The vertical velocity (vy)

        # Initialize state variables
        self.state = None
        self.viewer = None
        self.seed()
        self.reset()

    def generate_minerals(self):
        # Place minerals at random locations
        num_minerals = 8
        return [(self.np_random.uniform(-1, 1), self.np_random.uniform(self.MIN_Y, self.MAX_Y)) 
                for _ in range(num_minerals)]

    def generate_mines(self):
        # Place mines at random locations
        num_mines = 2
        return [(self.np_random.uniform(-1, 1), self.np_random.uniform(self.MIN_Y, self.MAX_Y)) 
                for _ in range(num_mines)]

    def reset(self):
        self.state = np.array([0, self.INITIAL_Y, 0, 0], dtype=np.float32)  # [x, y, vx, vy]
        self.minerals_collected = 0
        self.mines_hit = 0
        self.minerals_locations = self.generate_minerals()
        self.mines_locations = self.generate_mines()
        return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        # Initialize reward at the start of the step
        reward = 0
        done = False

        # Unpack the state
        x, y, vx, vy = self.state
        
        # Calculate the center of the landing pad
        landing_pad_x = (self.LANDING_PAD_X_RANGE[0] + self.LANDING_PAD_X_RANGE[1]) / 2
    
        # Apply gravity
        vy += self.GRAVITY

        # Update velocities based on action
        if action == 1:  # Fire left engine
            vx += self.THRUST_SIDE
        elif action == 2:  # Fire main engine
            vy += self.THRUST_MAIN
        elif action == 3:  # Fire right engine
            vx -= self.THRUST_SIDE

        # Update position based on velocity
        x += vx
        if abs(x) > 1:
            x = np.sign(x)*1
        #print(x)
        y = max(self.MIN_Y, y + vy)  # y should never be below the ground
        if y > 1:
            y = 1
        
        # Check for mineral collection
        mineral_collected = False
        for mineral in self.minerals_locations:
            if np.linalg.norm([x - mineral[0], y - mineral[1]]) < self.LANDING_THRESHOLD:
                reward += 5  # Collect mineral reward
                self.minerals_collected += 1
                self.minerals_locations.remove(mineral)  # Remove the collected mineral
                mineral_collected = True
                break  # Assuming only one collection per step

        # Check for mine collision
        mine_hit = False
        for mine in self.mines_locations:
            if np.linalg.norm([x - mine[0], y - mine[1]]) < self.LANDING_THRESHOLD:
                reward -= 10  # Mine hit penalty
                self.mines_hit += 1
                mine_hit = True
                break  # Assuming only one collision per step

        # Check for landing
        #if y <= self.MIN_Y and self.LANDING_PAD_X_RANGE[0] <= x <= self.LANDING_PAD_X_RANGE[1]:
        #    done = True
        #    if np.abs(vx) < self.LANDING_THRESHOLD and np.abs(vy) < self.LANDING_THRESHOLD:
        #        reward += 10  # Successful landing reward
        #    else:
        #        reward -= 2  # Crash penalty
        # Check for landing
        if y <= self.MIN_Y:
            done = True
            landing_accuracy = max(0, self.LANDING_THRESHOLD - np.abs(x - landing_pad_x))
            velocity_accuracy = max(0, self.LANDING_THRESHOLD - np.sqrt(vx**2 + vy**2))
            reward += landing_accuracy * 5  # Scale the reward for accuracy
            reward += velocity_accuracy * 5  # Scale the reward for gentle landing

            if np.abs(vx) < self.LANDING_THRESHOLD and np.abs(vy) < self.LANDING_THRESHOLD:
                reward += 50  # Bonus for perfect landing
            else:
                reward -= 10#2  # Penalty for crash
        else:
            # Penalize or reward the agent based on its distance to the landing pad
            # and its current velocity, scaled by some factor to control the influence on the reward
            reward -= np.abs(x - landing_pad_x) * 0.1  # Penalize being far from the pad
            reward -= np.sqrt(vx**2 + vy**2) * 0.1  # Penalize high velocity when close to landing

        
        # Update the state
        self.state = [x, y, vx, vy]

        info = {
            'mineral_collected': mineral_collected,
            'mine_hit': mine_hit,
        }
        
        return self.state, reward, done, info


    # def render(self, mode='human'):
    #    # Implement rendering logic here (e.g., using Pygame)
    #    screen = pygame.display.set_mode([-1, 1])
    #    screen.fill((255, 255, 255))
    #    pygame.draw.circle(screen, (0, 0, 255), (250, 250), 75)
    #    pass

    #implementation of render method to visualize the environment
    def render(self, mode='human', close=False):
        if close:
            plt.close()
            return

        if self.viewer is None:
            plt.ion()
            self.viewer = plt.figure()

        plt.clf()
        ax = self.viewer.add_subplot(111)
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, self.MAX_Y)

        # Draw the lander
        lander = Circle((self.state[0], self.state[1]), 0.05, color='black')
        ax.add_patch(lander)

        # Draw the minerals
        for mineral in self.minerals_locations:
            min_patch = Circle(mineral, 0.02, color='blue')
            ax.add_patch(min_patch)

        # Draw the mines
        for mine in self.mines_locations:
            mine_patch = Circle(mine, 0.02, color='red')
            ax.add_patch(mine_patch)

        # Draw the landing pad
        landing_pad = plt.Rectangle((self.LANDING_PAD_X_RANGE[0], 0), 
                                    self.LANDING_PAD_X_RANGE[1] - self.LANDING_PAD_X_RANGE[0],
                                    0.02, color='green')
        ax.add_patch(landing_pad)

        plt.draw()
        plt.pause(0.001)  # pause to update the plot

        if mode != 'rgb_array':
            plt.show(block=False)

'''
# Test the environment
if __name__ == "__main__":
    env = LunarHeistEnv()
    # Print the shape of the observation space
    print("Shape of observation space:", env.observation_space.shape)

    # If the observation space is discrete and not multi-dimensional (e.g., Box), you can use:
    print("Size of observation space:", env.observation_space.shape)
    for _ in range(1000):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        
        if done:
            env.reset()
'''
