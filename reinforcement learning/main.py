import numpy as np
import gym
from gym import spaces
from lunar_heist_env import LunarHeistEnv  
# from Agents import QLearningAgent, DQNAgent  # Assume you complete the agents.py file
import matplotlib.pyplot as plt
from agents import QLearningAgent

# TODO: Import your completed QLearningAgent and DQNAgent classes from Agents.py

def main():
    # Hyperparameters
    alpha = 0.1  # Learning rate
    gamma = 0.95  # Discount factor
    epsilon = 1.0  # Exploration rate
    min_epsilon = 0.01  # Minimum exploration probability
    decay_rate = 0.995  # Exponential decay rate for exploration prob
    num_episodes = 10000
    max_steps_per_episode = 1000
    # Environment setup
    env = LunarHeistEnv()
    actions = env.action_space
    # TODO: Define the state_bins based on the observation space of the environment
    state_bins = np.arange(-1,1,0.01)  # Placeholder for student code
    
    # TODO: Instantiate the QLearningAgent and DQNAgent here
    q_learning_agent = QLearningAgent(actions,alpha,gamma,epsilon,state_bins)  # Placeholder for student code
    dqn_agent = None  # Placeholder for student code
    
    # Initialize lists for detailed logs
    # ...

    # Training loop
    # ...

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0
        episode_minerals_collected = 0
        episode_mines_hit = 0
        
        # env.render()  # Optional rendering

        while not done and step < max_steps_per_episode:
            # TODO: Choose an action using the QLearningAgent or DQNAgent
            #if step%2 == 0:

            action = q_learning_agent.choose_action(state)  # Placeholder for student code
            
            # TODO: Take a step in the environment using the chosen action
            next_state, reward, done, info = env.step(action)  # Placeholder for student code
            
            # TODO: Update the QLearningAgent or DQNAgent with the new experience
            # Placeholder for student code
            q_learning_agent.update(state,action,reward,next_state,done)
            state = next_state



            # Render and collect logs
            # ...
        episode_minerals_collected = env.minerals_collected
        episode_mines_hit = env.episode_mines_hit
        # Log episode details and update plot
        # ...

    # After training loop
    # ...

if __name__ == "__main__":
    main()
