import numpy as np
import gym
from gym import spaces
from lunar_heist_env import LunarHeistEnv  
# from Agents import QLearningAgent, DQNAgent  # Assume you complete the agents.py file
import matplotlib.pyplot as plt
from agents import QLearningAgent, DQNAgent
from icecream import ic
import pandas as pd
from stable_baselines3 import DQN
# TODO: Import your completed QLearningAgent and DQNAgent classes from Agents.py

def main():
    # Hyperparameters
    alpha = 0.1  # Learning rate
    gamma = 0.95  # Discount factor
    epsilon = 1.0  # Exploration rate
    min_epsilon = 0.01  # Minimum exploration probability
    decay_rate = 0.995  # Exponential decay rate for exploration prob
    num_episodes = 5
    max_steps_per_episode = 1000
    # Environment setup
    env = LunarHeistEnv()
    actions = env.action_space
    df_ql = pd.DataFrame(columns=['episode', 'steps', 'reward', 'minerals', 'mines'])
    model = DQN("MlpPolicy", env, verbose=1,tensorboard_log='./log/')
    model.learn(total_timesteps=10000, log_interval=4)
    model.save("dqn_cartpole")

    #del model # remove to demonstrate saving and loading
    #
    #model = DQN.load("dqn_cartpole")
    #

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0
        episode_minerals_collected = 0
        episode_mines_hit = 0
        while not done and step < max_steps_per_episode:
            action, _states = model.predict(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
        
            state = next_state
            total_reward +=reward

            step +=1
            # Render and collect logs
            #env.render()
        # Log episode details and update plot
        episode_minerals_collected = env.minerals_collected
        episode_mines_hit = env.mines_hit
        print(f'episode num {episode},'+ f'minerals_collected = {episode_minerals_collected},' +f'mines_hit = {episode_mines_hit},' +f'total reward = {total_reward}')
        df_ql.loc[len(df_ql.index)] = [episode, step, total_reward, episode_minerals_collected, episode_mines_hit] 
    # TODO: Define the state_bins based on the observation space of the environment
    #bin_1 = np.round(np.arange(-1.02,1.01+0.02,0.02),2).tolist()
    #bin_2 = np.round(np.arange(-0.01,1.01+0.01,0.02),2).tolist()
    #state_bins = [bin_1, bin_2,bin_1,bin_1]  # Placeholder for student code
    
    # TODO: Instantiate the QLearningAgent and DQNAgent here
    #q_learning_agent = QLearningAgent(actions,alpha,gamma,epsilon,min_epsilon,decay_rate,state_bins)  # Placeholder for student code
    
    Reward = df_ql.loc[:,'reward']
    Episode = df_ql.loc[:,'episode']
    Step = df_ql.loc[:,'steps']
    Minerals = df_ql.loc[:,'minerals']
    Mines = df_ql.loc[:,'mines']
    ax1 = plt.subplot(2,2,1)
    plt.plot(Episode,Step)
    plt.xlabel('Num of episodes')
    plt.ylabel('Num steps')
    plt.title('Number of steps per episode')
    ax2 = plt.subplot(2,2,2)
    plt.plot(Episode,Reward)
    plt.xlabel('Number of episodes')
    plt.ylabel('Reward')
    plt.title('Reward per episode')
    ax3 = plt.subplot(2,2,3)
    plt.plot(Episode,Minerals)
    plt.xlabel('Number of episodes')
    plt.ylabel('Minerals')
    plt.title('Number of minerals collected per episode')
    ax3 = plt.subplot(2,2,4)
    plt.plot(Episode,Mines)
    plt.xlabel('Number of episodes')
    plt.ylabel('Mines')
    plt.title('Number of mines hit per episode')

    #plt.show()

if __name__ == "__main__":
    main()
