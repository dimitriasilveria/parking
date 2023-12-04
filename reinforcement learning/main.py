import numpy as np
import gym
from gym import spaces
from lunar_heist_env import LunarHeistEnv  
# from Agents import QLearningAgent, DQNAgent  # Assume you complete the agents.py file
import matplotlib.pyplot as plt
from agents import QLearningAgent, DQNAgent
from icecream import ic
import pandas as pd

# TODO: Import your completed QLearningAgent and DQNAgent classes from Agents.py

def main():
    # Hyperparameters
    alpha = 0.1  # Learning rate
    gamma = 0.95  # Discount factor
    epsilon = 1.0  # Exploration rate
    min_epsilon = 0.01  # Minimum exploration probability
    decay_rate = 0.995  # Exponential decay rate for exploration prob
    num_episodes = 1000
    max_steps_per_episode = 1000
    # Environment setup
    env = LunarHeistEnv()
    actions = env.action_space
    # TODO: Define the state_bins based on the observation space of the environment
    bin_1 = np.round(np.arange(-1.02,1.01+0.02,0.02),2).tolist()
    bin_2 = np.round(np.arange(-0.01,1.01+0.01,0.02),2).tolist()
    state_bins = [bin_1, bin_2,bin_1,bin_1]  # Placeholder for student code
    
    # TODO: Instantiate the QLearningAgent and DQNAgent here
    q_learning_agent = QLearningAgent(actions,alpha,gamma,epsilon,min_epsilon,decay_rate,state_bins)  # Placeholder for student code
    dqn_agent = DQNAgent(env.observation_space.shape[0], actions.n, alpha, gamma, min_epsilon, epsilon, decay_rate)  # Placeholder for student code
    batch_size = 128

    # Initialize lists for detailed logs
    df = pd.DataFrame(columns=['episode', 'steps', 'reward', 'minerals', 'mines'])

    # Training loop
    # ...
    rollout = 0
    mean = 0
    for episode in range(1, num_episodes+1):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0
        episode_minerals_collected = 0
        episode_mines_hit = 0
        
          # Optional rendering

        while not done and step < max_steps_per_episode:
            # TODO: Choose an action using the QLearningAgent or DQNAgent

            #q learning
            action = q_learning_agent.choose_action(state)  # Placeholder for student code
            # TODO: Take a step in the environment using the chosen action
            next_state, reward, done, info = env.step(action)  # Placeholder for student code
            # TODO: Update the QLearningAgent or DQNAgent with the new experience
            # Placeholder for student code
            q_learning_agent.update(state,action,reward,next_state,done)
            q_learning_agent.add_step()
            
            #dqn
            action = dqn_agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            dqn_agent.remember(state, action, reward, next_state, done)
            dqn_agent.add_step()
            dqn_agent.replay(batch_size)

            state = next_state
            total_reward +=reward

            step +=1
        #calculating the mean reward for the last 5 episodes
        if episode%5 == 0:

            rollout +=reward
            mean = rollout/5
            rollout = 0
            print(f'episode num {episode},'+ f'minerals_collected = {episode_minerals_collected},' +f'mines_hit = {episode_mines_hit},' +f'total reward = {mean}')
        else:
            rollout +=reward
            
            # Render and collect logs
            #env.render()
        # Log episode details and update plot
        episode_minerals_collected = env.minerals_collected
        episode_mines_hit = env.mines_hit
        #print(f'episode num {episode},'+ f'minerals_collected = {episode_minerals_collected},' +f'mines_hit = {episode_mines_hit},' +f'total reward = {total_reward}')
        
        #columns=['episode', 'steps', 'reward', 'minerals', 'mines']
        if episode >= 5: #saving data from the 5 onwards, because the mean reward is set to 0 before that
            df.loc[len(df.index)] = [episode, step, mean, episode_minerals_collected, episode_mines_hit] 
        
        # ...
        
    # After training loop
    #dqn_agent.load('./whatever.pth')
    Reward = df.loc[:,'reward']
    Episode = df.loc[:,'episode']
    Step = df.loc[:,'steps']
    Minerals = df.loc[:,'minerals']
    Mines = df.loc[:,'mines']
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

    plt.show()

if __name__ == "__main__":
    main()
