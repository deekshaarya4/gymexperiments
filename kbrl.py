import numpy as np
import gym
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='KBRL with KNN')
parser.add_argument('--episodes', nargs='?', type=int, default=500)
parser.add_argument('--max_timesteps', nargs='?', type=int, default=200)
parser.add_argument('environment')
args = parser.parse_args()

env = gym.make(args.environment).env
action_space = env.action_space

# hyperparameters:
epsilon = 1.0
exploration_decay = 0.98
k = 500 # number of nearest neighbors
minimum_num_iters = 500 # number of iterations used for training
num_iter = 0
max_iters = 0
gamma = 0.95
max_state_size = 15000 # because we don't know the state space size in continuous environments

# learning-related variables
states = None
actions = {}
rewards = {}
values = {}

# episode-related variables
episode_beginning = 0

def make_move(observation, reward, done):
    global states, actions, values, rewards, num_iter, episode_beginning, max_iters, epsilon
    if states is None:
        # first state observed
        states = np.zeros((max_state_size, observation.size))

    if num_iter > minimum_num_iters and np.random.rand() > epsilon and values:
        # if amount of data is sufficient and values is populated (atleast one episode has been run)
        # testing phase: exploitation
        # Uses k=500 nearest neighbors to pick the action which has the highest reward

        nbrs = NearestNeighbors(n_neighbors=min(k,max_iters)).fit(states[:max_iters])
        distances, indices = nbrs.kneighbors(observation)

        # find the best action
        action_list = {}
        freq_list = {}
        for i in indices[0]:
            v = values[i]
            a = actions[i]
            vnew = action_list.get(a, 0) + v
            action_list[a] = vnew
            freq_list[a] = freq_list.get(a, 0) + 1

        # normalize by number of times action occured and take action with highest value
        for act in action_list:
            action_list[act] = action_list[act] / freq_list[act]
        sorted_list = [(y,x) for x,y in action_list.items()]
        sorted_list.sort(reverse=True)
        take_action = sorted_list[0][1]

    else:
        # training phase: exploration randomly picks an action
        take_action = action_space.sample()

    # populate the state present, action taken and reward obtained
    if num_iter < max_state_size:
        states[num_iter] = observation # save the state
        actions[num_iter] = take_action # and the action we took
        rewards[num_iter-1] = reward # and the reward we obtained last time step
        values[num_iter-1] = 0

    num_iter += 1

    if done:
        # end of episode: calculate the value function for this episode
        val = 0
        for t in reversed(range(episode_beginning, num_iter)):
            val = gamma * val + rewards.get(t,0)
            values[t] = val
        episode_beginning = num_iter
        max_iters = min(max(max_iters, num_iter), max_state_size)

        # decay exploration probability
        epsilon *= exploration_decay
        # do not decay below 0
        epsilon = max(epsilon, 0)

    return take_action

# Ignore sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

reward = 0
episode_reward = 0
done = False
cumulative_reward_list = []

for i in range(args.episodes):
    observation = env.reset()
    sum_reward = 0

    for j in range(args.max_timesteps):
        env.render()
        action = make_move(observation, reward, done)
        observation, reward, done, _ = env.step(action)
        sum_reward += reward

        if done:
            break

    episode_reward = episode_reward * 0.95 + sum_reward * 0.05
    print('Reward for episode '+ str(i)+' : '+str(episode_reward))
    cumulative_reward_list.append(episode_reward)

# env.render()

plt.plot(range(0,500), cumulative_reward_list, linewidth=2)
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.title("Performance")
plt.show()
plt.close()
