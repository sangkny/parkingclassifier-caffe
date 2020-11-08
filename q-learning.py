# -*- coding: utf-8 -*-
'''
#	Reinforcement Q-learning
#	Jon Eivind Stranden (@) 2018
Rewards
각 action마다 -1을 reward로 받는데 승객을 목적지에 내려주면 +20을 reward로 받는다. 단 승객을 잘못 태우거나 잘못 내리면 -10을 reward로 받는다.
그리고 state space는  (택시의 행, 택시의 열, 승객의 위치, 목적지)로 이루어져있다는데 이게 0~500의 스칼라값으로 변환된 것임.

more details: https://apincan.tistory.com/29
modified by sangkny
'''

import os
import gym
import numpy as np
import random
import time
import csv

env = None
q_table = None


def init():
    global env
    global q_table

    # Import environment
    env = gym.make("Taxi-v3").env

    # Initialize Q-table
    q_table = np.zeros([env.observation_space.n, env.action_space.n])


def action_space_sample(num_actions):
    action = random.randint(0, num_actions - 1)
    return action


def environment(action):
    # Define environment here
    pass
    #return next_state, reward, done, info


def train_agent(num_epochs=100001, alpha=0.1, gamma=0.6, epsilon=0.1):
    # Alpha: 	Learning rate - Step size of the iteration
    # Gamma: 	Discount factor - Determines how much importance we want to give to future rewards
    # Epsilon:	How much to explore - Higher is more dependent of what is already learned, lower is more exploration

    global q_table

    for i in range(1, num_epochs):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0
        done = False

        while not done:
            # Decide an action
            if random.uniform(0, 1) < epsilon:
                action = action_space_sample(6)  # env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values

            next_state, reward, done, info = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            # Update penalty
            if reward is -10:
                penalties += 1

            # Update state
            state = next_state

            # Update epochs
            epochs += 1

            if i % 100 is 0:
                # Clear the terminal window
                # os.system('clear')
                print("*** Training agent ***")
                print('Episode: ' + str(i) + ' of ' + str(num_epochs - 1))

    # Save Q-table to file
    np.savetxt('q_table.txt', q_table)
    print('Training done!\n')


def run_agent(episodes=100):
    global q_table

    q_table = np.loadtxt('q_table.txt')  # Load Q-table from file

    for episode in range(episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0
        done = False

        while not done:
            action = np.argmax(q_table[state])  # Get the action with the highest Q-value based on the current state
            state, reward, done, info = env.step(action)

            # Update penalty
            if reward is -10:
                penalties += 1

            epochs += 1

            # Clear the terminal window
            os.system('cls')
            env.render()

            print('\nEpisode: ' + str(episode) + ' of ' + str(episodes) + '\n')

            time.sleep(0.1)


if __name__ == "__main__":
    init()
    train_agent(100001, 0.1, 0.6, 0.1) # at least you need 1000000 iterations
    run_agent(100)