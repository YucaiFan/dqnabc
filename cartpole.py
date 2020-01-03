import gym
import tensorflow as tf
import random
import numpy as np
from collections import deque

ENV_NAME = 'CartPole-v0'
EPISODE = 10000
STEP = 300

class dqn():

    def __init__(self, env):
        self.cache = deque()
        self.step = 0
        self.eps = 0.5
        self.dim_state = env.observation_space.shape[0]
        self.dim_action = env.action_space.n
        self.build()
        self.optim = ;;;

        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
    def build(self):
        W1 = self.weight_var([self.dim_state, 20])
        b1 = self.bias_var([20])
        W2 = self.weight_var([20, self.dim_action])
        b2 = self.bias_var([self.dim_action])
        self.action_input = tf.placeholder("float", [None, self.dim_state])
        self.hidden = tf.nn.relu(tf.matmul(self.action_input, W1) + b1)
        self.Q_value = tf.matmul(hidden, W2) + b2

    def bias_var(self, shape):
        return tf.Variable(tf.constant(0.01, shape=shape))
    def weight_var(self, shape):
        return tf.Variable(tf.truncated_normal(shape))

    def perceive(self, state, action, reward, next_state, done):
        one_hot = np.zeros(self.dim_action)
        one_hot[action] = 1
        self.cache.append((state, one_hot, reward, next_state, done))
        if len(self.cache) > REPLAY_SIZE:
            self.cache.popleft()
        if len(self.cache) > BATCH_SIZE:
            self.train()

    def train(self, env):



    # eps-greedy
    def action(self, env):
        return np.argmax(self.Q_value.eval(feed_dict = {
            self.intput: [state]
        })[0])

def main():
    env = gym.make(ENV_NAME)
    agent = nqn(env)
    agent.action_input = tf.placeholder("float", [None, agent.dim_action])
    agent.y_input = tf.placeholder("float", [None])
    Q_action = tf.reduce_sum(tf.mul(agent.Q_value, agent.action_input), reduction_indices=1)
    agent.cost = tf.reduce_mean(tf.sqaure(agent.y_input, Q_action))
    agent.optimizer = tf.train.AdamOptimizer(0.005).minimize(self.cost)
    agent.train()
