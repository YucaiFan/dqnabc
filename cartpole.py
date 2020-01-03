import gym
import tensorflow as tf
import random
import numpy as np
from collections import deque


class dqn():

    def __init__(self, gamma, batch_size, env):
        self.gamma = gamma
        self.batch_size = batch_size
        self.iter_times = 0
        self.cache = deque()
        self.eps = 0.5
        self.dim_state = env.observation_space.shape[0]
        self.dim_action = env.action_space.n
        self.build()

    def build(self):
        W1 = self.weight_var([self.dim_state, 20])
        b1 = self.bias_var([20])
        W2 = self.weight_var([20, self.dim_action])
        b2 = self.bias_var([self.dim_action])
        self.state_input = tf.placeholder("float", [None, self.dim_state])
        self.hidden = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        self.Q_value = tf.matmul(self.hidden, W2) + b2

    def bias_var(self, shape):
        return tf.Variable(tf.constant(0.01, shape=shape))
    def weight_var(self, shape):
        return tf.Variable(tf.truncated_normal(shape))

    def perceive(self, state, action, reward, next_state, done):
        one_hot = np.zeros(self.dim_action)
        one_hot[action] = 1
        self.cache.append((state, one_hot, reward, next_state, done))
        if len(self.cache) > 10000:
            self.cache.popleft()
        if len(self.cache) > self.batch_size:
            self.train()

    def train(self):
        self.iter_times += 1

        minibatch = random.sample(self.cache, self.batch_size)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
        for i in range(0, self.batch_size):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.gamma*np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
            })

    # eps-greedy
    def greedy_action(self, state):
        Q_value = self.Q_value.eval(feed_dict = {
            self.state_input:[state]
            })[0]
        if random.random() < self.eps:
            return random.randint(0, self.dim_action - 1)
        else:
            return np.argmax(Q_value)
        self.eps -= (0.5-0.01)/10000

    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict = {
            self.state_input: [state]
        })[0])

def main():
    # ENV_NAME = 'CartPole-v0'
    ENV_NAME = 'MountainCar-v0'
    max_iter = 300
    max_epoch = 10000
    TEST = 10

    env = gym.make(ENV_NAME)
    agent = dqn(gamma=0.9, batch_size=32, env=env)

    agent.action_input = tf.placeholder("float", [None, agent.dim_action])
    agent.y_input = tf.placeholder("float", [None])
    Q_action = tf.reduce_sum(tf.multiply(agent.Q_value, agent.action_input), reduction_indices=1)
    agent.cost = tf.reduce_mean(tf.square(agent.y_input - Q_action))
    agent.optimizer = tf.train.AdamOptimizer(0.005).minimize(agent.cost)

    agent.session = tf.InteractiveSession()
    agent.session.run(tf.initialize_all_variables())

    for epoch in range(max_epoch):
        state = env.reset()

        for iter_times in range(max_iter):
            action = agent.greedy_action(state)
            next_state, reward, done, _ = env.step(action)
            reward_agent = -1 if done else 0.1
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        if epoch % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(max_iter):
                    env.render()
                    action = agent.action(state)
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            avg_reward = total_reward/TEST
            print("Epoch: ", epoch, "avg_reward: ", avg_reward)
            if avg_reward >= 400:
                break

if __name__ == '__main__':
    main()
