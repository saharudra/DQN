from __future__ import print_function
import gym
import tensorflow as tf
import numpy as np
from itertools import count
from replay_memory import ReplayMemory, Transition
import env_wrappers
import random
import os
import argparse
from collections import deque
from gym.wrappers import Monitor

parser = argparse.ArgumentParser()
parser.add_argument('--eval', action="store_true", default=False, help='Run in eval mode')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

class DQN(object):
    """
    A starter class to implement the Deep Q Network algorithm

    TODOs specify the main areas where logic needs to be added.

    If you get an error a Box2D error using the pip version try installing from source:
    > git clone https://github.com/pybox2d/pybox2d
    > pip install -e .

    """

    def __init__(self, env):

        self.env = env

        self.monitor_dir = os.path.join(os.path.dirname(__file__), 'models/monitor_dir/')

        # A few starter hyperparameters
        self.batch_size = 512
        self.gamma = 0.99
        # If using e-greedy exploration
        self.eps_start = initial_eps
        self.eps_end = final_eps
        self.eps_mid = mid_eps
        self.eps_decay = 2000
        self.eps_decay_later = 3000# in episodes
        # If using a target network
        self.clone_steps = 5000

        self.sanity_epochs = 100
        self.max_episode = 1000000

        # using a deque for replay memory instead of provided implementation
        self.replay_memory = deque()
        # Perhaps you want to have some samples in the memory before starting to train?
        self.min_replay_size = 100000

        # define yours training operations here...
        self.observation_input = tf.placeholder(tf.float32, shape=[None] + list(env.observation_space.shape),
                                                name="input")
        self.action_input = tf.placeholder(tf.float32, [None, env.action_space.n])
        self.y = tf.placeholder(tf.float32, [None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.build_model()
        self.update()

        # define your update operations here...
        self.ini_random_walk_prob = 1.0
        self.change_random_walk_prob = 0.1
        self.follow_policy_prob = 0.01

        self.num_episodes = 0
        self.num_steps = 0
        self.sess = tf.Session()
        self.saver = tf.train.Saver(tf.trainable_variables())
        self.sess.run(tf.global_variables_initializer())


    def build_model(self, scope='train'):
        """
        TODO: Define the tensorflow model

        Hint: You will need to define and input placeholder and output Q-values

        Currently returns an op that gives all zeros.
        """
        with tf.variable_scope(scope):
            self.w1 = tf.get_variable('w1', list(env.observation_space.shape) + [512],
                                      initializer=tf.random_uniform_initializer(0, 0.1))
            self.b1 = tf.Variable(tf.constant(0.01, shape=[512, ]), name='b1')
            self.z1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(self.observation_input, self.w1), self.b1)),
                                    keep_prob=self.keep_prob)

            self.w2 = tf.get_variable('w2', [512, 256], initializer=tf.random_uniform_initializer(0, 0.1))
            self.b2 = tf.Variable(tf.constant(0.01, shape=[256, ]), name='b2')
            self.z2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(self.z1, self.w2), self.b2)),
                                    keep_prob=self.keep_prob)

            self.w3 = tf.get_variable('w3', [256, env.action_space.n], initializer=tf.random_uniform_initializer(0, 0.1))
            self.b3 = tf.Variable(tf.constant(0.01, shape=[env.action_space.n, ]))
            self.q_val = tf.add(tf.matmul(self.z2, self.w3), self.b3)
            return self.q_val

    def select_action(self, obs, evaluation_mode=False):
        """
        TODO: Select an action given an observation using your model. This
        should include any exploration strategy you wish to implement

        If evaluation_mode=True, then this function should behave as if training is
        finished. This may be reducing exploration, etc.

        Currently returns a random action.
        
        Return either a random action or the argmax action based upon whether doing exploration or not
        """
        q_val = self.q_val.eval(session=self.sess, feed_dict={self.observation_input: [obs], self.keep_prob: .85})[0]
        if evaluation_mode == True:
            return np.argmax(q_val)[0]
        if random.random() <= self.eps_start:
            return random.randint(0, env.action_space.n - 1)
        else:
            return np.argmax(q_val)

    def train_network(self, obs, action, reward, next_obs, done):
        """
        Helper code to call the optimizer
        :param obs: current observations
        :param action: current action to perform
        :param reward: reward in the current observation state
        :param next_obs: next observation from the environment
        :param done: reached the goal or not
        """
        encoded_action = np.zeros(env.action_space.n)
        encoded_action[action] = 1
        self.replay_memory.append((obs, encoded_action, reward, next_obs, done))
        if len(self.replay_memory) > self.min_replay_size:
            self.replay_memory.popleft()
        if len(self.replay_memory) > self.batch_size:
            self.perform_optim()

    def perform_optim(self):
        self.num_steps += 1
        for i in range(1):
            curr_batch = random.sample(self.replay_memory, self.batch_size)  # Sampling a batch randomly from memory
            obs_batch = [curr_data[0] for curr_data in curr_batch]
            action_batch = [curr_data[1] for curr_data in curr_batch]
            reward_batch = [curr_data[2] for curr_data in curr_batch]
            next_obs_batch = [curr_data[3] for curr_data in curr_batch]

            y_batch = []
            q_val_batch = self.q_val.eval(session = self.sess, feed_dict = {self.observation_input: next_obs_batch, self.keep_prob: 0.85})
            for j in range(0, self.batch_size):
                curr_run = curr_batch[j][4]
                if curr_run:
                    y_batch.append(reward_batch[j])
                else:
                    y_batch.append(reward_batch[j] + self.gamma * np.max(q_val_batch[j]))
            feed_out = [self.optimizer]
            feed_dict = {self.y: y_batch,
                             self.action_input: action_batch,
                             self.observation_input: obs_batch,
                             self.keep_prob: 0.85}
            _ = self.sess.run(feed_out, feed_dict)


    def update(self):
        """
        TODO: Implement the functionality to update the network according to the
        Q-learning rule
        Calculates the l2 loss and calls the optimizer
        """
        q_learning_action = tf.reduce_sum(tf.multiply(self.q_val, self.action_input), reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(self.y - q_learning_action))
        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)


    def train(self):
        """
        The training loop. This runs a single episode.

        TODO: Implement the following as desired:
            1. Storing transitions to the ReplayMemory
            2. Updating the network at some frequency
            3. Backing up the current parameters to a reference, target network
        """
        # Initially perform some random walks and make a replay memory
        env = Monitor(self.env, self.monitor_dir, force=True)
        for episode in range(1000):
            done = False
            obs = env.reset()
            while not done:
                action = random.randint(0, env.action_space.n - 1)
                encoded_action = np.zeros(env.action_space.n)
                encoded_action[action] = 1
                next_obs, reward, done, info = env.step(action)
                self.replay_memory.append((obs, encoded_action, reward, next_obs, done))
                obs = next_obs
                if len(self.replay_memory) > self.min_replay_size:
                    self.replay_memory.popleft()

        sum_of_reward = 0
        for episode in range(self.max_episode + 1):
            obs = env.reset()
            if self.eps_start > self.eps_mid:
                self.eps_start -= (initial_eps - mid_eps) / self.eps_decay  # Linear decay of exploration
            elif self.eps_start > self.eps_end:
                self.eps_start -= (mid_eps - final_eps) / self.eps_decay_later
            done = False            #     self.num_steps += 1
            # self.num_episodes += 1
            reward_per_episode = 0
            while not done:
                action = self.select_action(obs)
                next_obs, reward, done, info = env.step(action)
                self.train_network(obs, action, reward, next_obs, done)
                obs = next_obs
                reward_per_episode += reward
            sum_of_reward += reward_per_episode
            if episode % 100 == 0:
                avg_reward = sum_of_reward / 100
                self.saver.save(self.sess, 'models/dqn-model')
                print("Avg reward: %s" % avg_reward)
                if avg_reward > 210:
                    test_reward = 0
                    for i in range(self.sanity_epochs):
                        obs = env.reset()
                        done = False
                        while not done:
                            action = self.select_action(obs, evaluation_mode=True)
                            next_obs, reward, done, info = env.step(action)
                            test_reward += reward
                    avg_test_reward = test_reward / self.sanity_epochs
                    print("Episode: ", episode, "Average test reward: ", avg_test_reward)
                    if avg_test_reward >= 200:
                        env.close()
                        break
                sum_of_reward = 0


    # Performing eval inside train after every 100 episodes
    # def eval(self, save_snapshot=True):
    #     """
    #     Run an evaluation episode, this will call
    #     """
    #     total_reward = 0.0
    #     ep_steps = 0
    #     done = False
    #     obs = env.reset()
    #     while not done:
    #         env.render()
    #         action = self.select_action(obs, evaluation_mode=True)
    #         obs, reward, done, info = env.step(action)
    #         total_reward += reward
    #     print ("Evaluation episode: ", total_reward)
    #     if save_snapshot:
    #         print ("Saving state with Saver")
    #         self.saver.save(self.sess, 'models/dqn-model', global_step=self.num_episodes)

def train(dqn):
    dqn.train()


def eval(dqn):
    """
    Load the latest model and run a test episode
    """
    ckpt_file = os.path.join(os.path.dirname(__file__), 'models/checkpoint')
    with open(ckpt_file, 'r') as f:
        first_line = f.readline()
        model_name = first_line.split()[-1].strip("\"")
    dqn.saver.restore(dqn.sess, os.path.join(os.path.dirname(__file__), 'models/'+model_name))
    dqn.eval(save_snapshot=False)


if __name__ == '__main__':
    # On the LunarLander-v2 env a near-optimal score is some where around 250.
    # Your agent should be able to get to a score >0 fairly quickly at which point
    # it may simply be hitting the ground too hard or a bit jerky. Getting to ~250
    # may require some fine tuning.
    env = gym.make('LunarLander-v2')
    env.seed(args.seed)
    initial_eps = 1
    mid_eps = 0.1
    final_eps = 0.01
    # Consider using this for the challenge portion
    # env = env_wrappers.wrap_env(env)

    dqn = DQN(env)
    if args.eval:
        eval(dqn)
    else:
        train(dqn)
