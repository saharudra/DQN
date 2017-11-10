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
        self.sess = tf.Session()

        # A few starter hyperparameters
        self.batch_size = 128
        self.gamma = 0.99
        # If using e-greedy exploration
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000 # in episodes
        # If using a target network
        self.clone_steps = 5000

        # memory
        self.replay_memory = ReplayMemory(100000)
        # Perhaps you want to have some samples in the memory before starting to train?
        self.min_replay_size = 10000

        # define yours training operations here...
        self.observation_input = tf.placeholder(tf.float32, shape=[None] + list(self.env.observation_space.shape))
        self.keep_prob = tf.placeholder(tf.float32)
        q_values = self.build_model(self.observation_input)

        # define your update operations here...
        self.ini_random_walk_prob = 1.0
        self.change_random_walk_prob = 0.1
        self.follow_policy_prob = 0.01

        self.num_episodes = 0
        self.num_steps = 0

        self.saver = tf.train.Saver(tf.trainable_variables())
        self.sess.run(tf.global_variables_initializer())

    def build_model(self, observation_input, scope='train'):
        """
        TODO: Define the tensorflow model

        Hint: You will need to define and input placeholder and output Q-values

        Currently returns an op that gives all zeros.
        """
        with tf.variable_scope(scope):
            self.observation_input = observation_input
            self.w1 = tf.get_variable('w1', list(self.env.observation_space.shape) + [256],
                                      initializer=tf.random_uniform_initializer(0, 1))
            self.b1 = tf.Variable(tf.constant(0.01, shape=[256, ]), name='b1')
            self.z1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(self.observation_input, self.w1), self.b1)),
                                    keep_prob=self.keep_prob)

            self.w2 = tf.get_variable('w2', [256, 256], initializer=tf.random_uniform_initializer(0, 1))
            self.b2 = tf.Variable(tf.constant(0.01, shape=[256, ]), name='b2')
            self.z2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(self.z1, self.w2), self.b2)), keep_prob=self.keep_prob)

            self.w3 = tf.get_variable('w2', [256, 512], initializer=tf.random_uniform_initializer(0, 1))
            self.b3 = tf.Variable(tf.constant(0.01, shape=[512, ]), name='b2')
            self.z3 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(self.z2, self.w3), self.b3)), keep_prob=self.keep_prob)

            self.w4 = tf.get_variable('w2', [512, self.env.action_space.n], initializer=tf.random_uniform_initializer(0, 1))
            self.b4 = tf.Variable(tf.constant(0.01, shape=[self.env.action_space.n, ]))
            self.q_val = tf.add(tf.matmul(self.z3, self.w4), self.b4)

            return self.q_val

    def select_action(self, obs, evaluation_mode=False):
        """
        TODO: Select an action given an observation using your model. This
        should include any exploration strategy you wish to implement

        If evaluation_mode=True, then this function should behave as if training is
        finished. This may be reducing exploration, etc.

        Currently returns a random action.
        """
        return env.action_space.sample()

    def update(self):
        """
        TODO: Implement the functionality to update the network according to the
        Q-learning rule
        """
        raise NotImplementedError

    def train(self):
        """
        The training loop. This runs a single episode.

        TODO: Implement the following as desired:
            1. Storing transitions to the ReplayMemory
            2. Updating the network at some frequency
            3. Backing up the current parameters to a reference, target network
        """
        done = False
        obs = env.reset()
        while not done:
            action = self.select_action(obs, evaluation_mode=False)
            next_obs, reward, done, info = env.step(action)
            self.num_steps += 1
        self.num_episodes += 1

    def eval(self, save_snapshot=True):
        """
        Run an evaluation episode, this will call
        """
        total_reward = 0.0
        ep_steps = 0
        done = False
        obs = env.reset()
        while not done:
            env.render()
            action = self.select_action(obs, evaluation_mode=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        print ("Evaluation episode: ", total_reward)
        if save_snapshot:
            print ("Saving state with Saver")
            self.saver.save(self.sess, 'models/dqn-model', global_step=self.num_episodes)

def train(dqn):
    for i in count(1):
        dqn.train()
        # every 10 episodes run an evaluation episode
        if i % 10 == 0:
            dqn.eval()

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
    # Consider using this for the challenge portion
    # env = env_wrappers.wrap_env(env)

    dqn = DQN(env)
    if args.eval:
        eval(dqn)
    else:
        train(dqn)
