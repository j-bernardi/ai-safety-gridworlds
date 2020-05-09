# FROM AI SAFETY SIDE CAMP
# https://github.com/side-grids/ai-safety-gridworlds/blob/master/side_grids_camp/agents/dqn.py

from __future__ import print_function
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
import keras
from collections import deque, namedtuple
import datetime

from my_agents.dqn_solver.standard_agent import (
    StandardAgent, EpisodeStats, Transition, StandardEstimator)

# %% StateProcessor
class StateProcessor():
    """
    Changes gridworld RGB frames to gray scale.
    """
    def __init__(self, x_size, y_size):
        self.x_size = x_size
        self.y_size = y_size
        # Build the Tensorflow graph
        with tf.compat.v1.variable_scope("state_processor"):
            self.input_state = tf.compat.v1.placeholder(shape=[x_size, y_size, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [self.x_size, self.y_size, 3] gridworld RGB State
        Returns:
            A processed [self.x_size, self.y_size] state representing grayscale values.
        """
        return sess.run(self.output, { self.input_state: state })


# %% Estimator
class Estimator(StandardEstimator):
    """
    Q-Value Estimator neural network.
    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, actions_num, x_shape, y_shape, frames_state, name="estimator", experiment_dir=None, checkpoint=False):

        self.scope = name
        self.model_name = name
        self.actions_num = actions_num
        self.x_shape = x_shape 
        self.y_shape = y_shape
        self.frames_state = frames_state
        self.checkpoint = checkpoint
    
        # Writes Tensorboard summaries to disk
        #self.global_step = tf.Variable(0, name=self.model_name + "_global_step", 
        #                               trainable=False)
        if self.model_name == "q":
            self.global_step = tf.compat.v1.train.create_global_step()
        
        with tf.compat.v1.variable_scope(self.scope):
            self._build_model()

        # Load a previous checkpoint if we find one (now done in agent init)
        #if checkpoint:
        #    self.load_model_cp(experiment_dir)

    def _build_model(self):
        # Builds the Tensorflow graph.
        print("\nBUILDING MODEL", self.model_name)

        # Placeholders for our input
        # Our input are FRAMES_STATE RGB frames of shape of the gridworld
        self.X_pl = tf.compat.v1.placeholder(shape=[None, self.x_shape, self.y_shape,
                                             self.frames_state], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.compat.v1.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        # NETWORK ARCHITECTURE
        # tf.nn.conv2d(input, num_outputs, kernel_size, stride)

        filter_size = 2 ;input_channels = self.frames_state ; output_filters = 64
        flt = tf.Variable(tf.compat.v1.truncated_normal([filter_size, filter_size, 
                                               input_channels, output_filters], 
                                              stddev=0.5))

        conv1 = tf.nn.conv2d(X, filters=flt, strides=1, padding='SAME') # , activation_fn=tf.nn.relu)
        activated_conv = tf.nn.relu(conv1)

        # try with padding = 'VALID'
        # pool1 = tf.contrib.layers.max_pool2d(conv1, 2)
        # conv2 = tf.contrib.layers.conv2d(pool1, 32, WX, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        flattened = tf.compat.v1.layers.flatten(activated_conv)
        fc1 = tf.compat.v1.layers.dense(flattened, 64)
        self.predictions = tf.compat.v1.layers.dense(fc1, self.actions_num)

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calcualte the loss between q values and the values of the max actions predicted
        self.losses = tf.compat.v1.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.compat.v1.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.compat.v1.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)

        if self.model_name == "q":
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def predict(self, sess, s):
        #Predicts action values.
        #Args:
        #  sess: Tensorflow session
        #  s: State input of shape [batch_size, FRAMES_STATE, 160, 160, 3]
        #Returns:
        #  Tensor of shape [batch_size, actions_num] containing the estimated
        #  action values.

        return sess.run(self.predictions, { self.X_pl: s })

    def update(self, sess, s, a, y):
        #Updates the estimator towards the given targets.
        #Args:
        #  sess: Tensorflow session object
        #  s: State input of shape [batch_size, FRAMES_STATE, 160, 160, 3]
        #  a: Chosen actions of shape [batch_size]
        #  y: Targets of shape [batch_size]
        #Returns:
        #  The calculated loss on the batch.


        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        global_step, _, loss = sess.run(
                        [self.global_step, 
                         self.train_op, self.loss],
                        feed_dict)

        return loss

    """
    def load_model_cp(self, checkpoint_path):

        # Load a previous checkpoint if we find one

        # TODO - HAX because latest_checkpoint not working - maybe needs epoch?
        if os.path.exists(checkpoint_path):
            print("Loading model checkpoint {}...\n".format(checkpoint_path))
            self.saver.restore(self.sess, checkpoint_path)
        else:
            print("No chekpoint", checkpoint_path, "detected! "
                  "Initializing model from scratch")

    """


# %% helper functions
def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.
    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.compat.v1.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.compat.v1.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.
    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


# %% DQNAgent
class DQNAgent(StandardAgent):
    """
    DQNAgent adjusted to ai-safety-gridworlds.
    """
    def __init__(self, sess, world_shape, actions_num, env, frames_state=2, experiment_dir=None, replay_memory_size=1500, replay_memory_init_size=500, update_target_estimator_every=250, discount_factor=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=50000, batch_size=32, checkpoint=False):

        # Estimator for Q value
        self.q = Estimator(actions_num, 
                           world_shape[0],
                           world_shape[1],
                           frames_state=frames_state, 
                           name="q",
                           experiment_dir=experiment_dir,
                           checkpoint=checkpoint)

        # self.q.global_step = tf.compat.v1.train.create_global_step() #tf.compat.v1.Variable(0, trainable=False, dtype=tf.uint8)
        
        # Estimator for the target Q value
        self.target_q = Estimator(actions_num,
                                  world_shape[0],
                                  world_shape[1], 
                                  frames_state=frames_state, 
                                  name="target_q",
                                  experiment_dir=experiment_dir,
                                  checkpoint=False)

        self.sp = StateProcessor(world_shape[0], world_shape[1])
        self.policy = make_epsilon_greedy_policy(self.q, actions_num)

        self.sess = sess
        self.sess.run(tf.compat.v1.global_variables_initializer())

        if checkpoint:
            self.saver = tf.compat.v1.train.Saver(sharded=False)


        super(DQNAgent, self).__init__(world_shape, actions_num, env, 
                         frames_state, experiment_dir, 
                         replay_memory_size, replay_memory_init_size, 
                         update_target_estimator_every, discount_factor, 
                         epsilon_start, epsilon_end, epsilon_decay_steps, 
                         batch_size, checkpoint=checkpoint)

        print("Initialised double dqn with global step", 
              tf.compat.v1.train.get_global_step().eval())

    def get_state(self, obs):
        frame = np.moveaxis(obs['RGB'], 0, -1)
        frame = self.sp.process(self.sess, frame)
        if self.prev_state is None:
            state = np.stack([frame] * self.frames_state, axis=2)
        else:
            state = np.stack([self.prev_state[:,:,self.frames_state - 1], frame], axis=2)
        return state

    def act(self, obs, eps=None):
        if eps is None:
            eps = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        state = self.get_state(obs)
        probs = self.policy(self.sess, state, eps)  # you want some very random experience to populate the replay memory
        self.prev_state = state
        return np.random.choice(self.actions_num, p=probs)

    def learn(self, time_step, action):

        if self.total_t % self.update_target_estimator_every == 0:
            copy_model_parameters(self.sess, self.q, self.target_q)

        next_state = self.get_state(time_step.observation)
        done = time_step.last()

        self.replay_memory.append(Transition(self.prev_state, action,
                                             time_step.reward, next_state, done))

        # finally! let's learn something:
        sample = np.random.choice(len(self.replay_memory), self.batch_size)
        sample = [self.replay_memory[i] for i in sample]

        sts, a, r, n_sts, d = tuple(map(np.array, zip(*sample)))

        qs = self.target_q.predict(self.sess, n_sts).max(axis=1)
        qs[d] = 0
        targets = r + self.discount_factor * qs
        loss = self.q.update(self.sess, sts, a, targets)

        self.total_t += 1
        if time_step.last():
            self.new_episode()

        return loss

    def load(self):

        try:
            self.saver.restore(self.sess, self.checkpoint_path)
            print("Model loaded", self.checkpoint_path)
        except ValueError as ve:
            if "not a valid checkpoint" in str(ve):
                print("No checkpoint found, init from scratch.")
            else:
                raise ve

        super(DQNAgent, self).load_dict()

    def save(self, solved=False):

        # Save the session
        if self.checkpoint_dir:
            out_name = self.saver.save(self.sess, 
                                       self.checkpoint_path) # , 
                            # global_step=tf.compat.v1.train.get_global_step())
            assert out_name == self.checkpoint_path
        
        # Save the dict
        super(DQNAgent, self).save()
