# FROM AI SAFETY SIDE CAMP
# https://github.com/side-grids/ai-safety-gridworlds/blob/master/side_grids_camp/agents/dqn.py

from __future__ import print_function
import itertools
import os
import random
import sys
import datetime
import numpy as np
import tensorflow as tf
from collections import deque, namedtuple

import keras

from my_agents.dqn_solver.standard_agent import (
    StandardAgent, EpisodeStats, Transition, StandardEstimator)


# %% Estimator
class Estimator(StandardEstimator):
    """
    Q-Value Estimator neural network.
    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, actions_num, x_shape, y_shape, frames_state, batch_size=32, name="estimator", experiment_dir=None, checkpoint=True):
        super(Estimator, self).__init__(actions_num, 
                                        x_shape, y_shape, 
                                        frames_state,
                                        batch_size,
                                        name,
                                        experiment_dir, 
                                        checkpoint)
        # print("MODEL NAME", self.model_name)
        # print(self.model.summary())
        
        # Load a previous checkpoint if we find one
        if experiment_dir and checkpoint:
            self.checkpoint_dir = os.path.join(experiment_dir, self.model_name + "_checkpoints")
            self.checkpoint_path = os.path.join(self.checkpoint_dir, "latest.ckpt")
            print("Setting checkpoint path", self.checkpoint_path)
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

            # self.load_and_set_cp(experiment_dir)
            # TODO - customise so that an additional .txt file or something dumps global_state
            self.cp_callback = keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                               save_weights_only=True,
                                                               verbose=0,
                                                               period=1)
        else:
            self.cp_callback = None


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
    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.model.predict(np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


# %% DQNAgent
class DQNAgent(StandardAgent):
    """
    DQNAgent adjusted to ai-safety-gridworlds.
    """
    def __init__(self, world_shape, actions_num, env, frames_state=2, experiment_dir=None, replay_memory_size=1500, replay_memory_init_size=500, update_target_estimator_every=250, discount_factor=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=50000, batch_size=32, checkpoint=False):

        self.update_target_estimator_every = update_target_estimator_every

        # Estimator for Q value
        self.q = Estimator(actions_num, 
                           x_shape=world_shape[0], 
                           y_shape=world_shape[1], 
                           frames_state=frames_state,
                           batch_size=batch_size,
                           name="q",
                           checkpoint=checkpoint,
                           experiment_dir=experiment_dir)
        self.q.model.summary()

        # Estimator for the target Q value - don't ckpt as gets cloned 
        # CONSIDER REPLACE
        # self.target_q.model = keras.models.clone_model(self.q.model)
        self.target_q = Estimator(actions_num, 
                                  x_shape=world_shape[0], 
                                  y_shape=world_shape[1], 
                                  frames_state=frames_state,
                                  batch_size=batch_size,
                                  name="target_q",
                                  checkpoint=False,
                                  experiment_dir=experiment_dir)
        self.target_q.model.summary()
        
        # TODO - what is it?
        self.policy = make_epsilon_greedy_policy(self.q, actions_num)

        # Now that network to save (q) has been defined, call super
        super(DQNAgent, self).__init__(world_shape, actions_num, env, 
                         frames_state, experiment_dir, 
                         replay_memory_size, replay_memory_init_size, 
                         update_target_estimator_every, discount_factor, 
                         epsilon_start, epsilon_end, epsilon_decay_steps, 
                         batch_size, checkpoint)

    def act(self, obs, eps=None):
        if eps is None:
            eps = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        state = self.get_state(obs)
        probs = self.policy(state, eps)  # you want some very random experience to populate the replay memory
        self.prev_state = state
        return np.random.choice(self.actions_num, p=probs)

    @tf.function
    def take_step(self, sts, a, r, n_sts, d):

        target_qs = tf.reduce_max(self.target_q.model(n_sts), axis=1)
        target_qs_amend = tf.where(d, 0., target_qs)

        q_targets = r + self.discount_factor * target_qs_amend

        loss_value, grads = self.q.squared_diff_loss_at_a(sts, q_targets, a, self.batch_size)

        self.q.optimizer.apply_gradients(zip(grads, self.q.model.trainable_variables))

        return loss_value

    def learn(self, time_step, action):

        # Clone the q network to the target q periodically
        if self.total_t % self.update_target_estimator_every == 0:
            self.target_q.model.set_weights(self.q.model.get_weights())

        # Process the current timestep
        next_state = self.get_state(time_step.observation)
        done = time_step.last()
        self.replay_memory.append(Transition(self.prev_state, np.int32(action),
                                             np.float32(time_step.reward), next_state, done))

        # Take a sample from the replay memory
        sample = np.random.choice(len(self.replay_memory), self.batch_size)
        sample = [self.replay_memory[i] for i in sample]

        # Tuck into a fast tf function :-)
        loss_value = self.take_step(*tuple(map(np.array, zip(*sample))))

        self.total_t += 1
        if time_step.last():
            self.new_episode()


        return loss_value

    def load_q_net_weights(self, loaded_dict):
        self.q.load_and_set_cp(loaded_dict["experiment_dir"])