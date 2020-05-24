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
    StandardAgent, EpisodeStats, Transition)

from my_agents.dqn_solver.standard_agent import (
    StandardEstimator as Estimator)


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
                           name="q")
        self.q.model.summary()

        # Estimator for the target Q value - don't ckpt as gets cloned 
        self.target_q = Estimator(actions_num, 
                                  x_shape=world_shape[0], 
                                  y_shape=world_shape[1], 
                                  frames_state=frames_state,
                                  name="target_q",
                                  )

        # Call super - this automatically loads network state
        super(DQNAgent, self).__init__(world_shape, actions_num, env, 
                         frames_state, experiment_dir, 
                         replay_memory_size, replay_memory_init_size, 
                         update_target_estimator_every, discount_factor, 
                         epsilon_start, epsilon_end, epsilon_decay_steps, 
                         batch_size, checkpoint)
        
        # Set target weights after loading in q network from super
        # self.target_q.model = tf.keras.models.clone_model(self.q.model)
        self.target_q.model.set_weights(self.q.model.get_weights())
        self.target_q.model.summary()

    def act(self, obs, eps=None):
        """
        Act with randomness
        """
        if eps is None:
            eps = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        state = self.get_state(obs)
        probs = self.policy_fn(state, eps) # Probability over actions
        self.prev_state = state
        choice = np.random.choice(self.actions_num, p=probs)
        if eps == 0.:
            assert np.max(probs) == 1.
            assert np.sum(probs) == 1.
            assert np.argmax(probs) == choice
        return choice

    # def act_determine(self, obs, eps=None):
    #     """
    #     Act without randomness
    #     """
    #     state = self.get_state(obs)
    #     self.prev_state = state
    #     return np.argmax(self.q.model(state))

    def policy_fn(self, observation, epsilon):
        """
        Creates an epsilon-greedy policy based on the Q-function approximator and epsilon.
        Args:
            observation: the current state observation
            epsilon: the degree of randomness to apply
        Returns:
            A function that returns the probabilities for each possible agent action 
            in the form of a numpy array.
        """
        A = np.ones(self.actions_num, dtype=float) * epsilon / self.actions_num
        q_values = self.q.model(np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    def learn(self, time_step, action):
        # Clone the q network to the target q periodically
        if self.total_t % self.update_target_estimator_every == 0:
            self.target_q.model.set_weights(self.q.model.get_weights())

        # Process the current timestep
        next_state = self.get_state(time_step.observation)
        done = time_step.last()
        sarnsd = Transition(self.prev_state, np.int32(action),
                            np.float32(time_step.reward), 
                            next_state, done)
        self.replay_memory.append(sarnsd)

        # Take a sample from the replay memory
        sample_i = np.random.choice(len(self.replay_memory), self.batch_size)
        sample = [self.replay_memory[i] for i in sample_i]

        # Update the Q network with this sample
        loss_value = self.take_training_step(*tuple(map(np.array, zip(*sample))))

        self.total_t += 1
        if time_step.last():
            self.new_episode()

        return loss_value

    @tf.function
    def take_training_step(self, sts, a, r, n_sts, d):

        future_q = tf.reduce_max(self.target_q.model(n_sts), axis=1)
        adj_future_q = tf.where(d, 0., future_q)
        
        q_targets = r + self.discount_factor * adj_future_q

        loss_value, grads = self.squared_diff_loss_at_a(sts, a, q_targets)

        self.q.optimizer.apply_gradients(zip(grads, self.q.model.trainable_variables))

        return loss_value

    @tf.function
    def squared_diff_loss_at_a(self, states, action_mask, targets_from_memory):
        """
        A squared difference loss function 
        Diffs the Q model's predicted values with 
        the actual values plus the discounted next state by the target Q network
        """
        with tf.GradientTape() as tape:
            q_predictions = self.q.model(states)
            
            gather_indices = tf.range(self.batch_size) * tf.shape(q_predictions)[1] + action_mask
            q_predictions_at_a = tf.gather(tf.reshape(q_predictions, [-1]), gather_indices)

            losses = tf.math.squared_difference(q_predictions_at_a, targets_from_memory)
            reduced_loss = tf.math.reduce_mean(losses)

        return reduced_loss, tape.gradient(reduced_loss, self.q.model.trainable_variables)

    def save(self):
        # Save the Q model (Target Q is a copy)
        if self.checkpoint:
            self.q.save_model(self.checkpoint_path)

        # And save the param dict
        super(DQNAgent, self).save_param_dict()

    def load(self):

        # Load the Q model
        successful_model_load = self.q.load_model_cp(self.checkpoint_path)

        # Load the experiment state
        successful_dict_load = super(DQNAgent, self).load_param_dict()

        self.loaded_model = successful_model_load and successful_dict_load