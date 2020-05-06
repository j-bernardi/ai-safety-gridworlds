import pickle, os, random, pprint

import numpy as np

from collections import deque

import keras

from my_agents.dqn_solver.standard_agent import (
    StandardAgent, EpisodeStats, Transition, StandardEstimator)

# TODO - add batch size to estimator

class Estimator(StandardEstimator):
    """
    Q-Value Estimator neural network.
    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, actions_num, x_shape, y_shape, frames_state, name="estimator", experiment_dir=None, checkpoint=True):
        super(Estimator, self).__init__(actions_num, 
                                        x_shape, y_shape, 
                                        frames_state, 
                                        name, experiment_dir, 
                                        checkpoint)

        # Load a previous checkpoint if we find one
        if experiment_dir and checkpoint:
            self.checkpoint_dir = os.path.join(experiment_dir, self.model_name + "_checkpoints")
            self.checkpoint_path = os.path.join(self.checkpoint_dir, "latest.ckpt")
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

            self.load_and_set_cp(experiment_dir)
            # TODO - customise so that an additional .txt file or something dumps global_state
            self.cp_callback = keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                               save_weights_only=True,
                                                               verbose=0,
                                                               period=1)
        else:
            self.cp_callback = None


class DQNSolver(StandardAgent):
    """A standard dqn_solver.
    Implements a simple DNN that predicts values.
    """
    
    def __init__(self, 
                 world_shape, 
                 actions_num,
                 env, 
                 frames_state=2,
                 experiment_dir=None,
                 replay_memory_size=1500,
                 replay_memory_init_size=500,
                 update_target_estimator_every=250,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=50000,
                 batch_size=32,
                 checkpoint=False):

        self.q = Estimator(actions_num, 
                           x_shape=world_shape[0], 
                           y_shape=world_shape[1], 
                           frames_state=frames_state,
                           name="q",
                           experiment_dir=experiment_dir,
                           checkpoint=checkpoint)

        # Now that network to save (q) has been defined, call super
        super(DQNSolver, self).__init__(world_shape, actions_num, env, 
                         frames_state, experiment_dir, 
                         replay_memory_size, replay_memory_init_size, 
                         update_target_estimator_every, discount_factor, 
                         epsilon_start, epsilon_end, epsilon_decay_steps, 
                         batch_size, checkpoint)
        
    def act(self, obs, eps=None):
        """Take a random action or the most valuable predicted
        action, based on the agent's model. 
        """

        # If in exploration
        if eps is None:
            eps = self.epsilons[min(self.total_t, 
                                    self.epsilon_decay_steps-1)]

        # Prepare the staet
        state = self.get_state(obs)
        if len(state.shape) == 3: 
            pass_state = np.reshape(state, (1, *state.shape))
        else:
            pass_state
        self.prev_state = state

        if np.random.rand() <= eps:
            return random.randrange(self.actions_num)
        else:
            return np.argmax(self.q.model.predict(pass_state).flatten())
    
    def learn(self, time_step, action):
        """Updated the agent's decision network based
        on a sample of previous decisions it has seen.
        Here, we combine the target and action networks.
        """

        next_state = self.get_state(time_step.observation)
        done = time_step.last()
        self.replay_memory.append(Transition(self.prev_state, action,
                                             time_step.reward, next_state, done))

        x_batch, y_batch = [], []

        # Assume batch size < initial replay memory size
        sample = np.random.choice(len(self.replay_memory), self.batch_size)
        minibatch = [self.replay_memory[i] for i in sample]

        # Process the mini batch
        for state, action, reward, next_state, done in minibatch:
            
            # Get the predicted value of the action
            y_target = self.q.model.predict(np.reshape(state, (1, *state.shape))).flatten()

            # Set the value (or label) for the remembered action
            # the as the (discounted) next predicted reward
            next_pred_reward = self.q.model.predict(np.reshape(next_state, (1, *next_state.shape))).flatten()
            y_target[action] = reward if done else\
                               reward + self.discount_factor * np.amax(next_pred_reward)
            
            x_batch.append(state)
            y_batch.append(y_target)

        cbks = [self.q.cp_callback] if hasattr(self.q, "cp_callback") else []
        
        # Batched training
        self.q.action_mask = minibatch[1] # array of actions
        info = self.q.model.fit(np.array(x_batch), 
                               np.array(y_batch), 
                               batch_size=len(x_batch),
                               verbose=0, 
                               epochs=1,
                               callbacks=cbks)

        self.total_t += 1

        if time_step.last():
            self.new_episode()

        return info.history['loss'][0]

    def load_q_net_weights(self, loaded_dict):
        self.q.load_and_set_cp(loaded_dict["experiment_dir"])
