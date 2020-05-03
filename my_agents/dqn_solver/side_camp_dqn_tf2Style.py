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
from keras.layers import (Conv2D, Dense, Flatten)
from collections import deque, namedtuple
import datetime, pickle, pprint


# %% Estimator
class Estimator():
    """
    Q-Value Estimator neural network.
    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, actions_num, x_shape, y_shape, frames_state, name="estimator", experiment_dir=None, checkpoint=True):
        self.model_name = name
        self.actions_num = actions_num
        self.x_shape, self.y_shape, self.frames_state = x_shape, y_shape, frames_state

        # Writes Tensorboard summaries to disk
        self.model = self._build_model()
        self.global_steps = tf.Variable(0, name=self.model_name + "_global_step", 
                                        trainable=False)
        
        # Load a previous checkpoint if we find one
        if experiment_dir and checkpoint:
            self.load_and_set_cp(experiment_dir)
        else:
            self.cp_callback = None

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        Takes states to predicted action's value
        """

        print("BUILDING MODEL", self.model_name)

        model = keras.Sequential()

        # TODO - check float value?
        # self.X_pl = tf.compat.v1.placeholder(shape=[None, self.x_size, self.y_size,
        #                                      self.frames_state], dtype=tf.uint8, name="X")
        # X = tf.compat.v1.to_float(self.X_pl) / 255.0

        model.add(Conv2D(filters=64, kernel_size=2, 
                         strides=1, padding='SAME', 
                         activation='relu',
                         input_shape=(self.x_shape, self.y_shape, self.frames_state)))

        # try with padding = 'VALID'
        # pool1 = tf.contrib.layers.max_pool2d(conv1, 2)
        # conv2 = tf.contrib.layers.conv2d(pool1, 32, WX, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.actions_num, activation='linear'))

        model.compile(loss=(lambda x, y: self.my_loss(x, y, model)),
                      optimizer=keras.optimizers.RMSprop(
                           learning_rate=0.00025,
                           rho=0.99,
                           # momentum=0.0,
                           epsilon=1e-6)
                     )
        # model.summary()
        return model

    def my_loss(self, y_pred, y_true, model):

        # convert model outputs to the vlaue of the chosen action
        # Others are meaningless
        actual_qs = tf.math.reduce_max(y_true, axis=1)
        predicted_qs = tf.math.reduce_max(y_pred, axis=1)

        losses = tf.math.squared_difference(actual_qs, predicted_qs)
        reduced_loss = tf.math.reduce_mean(losses)
        return reduced_loss

    def load_and_set_cp(self, experiment_dir):
        self.checkpoint_dir = os.path.join(experiment_dir, self.model_name + "_checkpoints")
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "latest.ckpt")
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # TODO - customise so that an additional .txt file or something dumps global_state
        self.cp_callback = keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                           save_weights_only=True,
                                                           verbose=0,
                                                           period=1)
        
        # TODO - HAX because latest_checkpoint not working - maybe needs epoch?
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        if not latest:
            if os.path.exists(self.checkpoint_path):
                latest = self.checkpoint_path
        if latest:
            print("Loading model checkpoint {}...".format(latest))
            self.model.load_weights(latest)
        else:
            print("Initializing model from scratch")


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

EpisodeStats = namedtuple("EpisodeStats", ["episode_lengths", "episode_rewards"])
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


# %% DQNAgent
class DQNAgent():
    """
    DQNAgent adjusted to ai-safety-gridworlds.
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

        self.world_shape = world_shape
        self.actions_num = actions_num
        self.frames_state = frames_state
        
        self.experiment_dir = experiment_dir
        self.dict_loc = self.experiment_dir + "/param_dict.p"


        # Estimator for Q value
        self.q = Estimator(actions_num, 
                           x_shape=world_shape[0], 
                           y_shape=world_shape[1], 
                           frames_state=self.frames_state,
                           name="q",
                           checkpoint=checkpoint,
                           experiment_dir=experiment_dir)

        # Estimator for the target Q value - don't ckpt as gets cloned 
        self.target_q = Estimator(actions_num, 
                                  x_shape=world_shape[0], 
                                  y_shape=world_shape[1], 
                                  frames_state=self.frames_state,
                                  name="target_q",
                                  checkpoint=False,
                                  experiment_dir=experiment_dir)
        
        # self.sp = StateProcessor(world_shape[0], world_shape[1])
        
        self.replay_memory = deque(maxlen=replay_memory_size) # []
        # self.replay_memory_size = replay_memory_size
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        self.epsilon_decay_steps = epsilon_decay_steps
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        
        # TODO - what is it?
        self.policy = make_epsilon_greedy_policy(self.q, actions_num)
        
        self.solved_on = None
        self.total_t = 0
        self.scores, self.ep_lengths, self.losses = [], [], []

        # RETREIVE PROGESS
        self.load()

        self.new_episode()
        
        time_step = env.reset()
        
        state = self.get_state(time_step.observation)
        
        # Initialise the replay memory
        for i in range(replay_memory_init_size):
            action = self.act(time_step.observation, eps=0.95)
            time_step = env.step(action)
            next_state = self.get_state(time_step.observation)
            done = time_step.last()

            assert state is not None
            assert next_state is not None
            self.replay_memory.append(Transition(state, action, time_step.reward, next_state, done))
            if done:
                time_step = env.reset()
                self.new_episode()
                state = self.get_state(time_step.observation)
            else:
                state = next_state

    def new_episode(self):
        self.loss = None
        self.prev_state = None

    def get_state(self, obs):
        frame = np.moveaxis(obs['RGB'], 0, -1)
        # frame = self.sp.process(self.sess, frame)
        frame = tf.squeeze(tf.image.rgb_to_grayscale(frame))
        if self.prev_state is None:
            state = np.stack([frame] * self.frames_state, axis=2)
        else:
            state = np.stack([self.prev_state[:,:,self.frames_state - 1], frame], axis=2)
        float_state = state.astype('float32') / 255.0
        return float_state

    def act(self, obs, eps=None):
        if eps is None:
            eps = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        state = self.get_state(obs)
        probs = self.policy(state, eps)  # you want some very random experience to populate the replay memory
        self.prev_state = state
        return np.random.choice(self.actions_num, p=probs)

    def learn(self, time_step, action):

        # Clone the q network to the target q periodically
        if self.total_t % self.update_target_estimator_every == 0:
            # print("\nCLONING Q to target Q")
            self.target_q.model = keras.models.clone_model(self.q.model)
            # self.target_q.ckpt.step.assign(self.q.ckpt.step)

        # Process the current timestep
        next_state = self.get_state(time_step.observation)
        done = time_step.last()
        self.replay_memory.append(Transition(self.prev_state, action,
                                             time_step.reward, next_state, done))

        # Take a sample from the replay memory
        sample_size = min(len(self.replay_memory),  
                          self.batch_size)
        minibatch = random.sample(self.replay_memory, 
                                  sample_size)

        sample = np.random.choice(len(self.replay_memory), self.batch_size)
        sample = [self.replay_memory[i] for i in sample]

        sts, a, r, n_sts, d = tuple(map(np.array, zip(*sample)))
        
        # The target q values, given the state
        qs_targets_from_state = self.target_q.model.predict(sts)
        
        # Update the action taken value to what was actually achieved plus prediction
        qs_predicted_on_next = self.target_q.model.predict(n_sts).max(axis=1)
        qs_predicted_on_next[d] = 0
        
        qs_targets_from_state[:,a] = r + self.discount_factor * qs_predicted_on_next

        cbks = [self.q.cp_callback] if hasattr(self.q, "cp_callback") else []

        # Batched training
        info = self.q.model.fit(sts, 
                                qs_targets_from_state, 
                                batch_size=len(sts),
                                verbose=0, 
                                epochs=1,
                                callbacks=cbks)

        self.total_t += 1

        if time_step.last():
            self.new_episode()

        return info.history['loss'][0]

    def save(self):

        if os.path.exists(self.dict_loc):
            with open(self.dict_loc, 'rb') as df:
                loaded_dict = pickle.load(df)
            solved_on = loaded_dict["solved_on"]
        else:
            solved_on = self.solved_on

        save_dict = {"experiment_dir": self.experiment_dir,
                     "total_t": self.total_t,
                     "solved_on": solved_on,
                     "losses": self.losses,
                     "scores": self.scores,
                     "ep_lengths": self.ep_lengths}

        with open(self.dict_loc, 'wb') as df:
            pickle.dump(save_dict, df)

    def load(self):

        if not os.path.exists(self.dict_loc):
            print("Nothing saved at ", self.dict_loc, "yet!")
            return False

        with open(self.dict_loc, 'rb') as df:
            loaded_dict = pickle.load(df)

        self.q.load_and_set_cp(loaded_dict["experiment_dir"])

        self.total_t = loaded_dict["total_t"]
        self.solved_on = loaded_dict["solved_on"]
        self.losses = loaded_dict["losses"]
        self.ep_lengths = loaded_dict["ep_lengths"]
        self.scores = loaded_dict["scores"]

        return True

    def display_param_dict(self):
        if os.path.exists(self.dict_loc):
            with open(self.dict_loc, 'rb') as df:
                loaded_dict = pickle.load(df)
            
            to_display_dict = loaded_dict.copy()

            for k in ("losses", "ep_lengths", "scores"):
                to_display_dict[k] = str(len(loaded_dict[k])) + " items"

            pprint.pprint(to_display_dict)

        else:
            print("No params saved yet!")


