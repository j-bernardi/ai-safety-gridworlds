import os
import pickle
import pprint
import numpy as np
import tensorflow as tf
from collections import namedtuple, deque

# import keras
# from keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation # , Lambda, Activation

EpisodeStats = namedtuple("EpisodeStats", ["episode_lengths", "episode_rewards"])
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class StandardAgent(object):
    
    def __init__(self, world_shape, actions_num, env, frames_state=2, experiment_dir=None, replay_memory_size=1500, replay_memory_init_size=500, update_target_estimator_every=250, discount_factor=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=50000, batch_size=32, checkpoint=False):

        self.world_shape = world_shape
        self.actions_num = actions_num
        self.frames_state = frames_state

        self.checkpoint = checkpoint
        self.experiment_dir = experiment_dir
        self.dict_loc = self.experiment_dir + "/param_dict.p"

        self.update_target_estimator_every = update_target_estimator_every
        self.replay_memory = deque(maxlen=replay_memory_size) # []
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        self.epsilon_decay_steps = epsilon_decay_steps
        self.discount_factor = discount_factor
        self.batch_size = batch_size

        self.solved_on = None
        self.total_t = 0
        self.scores, self.ep_lengths, self.losses = [], [], []

        # RETREIVE PROGESS if it exists
        if checkpoint:
            
            self.checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
            self.checkpoint_path = os.path.join(self.checkpoint_dir, "model")
            if not os.path.exists(self.checkpoint_dir):
                print("Making path", self.checkpoint_dir)
                os.makedirs(self.checkpoint_dir)

            self.load()

        self.new_episode()
        
        # Initialise the replay memory with random steps
        self.initialise_replay(env, replay_memory_init_size)

    def initialise_replay(self, env, replay_memory_init_size):
        time_step = env.reset()
        state = self.get_state(time_step.observation)
        for i in range(replay_memory_init_size):
            action = self.act_random(time_step.observation, eps=0.95)
            time_step = env.step(action)
            next_state = self.get_state(time_step.observation)
            done = time_step.last()

            assert state is not None
            assert next_state is not None
            self.replay_memory.append(Transition(state, np.int32(action), np.float32(time_step.reward), next_state, done))
            if done:
                time_step = env.reset()
                self.new_episode()
                state = self.get_state(time_step.observation)
            else:
                state = next_state

    def new_episode(self):
        self.prev_state = None

    def get_state(self, obs):
        frame = np.moveaxis(obs['RGB'], 0, -1)
        frame = tf.squeeze(tf.image.rgb_to_grayscale(frame))
        frame = tf.dtypes.cast(frame, tf.float32) / 255.0

        if self.prev_state is None:
            state = np.stack([frame] * self.frames_state, axis=2)
        else:
            state = np.stack([self.prev_state[:,:,self.frames_state - 1], frame], axis=2)
        return state

    def save_param_dict(self):
        """Save params only!
        Model saving should be implemented by child class
        """

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

    def load_param_dict(self):

        if not os.path.exists(self.dict_loc):
            print("Nothing saved at ", self.dict_loc, "yet!")
            return False

        with open(self.dict_loc, 'rb') as df:
            loaded_dict = pickle.load(df)
        
        # Restore some params
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


class StandardEstimator(object):
    """
    Q-Value Estimator neural network.
    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, actions_num, x_shape, y_shape, frames_state, name="estimator"):
        
        self.model_name = name
        
        self.actions_num = actions_num
        self.x_shape = x_shape 
        self.y_shape = y_shape
        self.frames_state = frames_state

        self.model = self._build_model()

        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=0.00025,
            rho=0.99, 
            momentum=0.0, 
            epsilon=1e-6)

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        Takes states to predicted action's value
        """

        print("BUILDING MODEL", self.model_name)

        model = tf.keras.Sequential()

        # model.add(Lambda(lambda x : tf.dtypes.cast(x, tf.float32) / 255.0))

        model.add(Conv2D(filters=64, kernel_size=2, 
                         strides=1, padding='SAME',
                         input_shape=(self.x_shape, 
                                      self.y_shape,
                                      self.frames_state),
                         data_format="channels_last",
                         activation='relu'
                         ))
        # model.add(Activation('relu'))

        # Fully connected layers
        model.add(Flatten())
        model.add(Dense(64, activation='linear'))
        model.add(Dense(self.actions_num, activation='linear')) # predictions

        return model

    def save_model(self, checkpoint_path, verbose=False):
        """
        Simply saves the model to the checkpoint path
        """
        if verbose:
            print("Saving model to", checkpoint_path, "...", end="")
        
        self.model.save_weights(checkpoint_path)
        
        if verbose:
            print("Saved")

    def load_model_cp(self, checkpoint_path):
        """
        Load the estimator's model state.
        If no checkpoint exists, initialise from scratch
        """

        try:
            self.model.load_weights(checkpoint_path)
            print("Loaded model checkpoint {}...".format(checkpoint_path))
        except Exception as e:
            if "Failed to find any matching file" in str(e):
                print("No checkpoint", checkpoint_path, "detected! "
                      "Initializing model from scratch")
            else:
                raise e
