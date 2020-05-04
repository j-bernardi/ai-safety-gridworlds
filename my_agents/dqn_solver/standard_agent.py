import os
import pickle
import pprint
import numpy as np
import tensorflow as tf
from collections import namedtuple, deque
import keras

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D

EpisodeStats = namedtuple("EpisodeStats", ["episode_lengths", "episode_rewards"])
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class StandardAgent(object):
    
    def __init__(self, world_shape, actions_num, env, frames_state=2, experiment_dir=None, replay_memory_size=1500, replay_memory_init_size=500, update_target_estimator_every=250, discount_factor=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=50000, batch_size=32, checkpoint=False):

        self.world_shape = world_shape
        self.actions_num = actions_num
        self.frames_state = frames_state
        
        self.experiment_dir = experiment_dir
        self.dict_loc = self.experiment_dir + "/param_dict.p"

        self.replay_memory = deque(maxlen=replay_memory_size) # []
        # self.replay_memory_size = replay_memory_size
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
        
        # Restore the q net
        self.load_q_net_weights(loaded_dict)
        
        # Restore some params
        self.total_t = loaded_dict["total_t"]
        self.solved_on = loaded_dict["solved_on"]
        self.losses = loaded_dict["losses"]
        self.ep_lengths = loaded_dict["ep_lengths"]
        self.scores = loaded_dict["scores"]

        return True

    def load_q_net_weights(self):
        raise NotImplementedError("Needs to be implemented by child class")

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

    def __init__(self, actions_num, x_shape, y_shape, frames_state, name="estimator", experiment_dir=None, checkpoint=True):
        self.model_name = name
        self.actions_num = actions_num
        self.x_shape = x_shape 
        self.y_shape = y_shape
        self.frames_state = frames_state

        self.model = self._build_model()

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