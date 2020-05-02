# Structure implemented from author
# https://colab.research.google.com/drive/1AEcyCKQgXhMP5AMHu00bYWhQe6zo8JWG#scrollTo=ceii7gkPLl5M

import numpy as np

import tensorflow_probability

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K


from tensorflow.keras.layers import Dense

device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

print("tf version", tf.__version__)

K.set_floatx('float64')

class ValueModel(keras.Model):
    """
    A model trained to predict a reward (single value) 
    given a state
    """
    
    def __init__(self, state_processor):
        super.__init__('value_model')
        
        # Function
        self.process_states = state_processor

        self.l1 = Dense(128, activation='elu')
        self.l2 = Dense(256, activation='elu')
        self.l3 = Dense(128, activation='elu')
        self.out_layer = Dense(1, activation='elu')

    @staticmethod
    def distributionsFromLocsScales(locs_scales):
        norm = tensorflow_probability.distributions.Normal
        return norm(loc=locs_scales[...,:1],
                    scale=1e-5 + K.softplus(locs_scales[...,1:]))

    def valueNet(self, states):

        x = states
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return self.out_layer

    @tf.function
    def call(self, states):
        processed_states = self.process_states(states)
        return self.valuesNet(processed_states)

    @tf.function
    def oneValue(self, state):
        return self(state[None,:])[0, 0]

class ActionModel(keras.Model):

    def __init__(self, state_processor, num_actions):

        super().__init__('action_model')
        
        # Function
        self.process_states = state_processor
        
        self.num_actions = num_actions

        self.l1 = Dense(32, activation='elu')
        self.l2 = Dense(64, activation='elu')
        self.l3 = Dense(32, activation='elu')
        self.l4 = Dense(2, activation='linear')
        self.out_layer = tensorflow_probability.layers.DistributionLambda(
            make_distribution_fn=distributionsFromLocsScales,
            convert_to_tensor_fn=self.limitedSample
        )

    def limitedSample(self, distributions):
        actions = K.squeeze(distributions.sample(1), axis=0)

    def actionDistributionsNet(self, states):
        x = self.l1(states)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return self.out_layer(x)  # distribution layer

    def distributions(self, states):
        processed_states = self.process_states(states) # (states - self.norm_loc) / self.norm_scale
        return self.actionDistributionsNet(processed_states)

    @tf.function  # returns sample, remove to return distribution
    def call(self, states):
        return self.distributions(states)

    @tf.function  # SUPERIMPORTANT use tf.function for speed
    def oneAction(self, state):
        return self(state[None, ...])[0, 0]

    def oneDistribution(self, state):
        return self.distributions(state[None, ...])

    def wideAction(self):
        # Assuming action range 0->num_actions
        # Takes an action either close to 0 or close to 1
        # Actually this is for cts
        return np.random.beta(0.01, 0.01) * (self.num_actions-1)


class Agent(object):

    # TODO - work through actor in 
    # https://colab.research.google.com/drive/1AEcyCKQgXhMP5AMHu00bYWhQe6zo8JWG#scrollTo=1d1iIL5_LwYC

    def __init__(self, num_actions, value_factor=1.0, lr_action=0.00001, lr_value=0.001, gamma=0.99):
        
        self.num_actions = num_actions
        self.gamma = gamma

        self.action_net = ActionModel(self.process_states, self.num_actions)
        self.action_optimiser = tf.keras.optimizers.Adam(learning_rate=lr_action)
        self.action_train_loss = tf.keras.metrics.Mean(name='action_train_loss')

        self.value_net = ValueModel(self.process_states)
        self.value_optimiser = tf.keras.optimizers.Adam(learning_rate=lr_value)
        self.value_train_loss = tf.keras.metrics.Mean(name='value_train_loss')

    @staticmethod
    def process_states(states):
        # TODO - process states if wish - something like a norm or greyscaling
        processed_states = states
        return processed_states

    def ValueLoss(self, values_true, values_pred):
        return self.value_factor * K.sum(K.square(values_true - values_pred))

    def actionLoss(self, actions_deltas_true, distribution):
        actions = actions_deltas_true[:, 0]
        # TODO is this actually categorical
        deltas = actions_deltas_true[:, 1]
        log_losses = -distribution.log_prob(actions)
        policy_loss = K.sum(log_losses * deltas)
        # Entropy gain here if wanted
        return policy_loss

    @tf.function
    def valueTrainStep(self, one_state, value_pseudo_true):
        with tf.GradientTape() as tape:
            value_pred = self.value_net(one_state)
            loss = self.ValueLoss(value_pseudo_true, value_pred)

        gradients = tape.gradient(loss, self.value_model.trainable_variables)

        self.value_optimiser.apply_gradients(zip(gradients, self.value_model.trainable_variables))

        self.value_train_loss(loss)

    def train(self):
        pass






