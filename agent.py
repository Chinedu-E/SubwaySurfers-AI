from keras.layers import Dense,\
    Conv2D, Input, LeakyReLU, Dropout, Lambda, GlobalAvgPool2D
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
from utils import NoisyDense
from buffer import MemoryBuffer


class Agent:
    def __init__(self, discount_rate=0.95, learning_rate=6e-5,
                 with_per=True, batch_size=129,
                 epsilon=0.0, epsilon_decay=0.0,
                 epsilon_min=0.0, target_update_freq=10):

        self.input_dim = (150, 150, 1)
        self.eps_decay = epsilon_decay
        self.eps_min = epsilon_min
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.with_per = with_per
        self.gamma = discount_rate
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq

        model = self._build_model()
        target_model = self._build_model()
        target_model.set_weights(model.get_weights())
        # model.load_weights(".h5")
        self.model = model
        self.target_model = target_model
        self.buffer = MemoryBuffer(100000, with_per)
        # Print the model summary if you want to see what it looks like
        print(self.model.summary())

        self.loss = []
        self.location = 0

    def _build_model(self):
        input_layer = Input(self.input_dim)
        x = Conv2D(filters=32, kernel_size=8
                   , strides=1, padding='same')(input_layer)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=64, kernel_size=5, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = GlobalAvgPool2D()(x)
        x = Dense(64)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(rate=0.1)(x)
        value_fc = NoisyDense(32)(x)
        value = NoisyDense(1)(value_fc)

        advantage_fc = NoisyDense(32)(x)
        advantage = NoisyDense(5)(advantage_fc)

        def d_output(args):
            a = args[0]
            v = args[1]
            return v + tf.math.subtract(a, tf.math.reduce_mean(a, axis=1, keepdims=True))

        output = Lambda(d_output)([advantage, value])

        model = Model(input_layer, output)
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.learning_rate, epsilon=1.5e-4))

        return model

    def make_action(self, obs):
        """ predict next action from Actor's Policy
        """
        obs = np.expand_dims(obs, axis=0)
        qvals = self.predict(obs)[0]
        # Exploitation
        action = np.argmax(qvals)  # Highest Q-value

        return action

    def memorize(self, obs, act, reward, done, new_obs):
        """store experience in the buffer"""
        if self.with_per:
            nobs = np.expand_dims(obs, 0)
            q_val = self.predict(nobs)
            q_val_t = self.target_predict(nobs)
            new_val = reward + self.gamma * q_val_t
            td_error = abs(new_val - q_val)[0]
        else:
            td_error = 0
        self.buffer.memorize(obs, act, reward, done, new_obs, td_error)

    def replay(self, replay_num_):
        if self.with_per and (self.buffer.size() <= self.batch_size): return
        losses = []

        for _ in range(replay_num_):
            # sample from buffer
            states, actions, rewards, dones, new_states, idx = self.sample_batch(self.batch_size)

            # get target q-value using target network
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)
            q_vals = self.target_predict(new_states)

            # bellman iteration for target critic value
            critic_target = np.asarray(q_vals)
            for i in range(q_vals.shape[0]):
                if dones[i]:
                    critic_target[i] = rewards[i]
                else:
                    critic_target[i] = rewards[i] + self.gamma * np.max(q_vals[i])

                if self.with_per:
                    self.buffer.update(idx[i], abs(q_vals[i] - critic_target[i])[0])

            # train(or update) the actor & critic and target networks
            loss = self.update_network(states, critic_target)
            losses.append(loss)
        return losses

    def sample_batch(self, batch_size):
        """ Sampling from the batch
        """
        return self.buffer.sample_batch(batch_size)

    def update_network(self, obs, critic_target):
        history = self.model.fit(
            obs, critic_target, epochs=1, verbose=0)
        loss = history.history.get("loss")[0]
        return loss

    def predict(self, state):
        qval = self.model.predict(state, verbose=0)
        return qval

    def target_predict(self, state):
        qval = self.target_model.predict(state, verbose=0)
        return qval

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def on_episode_end(self):
        """Performing epsilon greedy parameter decay
        """
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_decay
        else:
            self.epsilon = self.eps_min

    def save_weights(self, path):
        self.model.save_weights(f"{path}.h5")
        self.target_model.save_weights(f"{path}_target.h5")

    def load_weights(self, path):
        self.model.load_weights(f"{path}.h5")
        self.target_model.load_weights(f"{path}_target.h5")
