from keras.layers import Dense, Flatten, Conv2D, Input, BatchNormalization, LeakyReLU, Activation, Dropout, Lambda
from keras.models import Model
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import random
from buffer import MemoryBuffer


class Agent:
    def __init__(self, discount_rate=0.99, learning_rate=1e-4, with_per=True, batch_size=64, epsilon=0.95, epsilon_decay=0.05, epsilon_min=0.05):

        self.input_dim = (150, 150, 1)
        self.eps_decay = epsilon_decay
        self.eps_min = epsilon_min
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.with_per = with_per
        self.gamma = discount_rate
        self.learning_rate = learning_rate

        model = self._build_model()
        target_model = self._build_model()
        target_model.set_weights(model.get_weights())
        # model.load_weights(".h5")
        self.model = model
        self.target_model = target_model
        self.buffer = MemoryBuffer(10000, with_per)
        # Print the model summary if you want to see what it looks like
        # print(self.model.summary())

        self.loss = []
        self.location = 0

    def _build_model(self):
        input_layer = Input(self.input_dim)
        x = Conv2D(filters=32, kernel_size=3
                   , strides=1, padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(128, kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(rate=0.1)(x)
        value = Dense(1, activation='linear', name="value")(x)
        advantage = Dense(5, activation='linear')(x)

        def d_output(args):
            a = args[0]
            v = args[1]
            return v + tf.math.subtract(a, tf.math.reduce_mean(a, axis=1, keepdims=True))

        output = Lambda(d_output)([advantage, value])

        model = Model(input_layer, output)
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def make_action(self, obs):
        """ predict next action from Actor's Policy
        """
        obs = np.expand_dims(obs, axis=0)
        qvals = self.predict(obs)[0]

        rnd = np.random.uniform()
        if rnd < self.epsilon:
            # Exploration
            action = random.choice(list(range(5)))
        else:
            # Exploitation
            action = np.argmax(qvals)  # Highest Q-value

        return action

    def memorize(self, obs, act, reward, done, new_obs):
        """store experience in the buffer"""
        if self.with_per:
            q_val = self.predict(obs)
            q_val_t = self.target_predict(obs)
            new_val = reward + self.gamma * q_val_t
            td_error = abs(new_val - q_val)[0]
        else:
            td_error = 0
        self.buffer.memorize(obs, act, reward, done, new_obs, td_error)

    def replay(self, replay_num_):
        if self.with_per and (self.buffer.size() <= self.batch_size): return

        for _ in range(replay_num_):
            # sample from buffer
            states, actions, rewards, dones, new_states, idx = self.sample_batch(self.batch_size)

            # get target q-value using target network
            q_vals = self.target_predict(new_states)

            # bellman iteration for target critic value
            critic_target = np.asarray(q_vals)
            for i in range(q_vals.shape[0]):
                if dones[i]:
                    critic_target[i] = rewards[i]
                else:
                    critic_target[i] = rewards[i] + self.gamma * np.max(q_vals[i])

                if self.with_per:
                    self.buffer.update(idx[i], abs(q_vals[i] - critic_target[i]))

            # train(or update) the actor & critic and target networks
            self.update_network(states, critic_target)

    def sample_batch(self, batch_size):
        """ Sampling from the batch
        """
        return self.buffer.sample_batch(batch_size)

    def update_network(self, obs, critic_target):
        history = self.model.fit(
            obs, critic_target, epochs=1, verbose=0)
        loss = history.history.get("loss")[0]
        print("LOSS: ", loss)

    def predict(self, state):
        qval = self.model.predict(state)
        return qval

    def target_predict(self, state):
        qval = self.target_model.predict(state)
        return qval

    def on_episode_end(self):
        """Performing epsiolon greedy parameter decay
        """
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_decay
        else:
            self.epsilon = self.eps_min
