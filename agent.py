from keras.layers import Dense,\
    Conv2D, Input, LeakyReLU, Dropout, Lambda, GlobalAvgPool2D
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
from utils import NoisyDense
from buffer import MemoryBuffer


from keras.layers import Dense, \
    Flatten, Input, LeakyReLU, Lambda, ReLU, ConvLSTM2D
from keras.models import Model
from keras.losses import MSE
from keras.optimizers import Adam
import numpy as np
import gym
import tensorflow as tf
from utils import NoisyDense
from buffer import MemoryBuffer


class Agent:
    def __init__(self, n_frames_stack: int,
                 env: gym.Env,
                 discount_rate=0.99,
                 learning_rate=1e-5,
                 with_per: bool = True,
                 batch_size: int = 32,
                 epsilon=0.0,
                 epsilon_decay=0.0,
                 epsilon_min=0.0,
                 target_update_freq: int = 4000,
                 buffer_size: int = 100000,
                 optimizer_epsilon: float = 1.5e-4
                 ):
        self.n_frames_stack = n_frames_stack
        # shape of (n_stacked_frames, width, height, channels)
        self.input_dim = (n_frames_stack, *env.observation_space.shape)
        self.n_actions = env.action_space.shape[0]
        self.eps_decay = epsilon_decay
        self.eps_min = epsilon_min
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.with_per = with_per
        self.gamma = discount_rate
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq
        self.optimizer_epsilon = optimizer_epsilon

        model = self._build_model()
        target_model = self._build_model()
        target_model.set_weights(model.get_weights())
        
        try:
            model.load_weights("weights/dqn")
            print("loaded existing weights")
        except FileNotFoundError:
            print("training from scratch")
            
        self.model = model
        self.target_model = target_model
        self.buffer = MemoryBuffer(buffer_size, with_per)
        # Print the model summary if you want to see what it looks like
        print(self.model.summary())

    def _build_model(self):
        input_layer = Input(self.input_dim)
        x = ConvLSTM2D(filters=32,
                       kernel_size=8,
                       strides=4,
                       return_sequences=True,
                       padding="same")(input_layer)
        x = ConvLSTM2D(filters=64,
                       kernel_size=4,
                       strides=2,
                       return_sequences=True,
                       padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = ConvLSTM2D(filters=64,
                       kernel_size=3,
                       strides=1,
                       return_sequences=False,
                       padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Flatten()(x)
        x = Dense(512)(x)
        x = LeakyReLU(alpha=0.2)(x)
        value_fc = NoisyDense(128)(x)
        value_fc = ReLU()(value_fc)
        value = NoisyDense(1)(value_fc)

        advantage_fc = NoisyDense(128)(x)
        advantage_fc = ReLU()(advantage_fc)
        advantage = NoisyDense(self.n_actions)(advantage_fc)
        self.advantage = Model(input_layer, advantage)

        def d_output(args):
            a = args[0]
            v = args[1]
            return v + tf.math.subtract(a, tf.math.reduce_mean(a, axis=1, keepdims=True))

        output = Lambda(d_output)([advantage, value])

        model = Model(input_layer, output)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate, epsilon=self.optimizer_epsilon),
                      loss="mse")

        return model

    def make_action(self, obs):
        """ predict next action from Actor's Policy
        """
        obs = np.expand_dims(obs, axis=0)
        qvals = self.predict(obs, advantage=True)[0]
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

            states = tf.convert_to_tensor(states, dtype=tf.float32)
            new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)

            q_pred = self.predict(states)
            q_vals = self.target_predict(new_states)
            past_actions = self.predict(new_states)
            max_actions = tf.math.argmax(past_actions, axis=1)

            target = q_pred.copy()
            for i in range(q_vals.shape[0]):
                if dones[i]:
                    target[i] = rewards[i]
                else:
                    target[i] = rewards[i] + self.gamma * q_vals[i, max_actions[i]]

                if self.with_per:
                    self.buffer.update(idx[i], abs(q_vals[i] - target[i])[0])

            # training
            loss = self.update_network(states, target)
            losses.append(loss)
        return losses

    def sample_batch(self, batch_size):
        """ Sampling from the batch
        """
        return self.buffer.sample_batch(batch_size)

    def update_network(self, obs, target):
        """Train Q-network for critic on sampled batch
        """
        loss = self.model.train_on_batch(obs, target)
        return loss

    def predict(self, state, advantage: bool = False):
        if advantage:
            qval = self.advantage.predict(state, verbose=0)
        else:
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
