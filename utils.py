import numpy as np
import mss
from keras.layers import Layer
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from keras.models import Model
import easyocr

READER = easyocr.Reader(['en'])


class NoisyDense(Layer):
    '''
    NoisyNet Layer (using Factorised Gaussian noise)
    (Fortunato et al. 2017)
    '''
    def __init__(self, units=32):
        super(NoisyDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        # Initializer of \mu and \sigma
        mu_init = tf.random_uniform_initializer(minval=-1 * 1 / np.power(input_shape[1], 0.5),
                                                maxval=1 * 1 / np.power(input_shape[1], 0.5))
        sigma_init = tf.constant_initializer(0.5 / np.power(input_shape[1], 0.5))

        # creating weights
        self.w_mu = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=mu_init,
            name="w_mu",
            trainable=True,
        )
        self.w_sigma = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=sigma_init,
            name="w_sigma",
            trainable=True,
        )

        self.b_mu = self.add_weight(
            shape=(self.units,),
            initializer=mu_init,
            name="b_mu",
            trainable=True,
        )

        self.b_sigma = self.add_weight(
            shape=(self.units,),
            initializer=sigma_init,
            name="b_sigma",
            trainable=True,
        )

    def call(self, inputs):
        p = self.sample_noise([inputs.shape[-1], 1])
        q = self.sample_noise([1, self.units])
        f_p = self.f(p)
        f_q = self.f(q)
        w_epsilon = f_p * f_q
        b_epsilon = tf.squeeze(f_q)
        # w = w_mu + w_sigma*w_epsilon
        w = self.w_mu + tf.multiply(self.w_sigma, w_epsilon)
        # w*x
        ret = tf.matmul(inputs, w)
        # bias
        b = self.b_mu + tf.multiply(self.b_sigma, b_epsilon)
        # y = w*x + b
        return ret + b

    @tf.function
    def sample_noise(self, shape):
        noise = tf.random.normal(shape)
        return noise

    @tf.function
    def f(self, x):
        return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))

def screenshot():
    mon = {'top': 150, 'left': 0, 'width': 400, 'height': 600}
    with mss.mss() as sct:
        state = np.array(sct.grab(mon))
    state = cv2.cvtColor(state, cv2.COLOR_BGRA2RGB)
    state = cv2.resize(state, (120, 80))
    state = state/255.
    return state


def read_coins() -> int:
    coin_portion = {'top': 75, 'left': 380, 'width': 80, 'height': 60}
    with mss.mss() as sct:
        img = np.array(sct.grab(coin_portion))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # rgb to hsv color space

    s_ch = hsv_img[:, :, 1]  # Get the saturation channel
    # Apply threshold - pixels above 5 are going to be 255, other are zeros.
    thesh = cv2.threshold(s_ch, 5, 255, cv2.THRESH_BINARY)[1]
    # Apply opening morphological operation for removing artifacts.
    thesh = cv2.morphologyEx(thesh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    # Fill the background in thesh with the value 128 (pixel in the foreground stays 0.
    cv2.floodFill(thesh, None, seedPoint=(0, 0), newVal=128, loDiff=1,
                  upDiff=1)
    # Set all the pixels where thesh=128 to black.
    img[thesh == 128] = (0, 0, 0)
    coins = READER.readtext(img)
    if len(coins) > 0:
        coins = int(coins[0][1])
    else:
        coins = 0
    return coins


def moving_average(x, n=4):
    return np.convolve(x, np.ones(n), 'valid') / n


def adjust_reward_and_memorize(experience: list[list], agent, look_back: int = 5) -> None:
    for i, exp in enumerate(experience):

        if i > len(experience) - look_back:
            exp[2] = -0.7

        agent.memorize(*exp)


def visualize_cnn(model, img):
    img = np.expand_dims(img, 0)
    output = [model.layers[1].output]
    cnn_model = Model(model.inputs, output)
    features = cnn_model.predict(img)

    for ftr in features:
        fig = plt.figure(figsize=(12, 12))
        for i in range(1, 33):
            fig = plt.subplot(16, 16, i)
            fig.set_xticks([])
            fig.set_yticks([])
            plt.imshow(ftr[0, :, :, i-1])
        plt.show()
